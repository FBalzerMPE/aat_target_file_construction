from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, unique, vstack
from astroquery.simbad import Simbad
from matplotlib.axes import Axes

from .helper_functions import (add_ra_dec_hms_dms_columns, calc_pm_tot,
                               convert_radec_to_hmsdms,
                               filter_for_existing_cols, filter_for_stars,
                               get_objects_in_circular_region)
from .paths import PATHS

##############################
# - Science targets:


def _sanitize_cluster_members(member_table: Table) -> Table:
    """Retrieve the vital information for all members associated with this cluster id"""
    member_table["obj_name"] = [
        f"erosource_{member['id']}" for member in member_table]
    # The magnitudes are given as triplets in the [mag] column (as grz), so we select the middle one:
    member_table["rmag"] = member_table["mag"][:, 1]
    # Since we're observing galaxies, their proper motions are 0:
    member_table["pmra"] = 0
    member_table["pmdec"] = 0
    relevant_cols = ["ra", "dec", "obj_name", "rmag", "pmra", "pmdec"]
    return member_table[relevant_cols]


def _sanitize_agn_candidates(agn_table: Table) -> Table:
    """Retrieve the vital information for all agn for this observation"""
    old_names = ["ERO_ID", "LS10_RA", "LS10_DEC", "RMAG"]
    new_names = ["obj_name", "ra", "dec", "rmag"]
    agn_table.rename_columns(old_names, new_names)
    # Since we're observing galaxies, their proper motions are 0:
    agn_table["pmra"] = 0
    agn_table["pmdec"] = 0
    relevant_cols = ["ra", "dec", "obj_name", "rmag", "pmra", "pmdec"]
    return agn_table[relevant_cols]

########################
# - White dwarfs:


def _sanitize_jacobs_white_dwarf_table(table: Table) -> Table:
    relevant_cols = ["_RAJ2000", "_DEJ2000",
                     "WD", "Source", "pmRA", "pmDE", "RPmag"]
    table.keep_columns(relevant_cols)
    # ! Note: We rename RPmag to rmag even though it's not the same to have more uniform names throughout
    new_names = ["ra", "dec", "obj_name", "source_id", "pmra", "pmdec", "rmag"]
    table.rename_columns(relevant_cols, new_names)
    return table

##########################
# - Simbad:


def _sanitize_simbad_table(table: Table) -> Table:
    # Sanitize the table by lowercasing colnames and turning ra and dec to degrees
    table.rename_columns(cols := table.colnames, [col.lower() for col in cols])
    sky_coords = SkyCoord(
        table["ra"], table["dec"], unit=(u.hourangle, u.deg))
    table["ra"] = sky_coords.ra
    table["dec"] = sky_coords.dec
    table.rename_column("flux_r", "rmag")
    table.rename_column("main_id", "obj_name")
    return table


def _retrieve_simbad_table(ra: float, dec: float, radius: float = 1) -> Table:
    """Retrieve the simbad sources around the given central `ra` and `dec` within
    the provided `radius` (in deg).
    Also obtain proper motion and r magnitude information.
    """
    custom_simbad = Simbad()
    # print(custom_simbad.get_votable_fields())
    custom_simbad.add_votable_fields(
        "fluxdata(r)", "pmra", "pmdec", "otype", "morphtype", "otypes", "otype(opt)", "rvz_type", "sptype")
    sources = custom_simbad.query_region(
        SkyCoord(ra, dec, unit="deg"), radius=radius * u.deg)
    return sources

###########################
# - Sky fibres:


def _sanitize_fibre_table(table: Table) -> Table:
    # Sanitize the table by lowercasing colnames and turning ra and dec to degrees
    table.rename_columns(cols := table.colnames, [col.lower() for col in cols])
    sky_coords = SkyCoord(table["ra"], table["dec"], unit="deg")
    table["ra"] = sky_coords.ra
    table["dec"] = sky_coords.dec
    table["obj_name"] = [f"skyfibre_{i}" for i in range(len(table))]
    table["rmag"] = 30
    table["pmra"] = 0
    table["pmdec"] = 0
    return table


#####################
# - Adding information

@np.vectorize
def _clean_object_name(name: str) -> str:
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    return name.replace(" ", "_")


def _sanitize_table_for_observation(table, obs_type: Literal["P", "S", "F"], priority=9) -> Table:
    # [P = 'Science targets', F = 'Guide stars', S = 'Sky fibres']
    assert obs_type in "PSF", f"Please choose a valid observation type and not {obs_type}"
    table["obj_name"] = _clean_object_name(table["obj_name"])
    table["obs_type"] = obs_type
    table["priority"] = priority
    # The program ID does not matter for us, but we need to provide it
    table["program_id"] = 0
    table = add_ra_dec_hms_dms_columns(table)
    relevant_cols = ["obj_name", "ra_hms", "dec_dms", "obs_type",
                     "priority", "rmag", "program_id", "pmra", "pmdec"]
    return table[relevant_cols]

#######################################
# - The actual TargetContainer class:


@dataclass
class TargetContainer:
    container_id: int
    obs_ra: float
    obs_dec: float
    selection_radius: float = 1.0  # The radius in degrees
    cluster_members: Optional[Table] = None
    agn_candidates: Optional[Table] = None
    white_dwarfs: Optional[Table] = None
    guide_stars: Optional[Table] = None
    sky_fibres: Optional[Table] = None

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def __len__(self):
        return sum([len(t) for t in self.get_available_tables().values()])

    def get_available_tables(self) -> dict[str, Table]:
        """Get all of the currently available tables."""
        keys = ["cluster_members", "agn_candidates", "white_dwarfs",
                "guide_stars", "sky_fibres"]
        return {key: table for key in keys if (table := self[key]) is not None}

    def get_cluster_members(self, mag_r_min=17.5, verbose=True):
        members = Table.read(PATHS.members)
        # This selection only works due to our setup:
        members = members[members["mem_match_id"] == self.container_id]
        members = _sanitize_cluster_members(members)
        count_initial = len(members)
        # Discard all rows where no rmag, pmra or pmdec information is available:
        members = filter_for_existing_cols(members)
        count_clean = len(members)
        # Discard WDs brighter than the [mag_r_min]
        brightness_cut = members["rmag"] >= mag_r_min
        members = members[brightness_cut]
        count_brightness_cut = len(members)

        self.cluster_members = members
        print(
            f"[{self.container_id}] {count_brightness_cut} science targets (cluster members) have been registered.")
        if not verbose:
            return
        info_string = f"\t{count_initial = }\n\t{count_clean = }"
        info_string += f"\n\t{count_brightness_cut = }"
        print(info_string)

    def get_agn(self, verbose=True):
        agn_candidates = PATHS.read_table(PATHS.agn_candidates)
        agn_candidates = _sanitize_agn_candidates(agn_candidates)
        count_initial = len(agn_candidates)
        # Restrict to circular region around pointing:
        agn_candidates = get_objects_in_circular_region(
            agn_candidates, self.obs_ra, self.obs_dec, self.selection_radius)
        count_in_region = len(agn_candidates)
        mask = agn_candidates["rmag"] >= 17.5
        mask &= agn_candidates["rmag"] <= 21.5
        agn_candidates = agn_candidates[mask]
        count_brightness_cut = len(agn_candidates)

        self.agn_candidates = agn_candidates
        print(
            f"[{self.container_id}] {count_brightness_cut} more science targets (AGN) have been registered.")
        if not verbose:
            return
        info_string = f"\t{count_initial = }\n\t{count_in_region = }"
        info_string += f"\n\t{count_brightness_cut = }"
        print(info_string)

    def get_white_dwarfs(self, mag_r_min=17.5, verbose=True):
        """Load the white dwarfs we need to observe for spectral calibration.
        Then, perform selection based on:
            - Area (circular region around central coordinates)
            - Cleanliness (discard sources w/o RPmag or proper motion)
            - Brightness (discard sources that are too bright to get a proper spectrum)

        Parameters
        ----------
        mag_r_min : float, optional
            The maximum brightness a white dwarf should have, by default 17.5
        verbose : bool, optional
            Whether additional information on the selection should be printed, by default True
        """
        white_dwarfs = PATHS.read_table(PATHS.white_dwarfs_jacob)
        # Rename the columns and only keep the ones necessary.
        # ! Note: We renamed RPmag to rmag for consistency with the other tables!
        white_dwarfs = _sanitize_jacobs_white_dwarf_table(white_dwarfs)
        count_initial = len(white_dwarfs)
        # Restrict to circular region around pointing:
        white_dwarfs = get_objects_in_circular_region(
            white_dwarfs, self.obs_ra, self.obs_dec, self.selection_radius)
        count_in_region = len(white_dwarfs)
        # Discard all rows where no rmag, pmra or pmdec information is available:
        white_dwarfs = filter_for_existing_cols(white_dwarfs)
        count_clean = len(white_dwarfs)
        # Discard WDs brighter than the [mag_r_min]
        brightness_mask = white_dwarfs["rmag"] >= mag_r_min
        white_dwarfs = white_dwarfs[brightness_mask]
        count_faint = len(white_dwarfs)

        # - Select the three white dwarfs closest to the median magnitude of our sources
        white_dwarfs.sort("rmag")
        median_mag = np.median(self.cluster_members["rmag"])
        closest_index = np.abs(white_dwarfs["rmag"] - median_mag).argmin()
        good_white_dwarfs = white_dwarfs[closest_index:closest_index + 3]

        count_good = len(good_white_dwarfs)

        self.white_dwarfs = good_white_dwarfs
        print(
            f"[{self.container_id}] {count_good} white dwarfs have been registered.")
        if not verbose:
            return
        info_string = f"\t{count_initial = }\n\t{count_in_region = }\n"
        info_string += f"\t{count_clean = }\n\t{count_faint = }"
        print(info_string)

    def get_guide_stars(self, mag_r_min: float, mag_r_max: float, pm_max: float, verbose=True):
        """Query the simbad database for the guide stars in a circular region around the central
        coordinates.
        Then, perform selection based on:
            - Cleanliness (discard sources w/o rmag or proper motion)
            - Magnitude and PM requirements (see parameters)
            - Type (discard all non-stellar sources)

        Parameters
        ----------
        mag_r_min : float
            The minimum magnitude a guide star should have
        mag_r_max : float
            The maximum magnitude a guide star should have
        pm_max : float
            The maximum proper motion a guide star should have
        verbose : bool, optional
            Whether additional information on the selection should be printed, by default True
        """
        simbad_sources = _retrieve_simbad_table(
            self.obs_ra, self.obs_dec, self.selection_radius)
        # Rename the columns and convert ra and dec to skycoord:
        simbad_sources = _sanitize_simbad_table(simbad_sources)
        simbad_sources["pm_tot"] = calc_pm_tot(
            simbad_sources["pmra"], simbad_sources["pmdec"])
        count_initial = len(simbad_sources)
        # Discard all rows where no rmag, pmra or pmdec information is available:
        simbad_sources = filter_for_existing_cols(
            simbad_sources, ("rmag", "pmra", "pmdec"))
        count_clean = len(simbad_sources)
        # Perform the desired brightness cuts
        mask = simbad_sources["rmag"] <= mag_r_max
        mask &= simbad_sources["rmag"] >= mag_r_min
        mask &= simbad_sources["pm_tot"] <= pm_max
        simbad_sources = simbad_sources[mask]
        count_flux_limited = len(simbad_sources)
        # Discard all sources that are not guide stars
        simbad_sources = filter_for_stars(simbad_sources)
        count_stars_only = len(simbad_sources)
        self.guide_stars = simbad_sources
        print(
            f"[{self.container_id}] {count_stars_only} guide stars have been registered.")
        if not verbose:
            return
        info_string = f"\t{count_initial = }\n\t{count_clean = }\n"
        info_string += f"\t{count_flux_limited = }\n\t{count_stars_only = }"
        print(info_string)

    def get_sky_fibres(self, verbose=True):
        sky_fibres = PATHS.read_table(PATHS.sky_fibres)
        sky_fibres = _sanitize_fibre_table(sky_fibres)
        count_initial = len(sky_fibres)
        # Restrict to circular region around pointing:
        sky_fibres = get_objects_in_circular_region(
            sky_fibres, self.obs_ra, self.obs_dec, self.selection_radius)
        count_in_region = len(sky_fibres)

        # TODO: Perform random selection and exclude fibers that are close to each other!
        self.sky_fibres = sky_fibres

        print(
            f"[{self.container_id}] {count_in_region} sky fibres have been registered.")
        if not verbose:
            return
        info_string = f"\t{count_initial = }\n\t{count_in_region = }"
        print(info_string)

    def plot_sources_on_ax(self, ax: Axes, **kwargs):
        labels = self.get_available_tables().keys()
        colors = ["r", "b", "y", "g", "gray"]
        for color, label in zip(colors, labels):
            if (table := self[label]) is not None:
                label = label.replace("_", " ") + f" ({len(table)})"
                ax.scatter(table["ra"], table["dec"],
                           label=label, c=color, **kwargs)
        ax.scatter(self.obs_ra, self.obs_dec, s=10, marker="x", color="k",
                   label="centre")
        ax.set_xlabel("RA")
        ax.set_ylabel("DEC")
        ax.set_aspect(True)
        if not ax.xaxis_inverted():
            ax.invert_xaxis()
        ax.legend()
        ax.set_title(f"Observation {self.container_id}")

    def pprint(self):
        print(f"[{self.container_id}] at [{self.obs_ra:.2f}°, {self.obs_dec:.2f}°] with {len(self)} total objects.")
        for key, table in self.get_available_tables().items():
            print(f"{key:18}-> {len(table)} sources")

    def get_full_target_file(self):
        if len((keys := self.get_available_tables().keys())) < 4:
            diff = 4 - len(keys)
            raise UserWarning(f"You are missing {diff} tables.\n"
                              f"So far, only the following tables are available: {keys}")
        cluster_members = _sanitize_table_for_observation(
            self.cluster_members, "P", priority=8)
        agn_candidates = _sanitize_table_for_observation(
            self.agn_candidates, "P", priority=6)
        white_dwarfs = _sanitize_table_for_observation(
            self.white_dwarfs, "P")
        guide_stars = _sanitize_table_for_observation(
            self.guide_stars, "F")
        sky_fibres = _sanitize_table_for_observation(
            self.sky_fibres, "S")
        tables = cluster_members, agn_candidates, white_dwarfs, guide_stars, sky_fibres
        full_table = vstack(tables)
        count_before = len(full_table)
        full_table = unique(full_table, "obj_name")
        if (diff := count_before - len(full_table)) > 0:
            print(f"Removed {diff} duplicate source(s).")
        return full_table

    def get_fld_file_header(self) -> str:
        label = f"eromapper cluster follow-up: {self.container_id}"
        utdate = "2023 02 19"
        ra, dec = convert_radec_to_hmsdms(
            self.obs_ra, self.obs_dec, " ", precision=2)
        file_header = (f"LABEL {label}\nUTDATE {utdate}\nCENTRE {ra} {dec}\nEQUINOX J2000\n"
                       f"WLEN1 6000\nPROPER_MOTIONS")
        return file_header

    def write_targets_to_disc(self, fpath: Optional[Path] = None, overwrite=True):
        if fpath is None:
            fpath = PATHS.get_fld_fname(self.container_id)
        all_targets = self.get_full_target_file()
        all_targets.write(fpath, format="csv", delimiter="\t", overwrite=overwrite,
                          formats={"obj_name": lambda x: x.upper(), "rmag": "%.3f"})
        header = self.get_fld_file_header()
        # Now that the file has been written, we have to prepend the file header:
        with open(fpath, 'r+', encoding="utf8") as f:
            content = f.read()
            f.seek(0, 0)
            f.write(header + '\n\n# ' + content)
