import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Sequence

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, unique, vstack
from astropy.utils.metadata import MergeConflictWarning
from matplotlib.axes import Axes

from .helper_functions import (add_ra_dec_hms_dms_columns, calc_pm_tot,
                               convert_radec_to_hmsdms,
                               filter_for_existing_cols, get_legacysurvey_url,
                               get_objects_in_circular_region)
from .paths import PATHS
from .region import RectangularRegion

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
# - Guide stars:


def _sanitize_sweep_for_guide_stars(table: Table) -> Table:
    # Filter for pointlike objects:
    table = table[table["type"] == "PSF"]
    table.rename_columns(cols := table.colnames, [col.lower() for col in cols])
    table.rename_column("gaia_phot_rp_mean_mag", "rmag")
    table["obj_name"] = [f"G_{objid}" for objid in table["ref_id"]]
    relevant_cols = ["obj_name", "ra", "dec", "pmra", "pmdec", "rmag"]
    return table[relevant_cols]

###########################
# - Sky fibres:


def _generate_sky_fibres_from_sweep(table: Table) -> Table:
    num_initial_points = 15000
    # Set an initial seed to achieve reproducible random results:
    np.random.seed(0)
    random_ra = np.random.uniform(
        np.min(table["ra"]), np.max(table["ra"]), num_initial_points)
    random_dec = np.random.uniform(
        np.min(table["dec"]), np.max(table["dec"]), num_initial_points)
    random_coords = SkyCoord(random_ra, random_dec, unit="deg")
    sweep_coords = SkyCoord(table["ra"], table["dec"])
    # Find all random coords far away from sweep sources
    idx, d2d, _ = random_coords.match_to_catalog_sky(sweep_coords)
    min_distance_to_others = 10 * u.arcsec
    sky_fibres = random_coords[d2d > min_distance_to_others]

    # Find all random coords far away from each other to prevent clustering:
    idx, d2d, _ = sky_fibres.match_to_catalog_sky(sky_fibres, nthneighbor=2)
    min_distance_to_others = 1 * u.arcmin
    sky_fibres = sky_fibres[d2d > min_distance_to_others]
    # Convert the pairs into a proper astropy table:
    sky_fibres = Table([sky_fibres.ra, sky_fibres.dec], names=["ra", "dec"])
    return sky_fibres


def _sanitize_fibre_table(table: Table) -> Table:
    # Sanitize the table by lowercasing colnames and turning ra and dec to degrees
    table.rename_columns(cols := table.colnames, [col.lower() for col in cols])
    sky_coords = SkyCoord(table["ra"], table["dec"], unit="deg")
    table["ra"] = sky_coords.ra
    table["dec"] = sky_coords.dec
    table["obj_name"] = [
        f"skyfibre_{str(i).zfill(3)}" for i in range(len(table))]
    table["rmag"] = 30
    table["pmra"] = 0
    table["pmdec"] = 0
    return table


#####################
# - Unifying the table:

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
    # Convert proper motions from mas/yr to arcsec/yr
    table["pmra"] = table["pmra"] / 1000.
    table["pmdec"] = table["pmdec"] / 1000.
    relevant_cols = ["obj_name", "ra_hms", "dec_dms", "obs_type",
                     "priority", "rmag", "program_id", "pmra", "pmdec"]
    table = unique(table, "obj_name")
    table.sort("obj_name")
    return table[relevant_cols]

#######################################
# - The actual TargetContainer class:


@dataclass
class TargetContainer:
    container_id: int
    obs_ra: float
    obs_dec: float
    selection_radius: float = 1.0  # The radius in degrees
    region: RectangularRegion = field(init=False)
    ls_url: str = field(init=False)
    cluster_members: Optional[Table] = None
    agn_candidates: Optional[Table] = None
    white_dwarfs: Optional[Table] = None
    guide_stars: Optional[Table] = None
    sky_fibres: Optional[Table] = None

    def __post_init__(self):
        self.region = RectangularRegion.from_centre_and_radius(
            self.obs_ra, self.obs_dec, self.selection_radius)
        self.ls_url = get_legacysurvey_url(self.obs_ra, self.obs_dec)

    def __getitem__(self, key):
        try:
            return self.__getattribute__(key)
        except AttributeError as exc:
            avail = ["cluster_members", "agn_candidates", "white_dwarfs",
                     "guide_stars", "sky_fibres"]
            raise AttributeError(
                f"{key} not found, please select one of the following:\n{avail}") from exc

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
        if not verbose:
            return
        print(
            f"[{self.container_id}] {count_brightness_cut} science targets (cluster members) have been registered.")
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
        if not verbose:
            return
        print(
            f"[{self.container_id}] {count_brightness_cut} more science targets (AGN) have been registered.")
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

        # - Select the 10 white dwarfs closest to the median magnitude of our sources
        white_dwarfs.sort("rmag")
        median_mag = np.median(self.cluster_members["rmag"])
        closest_index = np.abs(white_dwarfs["rmag"] - median_mag).argmin()
        good_white_dwarfs = white_dwarfs[closest_index:closest_index + 10]

        count_good = len(good_white_dwarfs)

        self.white_dwarfs = good_white_dwarfs
        if not verbose:
            return
        print(
            f"[{self.container_id}] {count_good} white dwarfs have been registered.")
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
        sweep = self.region.get_included_sweep_table()
        sweep = get_objects_in_circular_region(
            sweep, self.obs_ra, self.obs_dec, self.selection_radius)
        # ! Note: We renamed RPmag to rmag for consistency with the other tables!
        guide_stars = _sanitize_sweep_for_guide_stars(sweep)

        guide_stars["pm_tot"] = calc_pm_tot(
            guide_stars["pmra"], guide_stars["pmdec"])
        count_initial = len(guide_stars)
        # Discard all rows where no rmag, pmra or pmdec information is available:
        guide_stars = filter_for_existing_cols(
            guide_stars, ("rmag", "pmra", "pmdec"))
        count_clean = len(guide_stars)
        # Perform the desired brightness cuts
        mask = guide_stars["rmag"] <= mag_r_max
        mask &= guide_stars["rmag"] >= mag_r_min
        mask &= guide_stars["pm_tot"] <= pm_max
        guide_stars = guide_stars[mask]
        count_flux_limited = len(guide_stars)
        # Select the 150 brightest ones
        guide_stars.sort("rmag")
        limiting_count = 150
        guide_stars = guide_stars[:limiting_count]

        self.guide_stars = guide_stars
        if not verbose:
            return
        print(
            f"[{self.container_id}] {limiting_count} guide stars have been registered.")
        info_string = f"\t{count_initial = }\n\t{count_clean = }\n"
        info_string += f"\t{count_flux_limited = }"
        print(info_string)

    def get_sky_fibres(self, verbose=True):
        """Generate 200 random sky fibres from the legacy DR10 imaging"""
        sweep = self.region.get_included_sweep_table()
        sweep = get_objects_in_circular_region(
            sweep, self.obs_ra, self.obs_dec, self.selection_radius)

        sky_fibres = _generate_sky_fibres_from_sweep(sweep)
        sky_fibres = get_objects_in_circular_region(
            sky_fibres, self.obs_ra, self.obs_dec, self.selection_radius)
        count_initial = len(sky_fibres)
        limiting_count = 150
        sky_fibres = sky_fibres[:limiting_count]
        sky_fibres = _sanitize_fibre_table(sky_fibres)

        self.sky_fibres = sky_fibres
        if not verbose:
            return
        print(
            f"[{self.container_id}] {limiting_count} sky fibres have been registered.")
        info_string = f"\t{count_initial = }\n\t{limiting_count = }"
        print(info_string)

    def plot_sources_on_ax(self, ax: Axes, types: Optional[Sequence[str]] = None, **kwargs):
        labels = self.get_available_tables().keys()
        if types is not None:
            labels = [label for label in labels if label in types]
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
        """Pretty-print the most important parameters of the container."""
        print(f"[{self.container_id}] at [{self.obs_ra:.2f}°, {self.obs_dec:.2f}°] with {len(self)} total objects.")
        for key, table in self.get_available_tables().items():
            print(f"\t{key:18}-> {len(table)} sources")

    def get_full_target_table(self):
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MergeConflictWarning)
            full_table = vstack(tables)
        count_before = len(full_table)
        if (diff := count_before - len(full_table)) > 0:
            print(f"Removed {diff} duplicate source(s).")
        return full_table

    def get_fld_file_header(self, observation_utdate: str, observation_label: Optional[str] = None) -> str:
        label = f"Observation {self.container_id}" if observation_label is None else observation_label
        ra, dec = convert_radec_to_hmsdms(
            self.obs_ra, self.obs_dec, " ", precision=2)
        file_header = (f"LABEL {label}\nUTDATE {observation_utdate}\nCENTRE {ra} {dec}\nEQUINOX J2000\n"
                       f"WLEN1 6000\nPROPER_MOTIONS\n\n")
        file_header += ("""
# Proper motions in arcsec/year

#			  R. Ascention 	 Declination			      Prog    Proper Motion		Comments
# Name 			  hh  mm ss.sss  dd  mm ss.sss 		      mag     ID      ra	dec\n""")
        return file_header

    def write_targets_to_disc(self, observation_utdate: str, fpath: Optional[Path] = None, overwrite=True, verbose=False):
        if fpath is None:
            fpath = PATHS.get_fld_fname(self.container_id)
        all_targets = self.get_full_target_table()
        all_targets.write(fpath, format="csv", delimiter="\t", overwrite=overwrite,
                          formats={"obj_name": lambda x: x.upper(), "rmag": "%.3f"})
        header = self.get_fld_file_header(observation_utdate)
        # Now that the file has been written, we have to prepend the file header:
        with open(fpath, 'r+', encoding="utf8") as f:
            content = f.readlines()
            f.seek(0, 0)
            f.write(header + "".join(content[1:]))
        if verbose:
            self.pprint()
            print(
                f"Successfully written the .fld file to {fpath} with the following header:\n{header}")
