from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astropy.utils.metadata import MergeConflictWarning
from matplotlib.axes import Axes

from .helper_functions import (
    RELEVANT_TABLE_COLUMNS, add_ra_dec_degree_columns, calc_pm_tot,
    convert_radec_to_hmsdms, filter_for_existing_cols, get_legacysurvey_url,
    get_objects_in_circular_region, read_obs_parameters_from_fld_file,
    reduce_table_to_relevant_columns_and_remove_duplicates,
    sanitize_table_for_observation)
from .load_science_targets import get_science_targets
from .paths import PATHS
from .region import RectangularRegion

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


@u.quantity_input(min_dist_to_sweep="angle", min_dist_to_other_fibres="angle")
def _generate_sky_fibres_from_sweep(table: Table, min_dist_to_sweep=u.Quantity,
                                    min_dist_to_other_fibres=u.Quantity) -> Table:
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
    sky_fibres = random_coords[d2d > min_dist_to_sweep]

    # Find all random coords far away from each other to prevent clustering:
    idx, d2d, _ = sky_fibres.match_to_catalog_sky(sky_fibres, nthneighbor=2)
    sky_fibres = sky_fibres[d2d > min_dist_to_other_fibres]
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


#######################################
# - The actual TargetContainer class:


@dataclass
class TargetContainer:
    """The main class to set up your observation.
    Initially only containing the observation-id, the central ra and dec and possibly
    the selection radius, it can be successively filled with the various target tables
    and finally be used to construct the .fld file.

    Attributes
    ----------
    observation_id: int
        The id of the observation, used to uniquely identify the observation (e. g. used
        to name the .fld file)
    obs_ra: float
        The central right ascension for the observation, expected in degrees
    obs_dec: float
        The central declination for the observation, expected in degrees
    selection_radius: float = 1.0
        The radius of the observation, expected in degrees
    region: RectangularRegion
        The square-shaped region taken up by the observation
    ls_url: str = field(init=False)
        A link to the legacy surveys in which the area can be displayed
    science_targets: Optional[Table] = None
        A table of the science targets [filled after calling `get_science_targets`]
    white_dwarfs: Optional[Table] = None
        A table of the white dwarfs [filled after calling `get_white_dwarfs`]
    guide_stars: Optional[Table] = None
        A table of the guide stars [filled after calling `get_guide_stars`]
    sky_fibres: Optional[Table] = None
        A table of the sky fibres [filled after calling `get_sky_fibres`]

    Example
    -------
    To use this class, simply initialise it with the desired parameters and
    successively call the functions to fill the tables:
        >>> container = TargetContainer(1234, 180, 20)
        >>> container.get_science_targets()
        >>> container.get_white_dwarfs()
        >>> container.get_guide_stars(mag_r_min=14, mag_r_max=14.5, pm_max=20)
        >>> container.get_sky_fibres()
        >>> container.write_targets_to_disc("2023 02 22")
    """
    observation_id: int
    obs_ra: float
    obs_dec: float
    selection_radius: float = 1.0  # The radius in degrees
    region: RectangularRegion = field(init=False)
    ls_url: str = field(init=False)
    science_targets: Optional[Table] = None
    white_dwarfs: Optional[Table] = None
    guide_stars: Optional[Table] = None
    sky_fibres: Optional[Table] = None

    def __post_init__(self):
        self.region = RectangularRegion.from_centre_and_radius(
            self.obs_ra, self.obs_dec, self.selection_radius)
        self.ls_url = get_legacysurvey_url(self.obs_ra, self.obs_dec)

    def __getitem__(self, key):
        """Enable dict-like access of the different subsets."""
        try:
            return self.__getattribute__(key)
        except AttributeError as exc:
            avail = ["science_targets", "white_dwarfs",
                     "guide_stars", "sky_fibres"]
            raise AttributeError(
                f"{key} not found, please select one of the following:\n{avail}") from exc

    def __len__(self):
        return sum([len(t) for t in self.get_available_tables().values()])

    @classmethod
    def from_existing_fld_file(cls, fpath: Path) -> TargetContainer:
        """Create an instance of a TargetContainer from an existing .fld file.
        Warning: This cannot infer the selection radius and will just default it to 1.

        Parameters
        ----------
        fpath : Path
            The Path to the .fld file [expected to be in the same format as written by this container]

        Returns
        -------
        TargetContainer
        """
        # Unfortunately we need pandas to properly handle the input
        table = pd.read_csv(fpath, delimiter="\t", skiprows=8,
                            comment="#", names=RELEVANT_TABLE_COLUMNS)
        table = Table.from_pandas(table)
        table = add_ra_dec_degree_columns(table)
        science_targets = table[table["obs_type"] == "P"]
        is_white_dwarf = np.array(
            ["WDJ" in name for name in science_targets["obj_name"]])
        obs_id, ra, dec = read_obs_parameters_from_fld_file(fpath)
        container = cls(obs_id, ra, dec)
        container.white_dwarfs = science_targets[is_white_dwarf]
        container.science_targets = science_targets[~is_white_dwarf]
        container.sky_fibres = table[table["obs_type"] == "S"]
        container.guide_stars = table[table["obs_type"] == "F"]
        return container

    def get_available_tables(self) -> dict[str, Table]:
        """Get all of the currently available tables."""
        keys = ["science_targets", "white_dwarfs",
                "guide_stars", "sky_fibres"]
        return {key: table for key in keys if (table := self[key]) is not None}

    def get_science_targets(self, verbose=True):
        """Register the science targets."""
        self.science_targets = get_science_targets(
            self.observation_id, self.obs_ra, self.obs_dec, self.selection_radius, verbose)

    def get_white_dwarfs(self, mag_r_min=17.5, num_to_keep: int = 10, verbose=True):
        """Load the white dwarfs we need to observe for spectral calibration.
        Then, perform selection based on:
            - Area (circular region around central coordinates)
            - Cleanliness (discard sources w/o RPmag or proper motion)
            - Brightness (discard sources that are too bright to get a proper spectrum)

        Parameters
        ----------
        mag_r_min : float, optional
            The maximum brightness a white dwarf should have, by default 17.5
        num_to_keep : int, optional
            The number of White Dwarfs to keep as only a small number will be needed
            for the spectroscopic calibration.
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
        median_mag = np.median(self.science_targets["rmag"])
        closest_index = np.abs(white_dwarfs["rmag"] - median_mag).argmin()
        good_white_dwarfs = white_dwarfs[closest_index:closest_index + num_to_keep]

        count_good = len(good_white_dwarfs)

        good_white_dwarfs = sanitize_table_for_observation(
            good_white_dwarfs, "P", 9)
        self.white_dwarfs = good_white_dwarfs
        if not verbose:
            return
        print(
            f"[{self.observation_id}] {count_good} white dwarfs have been registered.")
        info_string = f"\t{count_initial = }\n\t{count_in_region = }\n"
        info_string += f"\t{count_clean = }\n\t{count_faint = }"
        print(info_string)

    def get_guide_stars(self, mag_r_min: float, mag_r_max: float, pm_max: float, verbose=True):
        """Select the Gaia sources of the SWEEP tables from a circular region around the central
        coordinates as guide stars.
        The selection is then based on:
            - Cleanliness (discard sources w/o rmag or proper motion)
            - Magnitude and PM requirements (see parameters)
            - Type (discard all non-stellar sources)

        For an alternative way (by querying Simbad) see `unused_functions.py`.

        Parameters
        ----------
        mag_r_min : float
            The minimum magnitude a guide star should have
        mag_r_max : float
            The maximum magnitude a guide star should have
        pm_max : float
            The maximum proper motion a guide star should have, in mas/year!
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
        guide_stars = sanitize_table_for_observation(guide_stars, "F", 9)
        self.guide_stars = guide_stars
        if not verbose:
            return
        print(
            f"[{self.observation_id}] {limiting_count} guide stars have been registered.")
        info_string = f"\t{count_initial = }\n\t{count_clean = }\n"
        info_string += f"\t{count_flux_limited = }"
        print(info_string)

    @u.quantity_input(min_dist_to_sweep="angle", min_dist_to_other_fibres="angle")
    def get_sky_fibres(self, min_dist_to_sweep=10 * u.arcsec,
                       min_dist_to_other_fibres=1 * u.arcmin,
                       limiting_count=150, verbose=True):
        """Generate a random set of sky fibres using the SWEEP catalogue of LS DR10.
        This is done by generating a sufficient of amount of random ra and dec coordinates
        in the area of interest, cross-matching these to the SWEEP sources, and discarding
        all random coordinates closer than `min_dist_to_sweep`.
        After that, only the top 150 of these are taken.
        NOTE: This approach produces lower sky fibre densities for more densely populated
        regions. If you find a good workaround that, feel free to send a mail or a PR!

        Parameters
        ----------
        min_dist_to_sweep : Quantity[angle], optional
            The minimum angular distance the sky fibres should have to sweep sources, by default 10*u.arcsec
        min_dist_to_other_fibres : Quantity[angle], optional
            The minimum angular distance the sky fibres should have to themselves, by default 1*u.arcmin
        limiting_count : int, optional
            The number of fibres to keep, by default 150
        verbose : bool, optional
            Print additional info if True., by default True
        """
        sweep = self.region.get_included_sweep_table()
        sweep = get_objects_in_circular_region(
            sweep, self.obs_ra, self.obs_dec, self.selection_radius)

        sky_fibres = _generate_sky_fibres_from_sweep(
            sweep, min_dist_to_sweep, min_dist_to_other_fibres)
        sky_fibres = get_objects_in_circular_region(
            sky_fibres, self.obs_ra, self.obs_dec, self.selection_radius)
        count_initial = len(sky_fibres)
        if count_initial < limiting_count:
            warnings.warn(
                f"There are only {count_initial} sky fibres while you requested {limiting_count}", UserWarning)
        sky_fibres = sky_fibres[:limiting_count]
        sky_fibres = _sanitize_fibre_table(sky_fibres)

        sky_fibres = sanitize_table_for_observation(sky_fibres, "S", 9)
        self.sky_fibres = sky_fibres
        if not verbose:
            return
        final_count = len(sky_fibres)
        print(
            f"[{self.observation_id}] {final_count} sky fibres have been registered.")
        info_string = f"\t{count_initial = }\n\t{final_count = }"
        print(info_string)

    def plot_sources_on_ax(self, ax: Axes, types: Optional[Sequence[str]] = None, **kwargs):
        """Construct a positional scatter plot of the sources on the given ax, labelled by
        their affiliation.

        Parameters
        ----------
        ax : Axes
            The matplotlib ax object to perform the plot on
        types : Optional[Sequence[str]], optional
            The types to plot (e.g. science_targets, white_dwarfs, sky_fibres, guide_stars).
            If not provided, all types are plotted., by default None
        **kwargs :
            Keyword arguments additionally handed to the scatter plot routine.
        """
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
        ax.set_title(f"Observation {self.observation_id}")

    def pprint(self):
        """Pretty-print the most important parameters of the container."""
        print(f"[{self.observation_id}] at [{self.obs_ra:.2f}°, {self.obs_dec:.2f}°] with {len(self)} total objects.")
        for key, table in self.get_available_tables().items():
            print(f"\t{key:18}-> {len(table)} sources")

    def get_full_target_table(self) -> Table:
        """Constructs a full target table from all available subtables.

        Returns
        -------
        Table
            The full target table

        Raises
        ------
        UserWarning
            If not all tables are available, this warning will be raised.
        """
        if len((keys := self.get_available_tables().keys())) < 4:
            diff = 4 - len(keys)
            raise UserWarning(f"You are missing {diff} tables.\n"
                              f"So far, only the following tables are available: {keys}")
        tables = self.science_targets, self.white_dwarfs, self.guide_stars, self.sky_fibres
        with warnings.catch_warnings():
            # Since we are only interested in the stack itself, metadata merge conflicts do not concern us:
            warnings.simplefilter("ignore", MergeConflictWarning)
            full_table = vstack(tables)
        full_table = reduce_table_to_relevant_columns_and_remove_duplicates(
            full_table)
        return full_table

    def get_fld_file_header(self, observation_utdate: str,
                            observation_label: Optional[str] = None) -> str:
        """Generate the file header for the .fld file.

        Parameters
        ----------
        observation_utdate : str
            The date of the observation in the YYYY MM DD format
        observation_label : str, optional, by default None
            The label for the observation. If not provided, the observation id is used (recommended)

        Returns
        -------
        str
            The label for the fld file."""
        label = f"Observation {self.observation_id}" if observation_label is None else observation_label
        ra, dec = convert_radec_to_hmsdms(
            self.obs_ra, self.obs_dec, " ", precision=2)
        file_header = (f"LABEL {label}\nUTDATE {observation_utdate}\nCENTRE {ra} {dec}\nEQUINOX J2000\n"
                       f"WLEN1 6000\nPROPER_MOTIONS\n\n")
        file_header += ("""
# Proper motions in arcsec/year

#			  R. Ascention 	 Declination			      Prog    Proper Motion		Comments
# Name 			  hh  mm ss.sss  dd  mm ss.sss 		      mag     ID      ra	dec\n""")
        return file_header

    def write_targets_to_disc(self, observation_utdate: str, fpath: Optional[Path] = None,
                              overwrite=True, verbose=False):
        """Constructs the final target table and writes them to disc.

        Parameters
        ----------
        observation_utdate : str
            The date of the observation in the YYYY MM DD format
        fpath : Optional[Path], optional
            The filepath where the data should be stored. If not provided,
            the default location for the observation id is used (recommended), by default None
        overwrite : bool, optional
            Whether an existing .fld file in the specified location should be
            overwritten, by default True
        verbose : bool, optional
            Whether additional information should be printed., by default False
        """
        if fpath is None:
            fpath = PATHS.get_fld_fname(self.observation_id)
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
                f"Successfully written the .fld file to {fpath} with the following header [cut after 50 characters]:\n{header[:50]}")
