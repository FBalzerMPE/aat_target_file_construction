from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from math import ceil, floor
from typing import Literal, Union

import numpy as np
from astropy.table import Table, vstack
from astropy.units import UnitsWarning
from matplotlib.patches import Polygon
from SciServer import Authentication, Files

from .paths import PATHS


def _log_in_to_sciserver() -> str:
    """Authenticate with my password at the sciserver"""
    pw_dir = os.environ["HANDYSTUFF"]
    with open(pw_dir + "/nice_file.txt", "r", encoding="utf-8") as f:
        login_name, login_pw = [l.split()[1] for l in f.readlines()[:2]]
    token1 = Authentication.login(login_name, login_pw)
    return token1


def get_filenames_in_sciserver_dir(sciserver_dir: str) -> dict[str: int]:
    """Retrieves the files in the given sciserver directory and returns them
    and the respective sizes.

    Returns
    -------
    dict[FullFilepath: int]
        A dictionary mapping the filepaths to the sizes (in Bytes)
    """
    _log_in_to_sciserver()
    file_service = Files.getFileServiceFromName("FileService")
    files = Files.dirList(
        file_service, sciserver_dir, level=2)["root"]["files"]
    file_size_dict = {sciserver_dir + "/" +
                      f["name"]: f["size"] for f in files}
    return file_size_dict


def download_file_from_sciserver(sciserver_filepath: str, destination: str):
    """Downloads the given file at the sciserver_filepath from the remote directory.

    Parameters
    ----------
    sciserver_filepath : FullFilepath
        The SciServer path of the file
    destination : FullFilepath
        The destination path for the file.
    """
    _log_in_to_sciserver()
    file_service = Files.getFileServiceFromName("FileService")
    Files.download(file_service, sciserver_filepath,
                   quiet=False, localFilePath=destination)


def _get_sweep_sgn_str(dec: float) -> str:
    """Returns 'p' if dec is positive, 'm' if it's negative."""
    return "p" if dec >= 0 else "m"


def _get_sweep_region_string(ra: int, dec: int) -> str:
    """Returns a string conforming to the naming convention of the SWEEP
    catalogues following the <AAA>c<BBB> pattern where AAA is the ra, c
    the p or m for the sign of dec and BBB is the dec."""
    return f"{ra:03}{_get_sweep_sgn_str(dec)}{abs(dec):03}"


@dataclass
class RectangularRegion:
    ra_min: float
    ra_max: float
    dec_min: float
    dec_max: float

    @classmethod
    def from_generic_table_and_rectangular(cls, table: Table) -> RectangularRegion:
        """Load a rectangular region using the table provided, scanning it for
        the min and max of ra and dec

        Parameters
        ----------
        table : Table
            The Table to constrain the region by

        Returns
        -------
        RectangularRegion
        """
        return cls(table["ra"].min(), table["ra"].max(), table["dec"].min(), table["dec"].max())

    @classmethod
    def from_centre_and_radius(cls, ra, dec, radius) -> RectangularRegion:
        """Load a rectangular region around the central coordinates
        Returns
        -------
        RectangularRegion
        """
        return cls(ra - radius, ra + radius, dec - radius, dec + radius)

    def get_included_sweep_bricks(self) -> list[str]:
        """Retrieve the relevant SWEEP bricks for this region.

        Returns
        -------
        list[str]
            A list of sweep brickstrings
        """
        brick_strings = []
        # Right ascension is taken in steps of 10, declination in steps of five
        ra_min = 5 * floor(self.ra_min / 5)
        ra_max = 5 * ceil(self.ra_max / 5)
        dec_min = 5 * floor(self.dec_min / 5)
        dec_max = 5 * ceil(self.dec_max / 5)
        for ra in range(ra_min, ra_max, 5):
            for dec in range(dec_min, dec_max, 5):
                reg_min = _get_sweep_region_string(ra, dec)
                reg_max = _get_sweep_region_string(ra + 5, dec + 5)
                brick_strings.append(f"{reg_min}-{reg_max}")
        return brick_strings

    def get_sweep_brick_filenames(self, filter_not_on_disk=False) -> list[str]:
        """Obtain the names of all sweep bricks covered by this cluster extent.

        Parameters
        ----------
        filter_not_on_disk : bool, optional
            If only the names of the locally NOT available bricks should be returned, by default False

        Returns
        -------
        list[str]
            The list of (if needed, except for local) sweep brick filenames.
        """
        filenames = [
            f"sweep-{brick}.fits" for brick in self.get_included_sweep_bricks()]
        if filter_not_on_disk:
            filenames = [fname for fname in filenames if not PATHS.sweep.joinpath(
                fname).is_file()]
        return filenames

    def download_sweep_files_for_region(self):
        """Download missing sweep files in this region
        """
        # Method to list sweep bricks:
        sweep_path = "external_catalogs/DECam/legacysurvey/DR10/X.X/"
        bricks_to_download = self.get_sweep_brick_filenames(
            filter_not_on_disk=True)
        if len(bricks_to_download) == 0:
            return
        file_size_dict = get_filenames_in_sciserver_dir(sweep_path)
        relevant_files = [
            f for brick in bricks_to_download for f in file_size_dict if brick in f]
        for filepath in relevant_files:
            destpath = PATHS.sweep.joinpath(filepath.split("/")[-1])
            filesize = file_size_dict[filepath] / 1024**3
            print(
                f"Downloading {filepath} (approx. size of {filesize:.2f} GB)...")
            download_file_from_sciserver(filepath, destpath)

    def get_stretched_bounds(self, direction: Literal["ra", "dec"], stretch_factor: float = 1.1) -> tuple[float, float]:
        """Provide rectangular right ascension bounds that are stretched by the given factor

        Parameters
        ----------
        stretch_factor : float
            The factor the total distance should be stretched by

        Returns
        ----------
        min, max : tuple[float, float]
        """
        assert direction in ["ra", "dec"]
        relevant_tuple = (self.ra_min, self.ra_max) if direction == "ra" else (
            self.dec_min, self.dec_max)
        dist = relevant_tuple[1] - relevant_tuple[0]
        extra = dist * (stretch_factor - 1)
        return relevant_tuple[0] - extra, relevant_tuple[1] + extra

    def get_mask_for_table(self, table: Table, stretch_factor: float = 1) -> np.ndarray:
        """Returns the table with the ra and dec constrained to the cluster extent

        Parameters
        ----------
        table : Table
            The input table, expected to have ra and dec columns
        stretch_factor : float, optional
            The factor that the bounds shall be stretched by, by default 1

        Returns
        -------
        Table
            A subset of the input table, constrained to the extent.
        """
        ra_min, ra_max = self.get_stretched_bounds("ra", stretch_factor)
        dec_min, dec_max = self.get_stretched_bounds("dec", stretch_factor)
        mask = (table["ra"] > ra_min) & (table["ra"] < ra_max)
        mask *= (table["dec"] > dec_min) & (table["dec"] < dec_max)
        return mask

    def get_constrained_table(self, table: Table, stretch_factor: float = 1) -> Table:
        return table[self.get_mask_for_table(table, stretch_factor)]

    def get_included_sweep_table(self) -> Table:
        self.download_sweep_files_for_region()
        tables = []
        for fname in self.get_sweep_brick_filenames():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UnitsWarning)
                t = Table.read(PATHS.sweep.joinpath(fname))
            tables.append(t)
        stack: Table = vstack(tables)
        stack.rename_columns((cols := stack.colnames), [
                             col.lower() for col in cols])
        # Constrain since the stack goes beyond the limits
        return self.get_constrained_table(stack, 1.1)
