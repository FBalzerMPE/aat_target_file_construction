
from functools import reduce
from typing import Optional

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table


def convert_radec_to_hmsdms(ra: float, dec: float,
                            delimiter: Optional[str] = ":",
                            precision: Optional[int] = None) -> tuple[str, str]:
    """Calculate the hms dms representation of the given coordinates, using
    the prescribed delimiter.

    Parameters
    ----------
    ra : float
        rigtht ascension in degrees
    dec : float
        declination in degrees
    delimiter : str?, optional
        Delimiter between the numbers, e. g. 120:23:32.123.
        If set to None, the default rep is given, by default ":"
    precision : int, optional
        The precision for ra and dec

    Returns
    -------
    tuple[str, str]
        The hms and dms
    """
    # Convert ra and dec to hms and dms employing astropy's SkyCoord utility:
    sky_coord = SkyCoord(ra, dec, unit="deg")
    if delimiter is None:
        hms, dms = sky_coord.to_string('hmsdms').split()
    else:
        hms, dms = [s.replace("d", delimiter).replace(
            "h", delimiter).replace("m", delimiter).replace("s", "") for s in sky_coord.to_string('hmsdms').split()]
    if precision is not None:
        static, sec = hms.split(".")
        hms = f"{static}.{float('0.' + sec):.{precision}f}"
        static, sec = dms.split(".")
        dms = f"{static}.{float('0.' + sec):.{precision}f}"
    return hms, dms


def get_legacysurvey_url(ra_cen: float, dec_cen: float, mark: bool = True,
                         zoom_factor=14, layer="ls-dr10") -> str:
    """Returns the URL to a LS10 viewer.

    Parameters
    ----------
    ra_cen : float
        The central right ascension in degrees
    layer : str, optional, by default ls-dr10-early
        The requested layer.
        Should be one of "ls-dr10-early", "ls-dr10-early-grz", "ls-dr9"
    dec_cen : float
        The central declination in degrees
    zoom_factor : int, optional
        The zoom factor, by default 15

    Returns
    -------
    Url
        The legacy survey url
    """
    mark_string = f"&mark={ra_cen},{dec_cen}" if mark else ""
    url = (f"https://www.legacysurvey.org/viewer?ra={ra_cen}&dec={dec_cen}"
           f"&layer={layer}&zoom={zoom_factor}{mark_string}")
    return url


def get_objects_in_circular_region(object_table: Table, ra: float, dec: float, radius: float = 1) -> Table:
    """Retrieve all objects in a radius of `radius` [deg] around the centre.

    Parameters
    ----------
    object_table : Table
        The table of which to select the sources from
    ra : float
        The ra to center around, expected in [deg]
    dec : float
        The dec to center the region around, expected in [deg]
    radius : float, optional
        The radius, expected in [deg], by default 1

    Returns
    -------
    Table
        A subset of the `object_table` with sources within the given radius.
    """
    coords_1 = SkyCoord(object_table["ra"], object_table["dec"])
    coords_2 = SkyCoord(ra, dec, unit="deg")
    sep = coords_1.separation(coords_2)

    return object_table[sep < radius * u.deg]


def filter_for_existing_cols(object_table: Table, cols=("RPmag", "pmra", "pmdec")) -> Table:
    """Remove all rows of the object table where entries in any of the given columns is not available.

    Parameters
    ----------
    object_table : Table
        The table to reduce
    cols : tuple, optional
        The columns to check for nan values, by default ("RPmag", "pmra", "pmdec")

    Returns
    -------
    Table
        A subset of `object_table` where all entries in the given columns are available.
    """
    # We take the intersection of the non-zero indices of each of the columns:
    okay_col_indices = reduce(
        np.intersect1d, (np.nonzero(object_table[col])[0] for col in cols))
    return object_table[okay_col_indices]
