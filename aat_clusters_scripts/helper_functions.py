
from functools import reduce
from typing import Optional, Sequence

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
    hms, dms = sky_coord.to_string('hmsdms', precision=precision).split()
    if delimiter is None:
        return hms, dms
    hms, dms = [s.replace("d", delimiter).replace(
        "h", delimiter).replace("m", delimiter).replace("s", "") for s in (hms, dms)]
    return hms, dms


def add_ra_dec_hms_dms_columns(table: Table) -> Table:
    hms_dms = np.array([convert_radec_to_hmsdms(
        member["ra"], member["dec"], delimiter=" ", precision=2) for member in table])
    table["ra_hms"] = hms_dms[:, 0]
    table["dec_dms"] = hms_dms[:, 1]
    return table


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
    coords_1 = SkyCoord(object_table["ra"], object_table["dec"], unit="deg")
    coords_2 = SkyCoord(ra, dec, unit="deg")
    sep = coords_1.separation(coords_2)

    return object_table[sep < radius * u.deg]


def filter_for_existing_cols(object_table: Table, cols=("rmag", "pmra", "pmdec")) -> Table:
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
    mask = True
    for col in cols:
        mask &= ~np.isnan(object_table[col])
    # mask = ~np.isnan(object_table[col]) for col in cols))
    # print(~np.isnan(object_table["rmag"]))
    # print(object_table)
    return object_table[mask]


def filter_for_stars(object_table: Table, more_star_names: Optional[Sequence[str]] = None):
    star_list = ["*", "HB*", "Ae*", "Em*", "Be*", "BS*",
                 "RG*", "AB*", "C*", "S*", "sg*", "s*r", "s*y", "HS*"]
    if more_star_names is not None:
        star_list += list(more_star_names)
    mask = np.isin(object_table["otype_opt"], star_list)
    return object_table[mask]


def calc_pm_tot(pmra: float, pmdec: float) -> float:
    """Calculate the total proper motion of a given source according to Jacob's formula"""
    return np.sqrt((0.3977 * pmra)**2 + pmdec**2)
