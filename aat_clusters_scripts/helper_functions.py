
import warnings
from pathlib import Path
from typing import Literal, Optional

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, unique
from IPython.display import HTML, display

# Register the table columns for the final .fld file
RELEVANT_TABLE_COLUMNS = ("obj_name", "ra_hms", "dec_dms", "obs_type",
                          "priority", "rmag", "program_id", "pmra", "pmdec")


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


def convert_radec_to_degree(ra_hms: str, dec_dms: str) -> tuple[float, float]:
    """Calculate the ra and dec in degree given their representation in hms and dms."""
    skycoord = SkyCoord(ra_hms, dec_dms, unit=(u.hourangle, u.degree))
    return skycoord.ra.value, skycoord.dec.value


def add_ra_dec_hms_dms_columns(table: Table) -> Table:
    """Adds 'ra_hms' and 'dec_dms' columns to a table with 'ra' and 'dec' columns
    which are assumed to be in degrees.

    Parameters
    ----------
    table : Table
        The input table with a ra and a dec column in degrees

    Returns
    -------
    Table
        The input table with two additional columns (ra_hms, dec_dms)
    """
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
    # We take the intersection of all not-nan entries of each of the columns:
    mask = True  # initialise so we can easily add to it.
    for col in cols:
        mask &= ~np.isnan(object_table[col])
    return object_table[mask]


def calc_pm_tot(pmra: float, pmdec: float) -> float:
    """Calculate the total proper motion of a given source according to Jacob's formula"""
    return np.sqrt((0.3977 * pmra)**2 + pmdec**2)


#######################################
# - Putting the tables into a unified form:

@np.vectorize
def _clean_object_name(name: str) -> str:
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    return name.replace(" ", "_")


def reduce_table_to_relevant_columns_and_remove_duplicates(table: Table, keep_old_radec=False) -> Table:
    """Reduces the given table to the relevant columns and removes duplicates in
    in the `obj_name` column."""
    relevant_cols = list(RELEVANT_TABLE_COLUMNS)
    if keep_old_radec:
        relevant_cols += ["ra", "dec"]
    if (diff := len(table) - len(np.unique(table["obj_name"]))) > 0:
        # TODO: Find the problematic sources and print them directly
        warnings.warn(
            f"Discarding {diff} sources that do not have unique names. You might want to check on that.")
        table = unique(table, "obj_name")
    table.sort("obj_name")
    return table[relevant_cols]


def sanitize_table_for_observation(table, obs_type: Literal["P", "S", "F"], priority=9) -> Table:
    """Performs cleaning operations on the given table, putting it into a format
    that can be used in the .fdl files.

    Parameters
    ----------
    table : Table
        An astropy table expected to have [ra, dec, obj_name, rmag, pmra, pmdec]
        columns.
    obs_type : Literal[P, S, F]
        P = 'Science targets' (and WDs), F = 'Guide stars', S = 'Sky fibres'
    priority : int, optional
        The priority the sources in the table should have [9 is highest], by default 9

    Returns
    -------
    Table
        A cleaned table containing only unique sources
    """
    assert obs_type in "PSF", f"Please choose a valid observation type and not {obs_type}"
    assert set(table.colnames).issuperset({"ra", "dec", "obj_name", "rmag", "pmra", "pmdec"}
                                          ), "The table provided for sanitising does not have the required columns (maybe you need to change the names?)."
    table["obj_name"] = _clean_object_name(table["obj_name"])
    table["obs_type"] = obs_type
    table["priority"] = priority
    # The program ID does not matter for us, but we need to provide it
    table["program_id"] = 0
    table = add_ra_dec_hms_dms_columns(table)
    # Convert proper motions from mas/yr to arcsec/yr
    table["pmra"] = table["pmra"] / 1000.
    table["pmdec"] = table["pmdec"] / 1000.
    table = reduce_table_to_relevant_columns_and_remove_duplicates(
        table, keep_old_radec=True)
    return table

########################################################
# - HTML stuff (just for display purposes)


def display_html_site(url: str, width: float = 800,
                      height: float = 600, provide_link: bool = False,
                      additional_text=""):
    """Displays the given url in the jupyter notebook

    Parameters
    ----------
    url : str
        The link of the requested page
    width : float, optional
        The width of the frame, by default 800
    height : float, optional
        The height of the frame, by default 600
    """
    output_string = _get_html_image_string(
        url, width, height) + additional_text
    if provide_link:
        link = f"<a href={url}>LINK</a><br>"
        output_string = link + output_string
    with warnings.catch_warnings():
        # We ignore this warning since IFrames don't work properly in my notebooks:
        warnings.simplefilter("ignore", UserWarning)
        display(HTML(output_string))


def _get_html_image_string(source: str, width: float = 600, height: float = 600) -> str:
    """Get the HTML frame for a given source (local image or url)

    Parameters
    ----------
    source : str
        The path to the image source
    width : float, optional
        The width of the image, by default 600
    height : float, optional
        The height of the image, by default 600

    Returns
    -------
    str
        The image enclosed in iframe html tags
    """
    is_url = source.startswith("http")
    tag = "iframe" if is_url else "img"
    width_height = f' width="{width}" height="{height}"' if is_url else f' width="{width}"'
    return f'<{tag} src="{source}"{width_height}></{tag}>'


def read_obs_parameters_from_fld_file(fpath: Path) -> tuple[int, float, float]:
    """Read the observation id and the ra and dec from the given fld file."""
    with fpath.open() as f:
        for line in f.readlines():
            if line.startswith("LABEL"):
                obs_id = line.split()[-1]
            if line.startswith("CENTRE"):
                break
    centre = line.split()[1:]
    ra, dec = ":".join(centre[:3]), ":".join(centre[3:])
    return obs_id, *convert_radec_to_degree(ra, dec)
