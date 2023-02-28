"""Script containing functions to load the science targets for the survey."""
from astropy.table import Table, vstack

from .helper_functions import (filter_for_existing_cols,
                               get_objects_in_circular_region,
                               sanitize_table_for_observation)
from .paths import PATHS


def get_science_targets(observation_id: int, ra: float, dec: float, radius: float, verbose=False) -> Table:
    """Load and return the science targets for the survey in an astropy table.
    --> This is the function you might need to reimplement for your use-case.

    Parameters
    ----------
    observation_id : int
        In the case of my observations, the observation_id corresponds to the cluster id
        and is used to select the cluster members associated with it.
    ra : float
    dec : float
    radius : float
        The ra, dec, and radius of the observation in degrees, in my case needed to select
        the agn candidates of the area.

    Returns
    -------
    Table
        The table containing the science targets, required to have the following columns:
        ra [deg], dec [deg], obj_name [str], rmag, pmra [arcsec/yr], pmdec [arcsec/yr].
        NOTE: `sanitize_table_for_observation` already converts the proper motions from mas/yr
            to arcsec/yr."""
    cluster_members = _get_cluster_members(observation_id, verbose=verbose)
    cluster_members = sanitize_table_for_observation(
        cluster_members, "P", priority=8)

    agn_candidates = _get_agn_candidates(ra, dec, radius, verbose=verbose)
    agn_candidates = sanitize_table_for_observation(
        agn_candidates, "P", priority=6)

    # Register the object type so we can distinguish the two later.
    cluster_members["object_type"] = "cluster_member"
    agn_candidates["object_type"] = "agn_candidate"

    table = vstack([cluster_members, agn_candidates])
    return table


####################################################################################
# My personal functions used to load the science targets for the AAT clusters observation:
# Since my cluster targets were clustered on a small area, we decided to supplement them with AGN candidates.

_CLUSTER_MEMBER_RMAG_MIN = 17.5
_AGN_RMAG_RANGE = (17.5, 21.5)


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


def _get_cluster_members(cluster_id: int, verbose=True):
    members = Table.read(PATHS.members)
    # This selection only works due to our setup:
    members = members[members["mem_match_id"] == cluster_id]
    members = _sanitize_cluster_members(members)
    count_initial = len(members)
    # Discard all rows where no rmag, pmra or pmdec information is available:
    members = filter_for_existing_cols(members)
    count_clean = len(members)
    # Discard WDs brighter than the [mag_r_min]
    brightness_cut = members["rmag"] >= _CLUSTER_MEMBER_RMAG_MIN
    members = members[brightness_cut]
    count_brightness_cut = len(members)

    if not verbose:
        return
    print(
        f"{count_brightness_cut} cluster members have been registered as science targets.")
    info_string = f"\t{count_initial = }\n\t{count_clean = }"
    info_string += f"\n\t{count_brightness_cut = }"
    print(info_string)


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


def _get_agn_candidates(ra, dec, radius, verbose=True):
    agn_candidates = PATHS.read_table(PATHS.agn_candidates)
    agn_candidates = _sanitize_agn_candidates(agn_candidates)
    count_initial = len(agn_candidates)
    # Restrict to circular region around pointing:
    agn_candidates = get_objects_in_circular_region(
        agn_candidates, ra, dec, radius)
    count_in_region = len(agn_candidates)
    mask = agn_candidates["rmag"] >= _AGN_RMAG_RANGE[0]
    mask &= agn_candidates["rmag"] <= _AGN_RMAG_RANGE[1]
    agn_candidates = agn_candidates[mask]
    count_brightness_cut = len(agn_candidates)

    if not verbose:
        return
    print(
        f"{count_brightness_cut} AGN candidates have been registered as science targets.")
    info_string = f"\t{count_initial = }\n\t{count_in_region = }"
    info_string += f"\n\t{count_brightness_cut = }"
    print(info_string)
