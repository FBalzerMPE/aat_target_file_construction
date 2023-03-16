"""Script to perform the target selection for an AAT observation."""
import argparse
from typing import Optional, Union

import aat_clusters_scripts as acs


def find_ra_and_dec_from_observation_id(observation_id: Union[int, str]) -> tuple[float, float]:
    """Get the central right ascension and declination for an observation from the observation id.

    Parameters
    ----------
    observation_id : Union[int, str]
        The ID of the observation from which the coordinates can be inferred

    Returns
    -------
    tuple[float, float]
        The ra and dec in degrees.
    """
    raise NotImplementedError(
        "Please provide the ra and dec directly or implement how to infer it from the observation id.")

    return ra, dec


def parse_args():
    parser = argparse.ArgumentParser(description="Script to generate the .fld files for AAT observations.\n"
                                     "SYNTAX:\n\t>>> python construct_target_file.py observation_id [ra, dec, radius]\n"
                                     "You may need to code how the science targets are loaded before the first run.")
    parser.add_argument(
        "--observation_id", type=Union[int, str], help="The ID of the observation, also used to identify the .fld file.")
    parser.add_argument("--ra", type=Optional[float], default=None,
                        help="The central right ascension in degrees. If not provided, please program a function to infer it from the observation_id.")
    parser.add_argument("--dec", type=Optional[float], default=None,
                        help="The central declination in degrees. If not provided, please program a function to infer it from the observation_id.")
    parser.add_argument("--utdate", type=Optional[str], default=None,
                        help="The UT Date [\"YYYY MM DD\"] when the observation shall not take place. If not provided, the .fld file is written with a placeholder.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Whether additional information should be printed on the screen.")
    args, _ = parser.parse_known_args()
    if args.ra is None or args.dec is None:
        args.ra, args.dec = find_ra_and_dec_from_observation_id(
            args.observation_id)
    return args


def main():
    args = parse_args()
    acs.set_up_directories()
    acs.check_successful_setup()
    container = acs.TargetContainer(args.observation_id, args.ra, args.dec)
    container.get_science_targets(verbose=args.verbose)
    container.get_white_dwarfs(verbose=args.verbose)
    container.get_guide_stars(verbose=args.verbose)
    container.get_sky_fibres(verbose=args.verbose)
    ut_date = args.utdate if args.utdate is not None else "YYYY MM DD, to be replaced"
    container.write_targets_to_disc(ut_date, verbose=args.verbose)
    if args.utdate is None:
        print("WARNING: Since you did not specify a date, the .fld file only contains a placeholder!")


if __name__ == "__main__":
    main()
