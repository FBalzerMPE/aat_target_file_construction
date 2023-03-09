"""Script to perform the target selection for an AAT observation."""
import argparse

import aat_clusters_scripts as acs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("observation_id", type=int)
    parser.add_argument("ra", type=float)
    parser.add_argument("dec", type=float)
    parser.add_argument("verbose", action="store_true")
    args = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    container = acs.TargetContainer(args.observation_id, args.ra, args.dec)
    container.get_science_targets()
    # TODO Finish this up


if __name__ == "__main__":
    main()
