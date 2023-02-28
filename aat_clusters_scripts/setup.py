"""Script that includes all functions necessary for the setup."""
import os
from pathlib import Path

from .paths import PATHS


def set_up_directories():
    """Set up the directories (i. e. a nested data- and plotting directory)
    that are necessary for the scripts to work.
    """
    # TODO: Find Path where aat_clusters_scripts lives in
    # TODO: Then, create data/plots/ and dataobservation_preparation/sweep
    # TODO: The sweep directory should only be created if SWEEP_FILEPATH is not an environment variable
    # TODO: Then, perform checks on the required files and instruct user provide them (and notify for incoming SWEEP download)
    pass


def check_successful_setup():
    # TODO: Check availability of all files.
    assert (p := PATHS.white_dwarfs_jacob).is_file(), (
        f"Please download the White Dwarf file and deposit it at {p}")
