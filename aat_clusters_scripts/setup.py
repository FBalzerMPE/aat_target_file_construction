"""Script that includes all functions necessary for the setup."""
import os
from pathlib import Path

from .paths import PATHS


def set_up_directories():
    """Set up the directories (i. e. a nested data- and plotting directory)
    that are necessary for the scripts to work.
    """
    run_dir = Path(os.getcwd())
    for subpath in ["data/observation_setup/", "data/plots"]:
        run_dir.joinpath(subpath).mkdir(parents=True, exist_ok=True)
    PATHS.sweep.mkdir(exist_ok=True)
    print("Successfully set up the necessary filepaths.")
    print(
        f"WARNING: The potentially big SWEEP files will be downloaded to {PATHS.sweep}.")


def check_successful_setup():
    assert (p := PATHS.white_dwarfs_jacob).is_file(), (
        f"Please download the White Dwarf file and deposit it at {p}")
    # You might want to assess whether your observation targets are available as well.
