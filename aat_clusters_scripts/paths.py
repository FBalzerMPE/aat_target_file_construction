"""Keep track of where files are stored via a PathProvider"""
import os
import warnings
from pathlib import Path

from astropy.table import Table
from astropy.units import UnitsWarning


class PathProvider:
    """Helper class to provide a handy interface for the path- and filenames used
    throughout this code."""

    def __init__(self):
        self.data = Path("./data")
        self.obs_setup = self.data.joinpath("observation_setup")
        self.sweep = self.obs_setup.joinpath(
            "sweep") if "SWEEP_PATH" not in os.environ else os.environ["SWEEP_PATH"]

        # Register the filepaths for the white dwarfs and the sky fibres:
        self.white_dwarfs_jacob = self.obs_setup.joinpath("WD_gaia.fits")
        self.sky_fibres = self.obs_setup.joinpath("sky_fibres.csv")
        # For an earlier implementation, I used the Gaia edr3 for the WDs.
        # self.white_dwarfs_edr = self.obs_setup.joinpath(
        #     "gaiaedr3_wd_main.fits")

        # The following three files were needed just for my own observation
        self.members = self.obs_setup.joinpath(
            "aat_observation_member_selection.fit")
        self.agn_candidates = self.obs_setup.joinpath("AGN_cluster123.fits")
        self.clusters = self.obs_setup.joinpath("aat_observation_clusters.fit")

    def read_table(self, path: Path):
        """Convenience function to read an astropy table at `path` while
        filtering out warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnitsWarning)
            t = Table.read(path)
        return t

    def get_fld_fname(self, observation_id: int) -> Path:
        """Provide the full filepath for the `.fld` file

        Parameters
        ----------
        observation_id : int
            The ID of the observation

        Returns
        -------
        Path
            The .fld-fileapth.
        """
        return self.obs_setup.joinpath(f"target_catalog_{observation_id}.fld")


PATHS = PathProvider()
