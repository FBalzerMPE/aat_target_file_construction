
import warnings
from pathlib import Path

from astropy.table import Table
from astropy.units import UnitsWarning


class PathProvider:
    """Helper class to keep track of the filenames used throughout this code."""

    def __init__(self):
        self.data = Path("./data")
        self.obs_setup = self.data.joinpath("observation_setup")
        self.members = self.obs_setup.joinpath(
            "aat_observation_member_selection.fit")
        self.agn_candidates = self.obs_setup.joinpath("AGN_cluster123.fits")
        self.clusters = self.obs_setup.joinpath("aat_observation_clusters.fit")
        self.white_dwarfs_edr = self.obs_setup.joinpath(
            "gaiaedr3_wd_main.fits")
        self.white_dwarfs_jacob = self.obs_setup.joinpath("WD_gaia.fits")
        # self.sky_fibres = self.obs_setup.joinpath("sky.sep_feb22.fit")
        self.sky_fibres = self.obs_setup.joinpath("SkyFibers.csv")
        self.sweep = self.obs_setup.joinpath("sweep")

    def read_table(self, path: Path):
        """Convenience function to read an astropy table while filtering out warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnitsWarning)
            t = Table.read(path)
        return t

    def get_fld_fname(self, cluster_id: int) -> Path:
        return self.obs_setup.joinpath(f"target_catalog_{cluster_id}.fld")


PATHS = PathProvider()
