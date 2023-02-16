from dataclasses import dataclass, field

from astropy.table import Table
from . import PATHS


@dataclass
class TargetContainer:
    obs_ra: float
    obs_dec: float
    white_dwarfs: Table = field(init=False)
    guide_stars: Table = field(init=False)
    sky_fibres: Table = field(init=False)
    science_targets: Table = field(init=False)

    def get_white_dwarfs(self):
        all_white_dwarfs = PATHS.read_table(PATHS.white_dwarfs_jacob)
        col_map = {"_RAJ2000": "ra", "_DEJ2000": "dec",
                   "WD": "WDJ_name", "Source": "source_id",
                   "pmRA": "pmra", "pmDE": "pmdec"}
        all_white_dwarfs.rename_columns(
            list(col_map.keys()), list(col_map.values()))
