"""Warning: These functions were convenient to use in the past, but are
deprecated as I've found better methods to tackle e. g. the problem of
finding guide stars."""
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad


def _retrieve_simbad_table(ra: float, dec: float, radius: float = 1) -> Table:
    """Retrieve the simbad sources around the given central `ra` and `dec` within
    the provided `radius` (in deg).
    Also obtain proper motion and r magnitude information.
    """
    custom_simbad = Simbad()
    # print(custom_simbad.get_votable_fields())
    custom_simbad.add_votable_fields(
        "fluxdata(r)", "pmra", "pmdec", "otype", "morphtype", "otypes", "otype(opt)", "rvz_type", "sptype")
    sources = custom_simbad.query_region(
        SkyCoord(ra, dec, unit="deg"), radius=radius * u.deg)
    return sources


def _sanitize_simbad_table(table: Table) -> Table:
    # Sanitize the table by lowercasing colnames and turning ra and dec to degrees
    table.rename_columns(cols := table.colnames, [col.lower() for col in cols])
    sky_coords = SkyCoord(
        table["ra"], table["dec"], unit=(u.hourangle, u.deg))
    table["ra"] = sky_coords.ra
    table["dec"] = sky_coords.dec
    table.rename_column("flux_r", "rmag")
    table.rename_column("main_id", "obj_name")
    return table


def _find_gaia_sources(ra, dec) -> Table:
    """This was used to find and select Gaia sources in the range, before reverting to SWEEP sources."""

    # print(gaiadr3_table)
    # print("\n".join([col.name for col in gaiadr3_table.columns]))
    job = Gaia.launch_job(f"""
    select source_id, ra, dec, ref_epoch, pmra, pmdec, phot_rp_mean_mag 

    from gaiadr3.gaia_source 
                        
    where phot_rp_mean_mag < 15 and phot_rp_mean_mag > 14
        and ra < {ra + 1} and ra > {ra - 1} and dec < {dec + 1} and dec > {dec -1}
    """)

    r = job.get_results()

    print(r)
