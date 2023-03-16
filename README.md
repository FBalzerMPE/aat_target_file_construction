# Setup of AAT observations

This repository shows how to set up an observation at the AAT.

For each observation (which, in my case, corresponds to one cluster), we need to produce a single `.fld` file in the correct format.\
This file needs to include:

- The targets to observe (`members`) as the science targets
- White Dwarfs for spectroscopic calibration
- Guide stars
- Sky fibres

Feel free to adapt it to your own observations; I tried to specify the points that might need to be modified for your observations, and they are also listed in [this section](#things_to_change) below.
  
## :arrow_forward: Set-up and running the program

I provide one (in the future maybe two) handles to run this program:

- A jupiter notebook [recommended] that includes the setup and all the steps described below.
- <s>A command-line script that can be invoked [not yet tested and only preliminary, sorry]</s>

These handles both include a setup routine to automatically generate a data and a plot directory relative to the directory of where the code is run.

### :exclamation: Requirements

For this script to work some requirements need to be met first:

#### Packages

The following packages need to be installed:

- `astropy`
- [`astroquery` (only necessary if you want to do Simbad or Gaia queries, see the 'unused functions.py' script for more information)]
- Standard modules like `numpy`, `scipy` (for the matching algorithm of astropy), `matplotlib`, and `pandas`
- Sufficiently large space on disk to save the downloaded SWEEP files (~2 GB/observation)
  
Also, a python version $>$ 3.9 is recommended, but not entirely necessary (with earlier versions, you might need to replace some types in the type hints with imports from the `typing` module, e. g. replace `tuple[str, str]` with `Tuple[str, str]` and include `from typing import Tuple` in the header.)

#### Space requirements

In addition to the python packages, the script will download necessary LS DR 10 sweep files!\
These can take up space of a few GB. If you want a custom directory, please set up a `SWEEP_PATH` environment variable (simply add

```
export SWEEP_PATH="your_filepath"
```

to your `.bashrc` (or equivalent setup of environment variables on other OS) for the files  to be saved to and retrieved from this specified path).

#### Other

The white dwarf file needs to be available and downloaded to the specified path.

## :telescope: Description of a single-observation setup

For each of the observations, the set up needs to follow the same procedure, which I have adapted from the scripts by Jacob.\
For this purpose, I have provided a class called `TargetContainer` which is successively filled with information.\
The main code for these operations can be found in `aat_clusters_scripts`.

### Initialise central observational coordinates

First, an instance of a `TargetContainers` with ID for the observation (in this case, this corresponds to the `cluster_id` associated with the cluster members) and the `ra` and `dec` for the centre of the observation is created.\
This is necessary as it keeps track of all of the sub-tables, allowing us to manipulate them separately.\
In my case, the coordinates correspond to the central coordinates of the clusters that I am about to observe.

### Specifying science targets

Then, the science targets need to be loaded in.\
In the case of this script, we have galaxy cluster members that are loaded depending on the id, and AGN, which we simply select from a 1 degree radius around the central RA and DEC.\
In my case, I selected the cluster members to observe and formatted the table to have the columns we need.\
In addition to that, I discard all sources with $r_\text{mag} < 17.5$ as they are too bright.\
Then, the positional information of AGN is loaded in to provide additional targets as the cluster members are only in the centre of the telescopes field of view.

### Spectroscopic calibration: Adding White Dwarfs

We need to observe a few White Dwarfs to later be able to calibrate the spectra of the science targets.

As a basis, I used `WD_gaia.fits` [citation/origin needed!] which has also been used by Jacob.

(The White Dwarf catalogue could possibly also be obtained from the Gaia EDR3 ([download link](https://warwick.ac.uk/fac/sci/physics/research/astro/research/catalogues/gaiaedr3_wd_main.fits.gz)), described Gentile Fusillo et al. 2021 (see [here](https://arxiv.org/pdf/2106.07669.pdf) for the pdf).)

For the selection, we follow the procedure adopted in `SelectWD1.py` with an additional restriction:

- We select all white dwarfs within a radius of `1 deg` of the central ra and dec for each cluster.
- Also, we require `RPmag`, `pmra` und `pmdec` to exist (although this seems to be ok for all sources).
- We discard all sources with $r_\text{mag} < 17.5$ as they might be too bright for proper calibration.
- Finally, we select 10 of them since there should not be more than two needed for the calibration, and this allows for optimal fibre selection by the telescope.

### Guide star selection

Guide stars are needed to guide the telescope during the obervation.\
*Note: Instead of following `Guidestargen1.py` (also described in [this section](#alternative-guide-star-selection)) to select Guide Stars from Simbad (which yielded insufficient coverage for my use case), I have developed an alternative approach.*

To obtain the guide stars, we make use of the SWEEP catalogue of the [Legacy Surveys DR 10](https://www.legacysurvey.org/dr10/description/), focusing on stars with Gaia magnitude information.

- First, all PSF-like objects of the SWEEP catalogues within 1 degree of the central coordinates are selected.
- Then, we filter for sources with `14.0 < rPmag < 14.5` and small proper motion of $pm < 50$ mas/yr, where we calculate $pm:=\sqrt{(0.3977 pm_\text{ra})^2+pm_\text{dec}^2}$.
- All sources without `rPmag`, `pmra`, or `pmdec` information are dropped.
- The sources are sorted by their `rP` magnitude and the 150 brightest ones are selected as more shouldn't be necessary.

### Sky fibre selection

Sky fibres are fibres that are pointed on spots of the sky without sources to later be able to perform realistic background subtraction.\
*Note: Instead of following `skyfibrefromJacobfile3.py` (also described in [this section](#alternative-sky-fibre-selection)), I have developed an alternative approach.*

We generate the sky fibres using the following procedure and data from the SWEEP catalogue of the [Legacy Surveys DR 10](https://www.legacysurvey.org/dr10/description/):

- First, we filter for all sources in the circular observation region.
- Then, we generate a uniform grid of 20 000 RA and DEC pairs in that region.
- Via crossmatching, we select only the coordinates that are further than 10 arcseconds away from the SWEEP sources.
- Then, we make sure that the sky fibres are not too close to each other by filtering all of the coordinates that are closer than 1 arcmin to another one of the coordinates.
- We pretend that each of them has an `rmag` of 30 and no proper motion for observational purposes.
- Finally, we select 150 of them as more shouldn't be necessary.

### Further steps to finish up the preparation

After all of the above steps have been performed, the sources can be stacked into one big table, and the `.fld` file can be created.\
Here, we make sure that priorities are assigned, that all sub-tables have the same column structure, and that the object types (science targets (`= P`), guide stars (`= F`), sky fibres (`= S`)) are correctly assigned.\
In addition to that, object names are sanitized so they don't contain any whitespaces, and RA and DEC are converted to the correct units (hms, dms), as well as a unit conversion for the proper motion to arcsec/yr.\
The `.fld` file should also be provided with the correct header, which is constructed on runtime.

The `TargetContainer` then also provides a convenience method to plot the sources.

## <a id="things_to_change" name="things_to_change"></a>:warning:Things to change for different set-ups

To make full use of this script for setting up your own observation, the only major thing to change lies in `load_science_targets.py`, as you'll have to modify the `get_science_targets` function.\
*Hint*: You may use the observation-id to your advantage for that purpose, or just select targets from your catalogue in a circular area around the central ra and dec.

After you have done so, you may simply run the `0_target_file_construction.ipynb` notebook to fully generate the `.fld` file for your observation.

## Alternative approaches

In Jacobs scripts, alternative methods for selecting guide stars and sky fibres have been suggested. I have applied them, but due to insufficient guide star coverage and the unavailability of sky fibres in our fields I have resorted to my own approach.\
Nevertheless, these methods are described below:

### Alternative guide star selection

To select guide stars to guide the telescope during the observation, a similar procedure as in `Guidestargen1.py` could also be employed:

- Here, the stars can first be obtained via `astroquery` which could also manually be done by performing a coordinate query [on the Simbad site](http://simbad.cds.unistra.fr/simbad/sim-fcoo), but the use of astroquery seems to be more convenient (I have briefly tested it, it yields the same results).\
A search radius of `1 degree` around the centre of each observation should be used.
- Then, all guide stars without `pmra`, `pmdec` or `rmag` values are removed.
- All sources too dim, too bright, or with a high proper motion are removed, corresponding to the following criteria:
  - $12 < r_\text{mag} < 14$
  - $pm < 20$ mas/yr, where $pm:=\sqrt{(0.3977 pm_\text{ra})^2+pm_\text{dec}^2}$
- Then, it is made sure that the type of the sources is stellar.
- Finally, the sources are sorted by their r magnitude.

### Alternative sky fibre selection

For the sky fibres,the procedure in `skyfibrefromJacobfile3.py` could also be employed:

- First, we load the general fibre file only containing their ra and dec.
- Again, we make sure to select only fibres in a `1 degree` range of the observational centre.
- We pretend that each of them has an `rmag` of 30 and no proper motion.
