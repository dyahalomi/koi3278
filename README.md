# KOI-3278

This repository provides the code used to model the self-lensing binary, KOI-3278. It will reproduce the bulk of our analysis, including the MCMC modeling and the key figures.

If you make use of this code please cite our work: [Yahalomi et al., 2019] (https://arxiv.org/abs/1904.11063).

We have also included the PARSEC isochrones used (PARSECv1.1/) as well as wget scripts to download the Kepler light curves (to lightcurve/) and white dwarf models from Pierre Bergeron (wdmodels/).

The code provided here was in part adapted from the analysis used in [Kruse and Agol, 2014 (https://science.sciencemag.org/content/344/6181/275) and that is available on their [GitHub](https://github.com/ethankruse/koi3278). The Gelman-Rubin function is adapted from the [RadVel Github](https://github.com/California-Planet-Search/radvel), [Fulton et al., 2017](http://adsabs.harvard.edu/abs/2018PASP..130d4504F).

KOI-3278 was discovered in 2014 in Kepler photometry ([Kruse and Agol, 2014 (https://science.sciencemag.org/content/344/6181/275)). KOI-3278 now has 16 follow-up spectroscopic observations from HIRES (8 observations) and TRES (8 observations). These observations provide us with spectroscopic estimates of the stellar primary parameters (surface gravity, metallicity, effective temperature, and v sin i) as well as radial velocity observations. With these new observations, we can model the system independently with Einsteinian microlensing models (using Kepler photometry and spectroscopic estimates of primary parameters), independently with Newtonian dynamical models (using spectroscopic estimates of the primary parameters and spectroscopic radial velocities), and a joint Einsteinian and Newtonian model (using Kepler photometry, spectroscopic estimates of primary parameters, and spectroscopic radial velocities).


The code is broken up into 5 folders: CompareWDmasses, EinsteinianModel, JointModel, NewtonianMassModel, and NewtonianModel.



## CompareWDmasses

This folder contains the python code used in order to create Figure 7 in the paper.


## EinsteinianModel

This folder contains the python code used in order to create the Einsteinian microlensing model. This code uses the Kepler photometry and the spectroscopic estimates of stellar primary parameters using SPC, Brewer, or SpecMatch analysis on HIRES spectra. The corresponding values from SPC, Brewer, and SpecMatch are in the "EinsteinMCMC_funcs.py" file.


## JointModel

This folder 
