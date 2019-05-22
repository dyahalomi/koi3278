# KOI-3278

This repository provides the code used to model the self-lensing binary, KOI-3278. It will reproduce the bulk of our analysis, including the MCMC modeling and the key figures.

If you make use of this code please cite our work: [Yahalomi et al., 2019](https://arxiv.org/abs/1904.11063).

We have also included the PARSEC isochrones used (PARSECv1.1/) as well as wget scripts to download the Kepler light curves (to lightcurve/) and white dwarf models from Pierre Bergeron (wdmodels/). We also provide the radial velocity observations from HIRES and TRES as well as the spectroscopic estimates of the stellar primary parameters from SPC, Brewer, and SpecMatch analysis of HIRES spectra.

The code provided here was in part adapted from the analysis used in [Kruse and Agol, 2014](https://science.sciencemag.org/content/344/6181/275) and that is available on their [GitHub](https://github.com/ethankruse/koi3278). The Gelman-Rubin function is adapted from the [RadVel Github](https://github.com/California-Planet-Search/radvel), as described in [Fulton et al., 2018](http://adsabs.harvard.edu/abs/2018PASP..130d4504F).

The MCMC modeling uses the emcee package: [GitHub](https://github.com/dfm/emcee) / [Paper](https://arxiv.org/abs/1202.3665).

KOI-3278 was discovered in 2014 in Kepler photometry ([Kruse and Agol, 2014](https://science.sciencemag.org/content/344/6181/275)). KOI-3278 now has 16 follow-up spectroscopic observations from HIRES (8 observations) and TRES (8 observations). These observations provide us with spectroscopic estimates of the stellar primary parameters (surface gravity, metallicity, effective temperature, and v sin i) as well as radial velocity observations. With these new observations, we can model the system independently with Einsteinian microlensing models (using Kepler photometry and spectroscopic estimates of primary parameters), independently with Newtonian dynamical models (using spectroscopic estimates of the primary parameters and spectroscopic radial velocities), and a joint Einsteinian and Newtonian model (using Kepler photometry, spectroscopic estimates of primary parameters, and spectroscopic radial velocities).


## General Tips
In order to run an MCMC model, you must set the `domcmc` variable in the `*_run.py` files to True. In order to create a LaTeX tabel with the median and/or best fit results from the MCMC output chain, set the `outfile` variable in the `*_analyze.py` files to the desired output file name. You can change the number of walkers and the number of iterations in the MCMC run in the `*_run.py` files. You can change the burnout length in the `*_analyze.py` files.



## Folders

The code is broken up into 5 folders: CompareWDmasses, EinsteinianModel, JointModel, NewtonianDynamicalModel, and NewtonianOrbitalModel.



### CompareWDmasses

This folder contains the python code used in order to create Figure 7 in the paper.


### EinsteinianModel

This folder contains the python code used in order to create the Einsteinian microlensing model. The supplementary functions are in "EinsteinMCMC_funcs.py". In order to run the MCMC model, use "EinsteinMCMC_run.py". In order to analyze the MCMC output chain, use "EinsteinMCMC_analyze.py". This code uses the Kepler photometry and the spectroscopic estimates of stellar primary parameters using SPC, Brewer, or SpecMatch analysis on HIRES spectra. The primary stellar parameter estimates from SPC, Brewer, and SpecMatch are in the "EinsteinMCMC_funcs.py" file.


### JointModel

This folder contains the python code used in order to create the Joint Einsteinian microlensing and Newtonian dynamical model.  The supplementary functions are in "JointMCMC_funcs.py" and the orbital fitting specific functions are in "JointMCMC_RVfuncs.py". In order to run the MCMC model, use "JointMCMC_run.py". In order to analyze the MCMC output chain, use "JointMCMC_analyze.py". This code uses the Kepler photometry, the spectroscopic estimates of stellar primary parameters using SPC, Brewer, or SpecMatch analysis on HIRES spectra, and the radial velocity observations from HIRES and TRES. The primary stellar parameter estimates from SPC, Brewer, and SpecMatch are in the "JointMCMC_funcs.py" file. The radial velocity observations from TRES and HIRES are in the "inputs.py" file.

### NewtonianDynamicalModel

This folder contains the python code used in order to create the complete Newtonian dynamical model. The supplementary functions are in "NewtonianDynamicalMCMC_funcs.py" and the orbital fitting specific functions are in "NewtonianDynamicalMCMC_RVfuncs.py". In order to run the MCMC model, use "NewtonianDynamicalMCMC_run.py". In order to analyze the MCMC output chain, use "NewtonianDynamicalMCMC_analyze.py". This code uses the spectroscopic estimates of stellar primary parameters using SPC, Brewer, or SpecMatch analysis on HIRES spectra and the radial velocity observations from HIRES and TRES. The primary stellar parameter estimates from SPC, Brewer, and SpecMatch are in the "NewtonianDynamicalMCMC_run.py" file. The radial velocity observations from TRES and HIRES are in the "NewtonianDynamicalMCMC_run.py" file.


### NewtonianOrbitalModel

This folder contains the python code used in order to create the orbital fit to the radial velocity observations only. The supplementary functions are in "NewtonianOrbitalMCMC_funcs.py". In order to run the MCMC model, use "NewtonianOrbitalMCMC_run.py". In order to analyze the MCMC ouput chain, use "NewtonianOrbitalMCMC_analyze.py". This code uses the radial velocity observations from HIRES and TRES. The radial velocity observations from TRES and HIRES are in the "NewtonianOrbitalMCMC_run.py" file.





