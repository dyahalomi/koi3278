"""
Supplementay functions needed to do the MCMC run and analysis,
including loading the isochrones.
"""

# TODO: switch out to a faster Mandel-Agol when needed
# import RV_funcs and RV observations
from glob import glob
import numpy as np
import sys
from scipy import interpolate
import numpy.polynomial.polynomial as poly
import NewtonianMassMCMC_RVfuncs as RV_f

def initrange(p):
	"""
	Return initial error estimates in each parameter.
	Used to start the MCMC chains in a small ball near an estimated solution.

	Input
	-----
	p : ndarray
		Model parameters.

	Returns
	-------
	errs : ndarray
		The standard deviation to use in each parameter
		for MCMC walker initialization.
	"""
	#p = (period, ttran, ecosw, esinw, K, M1, FeH, gamma, gamma_offset, sigma_jitter1_sqrd, sigma_jitter2_sqrd)
	return np.array([5.8e-03,   1.4e-01,   2.5e-03,   2.5e-03,   3.8e-02,   9.2e-02,   2.3e-01,   4.8e-02,  5.7e-02,  1e-5,  1e-5])


def loadisos():
	"""
	Loads all objects needed to interpolate and analyze the isochrones.
	All inputs are listed in the input file to make things easier and
	because they are referenced elsewhere.

	Returns
	-------
	isobundle : tuple
		(magname, interps, limits, fehs, ages, maxmasses)
	"""
	# get the inputs
	from inputs import (isodir, inds, magname)
	import warnings

	# we want these to be calculated by the isochrones too
	magname.append('Rad')
	magname.append('logg')
	magname.append('teff')

	magname = np.array(magname)

	padsdssfiles = isodir + '*sdss'
	padwisefiles = isodir + '*wise'
	# find the isochrones and make sure you have equivalent ones
	# in each filter system
	sdssisos = glob(padsdssfiles)
	sdssisos.sort()
	wiseisos = glob(padwisefiles)
	wiseisos.sort()

	if len(sdssisos) != len(wiseisos):
		print 'Error! Mismatched isochrones.'
		sys.exit(1)
	if len(sdssisos) == 0:
		print 'Cannot find isochrones!'

	# load and concatenate the isochrones
	for ii in np.arange(len(sdssisos)):
		iso1 = np.loadtxt(sdssisos[ii])
		iso2 = np.loadtxt(wiseisos[ii])
		# first 8 columns are the same, then you get into the bands
		# next 3 are repeats of JHK, so ignore those too
		together = np.concatenate((iso1, iso2[:, 11:]), axis=1)
		if ii == 0:
			fulliso = together * 1.
		else:
			fulliso = np.concatenate((fulliso, together))

	# pull out the indices in the list we want to evaluate
	maginds = []
	for ii in magname:
		maginds.append(inds[ii])
	maginds = np.array(maginds)
	maginds.astype(int)

	# convert from Padova Z into metallicity
	zs = fulliso[:, inds['feh']]
	fesol = 0.0147
	hsol = 0.7106
	hnew = 1. - 0.2485 - 2.78 * zs
	fulliso[:, inds['feh']] = np.log10((zs/hnew)/(fesol/hsol))

	# calculate radii two different ways
	G = 6.67e-8  # cgs units
	Msun = 1.9884e33  # g
	Rsun = 6.955e10  # cm
	R = np.sqrt(G * fulliso[:, inds['M']] * Msun /
				10.**fulliso[:, inds['logg']]) / Rsun
	sigma = 5.6704e-5  # cgs
	Lsun = 3.846e33  # erg/s
	R2 = np.sqrt(10.**fulliso[:, inds['lum']]*Lsun /
				 (4.*np.pi*sigma*(10.**fulliso[:, inds['teff']])**4.)) / Rsun
	# use the average of these two radii measures
	R = (R + R2)/2.
	# add this to the isochrone
	fulliso = np.concatenate((fulliso, R[:, np.newaxis]), axis=1)

	# what are the metallicities and ages in this set?
	fehs = np.unique(fulliso[:, inds['feh']])
	ages = np.unique(fulliso[:, inds['age']])
	minfeh = fehs.min()
	maxfeh = fehs.max()
	minage = ages.min()
	maxage = ages.max()

	# set up the mass interpolations
	interps = np.zeros((len(fehs), len(ages)), dtype=object)
	maxmasses = np.zeros((len(fehs), len(ages)))

	# for each Fe/H and age, create a mass interpolation function and
	# record the maximum mass still alive
	# interps[ii,jj](M) will return the desired parameters
	# (listed in magname) at a given mass
	for ii in np.arange(len(fehs)):
		for jj in np.arange(len(ages)):
			small = np.where((fulliso[:, inds['feh']] == fehs[ii]) &
							 (fulliso[:, inds['age']] == ages[jj]))[0]
			interps[ii, jj] = interpolate.interp1d(
				fulliso[small, inds['M']], fulliso[small][:, maginds],
				axis=0, bounds_error=False)
			maxmasses[ii, jj] = fulliso[small, inds['M']].max()


	# bounds where we trust the results
	limits = (minfeh, maxfeh, minage, maxage)

	# bundle all the important bits of the model together to feed
	# to functions that need them
	isobundle = (magname, interps, limits, fehs,
				 ages, maxmasses)



	return isobundle





def isointerp(M, FeH, age, isobundle, testvalid=False):
	"""
	Interpolate the isochrones to predict the desired observables
	(listed in the variable magname) at a particular mass, age, and
	Fe/H. We find the bounding 4 combinations of Fe/H and age in the
	isochrone grid, and predict the observables at these 4 (Fe/H,age)
	combinations at the input mass using the mass interpolation
	[e.g. interps[Fe/H,age](Mass)]. We then perform a bilinear
	interpolation of these 4 (Fe/H,age) locations to get to the
	predicted observables at the input Fe/H and age.

	Input
	-----
	M : float
		Input mass (solar)
	FeH : float
		Input metallicity [Fe/H]
	age : float
		Input age (log10(yr))
	isobundle : tuple
		Contains everything needed for the isochrones
	testvalid : bool, optional
		If True, returns a boolean indicating whether this combination of
		inputs is within the bounds of the isochrones.

	Returns
	-------
	result : ndarray
		Interpolated values of all parameters listed in magname.
	"""
	# unpack the model bundle

	(magname, interps, limits, fehs, ages, maxmasses) = isobundle
	minfeh, maxfeh, minage, maxage = limits

	# make sure this is a valid set of inputs
	if age >= minage and age <= maxage and FeH >= minfeh and FeH <= maxfeh:
		# what grid points of Fe/H and age is this input between
		fehinds = np.digitize([FeH], fehs)
		fehinds = np.concatenate((fehinds-1, fehinds))
		ageinds = np.digitize([age], ages)
		ageinds = np.concatenate((ageinds-1, ageinds))

		# bilinear interpolation done by hand
		fehdiff = np.diff(fehs[fehinds])[0]
		agediff = np.diff(ages[ageinds])[0]

		# step 1
		interp1 = (interps[fehinds[0], ageinds[0]](M) *
				   (fehs[fehinds[1]] - FeH) +
				   interps[fehinds[1], ageinds[0]](M) *
				   (FeH - fehs[fehinds[0]])) / fehdiff
		# step 2
		interp2 = (interps[fehinds[0], ageinds[1]](M) *
				   (fehs[fehinds[1]] - FeH) +
				   interps[fehinds[1], ageinds[1]](M) *
				   (FeH - fehs[fehinds[0]])) / fehdiff
		# step 3 of the bilinear interpolation
		result = ((interp1 * (ages[ageinds[1]] - age) +
				  interp2 * (age - ages[ageinds[0]])) / agediff)
	# otherwise return bad values
	else:
		result = np.zeros(len(magname)+3)
		result[:] = np.nan
	if testvalid:
		return np.isfinite(result).all()

	return result


def msage(M, FeH, isobundle):
	"""
	Return the liftime of a star of mass M and metallicity FeH based
	on the isochrones in the isobundle.

	Returns
	Input
	-----
	M : float
		Input mass (solar)
	FeH : float
		Input metallicity [Fe/H]
	isobundle : tuple
		Contains everything needed for the isochrones

	Returns
	-------
	finalage : float
		log10(lifetime [years]) of the star
	"""
	(magname, interps, limits, fehs, ages, maxmasses) = isobundle
	minfeh, maxfeh, minage, maxage = limits

	if FeH >= minfeh and FeH <= maxfeh:
		# which two isochrones is this star between
		fehinds = np.digitize([FeH], fehs)
		fehinds = np.concatenate((fehinds-1, fehinds))

		twoages = np.zeros(len(fehinds))
		# for each bounding metallicity
		for ii in np.arange(len(fehinds)):
			# the ages where the max mass is still bigger
			# than the current guess
			srch = np.where(maxmasses[fehinds[ii], :] >= M)[0]
			# very short MS lifetime, not even on the isochrones
			if len(srch) == 0:
				twoages[ii] = minage
			# hasn't evolved yet!
			elif srch[-1] == len(ages) - 1:
				twoages[ii] = maxage
			else:
				srch = srch[-1]
				# mass in the grid that died at the previous age and this age
				bounds = maxmasses[fehinds[ii], srch:srch+2]
				# do a linear interpolation to get the age this mass
				# would have died
				diff = bounds[0] - bounds[1]
				twoages[ii] = ((bounds[0] - M)/diff * ages[srch+1] +
							   (M - bounds[1])/diff * ages[srch])

		# do a linear interpolation between bounding metallicities to figure
		# out the age a star of this mass and metallicity would die
		diff = fehs[fehinds[1]] - fehs[fehinds[0]]
		finalage = ((fehs[fehinds[1]] - FeH)/diff * twoages[0] +
					(FeH - fehs[fehinds[0]])/diff * twoages[1])
		return finalage
	# the star isn't within the range of the isochrones
	return 0.




def logprior(p, isobundle):
	"""
	Priors on the input parameters.

	Input
	-----
	p : ndarray
		Orbital parameters. 
	isobundle : tuple
		Contains everything needed for the isochrones

	Returns
	-------
	prior : float
		Log likelihood of this set of input parameters based on the
		priors.
	"""
	# fix limb darkening

	(period, ttran, ecosw, esinw, K, M1, FeH, gamma, gamma_offset, sigma_jitter1_sqrd, sigma_jitter2_sqrd) = p
	(magname, interps, limits, fehs, ages, maxmasses) = isobundle
	
	age = msage(M1, FeH, isobundle)
	# to get in log(age) like the interpolation needs
	age = np.log10(age * 1e9)

	# check to make sure that it's valid within the models.
	if not isointerp(M1, FeH, age, isobundle, testvalid=True):
		return -np.inf

	# reconvert into more useful orbital elements
	e = np.sqrt(ecosw**2. + esinw**2.)
	omega = np.arctan2(esinw, ecosw)

	# if any of the parameters are unphysical, return negative infinity
	# log likelihood (impossible)
	if (period < 0. or e < 0. or e >= 1. or M1 < 0 or omega < -np.pi/2 or omega > np.pi/2 or sigma_jitter1_sqrd**.5 > 1 or sigma_jitter1_sqrd < 0 or sigma_jitter2_sqrd**.5 > 1 or sigma_jitter2_sqrd < 0):
		return -np.inf
	# otherwise return a uniform prior (except modify the eccentricity to
	# ensure the prior is uniform in e)
	
	totchisq = 0
	return -totchisq / 2. - np.log(e)

	



def loglikeli(p, isobundle, RVbundle, specbundle, minimize=False,
			  retmodel=False, retpoly=False,  indchi=False, **kwargs):
	"""
	Compute the log likelihood of an RV event with these orbital
	parameters given the data.

	Input
	-----
	p : ndarray
		Orbital parameters.
	isobundle : tuple
		Contains everything needed for the isochrones
	RVbundle : list
		Contains all the RV data
		t1, RV1 RVerr1, t2, RV2, RVerr2
	specbundle : list
		Contains all the spectroscopy data
		logg_spec, logg_err_spec, Teff_spec, Teff_err_spec, FeH_spec, FeH_err_spec
	minimize : boolean, optional
		If True, we are trying to minimize the chi-square rather than
		maximize the likelihood. Default False.
	retmodel : boolean, optional
		If True, return the model fluxes instead of the log likelihood.
		Default False.
	retpoly : boolean, optional
		If True, return the polynomial portion of the model fluxes
		instead of the log likelihood. Default False.
	indchi : boolean, optional
		If True, return the chi-square of each individual event.
		Default False.

	Returns
	------
	likeli : float
		Log likelihood that the model fits the data.
	"""  
	(period, ttran, ecosw, esinw, K, M1, FeH, gamma, gamma_offset, sigma_jitter1_sqrd, sigma_jitter2_sqrd) = p
	(t_RV1, RV1, RVerr1, t_RV2, RV2, RVerr2) = RVbundle
	(logg_spec, logg_err_spec, Teff_spec, Teff_err_spec, FeH_spec, FeH_err_spec) = specbundle


	# calculate the chi-square from the magnitudes
	(magname, interps, limits, fehs, ages, maxmasses) = isobundle
	
	age = msage(M1, FeH, isobundle)

	# to get in log(age) like the interpolation needs
	age = np.log10(age * 1e9)

	mags = isointerp(M1, FeH, age, isobundle)
	

	# save radius, logg, teff measurements
	R1 = mags[-3]
	logg = mags[-2]
	Teff = 10. **mags[-1]


	totchisq = (logg - logg_spec)**2. / (logg_err_spec**2.)
	totchisq += (Teff - Teff_spec)**2. / (Teff_err_spec**2.)
	totchisq += (FeH - FeH_spec)**2. / (FeH_err_spec**2.)

	#see Eastman et al., 2013 equation 2
	#see Christiansen et al., 2017 eq 1 (sec 3.2)
	loglikelihood = - (np.log(np.sqrt(2 * np.pi * (logg_err_spec**2.))) + 
		np.log(np.sqrt(2 * np.pi * (Teff_err_spec**2.))) + 
		np.log(np.sqrt(2 * np.pi * (FeH_err_spec**2.))) + 
		(0.5 * totchisq)) 

	# Add constraint that G dwarf age should be greater than or equal to
	# spin-down age of 0.89+-0.15 Gyr:
	#see Christiansen et al., 2017 eq 1 (sec 3.2)
	age = (10.**age) / 1e9
	if age < 0.89:
		agechisq = (age - 0.89)**2. / (0.15**2.)
		totchisq += agechisq
		loglikelihood += - (np.log(np.sqrt(2 * np.pi * (0.15**2.))) + (0.5 * agechisq))


	#Add RV fitting
	e = np.sqrt(ecosw**2. + esinw**2.)
	omega = np.arctan2(esinw, ecosw)


	# set RV parameters for RV_model...assumes using both TRES and HIRES data
	p_RV = (period, ttran, ecosw, esinw, K, gamma, gamma_offset, sigma_jitter1_sqrd, sigma_jitter2_sqrd)


	loglikelihood += RV_f.loglikelihood_2obs(p_RV, t_RV1, RV1, RVerr1, t_RV2, RV2, RVerr2)[0]
	totchisq += RV_f.loglikelihood_2obs(p_RV, t_RV1, RV1, RVerr1, t_RV2, RV2, RVerr2)[1]


	# if we're minimizing chi-square instead of maximizing likelihood
	if minimize:
		return totchisq


	#return loglikelihood
	return loglikelihood


def logprob(p, isobundle, RVbundle, specbundle, minimize=False):
	"""
	Get the log probability of the data given the priors and the model.
	See loglikeli for the input parameters.

	Also requires npert (the number of subsamples to divide a cadence into).

	Returns
	-------
	prob : float
		Log likelihood of the model given the data and priors, up to a
		constant.
	"""

	lp = logprior(p, isobundle)
	if not np.isfinite(lp):
		# minimization routines don't handle infinities very well, so
		# just penalize impossible parameter space
		if minimize:
			return 1e6
		return -np.inf
	if minimize:
		return -lp + loglikeli(p, isobundle, RVbundle, specbundle,
							   minimize=minimize)
	return lp + loglikeli(p, isobundle, RVbundle, specbundle,
						  minimize=minimize)

