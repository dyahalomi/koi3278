import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner


def initrange(p):
	"""
	Return initial error estimates in each parameter.
	Used to start the MCMC chains in a small ball near an estimated solution.

	Input
	-----
	p : ndarray
		Model parameters. See light_curve_model for the order.

	Returns
	-------
	errs : ndarray
		The standard deviation to use in each parameter
		for MCMC walker initialization.
	"""

	if len(p) == 7:
		return np.array([5.8e-03,   1.4e-01,   2.5e-03,   2.5e-03,   3.8e-02,   4.8e-02,  1e-5])
	if len(p) == 9:
		return np.array([5.8e-03,   1.4e-01,   2.5e-03,   2.5e-03,   3.8e-02,   4.8e-02,  5.7e-02,  1e-5,  1e-5])

def kepler(M, e):
	"""
	Simple Kepler solver.
	Iterative solution via Newton's method. Could likely be sped up,
	but this works for now; it's not the major roadblock in the code.

	Input
	-----
	M : ndarray
	e : float or ndarray of same size as M

	Returns
	-------
	E : ndarray
	"""

	M = np.array(M)
	E = M * 1.
	err = M * 0. + 1.

	while err.max() > 1e-8:
		#solve using Newton's method
		guess = E - (E - e * np.sin(E) - M) / (1. - e * np.cos(E))
		err = np.abs(guess - E)
		E = guess

	return E


def RV_model(t, p):
	"""
	Given the orbital parameters compute the RV at times t.

	Input
	-----
	t : ndarray
		Times to return the model RV.
	p : ndarray
		RV parameters.
		period [days], ttran [days], e,  omega [radians]
		K [km/s], gamma [km/s]

	Returns
	-------
	RV_model : ndarray
		RV corresponding to the times in t [km/s].

	"""

	(period, ttran, ecosomega, esinomega, K, gamma, gamma_offset, sigma_jitter1_sqrd, sigma_jitter2_sqrd) = p
	e = np.sqrt(ecosomega**2. + esinomega**2.)
	omega = np.arctan2(esinomega, ecosomega)

	#mean motion: n = 2pi/period
	n = 2. * np.pi / period

	# Sudarsky 2005 Eq. 9 to convert between center of transit
	# and pericenter passage (tau)



	edif = 1. - e**2.
	fcen = np.pi/2. - omega
	tau = (ttran + np.sqrt(edif) * period / (2 * np.pi) * 
		  (e * np.sin(fcen) / (1. + e * np.cos(fcen)) - 2. / np.sqrt(edif) * 
		  np.arctan(np.sqrt(edif) * np.tan(fcen / 2.) / (1. + e))))


	#Define mean anomaly: M
	M = (n * (t - tau)) % (2. * np.pi)

	#Determine the Energy: E
	E = kepler(M, e)

	#Solve for fanom (measure of location on orbit)
	tanf2 = np.sqrt((1. + e) / (1. - e)) * np.tan(E / 2.)
	fanom = (np.arctan(tanf2) * 2.) % (2. * np.pi)

	#Calculate RV at given location on orbit
	RV = K * (e * np.cos(omega) + np.cos(fanom + omega)) + gamma

	return RV



def loglikelihood(p, t1, RV1, RVerr1, t2, RV2, RVerr2):
	"""
	Compute the log likelihood of a RV signal with these orbital
	parameters given the data. 
	
	Input
	-----
	p : ndarray
		Orbital parameters. See RV model for order
	t, RV, RVerr : ndarray
		times, RV, and RV errors of the data.
		
	Returns
	------
	chisq, likeli : tuple of floats
		(
		Log likelihood that the model fits the data, 
		chisq that model fits data
		)
	"""

	(period, ttran, ecosomega, esinomega, K, gamma, gamma_offset, sigma_jitter1_sqrd, sigma_jitter2_sqrd) = p

	#compute RV model light curve for dataset 1
	model1 = RV_model(t1, p)

	#compute RV model light curve for dataset 2
	model2 = RV_model(t2, p)

	#compute loglikelihood for model 1
	#Eastman et al., 2013 equation 
	#Christiansen et al., 2017 sec. 3.2 eq. 1
	totchisq = np.sum((RV1-model1)**2. / ( (RVerr1**2. + sigma_jitter1_sqrd) ))
	loglikelihood1 = -np.sum( 
		(RV1-model1)**2. / ( 2. * (RVerr1**2. + sigma_jitter1_sqrd) ) +
		np.log(np.sqrt(2. * np.pi * (RVerr1**2. + sigma_jitter1_sqrd)))
		)

	#compute loglikelihood for model 2
	#Eastman et al., 2013 equation 
	#Christiansen et al., 2017 sec. 3.2 eq. 1
	totchisq += np.sum((RV2-model2+gamma_offset)**2. / ( (RVerr2**2. + sigma_jitter2_sqrd) ))
	loglikelihood2 = -np.sum( 
		(RV2-model2+gamma_offset)**2. / ( (RVerr2**2. + sigma_jitter2_sqrd) ) +
		np.log(np.sqrt(2. * np.pi * (RVerr2**2. + sigma_jitter2_sqrd)))
		)

	#sum the loglikelihoods
	loglikelihood = loglikelihood1 + loglikelihood2
	#print loglikelihood
	return (loglikelihood, totchisq)


def logprior(p):
	"""
	Priors on the input parameters.

	Input
	-----
	p : ndarray
		Orbital parameters. RV_model for the order.
		
	Returns
	-------
	prior : float
		Log likelihood of this set of input parameters based on the
		priors.
	"""

	(period, ttran, ecosomega, esinomega, K, gamma, gamma_offset, sigma_jitter1_sqrd, sigma_jitter2_sqrd) = p
	e = np.sqrt(ecosomega**2. + esinomega**2.)
	omega = np.arctan2(esinomega, ecosomega)

	#If any parameters not physically possible, return negative infinity
	if (period < 0. or e < 0. or e >= 1. or omega < -np.pi/2 or omega > np.pi/2 or 
		sigma_jitter1_sqrd**.5 > 1 or sigma_jitter2_sqrd**.5 > 1 or 
		sigma_jitter1_sqrd < 0 or sigma_jitter2_sqrd < 0):
		
		return -np.inf


	totchisq = 0.



	# otherwise return a uniform prior (except modify the eccentricity to
	# ensure the prior is uniform in e)
	return -totchisq / 2. - np.log(e)




def logprob(p, t1, RV1, RVerr1, t2, RV2, RVerr2):
	"""
	Get the log probability of the data given the priors and the model.
	See loglikeli for the input parameters.
	
	Returns
	-------
	prob : float
		Log likelihood of the model given the data and priors, up to a
		constant.
	"""

	lp = logprior(p)
	llike = loglikelihood(p, t1, RV1, RVerr1, t2, RV2, RVerr2)


	if not np.isfinite(lp):
		return -np.inf

	if not np.isfinite(llike):
		return -np.inf

	return lp + llike 


