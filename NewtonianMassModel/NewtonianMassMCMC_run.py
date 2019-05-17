import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from NewtonianMassMCMC_funcs import *



def run_MCMC_RV_mass(p, isobundle, RVbundle, specbundle):
	#set to True if you want to run full MCMC

	runmcmc = True

	#name output file
	outfile = 'RVchain_mass_SpecMatch_April1.txt'

	#number of iterations
	niter = 100000

	if runmcmc:
		ndim = len(p)
		nwalkers = 100

		#start walkers in a ball near the optimal solution
		startlocs = [p + initrange(p)*np.random.randn(ndim) for i in np.arange(nwalkers)]

		#run emcee MCMC code
		#run both data sets
		sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob, args = [isobundle, RVbundle, specbundle])

		#clear output file
		ofile = open(outfile, 'w')
		ofile.close()

		#run the MCMC...record parameters for every walker at every step
		for result in sampler.sample(startlocs, iterations = niter, storechain = False):
			pos = result[0]
			iternum = sampler.iterations
			ofile = open(outfile, 'a')

			#write iteration number, walker number, and log likelihood
			#and value of parameters for the step
			for walker in np.arange(pos.shape[0]):
				ofile.write('{0} {1} {2} {3}\n'.format(iternum, walker, str(result[1][walker]), " ".join([str(x) for x in pos[walker]])))

			ofile.close()


			#keep track of step number
			mod = iternum % 100
			if mod == 0:
				print iternum
				print pos[0]


	return "MCMC complete"

#set priors
#p = (period, ttran, ecosw, esinw, K, M1, FeH, gamma, gamma_offset, sigma_jitter1_sqrd, sigma_jitter2_sqrd)
p = np.array([88.18,   5000,   0.015,  0.000,   21,   1.042,   0.39,  -27,   -43,   0.0,   0.0])



#-------------------------------------------------------------------------#
#RV Data
#-------------------------------------------------------------------------#

# RV data HIRES
t_RV_H = np.array([6585.763935, 6909.848497, 7579.984325,
				 7581.005670, 7652.901655, 7703.779060,
				 7829.106551, 7853.094255])
RV_H	= np.array([-28.888, -9.044, -46.575, -46.524, -40.145,
				   -8.813, -39.762, -40.780])
RVerr_H = np.array([  0.089,  0.086,   0.118,   0.139,   0.133,
					0.072,  0.168,   0.149])


# RV data TRES -- Sam Multi-Order
t_RV_T = np.array([8006.664944, 8009.684164, 8019.772179, 8038.615663, 8052.616284, 8063.641604, 8070.641157, 8081.601247])
RV_T = np.array([2.5256, 0.0505, -3.0988, 14.2614, 32.0776, 35.4823, 32.3973, 19.3524])
RVerr_T = np.array([0.0445, 0.064, 0.0593, 0.0787, 0.0558, 0.0627, 0.0804, 0.0702])





#-------------------------------------------------------------------------#
#Stellar Estimates from HIRES Spectra Analysis
#-------------------------------------------------------------------------#
# Petigura
#    FeH [dex] = 0.16 +/- 0.04
#    logg [dex] = 4.62 +/- 0.07
#    Teff [K] = 5490.0 +/- 60
#    Vsini [km/s] = 3.4 +/-1
logg_spec = 4.62
logg_err_spec = 0.07
Teff_spec = 5490.0
Teff_err_spec = 60
FeH_spec = 0.16
FeH_err_spec = 0.04
#
# Brewer
#    FeH [dex] = 0.12 +/- 0.04
#    logg [dex] = 4.55 +/- 0.05
#    Teff [K] = 5384.0 +/- 45
#    Vsini [km/s] = 3.6 +/- 1.0
#
#logg_spec = 4.55
#logg_err_spec = 0.05
#Teff_spec = 5384.0
#Teff_err_spec = 45
#FeH_spec = 0.12
#FeH_err_spec = 0.04
#
# SPC
#    FeH [dex] = 0.22 +/- 0.08
#    logg [dex] = 4.59  +/- 0.10
#    Teff [K] = 5435 +/-  50
#    Vsini [km/s] = 3.2 +/- 0.5
#
#logg_spec = 4.59
#logg_err_spec = 0.10
#Teff_spec = 5435.0
#Teff_err_spec = 50
#FeH_spec = 0.22
#FeH_err_spec = 0.08

specbundle = (logg_spec, logg_err_spec, Teff_spec, Teff_err_spec, FeH_spec, FeH_err_spec)
RVbundle = (t_RV_H, RV_H, RVerr_H, t_RV_T, RV_T, RVerr_T)

# this takes a bit, so if you've already loaded things once, don't bother again
try:
    loaded
except NameError:
    loaded = 1

    isobundle = loadisos()

print 'Done loading isochrones'


#run MCMC RV+MASS
print run_MCMC_RV_mass(p, isobundle, RVbundle, specbundle)







