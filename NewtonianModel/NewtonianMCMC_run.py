import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from NewtonianMCMC_funcs import *

def run_MCMC_RV_single(p, t_RV, RV, RVerr):
	#set to True if you want to run full MCMC
	runmcmc = True

	#name output file
	outfile = 'RVchain_TRES_mgb_dec6.txt'

	#number of iterations
	niter = 100000

	if runmcmc:
		ndim = len(p)
		nwalkers = 100

		#start walkers in a ball near the optimal solution
		startlocs = [p + initrange(p)*np.random.randn(ndim) for i in np.arange(nwalkers)]
		#run emcee MCMC code
		#run single data set
		sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob_1obs, args = [t_RV, RV, RVerr])

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


def run_MCMC_RV_double(p, t_RV1, RV1, RVerr1, t_RV2, RV2, RVerr2):
	#set to True if you want to run full MCMC
	runmcmc = False

	#name output file
	outfile = 'RVchain_both.txt'

	#number of iterations
	niter = 100000

	if runmcmc:
		ndim = len(p)
		nwalkers = 100

		#start walkers in a ball near the optimal solution
		startlocs = [p + 1e-4 * np.random.randn(ndim) for i in np.arange(nwalkers)]

		#run emcee MCMC code
		#run both data sets
		sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob_2obs, args = [t_RV1, RV1, RVerr1, t_RV2, RV2, RVerr2])

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

#double p = (period, ttran, ecosomega, esinomega, K, gamma, gamma_offset, sigma_jitter1, sigma_jitter2)
p_double = np.array([88.18,   5000,   0.015,  0.000,   21,  -27,   -43, 0.0, 0.0])

#single p = (period, ttran, ecosomega, esinomega, K, gamma, sigma_jitter)
p_single = np.array([88.18,   5000,   0.015,  0.000,   21,  -27, 0.0])


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


# RV data TRES -- Sam Mg-b
t_RV_Tmgb = np.array([8006.664944, 8009.684164, 8019.772179, 8038.615663, 8052.616284, 8063.641604, 8070.641157, 8081.601247])
RV_Tmgb = np.array([-40.7415, -42.6037, -45.8447, -29.1316, -11.1451, -7.0490, -10.1248, -23.5211])
RVerr_Tmgb = np.array([0.112, 0.112, 0.118, 0.114, 0.117, 0.111, 0.122, 0.113])



#run MCMC RV Double Model
#print run_MCMC_RV_double(p_double, t_RV_H, RV_H, RVerr_H, t_RV_T, RV_T, RVerr_T)

#run MCMC RV single HIRES
#print run_MCMC_RV_single(p_single, t_RV_H, RV_H, RVerr_H)

#run MCMC RV single TRES multi-order
#print run_MCMC_RV_single(p_single, t_RV_T, RV_T, RVerr_T)

#run MCMC RV single TRES mgb-order
print run_MCMC_RV_single(p_single, t_RV_Tmgb, RV_Tmgb, RVerr_Tmgb)





