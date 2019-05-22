"""
Analyze the results of an MCMC run.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from RV_funcs_wJitter import RV_model_2obs
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from scipy.optimize import fsolve
from RV_mass_funcs import *


pRV_median_SPC = [  8.81888848e+01,   4.99720841e+03,   4.53771551e-03,  -6.33402532e-03,
   1.97466051e+01,  -2.73888308e+01, -4.36388788e+01,   3.50609998e-02,   1.05965734e-01]

pRV_best_Brewer = [ 8.81878646e+01,  4.99725480e+03,  4.81054510e-03, -6.31299791e-03,
  1.97662515e+01,   -2.73612547e+01, -4.36493244e+01,  7.61431551e-04,  4.07185742e-02]

# RV data HIRES
t_RV_H = np.array([6585.763935, 6909.848497, 7579.984325,
                 7581.005670, 7652.901655, 7703.779060,
                 7829.106551, 7853.094255])
RV_H    = np.array([-28.888, -9.044, -46.575, -46.524, -40.145,
                   -8.813, -39.762, -40.780])
RVerr_H = np.array([  0.089,  0.086,   0.118,   0.139,   0.133,
                    0.072,  0.168,   0.149])


# RV data TRES -- Sam Multi-Order
t_RV_T = np.array([8006.664944, 8009.684164, 8019.772179, 8038.615663, 8052.616284, 8063.641604, 8070.641157, 8081.601247])
RV_T = np.array([2.5256, 0.0505, -3.0988, 14.2614, 32.0776, 35.4823, 32.3973, 19.3524])
RVerr_T = np.array([0.0445, 0.064, 0.0593, 0.0787, 0.0558, 0.0627, 0.0804, 0.0702])


def plot_RV(p, t_RV1, RV1, RVerr1, t_RV2, RV2, RVerr2):
    '''
    Plot the RV data against RV model

    '''
    (period, ttran, ecosomega, esinomega, K, gamma, gamma_offset, sigma_jitter1_sqrd, sigma_jitter2_sqrd) = p

    plt.figure(figsize=(15,10))
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[2,1], wspace=0.03)
    gs.update(hspace=0.)

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    #phase_RV1 = ((t_RV1 - p[1]) % p[0]) / p[0]
    ax0.errorbar(t_RV1, RV1, yerr = np.sqrt(RVerr1**2. + sigma_jitter1_sqrd), fmt = 'o', color = 'b',  markersize = 10, label = "HIRES")

    #phase_RV2 = ((t_RV2-p[1]) % p[0])/p[0]
    ax0.errorbar(t_RV2, RV2 +p[6], yerr=np.sqrt(RVerr2**2. + sigma_jitter2_sqrd), fmt='o', color = 'g',  markersize = 10, label = "TRES")

    t = np.arange(6000, 8500)
    model = RV_model_2obs(t, p)
    #phase = ((t-p[1]) % p[0]) / p[0]
    #lsort = np.argsort(phase)
    ax0.plot(t, model, color = 'k')


    RV_model1 = RV_model_2obs(t_RV1, p)
    RV_model2 = RV_model_2obs(t_RV2, p)
    
    ax1.plot([6000, 8500], [0., 0.], color = 'k')
    ax1.errorbar(t_RV1, RV1 - RV_model1, yerr = np.sqrt(RVerr1**2. + sigma_jitter1_sqrd), fmt = 'o', markersize = 10,  color = 'b')
    ax1.errorbar(t_RV2, RV2 + p[6] - RV_model2, yerr = np.sqrt(RVerr2**2. + sigma_jitter2_sqrd), fmt = 'o',  markersize = 10, color = 'g')

    ax1.set_xlabel("Time (BJD - 2,450,000)", fontsize = 18)
    ax0.set_ylabel("Radial Velocity (km/s)", fontsize = 18)
    ax1.set_ylabel("Residuals (km/s)", fontsize = 18)
    yticks = ax0.yaxis.get_major_ticks()
    xticks = ax0.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)
    xticks[-1].label1.set_visible(False)
    yticks[0].label1.set_visible(False)
    ax0.legend(numpoints = 1, loc = 2, fontsize = 18)
    plt.xlim(6000,8500)

    plt.show()




def get_RMS_residuals(p, t_RV1, RV1, RVerr1, t_RV2, RV2, RVerr2):
    '''
    p: input parameters
    the rest are observations
    '''
    predicted_RV1 = RV_model_2obs(t_RV1, p)
    predicted_RV2 = RV_model_2obs(t_RV2, p)

    n = len(t_RV1) + len(t_RV2)

    rms = np.sqrt( (np.sum((RV1 - predicted_RV1)**2) + np.sum((RV2+p[6] - predicted_RV2) **2)) / n)

    print RV1 - predicted_RV1
    print RV2 + p[6] - predicted_RV2
    mean1 = np.mean(RV1 - predicted_RV1)
    mean2 = np.mean(RV2 + p[6] - predicted_RV2)
    mean = np.mean(np.array([mean1, mean2]))

    return rms, mean


#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################






#plot
print plot_RV(pRV_best_Brewer, t_RV_H, RV_H, RVerr_H, t_RV_T, RV_T, RVerr_T)

#rms
print "TRES and HIRES RMS"
print np.round(get_RMS_residuals(pRV_best_Brewer, t_RV_H, RV_H, RVerr_H, t_RV_T, RV_T, RVerr_T), 2)

