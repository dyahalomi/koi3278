"""
Analyze the results of an MCMC run.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from NewtonianMassMCMC_RVfuncs import RV_model_2obs
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from scipy.optimize import fsolve
from NewtonianMassMCMC_funcs import *




# current parameters for the model and their order
#labels model parameters
labels = ['$P$ (days)', '$t_{tran}$ (days)', '$e cos\omega$', '$e sin\omega$',
      '$K_1$ (km/s)', '$M_1$', '$[Fe/H]$', '$\gamma$ (km/s)', '$\gamma_os$ (km/s)', '$\sigma_{j, HIRES}^2$', '$\sigma_{j, TRES}^2$']


# the file with the MCMC chain results
#input file name
infile_SpecMatch = './RVchain_mass_full_Petigura_oct24.txt'
infile_SPC =  './RVchain_mass_full_SPC_oct25.txt'
infile_Brewer = './RVchain_mass_full_Brewer_oct25.txt'


# after the burn in, only use every thin amount for speed
nthin = 1

# output the median and 1-sigma error results to a TeX file
# use None if not desired
texout = None

#outfile 
#texout = './MCMC_fit_RV_mass_SpecMatch_April1.tex'
#texout = 'SpecMatch_oct24_20000.tex'

#should outfile include inferred parameters too
inferredparams = False


# iteration where burn-in stops
burnin = 2000
# make the triangle plot
maketriangle = True

#only get chisq
onlyChisq = False

#########################

nparams = len(labels)



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


RVbundle = (t_RV_H, RV_H, RVerr_H, t_RV_T, RV_T, RVerr_T)



#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################




#Brewer -- 2,000 burn out
#Best model parameters:
#[ 8.81878646e+01  4.99725480e+03  4.81054510e-03 -6.31299791e-03
#  1.97662515e+01  8.68142570e-01  1.06489976e-01 -2.73612547e+01
# -4.36493244e+01  7.61431551e-04  4.07185742e-02]
#Median model parameters:
p_Brewer = [ 8.81889234e+01,  4.99720808e+03,  4.52283556e-03, -6.23520406e-03,
             1.97474684e+01,  8.70291838e-01,  1.09079673e-01, -2.73883020e+01,
            -4.36370273e+01,  3.46368212e-02,  1.06935279e-01]


#SPC -- 2,000 burn out
#Best model parameters:
#[ 8.81990945e+01  4.99704906e+03  8.80940957e-05  4.14087912e-05
#  1.98426632e+01  8.91354257e-01  1.52843456e-01 -2.72732194e+01
# -4.35555369e+01  5.17855875e-04  1.74014263e-01]
#Median model parameters:
p_SPC = [ 8.81888848e+01,  4.99720841e+03,  4.53771551e-03, -6.33402532e-03,
          1.97466051e+01,  9.00331083e-01,  1.77803674e-01, -2.73888308e+01,
         -4.36388788e+01,  3.50609998e-02,  1.05965734e-01]




#SpecMatch -- 2,000 burn out
#Best model parameters:
#pB_SpecMatch = [ 8.81973589e+01,  4.99712726e+03,  4.50556195e-06,  4.44276220e-06,
#                 1.98681384e+01,  9.06214903e-01,  1.61751877e-01, -2.73218905e+01,
#                -4.36450172e+01,  5.12692245e-04,  6.49430352e-02]
#Median model parameters:
p_SpecMatch = [ 8.81889329e+01,  4.99720715e+03,  4.49867400e-03, -6.28462580e-03,
                1.97462942e+01,  8.96284596e-01,  1.44410346e-01, -2.73882267e+01,
               -4.36369016e+01,  3.49237131e-02,  1.05925337e-01]






#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################

# SpecMatch
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


isobundle = loadisos()

p = p_SpecMatch
#+3 for the 3 spectral DOF
print 'Reduced chi-square: ',  (loglikeli(
    p, isobundle, RVbundle, specbundle, minimize=True) /
    (len(t_RV_T) + len(t_RV_H) - len(p) + 3 ))

def RV_WDmass_eq(M2, K, P, M1, ecosw, esinw, i):
    #solve for e
    e = np.sqrt(ecosw**2 + esinw**2)

    #convert period[days] to period[seconds]
    P = P * 24 * 60 * 60

    #convert masses[M_sun] to masses[kg]
    Msun = 1.989 * 10**30
    M1 = M1 * Msun
    M2 = M2 * Msun


    #convert K[km/s] to K[m/s]
    K = K * 1000

    #define grav const
    G = 6.674*10**(-11)


    return ((2. * np.pi * G) / (P * (M1 + M2)**2))**(1. / 3.) * ((M2 * np.sin(i)) / np.sqrt(1. - e**2.)) - K


def solve_WDmass(K, P, M1, ecosw, esinw, i = np.pi/2.):

    return fsolve(RV_WDmass_eq, 0.5, args = (K, P, M1, ecosw, esinw, i))[0]





if onlyChisq:
    quit()
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################




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

    phase_RV1 = ((t_RV1 - p[1]) % p[0]) / p[0]
    ax0.errorbar(phase_RV1, RV1, yerr = np.sqrt(RVerr1**2. + sigma_jitter1_sqrd), fmt = 'o', color = 'b',  markersize = 10, label = "HIRES")

    phase_RV2 = ((t_RV2-p[1]) % p[0])/p[0]
    ax0.errorbar(phase_RV2, RV2 +p [6], yerr=np.sqrt(RVerr2**2. + sigma_jitter2_sqrd), fmt='o', color = 'g',  markersize = 10, label = "TRES")

    t = np.arange(p[0], p[0] + p[1])
    model = RV_model_2obs(t, p)
    phase = ((t-p[1]) % p[0]) / p[0]
    lsort = np.argsort(phase)
    ax0.plot(phase[lsort], model[lsort], color = 'k')


    RV_model1 = RV_model_2obs(t_RV1, p)
    RV_model2 = RV_model_2obs(t_RV2, p)
    
    ax1.plot([0., 1.], [0., 0.], color = 'k')
    ax1.errorbar(phase_RV1, RV1 - RV_model1, yerr = np.sqrt(RVerr1**2. + sigma_jitter1_sqrd), fmt = 'o', markersize = 10,  color = 'b')
    ax1.errorbar(phase_RV2, RV2 + p[6] - RV_model2, yerr = np.sqrt(RVerr2**2. + sigma_jitter2_sqrd), fmt = 'o',  markersize = 10, color = 'g')

    ax1.set_xlabel("Phase", fontsize = 18)
    ax0.set_ylabel("Radial Velocity (km/s)", fontsize = 18)
    ax1.set_ylabel("Residuals (km/s)", fontsize = 18)
    yticks = ax0.yaxis.get_major_ticks()
    xticks = ax0.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)
    xticks[-1].label1.set_visible(False)
    yticks[0].label1.set_visible(False)
    ax0.legend(numpoints = 1, loc = 2, fontsize = 18)

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
#print plot_RV(pRV_median_SPC, t_RV_H, RV_H, RVerr_H, t_RV_T, RV_T, RVerr_T)

#rms
#print "TRES and HIRES RMS w SPC"
#print np.round(get_RMS_residuals(pRV_median_SPC, t_RV_H, RV_H, RVerr_H, t_RV_T, RV_T, RVerr_T), 2)






x = np.loadtxt(infile_Brewer)
print 'File loaded'



# split the metadata from the chain results
iteration = x[:, 0]
walkers = x[:, 1]
uwalkers = np.unique(walkers)
loglike = x[:, 2]
x = x[:, 3:]

# thin the file if we want to speed things up
thin = np.arange(0, iteration.max(), nthin)
good = np.in1d(iteration, thin)
x = x[good, :]
iteration = iteration[good]
walkers = walkers[good]
loglike = loglike[good]

# plot the value of each chain for each parameter as well as its log likelihood
plt.figure()
plt.clf()
for ii in np.arange(nparams+1):
    # use 3 columns of plots
    ax = plt.subplot(np.ceil((nparams+1)/3.), 3, ii+1)
    for jj in uwalkers:
        this = np.where(walkers == jj)[0]
        if ii < nparams:
            # if this chain is really long, cut down on plotting time by only
            # plotting every tenth element
            if len(iteration[this]) > 5000:
                plt.plot(iteration[this][::10],
                         x[this, ii].reshape((-1,))[::10])
            else:
                plt.plot(iteration[this], x[this, ii].reshape((-1,)))
        # plot the likelihood
        else:
            if len(iteration[this]) > 5000:
                plt.plot(iteration[this][::10], loglike[this][::10])
            else:
                plt.plot(iteration[this], loglike[this])
    # show the burnin location
    plt.plot([burnin, burnin], plt.ylim(), lw=2)
    # add the labels
    if ii < nparams:
        plt.ylabel(labels[ii])
    else:
        plt.ylabel('Log Likelihood')
        plt.xlabel('Iterations')
    ax.ticklabel_format(useOffset=False)

# now remove the burnin phase
pastburn = np.where(iteration > burnin)[0]
iteration = iteration[pastburn]
walkers = walkers[pastburn]
loglike = loglike[pastburn]
x = x[pastburn, :]



# ========================================================================== #


# Taken from RadVel Github, April 16, 2019
def gelman_rubin(pars0, minTz, maxGR):
    '''Gelman-Rubin Statistic
    Calculates the Gelman-Rubin statistic and the number of
    independent draws for each parameter, as defined by Ford et
    al. (2006) (http://adsabs.harvard.edu/abs/2006ApJ...642..505F).
    The chain is considered well-mixed if all parameters have a
    Gelman-Rubin statistic of <= 1.03 and >= 1000 independent draws.
    Args:
        pars0 (array): A 3 dimensional array (NPARS,NSTEPS,NCHAINS) of
            parameter values
        minTz (int): minimum Tz to consider well-mixed
        maxGR (float): maximum Gelman-Rubin statistic to
            consider well-mixed
    Returns:
        tuple: tuple containing:
            ismixed (bool):
                Are the chains well-mixed?
            gelmanrubin (array):
                An NPARS element array containing the
                Gelman-Rubin statistic for each parameter (equation
                25)
            Tz (array):
                An NPARS element array containing the number
                of independent draws for each parameter (equation 26)
    History:
        2010/03/01:
            Written: Jason Eastman - The Ohio State University
        2012/10/08:
            Ported to Python by BJ Fulton - University of Hawaii,
            Institute for Astronomy
        2016/04/20:
            Adapted for use in RadVel. Removed "angular" parameter.
    '''


    pars = pars0.copy() # don't modify input parameters

    sz = pars.shape
    msg = 'MCMC: GELMAN_RUBIN: ERROR: pars must have 3 dimensions'
    assert pars.ndim == 3, msg

    npars = float(sz[0])
    nsteps = float(sz[1])
    nchains = float(sz[2])

    msg = 'MCMC: GELMAN_RUBIN: ERROR: NSTEPS must be greater than 1'
    assert nsteps > 1, msg

    # Equation 21: W(z) in Ford 2006
    variances = np.var(pars,axis=1, dtype=np.float64)
    meanofvariances = np.mean(variances,axis=1)
    withinChainVariances = np.mean(variances, axis=1)

    # Equation 23: B(z) in Ford 2006
    means = np.mean(pars,axis=1)
    betweenChainVariances = np.var(means,axis=1, dtype=np.float64) * nsteps
    varianceofmeans = np.var(means,axis=1, dtype=np.float64) / (nchains-1)
    varEstimate = (
        (1.0 - 1.0/nsteps) * withinChainVariances
        + 1.0 / nsteps * betweenChainVariances
    )

    bz = varianceofmeans * nsteps

    # Equation 24: varhat+(z) in Ford 2006
    varz = (nsteps-1.0)/bz + varianceofmeans

    # Equation 25: Rhat(z) in Ford 2006
    gelmanrubin = np.sqrt(varEstimate/withinChainVariances)

    # Equation 26: T(z) in Ford 2006
    vbz = varEstimate / bz
    tz = nchains*nsteps*vbz[vbz < 1]
    if tz.size == 0:
        tz = [-1]

    # well-mixed criteria
    ismixed = min(tz) > minTz and max(gelmanrubin) < maxGR

    return (ismixed, gelmanrubin, tz)


# ========================================================================== #



pars0 = np.reshape(x.T, (nparams, 100000-burnin-1, 100))

print gelman_rubin(pars0, 1000, 1.2)




# sort the results by likelihood for the triangle plot
lsort = np.argsort(loglike)
lsort = lsort[::-1]
iteration = iteration[lsort]
walkers = walkers[lsort]
loglike = loglike[lsort]
x = x[lsort, :]

        

if maketriangle:
    plt.figure(figsize = (18, 18))
    plt.clf()
    # set unrealistic default mins and maxes
    maxes = np.zeros(len(x[0, :])) - 9e9
    mins = np.zeros(len(x[0, :])) + 9e9
    nbins = 50
    # go through each combination of parameters
    for jj in np.arange(len(x[0, :])):
        for kk in np.arange(len(x[0, :])):
            # only handle each combination once
            if kk < jj:
                # pick the right subplot
                ax = plt.subplot(len(x[0, :]), len(x[0, :]),
                                 jj * len(x[0, :]) + kk + 1)
                # 3, 2, and 1 sigma levels
                sigmas = np.array([0.9973002, 0.9544997, 0.6826895])
                # put each sample into 2D bins
                hist2d, xedge, yedge = np.histogram2d(x[:, jj], x[:, kk],
                                                      bins=[nbins, nbins],
                                                      normed=False)
                # convert the bins to frequency
                hist2d /= len(x[:, jj])
                flat = hist2d.flatten()
                # get descending bin frequency
                fargs = flat.argsort()[::-1]
                flat = flat[fargs]
                # cumulative fraction up to each bin
                cums = np.cumsum(flat)
                levels = []
                # figure out where each sigma cutoff bin is
                for ii in np.arange(len(sigmas)):
                        above = np.where(cums > sigmas[ii])[0][0]
                        levels.append(flat[above])
                levels.append(1.)
                # figure out the min and max range needed for this plot
                # then see if this is beyond the range of previous plots.
                # this is necessary so that we can have a common axis
                # range for each row/column
                above = np.where(hist2d > levels[0])
                thismin = xedge[above[0]].min()
                if thismin < mins[jj]:
                    mins[jj] = thismin
                thismax = xedge[above[0]].max()
                if thismax > maxes[jj]:
                    maxes[jj] = thismax
                thismin = yedge[above[1]].min()
                if thismin < mins[kk]:
                    mins[kk] = thismin
                thismax = yedge[above[1]].max()
                if thismax > maxes[kk]:
                    maxes[kk] = thismax
                # make the contour plot for these two parameters
                plt.contourf(yedge[1:]-np.diff(yedge)/2.,
                             xedge[1:]-np.diff(xedge)/2., hist2d,
                             levels=levels,
                             colors=('k', '#444444', '#888888'))
            # plot the distribution of each parameter
            if jj == kk:
                ax = plt.subplot(len(x[0, :]), len(x[0, :]),
                                 jj*len(x[0, :]) + kk + 1)
                plt.hist(x[:, jj], bins=nbins, facecolor='k')

    # allow for some empty space on the sides
    diffs = maxes - mins
    mins -= 0.05*diffs
    maxes += 0.05*diffs
    # go back through each figure and clean it up
    for jj in np.arange(len(x[0, :])):
        for kk in np.arange(len(x[0, :])):
            if kk < jj or jj == kk:
                ax = plt.subplot(len(x[0, :]), len(x[0, :]),
                                 jj*len(x[0, :]) + kk + 1)
                # set the proper limits
                if kk < jj:
                    ax.set_ylim(mins[jj], maxes[jj])
                ax.set_xlim(mins[kk], maxes[kk])
                # make sure tick labels don't overlap between subplots
                ax.yaxis.set_major_locator(plticker.MaxNLocator(nbins=4,
                                                                prune='both'))
                # only show tick labels on the edges
                if kk != 0 or jj == 0:
                    ax.set_yticklabels([])
                else:
                    # tweak the formatting
                    plt.ylabel(labels[jj])
                    locs, labs = plt.yticks()
                    plt.setp(labs, rotation=0, va='center')
                    yformatter = plticker.ScalarFormatter(useOffset=False)
                    ax.yaxis.set_major_formatter(yformatter)
                # do the same with the x-axis ticks
                ax.xaxis.set_major_locator(plticker.MaxNLocator(nbins=4,
                                                                prune='both'))
                if jj != len(x[0, :])-1:
                    ax.set_xticklabels([])
                else:
                    plt.xlabel(labels[kk])
                    locs, labs = plt.xticks()
                    plt.setp(labs, rotation=90, ha='center')
                    yformatter = plticker.ScalarFormatter(useOffset=False)
                    ax.xaxis.set_major_formatter(yformatter)
    # remove the space between plots
    plt.subplots_adjust(hspace=0.0, wspace=0.0)


# the best, median, and standard deviation of the input parameters
# used to feed back to model_funcs for initrange, and plotting the best fit
# model for publication figures in mcmc_run
best = x[0, :]
meds = np.median(x, axis=0)
devs = np.std(x, axis=0)
print 'Best model parameters: '
print best

print 'Median model parameters: '
print meds




# load the isochrones if we need them
if inferredparams and texout is not None:

    try:
        loaded
    except NameError:
        loaded = 1
        isobundle = loadisos()
        # unpack the model bundle
        (magname, interps, limits, fehs, ages, maxmasses) = isobundle
        minfeh, maxfeh, minage, maxage = limits

# put the MCMC results into a TeX table
if texout is not None:
    best_out = best.copy()
    best_out = list(best_out)

    # calculate eccentricity and add it to the list of parameters
    e = (np.sqrt(x[:, 2]**2. + x[:, 3]**2.)).reshape((len(x[:, 0]), 1))
    e_best = np.sqrt(best[2]**2. + best[3]**2.)
    best_out.append(e_best)
    x = np.concatenate((x, e), axis=1)
    labels.append('$e$')

    # add omega to the list
    omega = np.arctan2(x[:, 3], x[:, 2]).reshape((len(x[:, 0]), 1))*180./np.pi
    omega_best = np.arctan2(best[3], best[2])*180./np.pi
    best_out.append(omega_best)
    x = np.concatenate((x, omega), axis=1)
    labels.append('$\omega$ (deg)')

    # if we want to get inferred value from the isochrones as well
    if inferredparams:
        # some important values
        FeH = x[:, 6]
        M1 = x[:, 5]

        age = np.zeros(len(FeH))
        logage = np.zeros(len(FeH))
        for ii in np.arange(len(M1)):
            age[ii] = msage(M1[ii], FeH[ii], isobundle)

            # convert to log(age) for the isochrone
            logage[ii] = np.log10(age[ii] * 1e9)
 
    

        # set up the output
        results = np.zeros((len(FeH), len(isointerp(M1[0], FeH[0],
                            logage[0], isobundle))))

        # get the isochrone values for each chain input
        # this is very time intensive
        M2 = np.zeros(len(FeH))
        for ii in np.arange(len(FeH)):
            results[ii, :] = isointerp(M1[ii], FeH[ii], logage[ii], isobundle)
            M2[ii] = solve_WDmass(x[:,4][ii], x[:,0][ii], x[:,5][ii], x[:,2][ii], x[:,3][ii])


        


        # add primary effective temperature
        Teff = (10.**results[:, -1]).reshape((len(x[:, 0]), 1))
        x = np.concatenate((x, Teff), axis=1)
        labels.append('$T_{eff,1}$ (K)')

        # add log(g)
        logg = (results[:, -2]).reshape((len(x[:, 0]), 1))
        x = np.concatenate((x, logg), axis=1)
        labels.append('log(g)')

        # add primary radius
        R1 = (results[:, -3]).reshape((len(x[:, 0]), 1))
        x = np.concatenate((x, R1), axis=1)
        labels.append('$R_1$')

        
        #add WD mass
        M2 = M2.reshape((len(x[:, 0]), 1))
        x = np.concatenate((x, M2), axis=1)
        labels.append('$M_2$')


        # calculate eccentricity and add it to the list of parameters
        e = (np.sqrt(x[:, 2]**2. + x[:, 3]**2.)).reshape((len(x[:, 0]), 1))
        x = np.concatenate((x, e), axis=1)
        labels.append('$e$')

        # add omega to the list
        omega = np.arctan2(x[:, 3], x[:, 2]).reshape((len(x[:, 0]), 1))*180./np.pi
        x = np.concatenate((x, omega), axis=1)
        labels.append('$\omega$ (deg)')




if texout is not None:
    # what are the median and 1-sigma limits of each parameter we care about
    stds = [15.87, 50., 84.13]
    neg1, med, plus1 = np.percentile(x, stds, axis=0)

    # get ready to write them out
    ofile = open(texout, 'w')
    ofile.write('\\documentclass{article}\n\\begin{document}\n\n')
    ofile.write('\\begin{tabular}{| c | c |}\n\\hline\n')

    # what decimal place the error bar is at in each direction
    sigfigslow = np.floor(np.log10(np.abs(plus1-med)))
    sigfigshigh = np.floor(np.log10(np.abs(med-neg1)))
    sigfigs = sigfigslow * 1
    # take the smallest of the two sides of the error bar
    lower = np.where(sigfigshigh < sigfigs)[0]
    sigfigs[lower] = sigfigshigh[lower]
    # go one digit farther
    sigfigs -= 1
    # switch from powers of ten to number of decimal places
    sigfigs *= -1.
    sigfigs = sigfigs.astype(int)

    # go through each parameter
    for ii in np.arange(len(labels)):
      
        # if we're rounding to certain decimal places, do it
        if sigfigs[ii] >= 0:
            val = '%.'+str(sigfigs[ii])+'f'
        else:
            val = '%.0f'
        # do the rounding to proper decimal place and write the result
        ostr = labels[ii]+' & $'
        ostr += str(val % np.around(med[ii], decimals=sigfigs[ii]))
        ostr += '^{+' + str(val % np.around(plus1[ii]-med[ii],
                                            decimals=sigfigs[ii]))
        ostr += '}_{-' + str(val % np.around(med[ii]-neg1[ii],
                                             decimals=sigfigs[ii]))
        ostr += '}$ \\\\\n\\hline\n'
        ofile.write(ostr)


        # add best fits in
        ostr = labels[ii]+' best & $'

        best_val = round(best_out[ii], sigfigs[ii])
        ostr += str(best_val)
        ostr += '$ \\\\\n\\hline\n'
        ofile.write(ostr)

    ofile.write('\\end{tabular}\n\\end{document}')
    ofile.close()


#plot 
plt.show()








