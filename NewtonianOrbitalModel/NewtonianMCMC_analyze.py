"""
Analyze the results of an MCMC run.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from NewtonianMCMC_funcs import RV_model_1obs
from NewtonianMCMC_funcs import RV_model_2obs
from NewtonianMCMC_funcs import loglikelihood_1obs as loglikeli
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

# current parameters for the model and their order

#labels HIRES or TRES
labels = ['$P$ (days)', '$t_{tran}$ (days)', '$ecos\omega$', '$esin\omega$',
          '$K_1$ (km/s)', '$\gamma$ (km/s)', '$\sigma_j^2$']

#labels both
#labels = ['$P$ (days)', '$t_{tran}$ (days)', '$ecos\omega$', '$esin\omega$',
#      '$K_1$ (km/s)', '$\gamma$ (km/s)', '$\gamma_os$ (km/s)', '$\sigma_j1$', '$\sigma_j2$']

# the file with the MCMC chain results
#file HIRES
infile_HIRES = './RVchain_HIRES_nov13.txt'
infile_TRES = './RVchain_TRES_multiOrder_nov13.txt'

#files TRES
#infile = './RVchain_both_nov13.txt'

#file both
#infile = './RVchain_both_nov13.txt'

# after the burn in, only use every thin amount for speed
nthin = 1

# output the median and 1-sigma error results to a TeX file
# use None if not desired
#texout = None

#outfile TRES
texout = './NewtonianMCMC_TRESnov13_best.tex'


#outfile both
#texout = './MCMC_fit_RV_both_oct12.tex'



# iteration where burn-in stops
burnin = 2000
# make the triangle plot
maketriangle = False



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


# RV data TRES -- Sam Mg-b
t_RV_Tmgb = np.array([8006.664944, 8009.684164, 8019.772179, 8038.615663, 8052.616284, 8063.641604, 8070.641157, 8081.601247])
RV_Tmgb = np.array([-40.7415, -42.6037, -45.8447, -29.1316, -11.1451, -7.0490, -10.1248, -23.5211])
RVerr_Tmgb = np.array([0.112, 0.112, 0.118, 0.114, 0.117, 0.111, 0.122, 0.113])






#########################

nparams = len(labels)

x = np.loadtxt(infile_TRES)
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


print 'Reduced chi-square TRES: ',  (loglikeli(
    meds, t_RV_T, RV_T, RVerr_T, minimize=True) /
    (len(t_RV_T) - len(meds) ))
'''
print 'Reduced chi-square HIRES: ',  (loglikeli(
    meds, t_RV_H, RV_H, RVerr_H, minimize=True) /
    (len(t_RV_H) - len(meds) ))
'''


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



plt.show()


def plot_RV_1obs(p, t_RV, RV, RVerr, obs):
    '''
    Plot the RV data against RV model

    '''
    if obs == 'HIRES':
        color = 'b'
        saveTitle = 'RVfit_HIRES.jpg'
        label = "HIRES"

    elif obs == "TRES Multi-Order":
        color = 'g'
        saveTitle = 'RVfit_TRES_multiOrder.jpg' 
        label = "TRES"

    elif obs == "TRES mgb":
        color = 'r'
        saveTitle = 'RVfit_TRES_mgb.jpg'
        label = "TRES Magnesium B" 

    (period, ttran, ecosomega, esinomega, K, gamma, sigma_jitter_sqrd) = p

    plt.figure(figsize=(15,10))
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[2,1], wspace=0.03)
    gs.update(hspace=0.)

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    phase_RV = ((t_RV - p[1]) % p[0]) / p[0]
    ax0.errorbar(phase_RV, RV, yerr = np.sqrt(RVerr**2. + sigma_jitter_sqrd), markersize = 10, fmt = 'o', color = color, label = label)

    t = np.arange(p[0], p[0] + p[1])
    model = RV_model_1obs(t, p)
    phase = ((t-p[1]) % p[0]) / p[0]
    lsort = np.argsort(phase)
    ax0.plot(phase[lsort], model[lsort], color = 'k')


    RV_model = RV_model_1obs(t_RV, p)
    
    ax1.plot([0., 1.], [0., 0.], color = 'k')
    ax1.errorbar(phase_RV, RV - RV_model, yerr = np.sqrt(RVerr**2. + sigma_jitter_sqrd), markersize = 10, fmt = 'o', color = color)
    ax1.set_xlabel("Phase", fontsize = 18)
    ax0.set_ylabel("Radial Velocity (km/s)", fontsize = 18)
    ax1.set_ylabel("Residuals (km/s)", fontsize = 18)
    yticks = ax0.yaxis.get_major_ticks()
    xticks = ax0.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)
    xticks[-1].label1.set_visible(False)
    yticks[0].label1.set_visible(False)
    ax0.legend(numpoints = 1, loc = 2, fontsize = 18)
    


    plt.savefig(saveTitle)
    plt.show()


def plot_RV_2obs(p, t_RV1, RV1, RVerr1, t_RV2, RV2, RVerr2):
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
    ax0.errorbar(phase_RV2, RV2 +p[6], yerr=np.sqrt(RVerr2**2. + sigma_jitter2_sqrd), fmt='o', color = 'g',  markersize = 10, label = "TRES Multi-Order")

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


    #plt.savefig('RVfit_both.jpg')
    plt.show()


def get_RMS_residuals_1obs(p, t_RV, RV, RVerr):
    '''
    p: input parameters
    the rest are observations
    '''
    predicted_RV = RV_model_1obs(t_RV, p)
    n = len(t_RV)

    rms = np.sqrt( np.sum((RV - predicted_RV)**2) / n)
    mean = np.mean(RV - predicted_RV)

    return rms, mean




def get_RMS_residuals_2obs(p, t_RV1, RV1, RVerr1, t_RV2, RV2, RVerr2):
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


#  p_single = (period, ttran, ecosomega, esinomega, K, gamma, sigma_jitter_sqrd)
p_med_TRES_multiOrder = [  8.83771178e+01,   4.99059505e+03,   7.98884286e-03,  -5.01718794e-03,
   1.96098065e+01,   1.62309184e+01,   2.36294195e-01]

p_med_TRES_mgb = [  8.86285969e+01,   4.98177052e+03,   1.88700359e-02,  -1.93688788e-03,
   1.96485126e+01,  -2.67297984e+01,   3.53298252e-01]

p_med_HIRES = [  8.81712630e+01,   4.99752688e+03,   9.84995039e-03,  -1.09856304e-02,
   1.97172125e+01,  -2.74781050e+01,   9.46259941e-02]






#plot HIRES
#print plot_RV_1obs(p_med_HIRES, t_RV_H, RV_H, RVerr_H, 'HIRES')

#plot TRES multi-order
#print plot_RV_1obs(p_med_TRES_multiOrder, t_RV_T, RV_T, RVerr_T, 'TRES Multi-Order')

#plot TRES mg b
#print plot_RV_1obs(p_med_TRES_mgb, t_RV_Tmgb, RV_Tmgb, RVerr_Tmgb, "TRES mgb")

#plot both (TRES and HIRES)
#print plot_RV_2obs(p_best_both_nov13, t_RV_H, RV_H, RVerr_H, t_RV_T, RV_T, RVerr_T)





#rms HIRES
#print "HIRES RMS"
#print np.round(get_RMS_residuals_1obs(p_med_HIRES, t_RV_H, RV_H, RVerr_H), 2)

#rms TRES multi-order
#print "TRES Multi-OrderRMS"
#print np.round(get_RMS_residuals_1obs(p_med_TRES_multiOrder, t_RV_T, RV_T, RVerr_T), 2)

#rms TRES mgb
#print "TRES mgb RMS"
#print np.round(get_RMS_residuals_1obs(p_med_TRES_mgb, t_RV_Tmgb, RV_Tmgb, RVerr_Tmgb), 2)


#rms both (TRES and HIRES)
#print "TRES and HIRES RMS"
#print np.round(get_RMS_residuals_2obs(p_best_both_nov13, t_RV_H, RV_H, RVerr_H, t_RV_T, RV_T, RVerr_T), 2)




