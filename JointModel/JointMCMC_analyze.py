"""
Analyze the results of an MCMC run.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from scipy import interpolate
from JointMCMC_funcs import msage, kepler_problem, isointerp, loadisos, solve_WDmassRV, RV_WDmass_eq
from inputs import labels

# the file with the MCMC chain results
infile_SPC = './JointChain_spcFeb27.txt'
infile_SpecMatch = './JointChain_SpecMatchFeb28.txt'
infile_Brewer = './JointChain_BrewerFeb28.txt'
# after the burn in, only use every thin amount for speed
nthin = 1
# does this include limb darkening as free parameters
fitlimb = False

# output the median and 1-sigma error results to a TeX file
# use None if not desired
texout = 'None'

# whether or not to evaluate all the isochrones to get inferred properties
# in the TeX file (adds a lot of time)
inferredparams = False

# iteration where burn-in stops
burnin = 20000
# make the triangle plot
maketriangle = True

# ========================================================================== #

if fitlimb:
    labels.append('$u_{S1,1}$')
    labels.append('$u_{S1,2}$')

nparams = len(labels)

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



pars0 = np.reshape(x.T, (nparams, 100000-burnin-1, 50))

print gelman_rubin(pars0, 1000, 1.1)



# sort the results by likelihood for the triangle plot
lsort = np.argsort(loglike)
lsort = lsort[::-1]
iteration = iteration[lsort]
walkers = walkers[lsort]
loglike = loglike[lsort]
x = x[lsort, :]

if maketriangle:
    plt.figure(figsize = (18,18))
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


# ========================================================================== #

# load the isochrones if we need them
if inferredparams and texout is not None:
    try:
        loaded
    except NameError:
        loaded = 1
        isobundle = loadisos()
        # unpack the model bundle
        (magname, interps, limits, fehs, ages,
         maxmasses) = isobundle
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
        FeH = x[:, 7]
        FeH_best = best[7]
        
        # convert to log(age) for the isochrone
        age = np.log10(x[:, 8] * 1e9)
        age_best = np.log10(best[8]*1e9)
        
        M1 = x[:, 6]
        M1_best = best[6]
        # set up the output
        results = np.zeros((len(FeH), len(isointerp(M1[0], FeH[0],
                            age[0], isobundle))))


        results_best = isointerp(M1_best, FeH_best, age_best, isobundle)
        M2_best = solve_WDmassRV(best[9], best[0], best[6], best[2], best[3])
        # get the isochrone values for each chain input
        # this is very time intensive
        M2 = np.zeros(len(FeH))
        for ii in np.arange(len(FeH)):
            results[ii, :] = isointerp(M1[ii], FeH[ii], age[ii], isobundle)
            M2[ii] = solve_WDmassRV(x[:,9][ii], x[:,0][ii], x[:,6][ii], x[:,2][ii], x[:,3][ii])


        #Add M_2
        best_out.append(M2_best)
        M2 = M2.reshape((len(x[:, 0]), 1))
        x = np.concatenate((x, M2), axis=1)
        labels.append('$M_2$')

        # add primary effective temperature
        Teff = (10.**results[:, -1]).reshape((len(x[:, 0]), 1))
        Teff_best = 10.**results_best[-1]
        best_out.append(Teff_best)
        x = np.concatenate((x, Teff), axis=1)
        labels.append('$T_{eff,1}$ (K)')
        
        # add log(g)
        logg = (results[:, -2]).reshape((len(x[:, 0]), 1))
        logg_best = results_best[-2]
        best_out.append(logg_best)
        x = np.concatenate((x, logg), axis=1)
        labels.append('log(g)')
        
        # add primary radius
        R1 = (results[:, -3]).reshape((len(x[:, 0]), 1))
        R1_best = results_best[-3]
        best_out.append(R1_best)
        x = np.concatenate((x, R1), axis=1)
        labels.append('$R_1$')
        
        # calculate and add the semi-major axis
        a = ((x[:, 0] * 86400.)**2.*6.67e-11 *
            (M2[:, 0] + M1)*1.988e30/(4.*np.pi**2.))**(1./3)  # in m
        aau = a * 6.685e-12  # in AU
        aau = aau.reshape((len(x[:, 0]), 1))  # in AU

        a_best = ((best[0] * 86400.)**2.*6.67e-11 *
            (M2_best + M1_best)*1.988e30/(4.*np.pi**2.))**(1./3)  # in m
        aau_best = a_best * 6.685e-12  # in AU

        best_out.append(aau_best)
        x = np.concatenate((x, aau), axis=1)
        labels.append('$a$ (AU)')


        # add a/R1 (in radii of the first star)
        a = (a / (6.955e8 * x[:, -2])).reshape((len(x[:, 0]), 1))
        a_over_R1_best = a_best / (6.955e8 * best[-2])
        best_out.append(a_over_R1_best)
        x = np.concatenate((x, a), axis=1)
        aind = len(labels)
        labels.append('$a/R_1$')

        # add inclination
        # Eq. 7 of Winn chapter from Exoplanets
        # inc = np.arccos(b/a * ((1. + esinw)/(1.-e**2.)))
        inc = np.arccos(x[:, 4]/x[:, aind])*180./np.pi
        inc_best = np.arccos(best[4]/a_best)*180./np.pi
        inc = inc.reshape((len(x[:, 0]), 1))
        best_out.append(inc_best)
        x = np.concatenate((x, inc), axis=1)
        labels.append('$i$ (deg)')

        # add the absolute magnitudes of the primary star
        results = results[:, :-3]
        magname = magname[:-3]
        x = np.concatenate((x, results), axis=1)
        for ii in magname:
            labels.append(ii)
            best_out.append(10000.000000)

        # predicted Kp magnitude of the primary star
        kpmag = np.zeros(len(results[:, 0]))
        blue = results[:, 0] - results[:, 1] <= 0.3
        kpmag[blue] = 0.25 * results[blue, 0] + 0.75 * results[blue, 1]
        kpmag[~blue] = 0.3 * results[~blue, 0] + 0.7 * results[~blue, 2]


        if not fitlimb:
            # add limb darkening parameters
            u1 = (0.44657704 - 0.00019632296*(Teff[:, 0]-5500.) +
                  0.0069222222 * (logg[:, 0]-4.5) + 0.086473504*FeH)
            u1_best = (0.44657704 - 0.00019632296*(Teff_best-5500.) +
                  0.0069222222 * (logg_best-4.5) + 0.086473504*FeH_best)

            u2 = (0.22779778 - 0.00012819556*(Teff[:, 0]-5500.) -
                  0.0045844444 * (logg[:, 0]-4.5) - 0.050554701*FeH)
            u2_best = (0.22779778 - 0.00012819556*(Teff_best-5500.) -
                  0.0045844444 * (logg_best-4.5) - 0.050554701*FeH_best)

            best_out.append(u1_best)
            u1 = u1.reshape((len(x[:, 0]), 1))
            x = np.concatenate((x, u1), axis=1)
            labels.append('$u_1$')
            
            best_out.append(u2_best)
            u2 = u2.reshape((len(x[:, 0]), 1))
            x = np.concatenate((x, u2), axis=1)
            labels.append('$u_2$')

        
        # solve for the Einstein radius
        n = 2. * np.pi / x[:, 0]
        n_best = 2. * np.pi / best[0]
        # Sudarsky 2005 Eq. 9 to convert between center of transit
        # and pericenter passage (tau)
        edif = 1.-e[:, 0]**2.
        edif_best = 1.-e_best**2.

        fcen = np.pi/2. - omega[:, 0] * np.pi/180.
        fcen_best = np.pi/2. - omega_best * np.pi/180.

        tau = (x[:, 1] + np.sqrt(edif)*x[:, 0] / (2.*np.pi) *
               (e[:, 0]*np.sin(fcen)/(1.+e[:, 0]*np.cos(fcen)) -
                2./np.sqrt(edif) * np.arctan(np.sqrt(edif)*np.tan(fcen/2.) /
                                            (1.+e[:, 0]))))
        tau_best = (best[1] + np.sqrt(edif_best)*best[0] / (2.*np.pi) *
               (e_best*np.sin(fcen_best)/(1.+e_best*np.cos(fcen_best)) -
                2./np.sqrt(edif_best) * np.arctan(np.sqrt(edif_best)*np.tan(fcen_best/2.) /
                                            (1.+e_best))))

        # define the mean anomaly
        M = (n * (x[:, 1] - tau)) % (2. * np.pi)
        M_best = (n_best * (best[1] - tau_best)) % (2. * np.pi)

        E = kepler_problem(M, e[:, 0])
        E_best = kepler_problem(M_best, e_best)

        # solve for f
        tanf2 = np.sqrt((1.+e[:, 0])/(1.-e[:, 0])) * np.tan(E/2.)
        tanf2_best = np.sqrt((1.+e_best)/(1.-e_best)) * np.tan(E_best/2.)

        f = (np.arctan(tanf2)*2.) % (2. * np.pi)
        f_best = (np.arctan(tanf2_best)*2.) % (2. * np.pi)

        r = a[:, 0] * (1. - e[:, 0]**2.) / (1. + e[:, 0] * np.cos(f))
        r_best = a_best * (1. - e_best**2.) / (1. + e_best * np.cos(f_best))

        # positive z means body 2 is in front (transit)
        Z = (r * np.sin(omega[:, 0]*np.pi/180. + f) *
             np.sin(inc[:, 0]*np.pi/180.))
        Z_best = (r_best * np.sin(omega_best*np.pi/180. + f_best) *
             np.sin(inc_best*np.pi/180.))


        # 1.6984903e-5 gives 2*Einstein radius^2/R1^2
        Rein = np.sqrt(1.6984903e-5 * M2[:,0] * np.abs(Z) * R1[:, 0] / 2.)
        Rein_best = np.sqrt(1.6984903e-5 * M2_best * np.abs(Z_best) * R1_best / 2.)

        # add the Einstein radius
        best_out.append(Rein_best)
        Rein = Rein.reshape((len(x[:, 0]), 1))
        x = np.concatenate((x, Rein), axis=1)
        labels.append('$R_E$')


        # add the predicted lens depth
        lensdeps = (1.6984903e-5 * M2[:, 0] * np.abs(Z) / R1[:, 0] -
                    (x[:,5]/R1[:, 0])**2.)
        lensdeps_best = (1.6984903e-5 * M2_best * np.abs(Z_best) / R1_best -
                    (best[5]/R1_best)**2.)

        lensdeps = lensdeps.reshape((len(x[:, 0]), 1))
        best_out.append(lensdeps_best)
        x = np.concatenate((x, lensdeps), axis=1)
        labels.append('Magnification - 1')


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

    best_out = np.array(best_out)
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


plt.savefig('triangle_BrewerFeb28.jpg')
plt.show()





####################################################
#Results from different runs
####################################################
# current parameters for the model and their order
#p = [period,    ttran,         ecosw,          esinw, 
#     b,         M2init,        R2,             M1, 
#     FeH,       age,           K,              gamma,
#     gamma_os,  jitter1_sqrd,  jitter2_sqrd,   F2F1]


####################################################
####################################################
####################################################
####################################################


#SPC run -- burn in = 20,000
#Best model parameters:
p_best_spcFeb27 = [ 8.81805914e+01,  8.54169504e+01,  1.47530464e-02, -1.11681806e-02,
                    6.42731750e-01,  1.26997108e-02,  9.54920734e-01,  1.84137231e-01,
                    8.83585507e-01,  1.97105634e+01, -2.74609089e+01, -4.36806929e+01,
                    1.04470452e-03,  4.93607200e-02,  1.11651906e-03]
#Median model parameters:
p_med_spcFeb27 = [ 8.81805270e+01,  8.54189802e+01,  1.47299860e-02, -8.19305402e-03,
                   6.85982362e-01,  8.85172593e-03,  9.50727159e-01,  2.08082632e-01,
                   3.48894376e+00,  1.97438200e+01, -2.74608620e+01, -4.36790593e+01,
                   2.75523725e-02,  1.05409155e-01,  1.12772515e-03]



#Brewer run -- burn in = 20,000
#Best model parameters:
p_best_BrewerFeb28 = [ 8.81806610e+01,  8.54191311e+01,  1.47081031e-02, -9.16521005e-03,
                       6.08509616e-01,  1.42680440e-02,  9.22592036e-01,  8.45473265e-02,
                       8.64184048e-01,  1.97608831e+01, -2.74656810e+01, -4.36830093e+01,
                       1.17900424e-03,  6.44299747e-02,  1.10938387e-03]
#Median model parameters:
p_med_BrewerFeb28 = [ 8.81805169e+01,  8.54190346e+01,  1.47297056e-02, -8.30129970e-03,
                      6.62562951e-01,  1.10547709e-02,  9.10988163e-01,  1.17873012e-01,
                      4.25140926e+00,  1.97420502e+01, -2.74634648e+01, -4.36813894e+01,
                      2.75580784e-02,  1.05841390e-01,  1.12766396e-03]



#SpecMatch run -- burn in = 20,000
#Best model parameters:
p_best_SpecMatchFeb28 = [ 8.81804780e+01,  8.54193976e+01,  1.47169087e-02, -9.15158069e-03,
                          6.56113248e-01,  1.20734411e-02,  9.73848205e-01,  1.49394260e-01,
                          8.83240340e-01,  1.96844235e+01, -2.74358603e+01, -4.36600464e+01,
                          2.81725582e-03,  5.25555570e-02,  1.14003226e-03]
#Median model parameters:
p_med_SpecMatchFeb28 = [ 8.81805219e+01,  8.54190148e+01,  1.47294168e-02, -8.12640188e-03,
                         6.79610983e-01,  9.90869452e-03,  9.54880511e-01,  1.55282788e-01,
                         2.67119194e+00,  1.97429010e+01, -2.74616664e+01, -4.36833695e+01,
                         2.86212816e-02,  1.04351968e-01,  1.12762111e-03]





