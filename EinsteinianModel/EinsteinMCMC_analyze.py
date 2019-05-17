"""
Analyze the results of an MCMC run.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from scipy import interpolate
from EinsteinMCMC_funcs import msage, kepler_problem, isointerp, loadisos, getwdmodels
from inputs import labels

# the file with the MCMC chain results
infile_Brewer = './lc+spec_chain_Brewer_feb13.txt'
infile_SpecMatch = './lc+spec_chain_specMatch_feb14.txt'
infile_SPC = './lc+spec_chain_spc_feb11_full.txt'
# after the burn in, only use every thin amount for speed
nthin = 1
# does this include limb darkening as free parameters
fitlimb = False

# output the median and 1-sigma error results to a TeX file
# use None if not desired
texout = None

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

print gelman_rubin(pars0, 1000, 1.2)




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
         maxmasses, wdmagfunc) = isobundle
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
        FeH = x[:, 8]
        FeH_best = best[8]

        # convert to log(age) for the isochrone
        age = np.log10(x[:, 9] * 1e9)
        age_best = np.log10(best[9] * 1e9)
        
        M = x[:, 7]
        M_best = best[7]

        # set up the output
        results = np.zeros((len(FeH), len(isointerp(M[0], FeH[0],
                            age[0], isobundle))))
        
        # get the isochrone values for each chain input
        # this is very time intensive
        for ii in np.arange(len(FeH)):
            results[ii, :] = isointerp(M[ii], FeH[ii], age[ii], isobundle)

        results_best = isointerp(M1_best, FeH_best, age_best, isobundle)


        # add primary effective temperature
        Teff = (10.**results[:, -1]).reshape((len(x[:, 0]), 1))
        Teff_best = (10. ** results_best[-1])
        best_out.append(Teff_best)
        x = np.concatenate((x, Teff), axis=1)
        labels.append('$T_{eff,1}$ (K)')
        
        # add log(g)
        logg = (results[:, -2]).reshape((len(x[:, 0]), 1))
        logg_best = (results_best[-2])
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
            (x[:, 6] + x[:, 7])*1.988e30/(4.*np.pi**2.))**(1./3)  # in m

        aau = a * 6.685e-12  # in AU
        aau = aau.reshape((len(x[:, 0]), 1))  # in AU

        a_best = ((best[0] * 86400.)**2.*6.67e-11 *
            (best[6] + best[7])*1.988e30/(4.*np.pi**2.))**(1./3)  # in m
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
        blue_best = results_best[0] - results_best[1] <= 0.3
        kpmag[blue] = 0.25 * results[blue, 0] + 0.75 * results[blue, 1]
        kpmag_best[blue_best] = 0.25 * results_best[blue_best, 0] + 0.75 * results_best[blue_best, 1]
        kpmag[~blue] = 0.3 * results[~blue, 0] + 0.7 * results[~blue, 2]
        kpmag_best[~blue_best] = 0.3 * results_best[~blue_best, 0] + 0.7 * results_best[~blue_best, 2]

        # get the WD properties and flux ratio
        wdage = np.zeros(len(age))
        msages = np.zeros(len(age))
        wdmag = np.zeros(len(age))
        F2F1 = np.zeros(len(age))
        for ii in np.arange(len(age)):
            msages[ii] = msage(x[ii, 5], x[ii, 8], isobundle)
            wdage[ii] = np.log10(10.**age[ii] - 10.**(msages[ii]))
            wdmag[ii] = wdmagfunc(np.array([[x[ii, 6], wdage[ii]]]))[0]
            F2F1[ii] = 10.**((wdmag[ii] - kpmag[ii])/(-2.5))

        msage_best = msage(best[5], best[8], isobundle)
        wdage_best = np.log10(10.**age_best[ii] - 10.**(msage_best))
        wdmag_best = wdmagfunc(best[6], wdage_best)[0]
        F2F1_best = 10.**((wdmag_best - kpmag_best)/(-2.5))
        # add the flux ratio
        best_out.appned(F2F1_best)
        F2F1 = F2F1.reshape((len(x[:, 0]), 1))
        x = np.concatenate((x, F2F1), axis=1)
        labels.append('$F_2/F_1$')

        wdpreds = np.zeros((len(wdage), 2))
        wdpreds[:, 0] = x[:, 6]
        wdpreds[:, 1] = wdage
        wdpreds_best = [best[6], wdage_best]
        # get the WD temperature
        wdmodels = getwdmodels()
        wdtemp = interpolate.griddata(wdmodels[:, 0:2],
                                      wdmodels[:, 2], wdpreds)
        wdtemp_best = interpolate.griddata(wdmodels[:, 0:2],
                                      wdmodels[:, 2], wdpreds_best)
        best_out.append(wdtemp_best)
        wdtemp = wdtemp.reshape((len(x[:, 0]), 1))
        x = np.concatenate((x, wdtemp), axis=1)
        labels.append('$T_{eff,2}$ (K)')

        # add the RV semi-amplitude (in km/s)
        # from Seager's Exoplanets Eq. 2.13
        K = (29.775 / np.sqrt(1. - e[:, 0]**2.) * x[:, 6] *
             np.sin(inc[:, 0] * np.pi/180.) / np.sqrt(x[:, 6] + x[:, 7]) /
             np.sqrt(a[:, 0] * R1[:, 0] * 0.004649))
        K_best = (29.775 / np.sqrt(1. - e_best**2.) * best[6] *
             np.sin(inc_best * np.pi/180.) / np.sqrt(best[6] + best[7]) /
             np.sqrt(a_best * R1_best * 0.004649))
        best_out.append(K_best)
        K = K.reshape((len(x[:, 0]), 1))
        x = np.concatenate((x, K), axis=1)
        labels.append('$K_1$ (km/s)')

       

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

        # add estimated WD radius
        MCh = 1.454
        # in Solar radii
        R2 = .0108*np.sqrt((MCh/x[:, 6])**(2./3.)-(x[:, 6]/MCh)**(2./3.))
        R2_best = .0108*np.sqrt((MCh/best[6])**(2./3.)-(best[6]/MCh)**(2./3.))
        best_out.append(R2_best)
        R2 = R2.reshape((len(x[:, 0]), 1))
        x = np.concatenate((x, R2), axis=1)
        labels.append('$R_2$')

        # get ages in Gyr
        msages = (10.**msages).reshape((len(x[:, 0]), 1)) / 1e9
        wdage = (10.**wdage).reshape((len(x[:, 0]), 1)) / 1e9

        msage_best = (10.**msage_best) / 1e9
        wdage_best = (10.**wdage_best) / 1e9

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
        Rein = np.sqrt(1.6984903e-5 * x[:,6] * np.abs(Z) * R1[:, 0] / 2.)
        Rein_best = np.sqrt(1.6984903e-5 * best[6] * np.abs(Z_best) * R1_best / 2.)

        # add the Einstein radius
        best_out.append(Rein_best)
        Rein = Rein.reshape((len(x[:, 0]), 1))
        x = np.concatenate((x, Rein), axis=1)
        labels.append('$R_E$')

        # add the ages
        best_out.append(msage_best)
        x = np.concatenate((x, msages), axis=1)
        labels.append('WD MS Age (Gyr)')

        best_out.append(wdage_best)
        x = np.concatenate((x, wdage), axis=1)
        labels.append('WD Cooling Age (Gyr)')

        # add the WD luminosity
        wdlum = (R2**2.) * ((wdtemp / 5777.)**4.)
        wdlum_best = (R2_best**2.) * ((wdtemp_best / 5777.)**4.)
        best_out.append(wdlum_best)
        x = np.concatenate((x, wdlum), axis=1)
        labels.append('$L_{WD} (L_\odot)$')


        # add the predicted lens depth
        lensdeps = (1.6984903e-5 * x[:, 6] * np.abs(Z) / R1[:, 0] -
                    (R2[:, 0]/R1[:, 0])**2.)
        lensdeps = lensdeps.reshape((len(x[:, 0]), 1))

        lensdeps_best = (1.6984903e-5 * best[6] * np.abs(Z_best) / R1_best -
                    (R2_best/R1_best)**2.)
        best_out.append(lensdeps_best)
        x = np.concatenate((x, lensdeps), axis=1)
        labels.append('Magnification - 1')

        # add in the A_V value (0.291 is maglam for V band)
        asubv = ((1.-np.exp(-x[:, 10]*np.sin(10.2869*np.pi/180.)/x[:, 12])) *
                 0.291*x[:, 13])
        asubv_best = ((1.-np.exp(-best[10]*np.sin(10.2869*np.pi/180.)/best[12])) *
                 0.291*best[13])
        asubv = asubv.reshape((len(x[:, 0]), 1))
        best_out.append(asubv_best)
        x = np.concatenate((x, asubv), axis=1)
        labels.append('$A_V$')
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


