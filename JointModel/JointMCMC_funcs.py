"""
Supplementary functions needed to do the MCMC run and analysis,
including loading the isochrones.
"""

# TODO: switch out to a faster Mandel-Agol when needed
from glob import glob
import numpy as np
import sys
from scipy import interpolate
from scipy.optimize import fsolve
import numpy.polynomial.polynomial as poly
from mandel_agol import mandel_agol

# Import RV functions and RV observations
import JointMCMC_RVfuncs as RV_f
from inputs import (t_RV1,RV1,RVerr1,t_RV2,RV2,RVerr2)


def loadisos():
    """
    Loads all objects needed to interpolate and analyze the isochrones.
    All inputs are listed in the input file to make things easier and
    because they are referenced elsewhere.

    Returns
    -------
    isobundle : tuple
        (magname, interps, limits, fehs,
         ages, maxmasses, wdmagfunc)
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

    '''
    # set up the WD section
    files = glob(wdfiles)
    if len(files) == 0:
        print 'Warning! White Dwarf models not found!'

    for ct, ii in enumerate(files):
        # ignore the warnings that header lines aren't the same length
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            iwdmods = np.genfromtxt(ii, skip_header=2, invalid_raise=False)
        # pull the mass out of the file name
        imass = float(ii[-3:])
        # only grab the H WDs, ignore the He ones
        iwdmods = iwdmods[:np.diff(iwdmods[:, wdinds['teff']]).argmin()+1, :]
        # all these have the same mass
        imass = np.ones(len(iwdmods[:, 0])) * imass

        if ct == 0:
            wdmods = iwdmods * 1.
            mass = imass * 1.
        else:
            mass = np.concatenate((mass, imass))
            wdmods = np.concatenate((wdmods, iwdmods))

    # get the Kp magnitude from the g,r,i bands
    kpmag = np.zeros(len(wdmods[:, 0]))
    blue = wdmods[:, wdinds['g']] - wdmods[:, wdinds['r']] <= 0.3
    kpmag[blue] = (0.25 * wdmods[blue, wdinds['g']] +
                   0.75 * wdmods[blue, wdinds['r']])
    kpmag[~blue] = (0.3 * wdmods[~blue, wdinds['g']] +
                    0.7 * wdmods[~blue, wdinds['i']])

    # WD models contains the WD mass, age, and Kp magnitude
    wdmodels = np.zeros((len(mass), 3))
    wdmodels[:, 0] = mass
    # log age like everything else
    wdmodels[:, 1] = np.log10(wdmods[:, wdinds['age']])
    wdmodels[:, 2] = kpmag
    # get an interpolator of Kp magnitudes based on age and mass
    wdmagfunc = interpolate.LinearNDInterpolator(wdmodels[:, 0:2],
                                                 wdmodels[:, 2])
    
    '''
    # bounds where we trust the results
    limits = (minfeh, maxfeh, minage, maxage)

    # bundle all the important bits of the model together to feed
    # to functions that need them
    isobundle = (magname, interps, limits, fehs,
                 ages, maxmasses)
    return isobundle



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


def solve_WDmassRV(K, P, M1, ecosw, esinw, i = np.pi/2.):

    return fsolve(RV_WDmass_eq, 0.5, args = (K, P, M1, ecosw, esinw, i))[0]



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
    if len(p) == 15:
        return np.array([2.41373687e-04, 2.15625144e-03, 5.42134862e-05,
                         4.95601162e-02, 4.33065931e-02, 3.50000000e-04, 
                         9.25556734e-02, 2.34386318e-01, 2.34386318e-01, 
                         3.8e-02,        4.8e-02,        5.7e-02,        
                         1e-5,           1e-5,           1e-5])
    
    if len(p) == 17:
        return np.array([2.41373687e-04, 2.15625144e-03, 5.42134862e-05,
                         4.95601162e-02, 4.33065931e-02, 6.90000000e-04, 
                         9.25556734e-02, 2.34386318e-01, 2.34386318e-01, 
                         3.8e-02,        4.8e-02,        5.7e-02,        
                         1e-5,           1e-5,           1e-5,             
                         0.1,            0.1])











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
    (magname, interps, limits, fehs, ages,
     maxmasses) = isobundle
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
        result = np.zeros(3)
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
    (magname, interps, limits, fehs, ages,
     maxmasses) = isobundle
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


def kepler_problem(M, e):
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
    import numpy as np
    # start with this guess
    M = np.array(M)
    E = M * 1.
    err = M * 0. + 1.
    while err.max() > 1e-8:
        # solve via Newton's method
        guess = E - (E - e * np.sin(E) - M) / (1. - e * np.cos(E))
        err = np.abs(guess - E)
        E = guess
    return E


def light_curve_model(t, p, isobundle, npert=1):
    """
    Given the orbital parameters in p, compute a model light curve at times
    t, sampling at the rate of npert.

    Input
    -----
    t : ndarray
        Times to return the model light curve.
    p : ndarray
        Orbital parameters. Currently must contain:
        period, time of center of transit, ecos(omega), esin(omega),
        impact parameter, initial mass of star 2, current mass of star 2,
        mass of star 1, metallicity of star 1, log age of the system.
        Can also add optional 2 parameters for quadratic limb
        darkening of star 1.
    isobundle : tuple
        (magname, interps, limits, fehs, ages,
         maxmasses)
        Contains everything needed for the isochrones
    npert : int, optional
        Sampling rate per cadence. Final light curve will average each
        cadence over this many samples.

    Returns
    -------
    fluxes : ndarray
        Light curve corresponding to the times in t.
    """
    # fix limb darkening
    if len(p) == 15:
        (period, ttran, ecosw, esinw, b, R2, M1, FeH, age, K, gamma, gamma_offset, sigma_jitter1_sqrd, sigma_jitter2_sqrd, F2F1) = p
        u20 = 0.
        u21 = 0.

    # fit limb darkening for primary star
    if len(p) == 17:
        (period, ttran, ecosw, esinw, b, R2, M1, FeH, age, K, gamma, gamma_offset, sigma_jitter1_sqrd, sigma_jitter2_sqrd, F2F1, u10, u11) = p
        u20 = 0.
        u21 = 0.




    #solve for the WD Mass using Newtonian orbital models -- assume inc = 90deg
    M2 = solve_WDmassRV(K, period, M1, ecosw, esinw)

    # unpack the isochrone info
    (magname, interps, limits, fehs, ages,
     maxmasses) = isobundle

    # to get in log(age) like the interpolations need
    age = np.log10(age * 1e9)

    # get the white dwarf age and Kp magnitude
    #wdage = np.log10(10.**age - 10.**(msage(M2init, FeH, isobundle)))
    #wdmag = wdmagfunc(np.array([[M2, wdage]]))[0]

    # get the estimated parameters of the primary star
    mags = isointerp(M1, FeH, age, isobundle)

    R1 = mags[-3]
    logg = mags[-2]
    Teff = 10.**mags[-1]

    if len(p) == 15:
        # get the limb darkening from the fit to Sing
        u10 = (0.44657704 - 0.00019632296 * (Teff-5500.) +
               0.0069222222 * (logg-4.5) + 0.086473504 * FeH)
        u11 = (0.22779778 - 0.00012819556 * (Teff-5500.) -
               0.0045844444 * (logg-4.5) - 0.050554701 * FeH)
    u1 = np.array([u10, u11])
    u2 = np.array([u20, u21])

    


    # reconvert into more useful orbital elements
    e = np.sqrt(ecosw**2. + esinw**2.)
    omega = np.arctan2(esinw, ecosw)
    a_meters = ((period * 86400.)**2. * 6.67e-11 * (M1 + M2) * 1.988e30 /
         (4.*np.pi**2.))**(1./3)  # in m
    a = a_meters / (6.955e8 * R1)  # in radii of the first star

    inc = np.arccos(b/a)


    # mean motion
    n = 2. * np.pi / period

    # cadence for this data set
    medt = np.median(np.diff(t))
    # generate npert subcadences, equally spaced
    tmfine = np.linspace(-medt/2., +medt/2., npert+1)
    tmfine = tmfine[:-1] + (tmfine[1] - tmfine[0])/2.
    # all times to evaluate fluxes at
    # has shape (t, npert)
    newt = t[:, np.newaxis] + tmfine
    # has to be a vector for Mandel-Agol function
    tt = newt.reshape((-1,))

    # Sudarsky 2005 Eq. 9 to convert between center of transit
    # and pericenter passage (tau)
    edif = 1.-e**2.
    fcen = np.pi/2. - omega
    tau = (ttran + np.sqrt(edif)*period / (2.*np.pi) *
           (e*np.sin(fcen)/(1.+e*np.cos(fcen)) - 2./np.sqrt(edif) *
            np.arctan(np.sqrt(edif)*np.tan(fcen/2.)/(1.+e))))

    # define the mean anomaly
    M = (n * (tt - tau)) % (2. * np.pi)
    E = kepler_problem(M, e)

    # solve for f
    tanf2 = np.sqrt((1.+e)/(1.-e)) * np.tan(E/2.)
    fanom = (np.arctan(tanf2)*2.) % (2. * np.pi)

    r = a * (1. - e**2.) / (1. + e * np.cos(fanom))
    # projected distance between the stars (in the same units as a)
    projdist = r * np.sqrt(1. - np.sin(omega + fanom)**2. * np.sin(inc)**2.)

    # positive z means body 2 is in front (transit)
    # see Han, 2016 for description
    Z = r * np.sin(omega + fanom) * np.sin(inc)


    # form of Rein = sqrt(4G*M_2*a/c^2) from Kruse+Agol mcmc_analyze.py
    Rein = np.sqrt(1.6984903e-5 * M2 * np.abs(Z) * R1 / 2.)

    
    # solve for R ratio
    rrat = R2 / R1


    # fluxes of each body, adjusted by their relative fluxes
    F1t = tt * 0. + 1.
    F2t = tt * 0. + F2F1


    # get the lens depths given this separation at transit
    # 1.6984903e-5 gives 2*Einstein radius^2/R1^2 = 8GMZ/(c^2 R^2)
    # with M, Z, R all scaled to solar values
    #lensdeps = 1.6984903e-5 * M2 * np.abs(Z) / R1 - rrat**2.

    lensdeps = ( (2. * Rein**2) - R2**2 ) / (R1**2)

    # Change lensdeps to the form used in Kruse & Agol, 2014
    lensdeps = (lensdeps / rrat**2.) + 1.


    # object 2 passes in front of object 1
    transits = np.where((projdist < 1. + rrat) & (Z > 0.))[0]
    if len(transits) > 0:
        # limb darkened light curves for object 1
        ldark = mandel_agol(projdist[transits], u1[0], u1[1], rrat)
        # object 1 also has microlensing effects
        F1t[transits] *= (ldark + (1. - ldark)*lensdeps[transits])

    # object 1 passes in front of object 2
    occults = np.where((projdist < 1. + rrat) & (Z < 0.))[0]
    if len(occults) > 0:
        # must be in units of the blocked star/object radius
        # for Mandel/Agol function, so divide by the radius ratio
        ldark = mandel_agol(projdist[occults]/rrat, u2[0], u2[1], 1./rrat)
        F2t[occults] *= ldark

    # get back to the proper shape
    F1t = F1t.reshape(newt.shape)
    F2t = F2t.reshape(newt.shape)
    # get the average value for each cadence
    F1t = F1t.mean(axis=1)
    F2t = F2t.mean(axis=1)

    # return a normalized light curve
    normed = (F1t + F2t)/(1. + F2F1)
    return normed


def logprior(p, isobundle):
    """
    Priors on the input parameters.

    Input
    -----
    p : ndarray
        Orbital parameters. See light_curve_model for the order.
    isobundle : tuple
        Contains everything needed for the isochrones

    Returns
    -------
    prior : float
        Log likelihood of this set of input parameters based on the
        priors.
    """
    # fix limb darkening

    if len(p) == 15:
        (period, ttran, ecosw, esinw, b, R2, M1, FeH, age, K, gamma, gamma_offset, sigma_jitter1_sqrd, sigma_jitter2_sqrd, F2F1) = p
        u20 = 0.
        u21 = 0.
        # for the sake of the limits below just make up a valid number
        # for these
        u10 = 0.1
        u11 = 0.1
    # fit limb darkening for primary star
    if len(p) == 17:
        (period, ttran, ecosw, esinw, b, R2, M1, FeH, age, K, gamma, gamma_offset, sigma_jitter1_sqrd, sigma_jitter2_sqrd, F2F1, u10, u11) = p
        u20 = 0.
        u21 = 0.

    


    #solve for the WD Mass using Newtonian orbital models
    M2 = solve_WDmassRV(K, period, M1, ecosw, esinw)


    # to get in log(age) like the interpolation needs
    age = np.log10(age * 1e9)

    # check to make sure that it's valid within the models.
    if not isointerp(M1, FeH, age, isobundle, testvalid=True):
        return -np.inf

    # reconvert into more useful orbital elements
    e = np.sqrt(ecosw**2. + esinw**2.)
    omega = np.arctan2(esinw, ecosw)


    a = ((period * 86400.)**2. * 6.67e-11 * (M1 + M2) * 1.988e30 /
         (4.*np.pi**2.))**(1./3)  # in m

    # if any of the parameters are unphysical, return negative infinity
    # log likelihood (impossible)
    if (period < 0. or e < 0. or e >= 1. or a < 0. or u10 + u11 >= 1 or
            u20 + u21 >= 1 or M2 < 0. or M1 < 0. or sigma_jitter1_sqrd < 0 
            or sigma_jitter2_sqrd < 0 or omega < -np.pi/2 or omega > np.pi/2 
            or sigma_jitter1_sqrd**.5 > 1 or sigma_jitter2_sqrd**.5 > 1 or R2 < 0.):
        return -np.inf
    


    # otherwise return a uniform prior (except modify the eccentricity to
    # ensure the prior is uniform in e)
    return 0 - np.log(e)





def loglikeli(p, t, f, ferr, cuts, crowding, isobundle,  minimize=False,
              retmodel=False, retpoly=False,  indchi=False, **kwargs):
    """
    Compute the log likelihood of a microlensing signal with these orbital
    parameters given the data. By default returns this value, but can
    optionally return the full model light curve or just the polynomial
    portion of the light curve instead.

    Input
    -----
    p : ndarray
        Orbital parameters. See light_curve_model for the order.
    t, f, ferr : ndarray
        times, fluxes, and flux errors of the data.
    cuts : ndarray
        Same length as t. Says which group each cadence belongs to,
        starting with 0. E.g. all cadences with cuts == 0 will be
        assumed to be one event. Each event must be equal length.
    isobundle : tuple
        Contains everything needed for the isochrones
    crowding : ndarray
        Must be an array of len(cuts)-1
        indicating what fraction of the light is due to the binary system.
        1 - crowding is the contamination from outside sources.
        If purely light from the system in question, should be just
        np.ones(len(cuts)-1)
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
    # don't modify the originals
    tt = t * 1.
    ff = f * 1.
    fferr = ferr * 1.
    

    # compute the model light curve
    model = light_curve_model(t, p, isobundle, **kwargs)

    ncuts = cuts[-1] + 1

    # add in the contamination from outside light sources
    if crowding is not None:
        model = model * crowding[cuts] + 1. - crowding[cuts]

    # now has shape (ncuts, tper)
    tt = tt.reshape((ncuts, -1))
    ff = ff.reshape((ncuts, -1))
    fferr = fferr.reshape((ncuts, -1))
    model = model.reshape((ncuts, -1))

    # center each event on the median time
    tmeds = np.median(tt, axis=1)
    tt -= tmeds[:, np.newaxis]

    # marginalize over the polynomial detrending

    # these are just all orders we want to compute
    pord = 2
    pords = np.arange(pord+1)
    # 1-d and 4-d blank arrays to allow for numpy array broadcasting
    # in a bit
    ones = np.ones((pord+1))
    ones4d = np.ones((ncuts, 1, pord+1, pord+1))

    # every time to every polynomial order power
    # has shape (ncuts, pert, pords)
    tpow = tt[:, :, np.newaxis] ** pords
    # this is the same for every polynomial order.
    # has shape (ncuts, pert, pords)
    prefix = ((ff/model) / (fferr/model)**2.)[:, :, np.newaxis] * ones

    # get the data side of the equation. Just has shape (ncuts, pords)
    # because we summed over (pert)
    Bmat = np.sum(prefix * tpow, axis=(1,))

    # get the time**pords for both the j and k indices
    # has shape (ncuts, pert, 1, pords)
    j = tt[:, :, np.newaxis, np.newaxis]**pords[np.newaxis, np.newaxis,
                                                np.newaxis, :]
    # has shape (ncuts, pert, pords, 1)
    k = tt[:, :, np.newaxis, np.newaxis]**pords[np.newaxis, np.newaxis,
                                                :, np.newaxis]
    # has shape (ncuts, pert, pords, pords)
    Mbig = j*k

    # this gets divided in. should be the same for each pord,
    # but needs to be the right shape
    # has shape (ncuts, pert, pords, pords)
    divider = ((fferr/model)**2.)[:, :, np.newaxis, np.newaxis] * ones4d
    Mbig = Mbig / divider
    # sum over all times so now
    # has shape (ncuts,pords,pords)
    Mfinal = np.sum(Mbig, axis=1)

    solution = np.array([np.linalg.lstsq(Mfinal[ii, :, :], Bmat[ii, :])[0]
                        for ii in np.arange(ncuts)]).swapaxes(0, 1)
    solution = solution[:, cuts]
    # get the optimal polynomial model for each segment of data
    polymodel = poly.polyval(tt.reshape((-1,)), solution, tensor=False)
    # if there are entire events that aren't seen, they will
    # produce polymodel == 0, which can later give divide by 0 errors.
    polymodel[polymodel == 0.] = 1.

    # compute the chi-square of each segment
    totchisq = np.sum(((ff-model*polymodel.reshape((ncuts, -1)))/fferr)**2.,
                      axis=1)
    
    
    # get rid of fferr = infinite data points in likelihood function by...
    # setting np.sqrt(2. * np.pi * (fferr_likelihood**2.)) = 1 for these data points
    # then log(val above) = 0 and it adds nothing to the likelihood
    fferr_likelihood = fferr * 1.
    for ii in np.arange(0, len(fferr)):
        for jj in np.arange(0, len(fferr[ii])):
            if not np.isfinite(fferr[ii][jj]):
                fferr_likelihood[ii][jj] = np.sqrt(1 / (2. * np.pi))

    #compute total likelihood
    #see Eastman et al., 2013 equation 2
    #see Christiansen et al., 2017 eq 1 (sec 3.2)
    loglikelihood = - np.sum( 
        ((ff-model*polymodel.reshape((ncuts, -1)))**2. ) / (2. * (fferr**2.)) +
        np.log(np.sqrt(2. * np.pi * (fferr_likelihood**2.))),
        axis=1)


    
    # return now if desired
    if retmodel:
        return model.reshape((-1,)) * polymodel
    if retpoly:
        return polymodel
    if indchi:
        return totchisq
    

    # start with the light curve chi-square before adding the other
    # photometric constraints
    totchisq = np.sum(totchisq)
    loglikelihood = np.sum(loglikelihood)

    if len(p) == 15:
        (period, ttran, ecosw, esinw, b, R2, M1, FeH, age, K, gamma, gamma_offset, sigma_jitter1_sqrd, sigma_jitter2_sqrd, F2F1) = p
        u20 = 0.
        u21 = 0.
        # for the sake of the limits below just make up a valid number
        # for these
        u10 = 0.1
        u11 = 0.1
    # fit limb darkening for primary star
    if len(p) == 17:
        (period, ttran, ecosw, esinw, b, R2, M1, FeH, age, K, gamma, gamma_offset, sigma_jitter1_sqrd, sigma_jitter2_sqrd, F2F1, u10, u11) = p
        u20 = 0.
        u21 = 0.

    


    # to get in log(age) like the interpolation needs
    age = np.log10(age * 1e9)

    # calculate the chi-square from the magnitudes
    (magname, interps, limits, fehs, ages,
     maxmasses) = isobundle
    mags = isointerp(M1, FeH, age, isobundle)
    

    # save radius, logg, teff measurements
    R1 = mags[-3]
    logg = mags[-2]
    Teff = 10.**mags[-1]



    # Add constraint that G dwarf age should be greater than or equal to
    # spin-down age of 0.89+-0.15 Gyr:
    age = (10.**age) / 1e9
    if age < 0.89:
        agechisq = (age - 0.89)**2. / (0.15**2.)
        totchisq += agechisq
        #compute total likelihood
        #see Eastman et al., 2013 equation 2
        #see Christiansen et al., 2017 eq 1 (sec 3.2)
        loglikelihood += - (np.log(np.sqrt(2 * np.pi * (0.15**2.))) + (0.5 * agechisq))

    # add constraints on logg, teff and FeH from spectra
    # Petigura
    #    FeH [dex] = 0.16 +/- 0.04
    #    logg [dex] = 4.62 +/- 0.07
    #    Teff [K] = 5490.0 +/- 60
    #    Vsini [km/s] = 3.4 +/-1
    #logg_spec = 4.62
    #logg_err_spec = 0.07
    #Teff_spec = 5490.0
    #Teff_err_spec = 60
    #FeH_spec = 0.16
    #FeH_err_spec = 0.04
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
    logg_spec = 4.59
    logg_err_spec = 0.10
    Teff_spec = 5435.0
    Teff_err_spec = 50
    FeH_spec = 0.22
    FeH_err_spec = 0.08

    specchisq = (logg - logg_spec)**2. / (logg_err_spec**2.)
    specchisq += (Teff - Teff_spec)**2. / (Teff_err_spec**2.)
    specchisq += (FeH - FeH_spec)**2. / (FeH_err_spec**2.)
    totchisq += specchisq
    
    #see Eastman et al., 2013 equation 2
    #see Christiansen et al., 2017 eq 1 (sec 3.2)
    loglikelihood += - (np.log(np.sqrt(2 * np.pi * (logg_err_spec**2.))) + 
        np.log(np.sqrt(2 * np.pi * (Teff_err_spec**2.))) + 
        np.log(np.sqrt(2 * np.pi * (FeH_err_spec**2.))) + 
        (0.5 * specchisq)) 




    # set RV parameters for RV_model...assumes using both TRES and HIRES data
    # ttran+5000 bc RV times are bjd-2,450,000 and photometry times are bjd-2,455,000
    p_RV = (period, ttran+5000, ecosw, esinw, K, gamma, gamma_offset, sigma_jitter1_sqrd, sigma_jitter2_sqrd)

    # add RV fits to loglikehood and to totchisq
    (RVloglikli, RVchisq) = RV_f.loglikelihood(p_RV, t_RV1, RV1, RVerr1, t_RV2, RV2, RVerr2)
    
    loglikelihood += RVloglikli
    totchisq += RVchisq


    # if we're minimizing chi-square instead of maximizing likelihood
    if minimize:
        return totchisq
    

    return loglikelihood


def logprob(p, t, f, ferr, cuts, crowding, npert, isobundle, minimize=False):
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
        return -lp + loglikeli(p, t, f, ferr, cuts, crowding, isobundle,
                               minimize=minimize,  npert=npert)
    return lp + loglikeli(p, t, f, ferr, cuts, crowding, isobundle,
                          minimize=minimize,  npert=npert)
