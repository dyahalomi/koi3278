"""
Take in oribtal and stellar parameters and turn them into an
eclipsing light curve.

Also run an MCMC analysis and/or find an initial fit.

This time we use isochrones and observed magnitudes. Also use age constraints
to make sure the masses/MS ages/etc are consistent.

Vectorized to run faster.
"""
import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import emcee
from scipy import optimize as opt
import matplotlib.ticker as plticker
from JointMCMC_funcs import (logprob, loglikeli, initrange, light_curve_model,
                         msage, kepler_problem, isointerp, loadisos)
from JointMCMC_RVfuncs import RV_model
from inputs import (labels,t_RV1,RV1,RVerr1,t_RV2,RV2,RVerr2)

# whether or not to use the adjustments for crowding (3rd light contamination)
usecrowd = True
# crowding value in Kepler for each quarter (1-17)
quartcontam = np.array([0.9619, 0.9184, 0.9245, 0.9381, 0.9505, 0.9187,
                        0.9246, 0.9384, 0.9598, 0.9187, 0.9248, 0.9259,
                        0.9591, 0.9186, 0.9244, 0.9383, 0.9578])
# quarter for each event
equarts = np.array([1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10,
                    10, 11, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16,
                    17])

# use scipy.optimize to get an initial fit
findfit = False
# run the full MCMC model
domcmc = False
# plot
doplot = True
# where to save the MCMC output
outfile = './JointChain.txt'
# number of MCMC iterations

niter = 100000

# fit quadratic limb darkening coefficients as free parameters
# if False, uses the Sing fits based on the stellar parameters
fitlimb = False

# subsample/npert: change sampling of times per cadence to obtain
# higher accuracy at ingress/egress
subsample = 10

# time/fluxes/flux errors for all events to fit
# must have equal numbers of points in each cut!
infile = './KOI3278_events_sap.txt'

# multiply the Kepler flux errors by this amount to get
# a reduced chi-square closer to 1.
expanderror = 1.13

# ========================================================================== #

# load in the sections of the light curve near transits
t, f, ferr = np.loadtxt(infile, unpack=True)
ferr *= expanderror
good = np.isfinite(ferr)

# this takes a bit, so if you've already loaded things once, don't bother again
try:
    loaded
except NameError:
    loaded = 1

    isobundle = loadisos()
    # unpack the model bundle
    (magname, interps, limits, fehs, ages,
     maxmasses) = isobundle
    minfeh, maxfeh, minage, maxage = limits
print 'Done loading isochrones'

# ========================================================================== #


# Starting parameters for the MCMC models
#p = [period,         ttran,          ecosw,          esinw, 
#     b,              R2,             M1,             FeH,
#     age,            K,              gamma,          gamma_os,
#     jitter1_sqrd,   jitter2_sqrd,   F2F1]

# p_start from spc median light curve fit (per, ttran, ecosw, esinw, b, R2, m1, feh, age, f2f1) 
# and RV SPC median fit (gamma, gamma_os, jitter1, jitter2)
p_start = [  8.81805033e+01,   8.54191297e+01,   1.47166426e-02,  -1.22324907e-02,
             6.72741838e-01,   1.249e-02     ,   9.64221684e-01,   2.29024719e-01,   
             2.55001938e+00,   1.97466051e+01,  -2.73888308e+01,  -4.36388788e+01,   
             3.50609998e-02,   1.05965734e-01,   0.001127   ]


#p = p_start

# add limb darkening parameters if we want to try to fit for them
if fitlimb:
    p = np.concatenate((p, np.array([5.64392567e-02,  5.07460729e-01])))
    labels.append('$u_{S1,1}$')
    labels.append('$u_{S1,2}$')

# set up the crowding parameters for each event
crowding = np.ones(len(equarts))
if usecrowd:
    for ii in np.arange(len(crowding)):
        crowding[ii] = quartcontam[equarts[ii]-1]

# just define segments of data as any data gap more than 4 days
edges = np.where(np.abs(np.diff(t)) > 4.)[0] + 1
cuts = np.zeros(len(t)).astype(np.int)
# increment the start of a new segment by 1
cuts[edges] = 1
cuts = np.cumsum(cuts)
ncuts = cuts[-1] + 1

# try to find a roughly optimal solution to start the MCMC off
if findfit:
    # use a minimization routine to get the solution
    result2 = opt.minimize(logprob, p, args=(t, f, ferr, cuts, crowding,
                                             subsample, isobundle, True),
                           method='TNC', options = {'maxiter': 1000,
                                                    'disp': True})
    p = result2['x']
    print logprob(p, t, f, ferr, cuts, crowding, subsample, isobundle,
                  minimize=True)
    print 'Minimization Fit:'
    print p

# run the full MCMC model
if domcmc:
    ndim = len(p)
    nwalkers = 50
    # set up the walkers in a ball near the optimal solution
    startlocs = [p + initrange(p)*np.random.randn(ndim)
                 for i in np.arange(nwalkers)]

    # set up the MCMC code
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob,
                                    args=(t, f, ferr, cuts, crowding,
                                          subsample, isobundle))
    # clear the output file
    ofile = open(outfile, 'w')
    ofile.close()

    # run the MCMC, recording parameters for every walker at every step
    for result in sampler.sample(startlocs, iterations=niter,
                                 storechain=False):
        position = result[0]
        iternum = sampler.iterations
        ofile = open(outfile, 'a')
        for k in np.arange(position.shape[0]):
            # write the iteration number, walker number, log likelihood
            # and the values for all parameters at this step
            ofile.write('{0} {1} {2} {3}\n'.format(iternum, k,
                        str(result[1][k]), " ".join([str(x)
                                                    for x in position[k]])))
        ofile.close()
        # keep track of how far along thing are
        mod = iternum % 10
        if mod == 0:
            print iternum



####################################################
####################################################
####################################################
####################################################
####################################################
####################################################
####################################################
####################################################


####################################################
#Results from different runs
####################################################
# current parameters for the model and their order
#p = [period,         ttran,          ecosw,          esinw, 
#     b,              R2,             M1,             FeH,       
#     age,            K,              gamma,          gamma_os,  
#     jitter1_sqrd,   jitter2_sqrd,   F2F1]


####################################################
####################################################
####################################################
####################################################


#Feb27 SPC run -- no M2init + F2F1 ...burn in = 20,000
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



#Feb28 Brewer run -- no M2init + F2F1 ...burn in = 20,000
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



#Feb28 SpecMatch run -- no M2init + F2F1 ...burn in = 20,000
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


####################################################
####################################################
####################################################
####################################################
####################################################
####################################################
####################################################
####################################################


p = p_best_BrewerFeb28

# get the values of the best fit parameters
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
# to get in log(age) like the interpolation needs
age = np.log10(age * 1e9)


print 'Reduced chi-square: ',  (loglikeli(
   p, t, f, ferr, cuts, crowding, isobundle, npert=subsample, minimize=True) /
  (len(t[good]) + len(RV1) + len(RV2) - len(p) + 3))





if not doplot:
     quit()


# the modeled light curve
pfullmod = loglikeli(p, t, f, ferr, cuts, crowding, isobundle,
                     retmodel=True, npert=500)
# the polynomial model only
polymodel = loglikeli(p, t, f, ferr, cuts, crowding, isobundle,
                      retpoly=True, npert=500)
# chi-square of each event on its own
indchis = loglikeli(p, t, f, ferr, cuts, crowding, isobundle,
                    indchi=True, npert=500)




# ========================================================================== #
# plot the raw data again the model to see how good the fit is by eye

plt.figure(2)
plt.clf()
# plot the raw data
plt.plot(t[good], f[good], 'b', lw=2)
plt.plot(t[good], polymodel[good], label='Polynomial only')
plt.plot(t[good], pfullmod[good], label='Full model')
plt.legend()
plt.show()

# ========================================================================== #

# the observed model includes crowding (3rd light, assumed constant) like so:
# modelobs = realmodel * crowding[ii] + 1. - crowding[ii]
# thus, we invert this to get
# realmodel = (modelobs + crowding - 1.) / crowding
realmodel = np.ones(len(f))
for ii in np.arange(ncuts):
    foo = np.where(cuts == ii)[0]
    realmodel[foo] = (((pfullmod/polymodel) + crowding[ii] - 1.) /
                      crowding[ii])[foo]
# rescale says how much you have to multiply each cadence of the
# observed model by to get the model with the third light removed
rescale = realmodel / (pfullmod/polymodel)

# set up the arrays of occultations and pulses
tocc = np.array([])
focc = np.array([])
ferrocc = np.array([])
resocc = np.array([])
tpul = np.array([])
fpul = np.array([])
ferrpul = np.array([])
respul = np.array([])

# go through each event
for ii in np.arange(ncuts):
    used = np.where(cuts == ii)[0]
    igood = np.isfinite(ferr[used])
    # every other event is an occultation
    if ii % 2:
        # line up the event in time
        tocc = np.concatenate((tocc, t[used][igood] % period))
        # adjust for crowding and divide out the polynomial trend
        # for the raw fluxes and their errors
        focc = np.concatenate((focc, f[used][igood]*rescale[used][igood] /
                               polymodel[used][igood]))
        ferrocc = np.concatenate((ferrocc, ferr[used][igood] *
                                 rescale[used][igood]/polymodel[used][igood]))
        # also record the residuals
        resocc = np.concatenate((resocc, f[used][igood]-pfullmod[used][igood]))
    # every other event is a pulse
    else:
        # line up the event in time
        tpul = np.concatenate((tpul, t[used][igood] % period))
        # adjust for crowding and divide out the polynomial trend
        # for the raw fluxes and their errors
        fpul = np.concatenate((fpul, f[used][igood]*rescale[used][igood] /
                              polymodel[used][igood]))
        ferrpul = np.concatenate((ferrpul, ferr[used][igood] *
                                 rescale[used][igood]/polymodel[used][igood]))
        # also record the residuals
        respul = np.concatenate((respul, f[used][igood]-pfullmod[used][igood]))

# this number was tuned to give ~5 bins covering the duration of each event
npts = 42
# the subtraction is so the bins don't straddle ingress/egress
tbinpul = np.linspace(min(tpul), max(tpul), npts) - 0.005
tbinocc = np.linspace(min(tocc), max(tocc), npts) - 0.007
bwidpul = tbinpul[1]-tbinpul[0]
bwidocc = tbinocc[1]-tbinocc[0]
# figure out what bin each cadence is in
digitspul = np.digitize(tpul, tbinpul)
digitsocc = np.digitize(tocc, tbinocc)
# get the median flux in each bin
fbinpul = np.array([np.median(fpul[digitspul == foo])
                    for foo in range(1, len(tbinpul))])
fbinocc = np.array([np.median(focc[digitsocc == foo])
                    for foo in range(1, len(tbinocc))])
# and the median residual in each bin
resbinpul = np.array([np.median(respul[digitspul == foo])
                      for foo in range(1, len(tbinpul))])
resbinocc = np.array([np.median(resocc[digitsocc == foo])
                      for foo in range(1, len(tbinocc))])
# use standard error of the mean as an error bar
errbinpul = np.array([np.std(fpul[digitspul == foo]) /
                      np.sqrt(len(fpul[digitspul == foo]))
                      for foo in range(1, len(tbinpul))])
errbinocc = np.array([np.std(focc[digitsocc == foo]) /
                      np.sqrt(len(focc[digitsocc == foo]))
                      for foo in range(1, len(tbinocc))])
# put the time stamp at the center of the bin
tbinpul = tbinpul[1:] - bwidpul/2.
tbinocc = tbinocc[1:] - bwidocc/2.

# ========================================================================== #
# Figure 2 of the main text in Science
plt.figure(3, figsize = (15,10))
plt.clf()
# setup to get a nice looking figure
gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], wspace=0.03)
gs.update(hspace=0.)
# get the axes
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
ax3 = plt.subplot(gs[3])

# plot the binned fluxes and residuals
ax0.errorbar(tbinpul, fbinpul, yerr=errbinpul, ls='none', color='#dd0000',
             marker='o', mew=0, zorder=3, ms=np.sqrt(50), elinewidth=4,
             capthick=0, capsize=0)
ax2.errorbar(tbinpul, resbinpul, yerr=errbinpul, ls='none', color='#dd0000',
             marker='o', mew=0, zorder=3, ms=np.sqrt(50), elinewidth=4,
             capthick=0, capsize=0)
ax1.errorbar(tbinocc, fbinocc, yerr=errbinocc, ls='none', color='#dd0000',
             mew=0, marker='o', zorder=3, ms=np.sqrt(50), elinewidth=4,
             capthick=0, capsize=0)
ax3.errorbar(tbinocc, resbinocc, yerr=errbinocc, ls='none', color='#dd0000',
             mew=0, marker='o', zorder=3, ms=np.sqrt(50), elinewidth=4,
             capthick=0, capsize=0)
# just to see where the bin edges are exactly
#for ii in np.arange(len(tbinpul)):
    #ax0.plot([tbinpul[ii]-bwidpul/2., tbinpul[ii]-bwidpul/2.], [0, 2], c='r')
#for ii in np.arange(len(tbinocc)):
    #ax1.plot([tbinocc[ii]-bwidocc/2., tbinocc[ii]-bwidocc/2.], [0, 2], c='r')

# plot the actual data and residuals in the background
ax0.scatter(tpul, fpul, s=3, zorder=2, c='k', lw=0)
ax1.scatter(tocc, focc, s=3, zorder=2, c='k', lw=0)
ax2.scatter(tpul, respul, s=3, zorder=2, c='k', lw=0)
ax3.scatter(tocc, resocc, s=3, zorder=2, c='k', lw=0)
# fix the x-range
ax0.set_xlim(tpul.min(), tpul.max())
ax1.set_xlim(tocc.min(), tocc.max())
ax2.set_xlim(tpul.min(), tpul.max())
ax3.set_xlim(tocc.min(), tocc.max())
# find the common y-range to use and set it
maxflux = np.array([fpul.max(), focc.max()]).max()
minflux = np.array([focc.min(), fpul.min()]).min()
ax0.set_ylim(minflux, maxflux)
ax1.set_ylim(minflux, maxflux)
# manual adjustment for maximum effect
ax0.set_ylim(0.9982, 1.0018)
ax1.set_ylim(0.9982, 1.0018)
# formatting the ticks and labels for Science
ax0.ticklabel_format(useOffset=False)
ax1.set_yticklabels([])
ax3.set_yticklabels([])
ax0.set_xticklabels([])
ax1.set_xticklabels([])
ax0.tick_params(labelsize=18, width=2, length=5)
ax1.tick_params(labelsize=18, width=2, length=5)
ax2.tick_params(labelsize=18, width=2, length=5)
ax3.tick_params(labelsize=18, width=2, length=5)
ax0.set_ylabel('Relative flux', fontsize=24)
ax2.set_ylabel('Residuals', fontsize=24)
ax2.set_xlabel('BJD - 2455000', fontsize=24)
ax3.set_xlabel('BJD - 2455000', fontsize=24)
# common y-range for the residuals panels
maxresid = np.array([np.abs(respul).max(), np.abs(resocc).max()]).max()
# manual fix
maxresid = 0.0015
ax2.set_ylim(-maxresid,  maxresid)
ax3.set_ylim(-maxresid,  maxresid)

# this plots the model convolved at the Kepler cadence
goodmod = np.where(polymodel > 0)[0]
tmod = t[goodmod]
# get the pure model without the polynomial continuum and crowding
pmod = pfullmod[goodmod]*rescale[goodmod]/polymodel[goodmod]
order = np.argsort(tmod % period)
# plot the model
ax0.plot(tmod[order] % period, pmod[order], c='#666666', lw=4, zorder=1)
ax1.plot(tmod[order] % period, pmod[order], c='#666666', lw=4, zorder=1)
ax2.plot(tmod[order] % period, np.zeros(len(tmod)), c='#666666',
         lw=4, zorder=1)
ax3.plot(tmod[order] % period, np.zeros(len(tmod)), c='#666666',
         lw=4, zorder=1)
plt.show()


# ========================================================================== #

# separate cadences into pulses, occultations, or neither
inpul = np.where(pmod > 1.00002)[0]
inocc = np.where(pmod < 0.99996)[0]
flat = np.where((pmod < 1.00002) & (pmod > 0.99996))[0]
allresids = f[goodmod] - pfullmod[goodmod]
# residuals during no event, occultations, and pulses
# this was a test to see if spots or something could cause higher residuals
#print (np.std(allresids[flat]), np.std(allresids[inocc]),
#       np.std(allresids[inpul]))

# ========================================================================== #
# plot residual distributions during occultations, pulses, or neither
# to see if there is a larger scatter during any phase
'''
plt.figure(7)
plt.clf()
plt.hist(allresids[flat], bins=350, alpha=0.5, facecolor='k',
         label='Out of Events')
plt.hist(allresids[inocc], bins=30, alpha=0.5, facecolor='r',
         label='Occultation')
plt.hist(allresids[inpul], bins=30, alpha=0.5, facecolor='g', label='Pulse')
plt.legend()
plt.xlabel('Residuals')
'''

# ========================================================================== #


# ========================================================================== #
# Figure S1 in the Science supplement

fig4 = plt.figure(4, figsize=(18,10))
fig4.clf()
fig5 = plt.figure(5, figsize=(18,10))
fig5.clf()
# which subplot we're on
f4ct = 1
f5ct = 1

for ii in np.arange(ncuts):
    used = np.where(cuts == ii)[0]
    igood = np.isfinite(ferr[used])
    # this is an observed occultation
    if ii % 2 and len(t[used][igood]) > 1:
        # get a new subplot and tweak some formatting
        ax = fig4.add_subplot(4, 4, f4ct)
        ax.ticklabel_format(useOffset=False)
        if f4ct % 4 != 1:
            ax.set_yticklabels([])
        f4ct += 1
    # this is an observed pulse
    elif len(t[used][igood]) > 1:
        # get a new subplot and tweak some formatting
        ax = fig5.add_subplot(4, 4, f5ct)
        ax.ticklabel_format(useOffset=False)
        if f5ct % 4 != 1:
            ax.set_yticklabels([])
        f5ct += 1
    # if this event has valid data
    if len(t[used][igood]) > 1:
        # plot the data
        ax.scatter(t[used][igood], f[used][igood]/polymodel[used][igood],
                   c='k', s=40, zorder=1)
        # get a very fine model
        t3 = np.linspace(t[used][igood].min(), t[used][igood].max(), 500)
        modelfine3 = light_curve_model(t3, p, isobundle, npert=50)
        ax.plot(t3, modelfine3, c='r', lw=3, zorder=2)
        # manually fix the y-limit to be the same in every plot
        ax.set_ylim(0.9982, 1.0015)
        ax.set_xlim(t[used][igood].min(), t[used][igood].max())
        # make sure the labels are all legible
        ax.xaxis.set_major_locator(plticker.MaxNLocator(nbins=5, prune='both'))
        ax.yaxis.set_major_locator(plticker.MaxNLocator(nbins=5, prune='both'))
        ax.tick_params(labelsize=18, width=2, length=5)

# adjust the formatting
fig4.subplots_adjust(wspace=0.03)
fig5.subplots_adjust(wspace=0.03)
fig4.text(0.5, 0.05, 'BJD - 2455000', ha='center', va='center', fontsize=24)
fig5.text(0.5, 0.05, 'BJD - 2455000', ha='center', va='center', fontsize=24)
fig4.text(0.07, 0.5, 'Relative Flux', ha='center', va='center',
          rotation=90, fontsize=24)
fig5.text(0.07, 0.5, 'Relative Flux', ha='center', va='center',
          rotation=90, fontsize=24)


plt.show()





# ========================================================================== #
# plot RV data against the RV model to see fit by eye

fig = plt.figure(8, figsize=(15,10))
fig.set_figheight(10)
fig.set_figwidth(15)
plt.clf()

gs = gridspec.GridSpec(2, 1, height_ratios=[2,1], wspace=0.03)
gs.update(hspace=0.)

ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])


#print period,ttran+5000.,ecosw,esinw,K,gamma, gamma_offset
p_RV = (period, ttran+5000., ecosw, esinw, K, gamma, gamma_offset, sigma_jitter1_sqrd, sigma_jitter2_sqrd)

phase_RV1 = ((t_RV1 - p_RV[1]) % p_RV[0]) / p_RV[0]
ax0.errorbar(phase_RV1, RV1, yerr = np.sqrt(RVerr1**2. + sigma_jitter1_sqrd), fmt = 'o', color = 'b',  markersize = 10, label = "HIRES")

phase_RV2 = ((t_RV2 - p_RV[1]) % p_RV[0])/p_RV[0]
ax0.errorbar(phase_RV2, RV2 +p_RV [6], yerr=np.sqrt(RVerr2**2. + sigma_jitter2_sqrd), fmt='o', color = 'g',  markersize = 10, label = "TRES")

t = np.arange(p_RV[0], p_RV[0] + p_RV[1])
model = RV_model(t, p_RV)
phase = ((t-p_RV[1]) % p_RV[0]) / p_RV[0]
lsort = np.argsort(phase)
ax0.plot(phase[lsort], model[lsort], color = 'k')


RV_model1 = RV_model(t_RV1, p_RV)
RV_model2 = RV_model(t_RV2, p_RV)

ax1.plot([0., 1.], [0., 0.], color = 'k')
ax1.errorbar(phase_RV1, RV1 - RV_model1, yerr = np.sqrt(RVerr1**2. + sigma_jitter1_sqrd), fmt = 'o', markersize = 10,  color = 'b')
ax1.errorbar(phase_RV2, RV2 + p_RV[6] - RV_model2, yerr = np.sqrt(RVerr2**2. + sigma_jitter2_sqrd), fmt = 'o',  markersize = 10, color = 'g')


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
# ========================================================================== #


predicted_RV1 = RV_model(t_RV1, p_RV)
predicted_RV2 = RV_model(t_RV2, p_RV)

n = len(t_RV1) + len(t_RV2)

rms = np.sqrt( (np.sum((RV1 - predicted_RV1)**2) + np.sum((RV2+p_RV[6] - predicted_RV2) **2)) / n)

print RV1 - predicted_RV1
print RV2 + p_RV[6] - predicted_RV2
mean1 = np.mean(RV1 - predicted_RV1)
mean2 = np.mean(RV2 + p_RV[6] - predicted_RV2)
mean = np.mean(np.array([mean1, mean2]))

print "rms"
print rms



