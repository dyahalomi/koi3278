
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from scipy import interpolate
from JointMCMC_funcs import isointerp, loadisos, solve_WDmassRV, RV_WDmass_eq
from inputs import labels


# the file with the MCMC chain results
infile = './JointChain_BrewerFeb28.txt'
# iteration where burn-in stops
burnin = 20000
nparams = len(labels)

isobundle = loadisos()
x = np.loadtxt(infile)
print 'File loaded'


# split the metadata from the chain results
iteration = x[:, 0]
walkers = x[:, 1]
uwalkers = np.unique(walkers)
loglike = x[:, 2]
x = x[:, 3:]

# now remove the burnin phase
pastburn = np.where(iteration > burnin)[0]
iteration = iteration[pastburn]
walkers = walkers[pastburn]
loglike = loglike[pastburn]
x = x[pastburn, :]

# sort the results by likelihood for the triangle plot
lsort = np.argsort(loglike)
lsort = lsort[::-1]
iteration = iteration[lsort]
walkers = walkers[lsort]
loglike = loglike[lsort]
x = x[lsort, :]

R2 = x[:, 5]




# some important values
FeH = x[:, 7]
# convert to log(age) for the isochrone
age = np.log10(x[:, 8] * 1e9)
M1 = x[:, 6]

# set up the output
results = np.zeros((len(FeH), len(isointerp(M1[0], FeH[0],
                    age[0], isobundle))))

# get the isochrone values for each chain input
# this is very time intensive
M2 = np.zeros(len(FeH))
for ii in np.arange(len(FeH)):
    results[ii, :] = isointerp(M1[ii], FeH[ii], age[ii], isobundle)
    M2[ii] = solve_WDmassRV(x[:,9][ii], x[:,0][ii], x[:,6][ii], x[:,2][ii], x[:,3][ii])




# set unrealistic default mins and maxes
maxes = np.zeros(2) - 9e9
mins = np.zeros(2) + 9e9
nbins = 1000
# 2, and 1 sigma levels
sigmas = np.array([0.9544997, 0.6826895])
# put each sample into 2D bins
hist2d, xedge, yedge = np.histogram2d(R2, M2,
                                      bins=[nbins, nbins],
                                      normed=False)
# convert the bins to frequency
hist2d /= len(M2)
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
# make the contour plot for these two parameters
plt.contourf(yedge[1:]-np.diff(yedge)/2.,
             xedge[1:]-np.diff(xedge)/2., hist2d,
             levels=levels,
             colors=('k', '#888888'), alpha = 0.5)


#M_KA = 0.634
#M_KA_errLow = 0.055
#M_KA_errHigh = 0.047

#Brewer Micro-Lensing M2  = $0.539^{+0.022}_{-0.020}$
#Brewer Micro-Lensing R2  = $0.01288^{+0.00029}_{-0.00029}$ 

#Brewer Joint Model M2 = $0.5250^{+0.0082}_{-0.0089}$
#Brewer Joint Model R2 = $0.0111^{+0.0026}_{-0.0048}$
M = np.array([[0.634], [0.539], [0.5250], [0.533], [0.623], [0.621]])
R = np.array([[0.01166], [0.01288], [0.0111], [0.0133], [0.0122], [0.0122]])
colors = ['#935116', '#0B5345', 'k', 'r', 'b', 'g']
labels = ["Kruse and Agol, 2014: KOI-3278", "Updated Einsteinian Model: KOI-3278", "Joint Model: KOI-3278", "Kawahara et al., 2018: SLB1", "Kawahara et al., 2018: SLB2", "Kawahara et al., 2018: SLB3"]
markers = ['.', '.', '.', '*', '*', '*']

M_errors = [[0.055, 0.047], [0.020, 0.022], [0.0089, 0.0082], [0.033, 0.078], [0.042, 0.053], [0.061, 0.090]]
np_M_Error = []
for err in M_errors:
	np_M_Error.append(np.array([err]).T)

R_errors = [[0.00056, 0.00069], [0.00029, 0.00029], [0.0048, 0.0026], [0.0010, 0.0005], [0.0006, 0.0005], [0.0011, 0.0008]]
np_R_Error = []
for err in R_errors:
	np_R_Error.append(np.array([err]).T)
#npError errors = np.array([[0.055, 0.047], [0.03, 0.03], [0.0085, 0.0085], [0.0121, 0.0121]])


for ii in range(0, len(colors)):
	plt.errorbar(M[ii], R[ii], xerr = np_M_Error[ii], yerr = np_R_Error[ii], label = labels[ii], fmt = 'o', ecolor = colors[ii], marker = markers[ii], markerfacecolor = colors [ii], markeredgecolor = colors [ii], capsize = 5, elinewidth = 2, markersize = 15, alpha = 0.9)


#R_2 = 0.0108 sqrt( [{M_{Ch}/{M_2}]^(2/3) - [M_2/M_{Ch}]^(2}/3) )
M_Ch = 1.454 
M_P = 0.00057

def graph(mass_radius_relation, x_range, label, linestyle):  

    #^ use x as range variable
    y = mass_radius_relation(x_range)
    #^          ^call the lambda expression with x
    #| use y as function result

    plt.plot(x_range,y, color = 'k', alpha = 0.7, linestyle = linestyle, linewidth = 2, label = label)  

graph(lambda x : (0.0108 * ( ((M_Ch/x)**float(2./3.)) - ((x/M_Ch)**float(2./3.)) ) ** .5) , np.arange(0, 0.75, 0.001), "Nauenberg Relation", '-')
#     ^use a lambda expression           
graph(lambda x : (0.0114 * ((( ((M_Ch/x)**float(2./3.)) - ((x/M_Ch)**float(2./3.)) ) ** .5) * (1. + (3.5* (x/M_P)**float(-2./3.)) + (x/M_P)**(-1.) )**float(-2./3.))) , np.arange(0, 0.75, 0.001), "Eggleton Relation", '--') 
#     ^use a lambda expression 




plt.xlabel("White Dwarf Mass M\\textsubscript{\(\odot\)")
plt.ylabel("White Dwarf Radius R\\textsubscript{\(\odot\)")
plt.xlim(0, 0.75)
plt.ylim(0, 0.025)
plt.legend(loc = 3, numpoints = 1, fontsize = 9)
plt.savefig("WDmassVradius_Brewer.pdf")
plt.show()















