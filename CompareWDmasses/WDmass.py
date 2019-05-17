
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.lines import Line2D


'''
M_KA = 0.634
M_KA_errLow = 0.055
M_KA_errHigh = 0.047

M_E = 0.558
M_E_errLow = 0.03
M_E_errHigh = 0.03

M_N = 0.5183
M_N_errLow = 0.0085
M_N_errHigh = 0.0085

M_J = 0.5340
M_J_errLow = 0.0121
M_J_errHigh = 0.0121
'''

#Einsteinian Models
	# SPC: $0.568^{+0.028}_{-0.027}$  & 
	# Brewer: $0.539^{+0.022}_{-0.020}$ &  
	# SpecMatch: $0.567^{+0.026}_{-0.025}$

#Newtonian Models
	# SPC: $0.5220^{+0.0081}_{-0.0081}$  
	# Brewer: $0.5122^{+0.0057}_{-0.0058}$ 
	# SpecMatch: $0.5207^{+0.0063}_{-0.0063}$

#Joint Models
	# SPC: $0.5379^{+0.0100}_{-0.0107}$  
	# Brewer: $0.5250^{+0.0082}_{-0.0089}$
	# SpecMatch: $0.5392^{+0.0081}_{-0.0088}$

M = np.array([[0.634], [0.568], [0.539], [0.567], [0.5220], [0.5122], [0.5207], [0.5379], [0.5250], [0.5392]])
Y = np.array([0.85, 0.7, 0.65, 0.6, 0.45, 0.4, 0.35, 0.2, 0.15, 0.1])
Y_labels = np.array([0.85, 0.7, 0.45, 0.2])

colors = ['#935116', '#0B5345', '#0B5345', '#0B5345', '#1B4F72', '#1B4F72', '#1B4F72', 'k', 'k', 'k']
colors_labels = ['#935116', '#0B5345', '#1B4F72', 'k']

labels = ["Kruse and Agol, 2014", "Updated Einsteinian Model", "Newtonian Model", "Joint Model"]
centerOffset = [0.027, 0.045, 0.04, 0.05]
errors = [[0.055, 0.047], [0.027, 0.028], [0.020, 0.022], [0.025, 0.026], [0.0081, 0.0081], [0.0058, 0.0057], [0.0063, 0.0063], [0.0107, 0.0100], [0.0089, 0.0082], [0.0088, 0.0081]]
npError = []
for err in errors:
	npError.append(np.array([err]).T)

#npError errors = np.array([[0.055, 0.047], [0.03, 0.03], [0.0085, 0.0085], [0.0121, 0.0121]])


for ii in range(0, len(colors)):
	if ii > 0:
		if (ii-1) % 3 == 0:
			erb1 = plt.errorbar(M[ii], Y[ii], xerr = npError[ii], fmt = 'o', ecolor = colors[ii], markerfacecolor = colors [ii], markeredgecolor = colors [ii], capsize = 3, elinewidth = 2, markersize = 7)
			erb1[-1][0].set_linestyle('--')
		elif (ii-1) % 3 == 1:
			erb2 = plt.errorbar(M[ii], Y[ii], xerr = npError[ii], fmt = 'o', ecolor = colors[ii], markerfacecolor = colors [ii], markeredgecolor = colors [ii], capsize = 3, elinewidth = 2, markersize = 7)
			erb2[-1][0].set_linestyle('-.')
		elif (ii-1) % 3 == 2:
			erb3 = plt.errorbar(M[ii], Y[ii], xerr = npError[ii], fmt = 'o', ecolor = colors[ii], markerfacecolor = colors [ii], markeredgecolor = colors [ii], capsize = 3, elinewidth = 2, markersize = 7)	
			erb3[-1][0].set_linestyle(':')

	else:
		plt.errorbar(M[ii], Y[ii], xerr = npError[ii], fmt = 'o', ecolor = colors[ii], markerfacecolor = colors [ii], markeredgecolor = colors [ii], capsize = 3, elinewidth = 2, markersize = 7)



for ii in range(0, 4):
	plt.text(M[ii]-centerOffset[ii], Y_labels[ii]+.03, labels[ii], color = colors_labels[ii])

lineKA = Line2D([0,1],[0,1],linestyle='-', color='k', linewidth = 2)
lineSPC = Line2D([0,1],[0,1],linestyle='--', color='k', linewidth = 2)
lineBrewer = Line2D([0,1],[0,1],linestyle='-.', color='k', linewidth = 2)
lineSpecMatch = Line2D([0,1],[0,1],linestyle=':', color='k', linewidth = 2)

plt.xlabel("White Dwarf Mass M\\textsubscript{\(\odot\)")
plt.ylim(0,1)
plt.xlim(0.49, 0.7)
ax = plt.axes()
ax.get_yaxis().set_visible(False)
plt.legend([lineKA, lineSPC, lineBrewer, lineSpecMatch], ['No Spectroscopy', 'SPC on HIRES Spectra', 'Brewer on HIRES Spectra', 'SpecMatch on HIRES Spectra'], handlelength=3, loc=4)

plt.savefig("WDmass_final.pdf")
plt.show()


