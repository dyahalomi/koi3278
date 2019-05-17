"""
Some inputs pointing to data locations and other variables that may change.
"""
import numpy as np

# current parameters for the model and their order
labels = ['$P$ (days)', '$t_{tran}$ (days)', '$e cos\omega$', '$e sin\omega$',
          '$b$', '$M_{2,init}$', '$M_2$', '$M_1$', '[Fe/H]', 'Age (Gyr)']

# BJD - timeoffset for all light curves
timeoffset = 55000.
# location of the light curve fits files
keplerdata = './lightcurve/'
# file with a list of times to ignore
baddata = './ignorelist.txt'

# directory containing the PARSEC isochrones used
isodir = './PARSECv1.1/'

# what's in the isochrone and what its column index is
inds = {'feh': 0, 'age': 1, 'M': 2, 'Mact': 3, 'lum': 4, 'teff': 5, 'logg': 6,
        'mbol': 7, 'Kp': 8, 'g': 9, 'r': 10, 'i': 11, 'z': 12, 'D51': 13,
        'J': 14, 'H': 15, 'Ks': 16, 'int_IMF1': 17, '3.6': 18, '4.5': 19,
        '5.8': 20, '8.0': 21, '24': 22, '70': 23, '160': 24, 'W1': 25,
        'W2': 26, 'W3': 27, 'W4': 28, 'int_IMF2': 29, 'Rad': 30}

# magnitude names, values, errors, and extinction of the system in other
# filters that we want to simulate in the isochrones
magname = ['g', 'r', 'i', 'z', 'J', 'H', 'Ks', 'W1', 'W2']


# white dwarf cooling models
wdfiles = './wdmodels/Table_Mass*'
# what's in the model and what its index is
wdinds = {'g': 13, 'r': 14, 'i': 15, 'teff': 0, 'logg': 1, 'age': 26}
