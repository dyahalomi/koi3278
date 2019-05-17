"""
Some inputs pointing to data locations and other variables that may change.
"""
import numpy as np

# current parameters for the model and their order
labels = ['$P$ (days)', '$t_{tran}$ (days)', '$ecos\omega$', '$esin\omega$',
      '$K_1$ (km/s)', '$M_1$', '$[Fe/H]$', '$\gamma$ (km/s)', '$\gamma_os$ (km/s)', '$\sigma_{j1}^2$', '$\sigma_{j2}^2$']

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

