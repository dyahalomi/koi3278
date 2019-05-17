"""
Some inputs pointing to data locations and other variables that may change.
"""
import numpy as np

# current parameters for the model and their order
#p = [period,    ttran,         ecosw,          esinw, 
#     b,         M2init,        R2,             M1, 
#     FeH,       age,           K,              gamma,
#     gamma_os,  jitter1_sqrd,  jitter2_sqrd]
labels = ['$P$ (days)', '$t_{tran}$ (days)', '$e \cos\omega$', '$e \sin\omega$',
          '$b$', '$R_2$', '$M_1$', '[Fe/H]', 'Age (Gyr)',
          'K', '$\gamma$','$\gamma_{o}$', '$\sigma_{j,1}^2$', '$\sigma_{j,2}^2$', '$\\frac{F_2}{F_1}$']

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


# RV data HIRES
t_RV1 = np.array([6585.763935, 6909.848497, 7579.984325,
                 7581.005670, 7652.901655, 7703.779060,
                 7829.106551, 7853.094255])
RV1    = np.array([-28.888, -9.044, -46.575, -46.524, -40.145,
                   -8.813, -39.762, -40.780])
RVerr1 = np.array([  0.089,  0.086,   0.118,   0.139,   0.133,
                    0.072,  0.168,   0.149])

# RV data TRES -- Sam Multi-Order
t_RV2 = np.array([8006.664944, 8009.684164, 8019.772179, 8038.615663, 8052.616284, 8063.641604, 8070.641157, 8081.601247])
RV2= np.array([2.5256, 0.0505, -3.0988, 14.2614, 32.0776, 35.4823, 32.3973, 19.3524])
RVerr2 = np.array([0.0445, 0.064, 0.0593, 0.0787, 0.0558, 0.0627, 0.0804, 0.0702])