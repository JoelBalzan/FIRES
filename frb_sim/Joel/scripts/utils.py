#
#	Functions for simulating scattering 
#
#								AB, May 2024
#
#	--------------------------	Import modules	---------------------------

import os
import pickle as pkl
import sys
from collections import namedtuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

central_frequency_mhz	=	1000.0	                        #	Central frequency in MHz
num_channels		    =	300		                        #	Number of frequency channels
channel_width_mhz		=	1		                        #	Channel width in MHz
time_window_ms		    =	10.0	                        #	Time window in ms
time_resolution_ms		=	0.1		                        #	Time resolution in ms
num_time_bins		    =	int(2*time_window_ms/time_resolution_ms)	#	Number of time bins
time_per_bin_ms		    =	time_window_ms / num_time_bins	#	Time per bin in ms
scattering_index		=	-4.0	                        #	Scattering index
reference_frequency_mhz	=	1000.0	                        #	Reference frequency for scattering

# Universal constants 
gravitational_constant_cgs	=	6.67430e-8					#	Universal gravitational constant in CGS
electron_charge_cgs		    =	4.8032047e-10				#	Absolute electronic charge in CGS
electron_mass_cgs			=	9.1093837e-28				#	Electron mass in CGS
speed_of_light_cgs		    =	2.99792458e10				#	Speed of light in CGS
parsec_cm				    =	3.0857e18					#	Parsec in cm							
omega_nu				    =	2 * np.pi					#	Omega / nu 
solar_mass_grams		    =	1.98847e33					#	Solar mass in grams
radian_to_arcsec		    =	180.0 * 3600 / np.pi		#	Radian in arcseconds
radian_to_picoarcsec	    =	180.0 * 3600 * 1.0e12 / np.pi	#	Radian in pico-arcseconds
solar_radius_cm			    =	6.957e10					#	Solar radius in cm
astronomical_unit_cm	    =	1.496e13					#	1 AU in cm
inch_to_cm				    =	2.54
data_directory			    =	'../simfrbs/'
plot_directory			    =	'../plots/'

# constants for scintillation application
mb2							=	2							#mb2: Max Born parameter for strength of scattering
rf							=	1							#rf: Fresnel scale
ds							=	0.01						#ds (or dx,dy): Spatial step sizes with respect to rf
alpha						=	5/3							#alpha: Structure function exponent (Kolmogorov = 5/3)
ar							=	1							#ar: Anisotropy axial ratio
psi							=	0							#psi: Anisotropy orientation
inner						=	0.001						#inner: Inner scale w.r.t rf - should generally be smaller than ds
ns							=	256							#ns (or nx,ny): Number of spatial steps
nf							=	256							#nf: Number of frequency steps.
dlam						=	0.25						#dlam: Fractional bandwidth relative to centre frequency
lamsteps					=	False						#lamsteps: Boolean to choose whether steps in lambda or freq
seed						=	1234 						#seed: Seed number, or use "-1" to shuffle
nx							=	None
ny							=	None
dx							=	None
dy							=	None
plot						=	False
verbose						=	False
dt							=	30



# 1 FRB data simulated, (taums as input command line),
# dspec4 = 4D
# create another sub directory called SIMFRB
simulated_frb	=	namedtuple('simulated_frb', ['frbname', 'frequency_mhz_array', 'time_ms_array', 'scattering_time_ms', 'reference_frequency_mhz', 'scattering_index', 'gaussian_params', 'dynamic_spectrum_4d'])

# time variation
frb_time_series	=	namedtuple('frbts',['iquvt','lts','elts','pts','epts','phits','dphits','psits','dpsits','qfrac','eqfrac','ufrac','eufrac','vfrac','evfrac','lfrac','elfrac','pfrac','epfrac'])

# spectra (anything varying with freq (hz))
frb_spectrum	=	namedtuple('frbspec',['iquvspec','diquvspec','lspec','dlspec','pspec','dpspec','qfracspec','dqfrac','ufracspec','dufrac','vfracspec','dvfrac',\
									  								'lfracspec','dlfrac','pfracspec','dpfrac','phispec','pshispec','psizpec','dpsispec'])




























































