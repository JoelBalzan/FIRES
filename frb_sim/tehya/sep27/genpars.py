#
#	Functions for simulating scattering 
#
#								AB, May 2024
#
#	--------------------------	Import modules	---------------------------

import os, sys
import matplotlib as mpl
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import pickle as pkl
from collections import namedtuple

cfreq		=	1000.0		#centre freq										#	Central frequency in MHz
nchan		=	64		#number channels (336 stand)							#	Number of fequency channels
df_mhz		=	5.25		#width freq channel (1MHz)								#	Channel width in MHz
time_win_ms		=	10.0		#time window (msec) 									#	Time window in ms
time_res_ms		=	0.1		#time res (msec) 										#	Time resolution in ms
scindex		=	-4.0		#scattering index 									#	Scattering index
f_ref_mhz		=	1000.0		# ref freq (MHz)									#	Reference frequency for scattering

# universal constants 
G_cgs		=	6.67430e-8						#	Universal gravitational constant in CGS
elecE		=	4.8032047e-10					#	Absolute electronic charge in CGS
mE			=	9.1093837e-28					#	Electron mass in CGS
c_cgs			=	2.99792458e10					#	Speed of light in CGS
pc_cm		=	3.0857e18						#	Persec / cm							
wbynu		=	2*np.pi							#	Omega / nu 
mSUN		=	1.98847e33						#	Solar mass in grams
radtosec	=	180.0*3600/np.pi				#	Radian in arcsecs
radtopas	=	180.0*3600*1.0e12/np.pi			#	Radian in pico-arcsecs
radsolar	=	6.957e10						#	Solar radius in cm
auincm		=	1.496e13						#	1 AU in cm
intocm		=	2.54
datadir		=	'../simfrbs/'

# 1 FRB data simulated, (taums as input command line),
# dspec4 = 4D
# create another sub directory called SIMFRB
simfrb		=	namedtuple('simfrb',['frbname','fmhzarr','tmsarr','taums','f_ref_mhz','scindex','gparams','dspec4'])























































