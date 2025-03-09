#
#	Global parameters for FRB HTR polarization analysis
#
#								AB, Sep 2024

import numpy as np
import pickle as pkl
from collections import namedtuple

#	-------------------		Constants	------------------------------------------------

uniG		=	6.67430e-8						#	Universal gravitational constant in CGS
elecE		=	4.8032047e-10					#	Absolute electronic charge in CGS
mE			=	9.1093837e-28					#	Electron mass in CGS
ccC			=	2.99792458e10					#	Speed of light in CGS
pcincm		=	3.0857e18						#	Persec / cm							
wbynu		=	2*np.pi							#	Omega / nu 
mSUN		=	1.98847e33						#	Solar mass in grams
radtosec	=	180.0*3600/np.pi				#	Radian in arcsecs
radtopas	=	180.0*3600*1.0e12/np.pi			#	Radian in pico-arcsecs
radsolar	=	6.957e10						#	Solar radius in cm
auincm		=	1.496e13						#	1 AU in cm
intocm		=	2.54							#	1 inch in cm
taums		=	75.0							#	Scattering timescale in ms

datadir		=	'../simfrbs/'
plotdir		=	'../plots/'

# same as other script 
simfrb		=	namedtuple('simfrb',['frbname','fmhzarr','tmsarr','taums','frefmhz','scindex','gparams','dspec4'])

# time variation
frbts		=	namedtuple('frbts',['iquvt','lts','elts','pts','epts','phits','dphits','psits','dpsits','qfrac','eqfrac','ufrac','eufrac','vfrac','evfrac','lfrac','elfrac','pfrac','epfrac'])

# spectra (anything varying with freq (hz))
frbspec		=	namedtuple('frbspec',['iquvspec','diquvspec','lspec','dlspec','pspec','dpspec','qfracspec','dqfrac','ufracspec','dufrac','vfracspec','dvfrac',\
									  								'lfracspec','dlfrac','pfracspec','dpfrac','phispec','pshispec','psizpec','dpsispec'])




























































