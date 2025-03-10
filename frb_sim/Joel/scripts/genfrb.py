#
#	Simulating scattering 
#
#								AB, Sep 2024
#                                                               
#	--------------------------	Import modules	---------------------------

import os, sys
import matplotlib as mpl
import numpy as np
from genfns import *
from utils import *

def print_instructions():
	"""
	Print instructions to terminal
	"""

	print("\n            You probably need some assistance here!\n")
	print("Arguments are       --- <scattering time scale> <name>\n")	
	print("\n            Now let's try again!\n")
	
	return(0)

#	--------------------------	Read inputs	-------------------------------

if(len(sys.argv)<3):
	print_instructions()
	sys.exit()

tau_ms		=	float(sys.argv[1])		# Scattering time scale (msec)
fname		=	sys.argv[2]				# FRB identifier

#	-------------------------	Execute steps	-------------------------------

f_mhzarr	=	np.arange( central_frequency_mhz-(num_channels*channel_width_mhz)/2.0 , central_frequency_mhz+(num_channels*channel_width_mhz)/2.0, channel_width_mhz, dtype=float)		# Array of frequency channels
t_msarr		=	np.arange( -time_window_ms, time_window_ms, time_resolution_ms, dtype=float )										# Array of time bins
gparams		=	np.loadtxt('gparams.txt')																# Load gaussians from gparams.txt

#	Generate Inital dispersed dynamic spectrum with Gassian components
dynspec0	=	gauss_dynspec(f_mhzarr, t_msarr, channel_width_mhz, time_resolution_ms, gparams[:,3], gparams[:,2], gparams[:,1], gparams[:,0], gparams[:,4], \
                                 gparams[:, 6], gparams[:, 7], gparams[:,8], gparams[:, 9], gparams[:,5])

#	Scatter the dynamic spectrum 
sc_dynspec	=	scatter_dynspec(dynspec0, f_mhzarr, t_msarr, channel_width_mhz, time_resolution_ms, tau_ms, scattering_index)

#	'Pickle' the simulated FRB and save it to the disk
fakefrb		=	simulated_frb(fname,f_mhzarr,t_msarr,tau_ms,reference_frequency_mhz,scattering_index,gparams,sc_dynspec)      

frbfile		=	open("{}{}_sc_{:.2f}.pkl".format(data_directory,fname,tau_ms),'wb')             # Create the data directory, keep all simulated frbs 
pkl.dump(fakefrb, frbfile)		
frbfile.close()


































































