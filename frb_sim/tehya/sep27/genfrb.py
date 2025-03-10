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
from genpars import *

def print_instructions():

	#	Print instructions to terminal
	
	print("\n            You probably need some assistance here!\n")
	print("Arguments are       --- <scattering time scale> <name>\n")	
	
	print("\n            Now let's try again!\n")
	
	return(0)

#	--------------------------	Read inputs	-------------------------------

if(len(sys.argv)<3):
	print_instructions()
	sys.exit()

taums		=	float(sys.argv[1])		# Scattering time scale (msec)
fname		=	sys.argv[2]				# FRB identifier

#	-------------------------	Execute steps	-------------------------------

fmhzarr		=	np.arange( cfreq-(nchan*df_mhz)/2.0 , cfreq+(nchan*df_mhz)/2.0, df_mhz, dtype=float)
tmsarr		=	np.arange( -time_win_ms, time_win_ms, time_res_ms, dtype=float )
gparams		=	np.loadtxt('gparams.txt')

#	Generate Inital dispersed dynamic spectrum with Gassian components
dynspec0	=	gauspuls(fmhzarr, tmsarr, df_mhz, time_res_ms, gparams[:,3], gparams[:,2], gparams[:,1], gparams[:,0], gparams[:,4], \
                                 gparams[:, 6], gparams[:, 7], gparams[:,8], gparams[:, 9], gparams[:,5])

#	Scatter the dynamic spectrum 
scdspec		=	scatter_dynspec(dynspec0, fmhzarr, tmsarr, df_mhz, time_res_ms, taums, scindex)

#	'Pickle' the simulated FRB and save it to the disk
fakefrb		=	simfrb(fname,fmhzarr,tmsarr,taums,f_ref_mhz,scindex,gparams,scdspec)      

frbfile		=	open("{}{}_sc_{:.2f}.pkl".format(datadir,fname,taums),'wb')             # Create the data directory, keep all simulated frbs 
pkl.dump(fakefrb, frbfile)		
frbfile.close()


































































