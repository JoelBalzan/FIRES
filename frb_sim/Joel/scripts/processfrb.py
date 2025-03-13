#
#	Script for FRB polarization analysis
#
#								AB, September 2024

#	--------------------------	Import modules	---------------------------

import os
import sys

import numpy as np
from basicfns import *
from plotfns import *
from utils import *


def print_instructions():

	#	Print instructions to terminal
	
	print("\n            You probably need some assistance here!\n")
	print("\n Arguments are       --- <Name> <mode> <taums> <startms> <stopms> <startchan> <endchan> <rm0>\n")	
	print(" Supported Modes are --- calcrm    (Estimate RM)")
	print("                         iquv    (Plot IQUV dynamic spectra)")
	print("                         lvpa    (Plot L V and PA profiles)")
	print("                         dpa     (Find PA change)")
	
	print("\n            Now let's try again!\n")
	
	return(0)

#	--------------------------	Read inputs	-------------------------------

if(len(sys.argv)<9):
	print_instructions()
	sys.exit()
	
fname		=	sys.argv[1]					#	Name
exmode		=	sys.argv[2]					#	What to do
taums		=	float(sys.argv[3])			#	Scattering timescale
startms		=	float(sys.argv[4])			#	Starting time from peak (ms)
stopms		=	float(sys.argv[5])			#	Stopping time from peak (ms)
startchan	=	int(sys.argv[6])
endchan		=	int(sys.argv[7])            # 	0, 0 = all channels 
rm0			=	float(sys.argv[8])			#	RM for derotation, set to 0.0 otherwise

#	-------------------------	Do steps	-------------------------------

dsfile	=	open("{}{}_sc_{:.2f}.pkl".format(data_directory,fname,taums),'rb')
dsdata	=	pkl.load(dsfile)
dsfile.close()

nchan	=	len(dsdata.fmhzarr)

if(startchan < 0):
	startchan	=	0 

if(endchan <= 0):
	endchan	=	nchan-1 

#	Estimate Noise spectra
noisespec	=	estimate_noise(dsdata.dspec4, dsdata.tmsarr, startms, stopms) # add the arguments here 
noistks		=	np.sqrt(np.nansum(noisespec[:,startchan:endchan]**2,axis=1))/len(dsdata.fmhzarr)

if(exmode=="calcrm"):
	#	Estimate RM
	resrmt		=	estimate_rm(dsdata.dspec4, dsdata.fmhzarr, dsdata.tmsarr, noisespec, startms, stopms, 1.0e3, 1.0, startchan, endchan)
	
	#	Otherwise correct fot the given RM
else:
	corrdspec	=	rm_correct_dynspec(dsdata.dspec4, dsdata.fmhzarr, rm0)
	tsdata		=	est_profiles(corrdspec, dsdata.fmhzarr, dsdata.tmsarr, noisespec, startchan, endchan)
	
	if(exmode=="iquv"):
		plot_stokes(plot_directory,corrdspec,tsdata.iquvt,dsdata.fmhzarr,dsdata.tmsarr,[0.0,0.0],[5.0,8.0])
	
	if(exmode=="lvpa"):
		plot_ilv_pa_ds(plot_directory,noistks,corrdspec,tsdata,dsdata.fmhzarr,dsdata.tmsarr,[0.0,0.0],[4.0,5.0])

	if(exmode=="dpa"):
		plot_dpa(plot_directory,noistks,tsdata,dsdata.tmsarr,[4.0,4.0],5)
        





































