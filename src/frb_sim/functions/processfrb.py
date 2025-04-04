#
#	Script for FRB polarization analysis
#
#								AB, September 2024

#	--------------------------	Import modules	---------------------------

import os
import sys

import numpy as np
from .basicfns import *
from .plotfns import *
from ..utils.utils import *


def plots(fname, FRB_data, mode, startms, stopms, startchan, endchan, rm, outdir, save):
	#	Plotting function
	#	Plots the dynamic spectrum and the IQUV profiles
	#	Plots the L V and PA profiles
	#	Plots the DPA

	#	-------------------------	Do steps	-------------------------------
	dsdata	=	FRB_data
	nchan	=	len(dsdata.frequency_mhz_array)

	if(startchan < 0):
		startchan	=	0 

	if(endchan <= 0):
		endchan	=	nchan-1 

	if(startms == 0):
		startms	=	dsdata.time_ms_array[0]

	if(stopms == 0):
		stopms	=	dsdata.time_ms_array[-1]

	#	Estimate Noise spectra
	noisespec	=	estimate_noise(dsdata.dynamic_spectrum, dsdata.time_ms_array, startms, stopms) # add the arguments here 
	noistks		=	np.sqrt(np.nansum(noisespec[:,startchan:endchan]**2,axis=1))/len(dsdata.frequency_mhz_array)
	
	# Use the RM with the largest magnitude
	max_rm = rm[np.argmax(np.abs(rm))]
	corrdspec	=	rm_correct_dynspec(dsdata.dynamic_spectrum, dsdata.frequency_mhz_array, max_rm)
	tsdata		=	est_profiles(corrdspec, dsdata.frequency_mhz_array, dsdata.time_ms_array, noisespec, startchan, endchan)
	if (mode == "all"):
		plot_ilv_pa_ds(corrdspec, dsdata.frequency_mhz_array, dsdata.time_ms_array, max_rm, save, fname, outdir, tsdata, noistks)
		plot_stokes(fname, outdir, corrdspec, tsdata.iquvt, dsdata.frequency_mhz_array, dsdata.time_ms_array, save)
		plot_ilv_pa_ds(fname, outdir, noistks, corrdspec, tsdata, dsdata.frequency_mhz_array, dsdata.time_ms_array, save)
		plot_dpa(fname, outdir, noistks, tsdata, dsdata.time_ms_array, 5, save)
		estimate_rm(dsdata.dynamic_spectrum, dsdata.frequency_mhz_array, dsdata.time_ms_array, noisespec, startms, stopms, 1.0e3, 1.0, startchan, endchan, outdir, save)

	else:
		if(mode=="iquv"):
			plot_stokes(fname, outdir, corrdspec, tsdata.iquvt, dsdata.frequency_mhz_array, dsdata.time_ms_array, save)

		if(mode=="lvpa"):
			plot_ilv_pa_ds(corrdspec, dsdata.frequency_mhz_array, dsdata.time_ms_array, save, fname, outdir, tsdata, noistks)

		if(mode=="dpa"):
			plot_dpa(fname, outdir, noistks, tsdata, dsdata.time_ms_array, 5, save)
		
		if(mode=="rm"):
			estimate_rm(dsdata.dynamic_spectrum, dsdata.frequency_mhz_array, dsdata.time_ms_array, noisespec, startms, stopms, 1.0e3, 1.0, startchan, endchan, outdir, save)

	





































