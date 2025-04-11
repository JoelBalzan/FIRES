#
#	Script for FRB polarization analysis
#
#								AB, September 2024

#	--------------------------	Import modules	---------------------------

import os
import sys

import numpy as np
from FIRES.functions.basicfns import *
from FIRES.functions.plotfns import *
from FIRES.utils.utils import *


def plots(fname, FRB_data, mode, startms, stopms, startchan, endchan, rm, outdir, save, figsize, scattering_timescale, pa_rms, dpa_rms, show_plots, width_ms):
	"""
	Plotting function for FRB data.
	Handles dynamic spectrum, IQUV profiles, L V PA profiles, and DPA.
	"""
	if mode == 'pa_rms':
		plot_pa_rms_vs_scatter(scattering_timescale, pa_rms, dpa_rms, save, fname, outdir, figsize, show_plots, width_ms)
		sys.exit(0)
	
	if FRB_data is None:
		print("Error: FRB data is not available for the selected plot mode. \n")
		return
	
	dsdata = FRB_data

	tsdata, corrdspec, noisespec, noistks = process_dynspec(
		dsdata.dynamic_spectrum, dsdata.frequency_mhz_array, dsdata.time_ms_array, startms, stopms, startchan, endchan, rm
	)
	if startchan == 0 and endchan == 0:
		startchan = 0
		endchan = len(dsdata.frequency_mhz_array)
	else:
		startchan, endchan = find_zoom_indices(
			dsdata.frequency_mhz_array, startchan, endchan
		)
	if startms == 0 and stopms == 0:
		startms = 0
		stopms = len(dsdata.time_ms_array)
	else:
		startms, stopms = find_zoom_indices(
			dsdata.time_ms_array, startms, stopms
		)
	#corrdspec = corrdspec[:, startchan:endchan, startms:stopms]
	#iquvt = tsdata.iquvt[:, startms:stopms]
	tmsarr = dsdata.time_ms_array[startms:stopms]
	fmhzarr = dsdata.frequency_mhz_array[startchan:endchan]

	if mode == "all":
		plot_ilv_pa_ds(corrdspec, fmhzarr, tmsarr, save, fname, outdir, tsdata, noistks, figsize, scattering_timescale, show_plots, startms, stopms, startchan, endchan)
		plot_stokes(fname, outdir, corrdspec, tsdata.iquvt, fmhzarr, tmsarr, save, figsize, show_plots, startms, stopms, startchan, endchan)
		plot_dpa(fname, outdir, noistks, tsdata, tmsarr, 5, save, figsize, show_plots, startms, stopms, startchan, endchan)
		estimate_rm(corrdspec, fmhzarr, tmsarr, noisespec, startms, stopms, 1.0e3, 1.0, outdir, save, show_plots)
	elif mode == "iquv":
		plot_stokes(fname, outdir, corrdspec, tsdata.iquvt, fmhzarr, tmsarr, save, figsize, show_plots, startms, stopms, startchan, endchan)
	elif mode == "lvpa":
		plot_ilv_pa_ds(corrdspec, fmhzarr, tmsarr, save, fname, outdir, tsdata, noistks, figsize, scattering_timescale, show_plots, startms, stopms, startchan, endchan)
	elif mode == "dpa":
		plot_dpa(fname, outdir, noistks, tsdata, tmsarr, 5, save, figsize, show_plots, startms, stopms, startchan, endchan)
	elif mode == "rm":
		estimate_rm(corrdspec, fmhzarr, tmsarr, noisespec, startms, stopms, 1.0e3, 1.0, outdir, save, show_plots)
	else:
		print(f"Invalid mode: {mode} \n")