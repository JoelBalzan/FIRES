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


def plots(fname, FRB_data, mode, startms, stopms, start_chan, end_chan, rm, outdir, save, figsize, scattering_timescale, pa_rms, dpa_rms, show_plots, width_ms, rms_pol_angle):
	"""
	Plotting function for FRB data.
	Handles dynamic spectrum, IQUV profiles, L V PA profiles, and DPA.
	"""
	if mode == 'pa_rms':
		plot_pa_rms_vs_scatter(scattering_timescale, pa_rms, dpa_rms, save, fname, outdir, figsize, show_plots, width_ms, rms_pol_angle)
		sys.exit(0)
	
	if FRB_data is None:
		print("Error: FRB data is not available for the selected plot mode. \n")
		return
	
	dsdata = FRB_data

	tsdata, corrdspec, noisespec, noistks = process_dynspec(
		dsdata.dynamic_spectrum, dsdata.frequency_mhz_array, dsdata.time_ms_array, rm
	)
	# Apply zooming
	if start_chan == 0 and end_chan == 0:
		start_chan = 0
		end_chan = len(dsdata.frequency_mhz_array)
	else:
		start_chan, end_chan = find_zoom_indices(
			dsdata.frequency_mhz_array, start_chan, end_chan
		)
	if startms == 0 and stopms == 0:
		startms = 0
		stopms = len(dsdata.time_ms_array)
	else:
		startms, stopms = find_zoom_indices(
			dsdata.time_ms_array, startms, stopms
		)

	# Slice the dynspec for zoomed range
	corrdspec = corrdspec[:, start_chan:end_chan, startms:stopms]
	noisespec = noisespec[:, start_chan:end_chan]
	iquvt = tsdata.iquvt[:, startms:stopms]

	tmsarr = dsdata.time_ms_array[startms:stopms]
	fmhzarr = dsdata.frequency_mhz_array[start_chan:end_chan]

	if mode == "all":
		plot_ilv_pa_ds(corrdspec, fmhzarr, tmsarr, save, fname, outdir, tsdata, noistks, figsize, scattering_timescale, show_plots, startms, stopms)
		plot_stokes(fname, outdir, corrdspec, iquvt, fmhzarr, tmsarr, save, figsize, show_plots)
		plot_dpa(fname, outdir, noistks, tsdata, tmsarr, 5, save, figsize, show_plots)
		estimate_rm(corrdspec, fmhzarr, tmsarr, noisespec, startms, stopms, 1.0e3, 1.0, outdir, save, show_plots)
	elif mode == "iquv":
		plot_stokes(fname, outdir, corrdspec, iquvt, fmhzarr, tmsarr, save, figsize, show_plots)
	elif mode == "lvpa":
		plot_ilv_pa_ds(corrdspec, fmhzarr, tmsarr, save, fname, outdir, tsdata, noistks, figsize, scattering_timescale, show_plots, startms, stopms)
	elif mode == "dpa":
		plot_dpa(fname, outdir, noistks, tsdata, tmsarr, 5, save, figsize, show_plots)
	elif mode == "rm":
		estimate_rm(corrdspec, fmhzarr, tmsarr, noisespec, startms, stopms, 1.0e3, 1.0, outdir, save, show_plots)
	else:
		print(f"Invalid mode: {mode} \n")