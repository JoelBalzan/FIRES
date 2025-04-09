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


def plots(fname, FRB_data, mode, startms, stopms, startchan, endchan, rm, outdir, save, figsize, scattering_timescale, pa_rms, dpa_rms):
    """
    Plotting function for FRB data.
    Handles dynamic spectrum, IQUV profiles, L V PA profiles, and DPA.
    """
    if mode == "pa_rms":
        plot_pa_rms_vs_scatter(scattering_timescale, pa_rms, dpa_rms, save, fname, outdir, figsize)
        sys.exit(0)

    dsdata = FRB_data
    nchan = len(dsdata.frequency_mhz_array)
    startchan = max(0, startchan)
    endchan = nchan - 1 if endchan <= 0 else endchan
    startms = startms or dsdata.time_ms_array[0]
    stopms = stopms or dsdata.time_ms_array[-1]

    tsdata, corrdspec, noisespec = process_dynspec(
        dsdata.dynamic_spectrum, dsdata.frequency_mhz_array, dsdata.time_ms_array, startms, stopms, startchan, endchan, rm
    )
    noistks = np.sqrt(np.nansum(noisespec[:, startchan:endchan]**2, axis=1)) / nchan
    max_rm = rm[np.argmax(np.abs(rm))]

    if mode == "all":
        plot_ilv_pa_ds(corrdspec, dsdata.frequency_mhz_array, dsdata.time_ms_array, max_rm, save, fname, outdir, tsdata, noistks, figsize)
        plot_stokes(fname, outdir, corrdspec, tsdata.iquvt, dsdata.frequency_mhz_array, dsdata.time_ms_array, save, figsize)
        plot_dpa(fname, outdir, noistks, tsdata, dsdata.time_ms_array, 5, save, figsize)
        estimate_rm(dsdata.dynamic_spectrum, dsdata.frequency_mhz_array, dsdata.time_ms_array, noisespec, startms, stopms, 1.0e3, 1.0, startchan, endchan, outdir, save, figsize)
    elif mode == "iquv":
        plot_stokes(fname, outdir, corrdspec, tsdata.iquvt, dsdata.frequency_mhz_array, dsdata.time_ms_array, save, figsize)
    elif mode == "lvpa":
        plot_ilv_pa_ds(corrdspec, dsdata.frequency_mhz_array, dsdata.time_ms_array, save, fname, outdir, tsdata, noistks, figsize, scattering_timescale)
    elif mode == "dpa":
        plot_dpa(fname, outdir, noistks, tsdata, dsdata.time_ms_array, 5, save, figsize)
    elif mode == "rm":
        estimate_rm(dsdata.dynamic_spectrum, dsdata.frequency_mhz_array, dsdata.time_ms_array, noisespec, startms, stopms, 1.0e3, 1.0, startchan, endchan, outdir, save, figsize)
    else:
        print(f"Invalid mode: {mode}")