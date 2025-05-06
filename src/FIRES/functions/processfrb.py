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


# ...existing imports...

def plots(fname, frb_data, mode, rm, out_dir, save, figsize, scatter_ms, pa_var_weighted, dpa_var_weighted, show_plots, width_ms):
    """
    Plotting function for FRB data.
    Handles dynamic spectrum, IQUV profiles, L V PA profiles, and DPA.
    """
    if mode == 'pa_var':
        plot_pa_var_vs_scatter(scatter_ms, pa_var_weighted, dpa_var_weighted, save, fname, out_dir, figsize, show_plots, width_ms)
        sys.exit(0)
    
    if frb_data is None:
        print("Error: FRB data is not available for the selected plot mode. \n")
        return
    
    ds_data = frb_data

    ts_data, corr_dspec, noise_spec, noise_stokes = process_dynspec(
        ds_data.dynamic_spectrum, ds_data.freq_mhz, ds_data.time_ms, rm
    )

    iquvt = ts_data.iquvt
    time_ms = ds_data.time_ms
    freq_mhz = ds_data.freq_mhz

    if mode == "all":
        plot_ilv_pa_ds(corr_dspec, freq_mhz, time_ms, save, fname, out_dir, ts_data, noise_stokes, figsize, scatter_ms, show_plots)
        plot_stokes(fname, out_dir, corr_dspec, iquvt, freq_mhz, time_ms, save, figsize, show_plots)
        plot_dpa(fname, out_dir, noise_stokes, ts_data, time_ms, 5, save, figsize, show_plots)
        estimate_rm(corr_dspec, freq_mhz, time_ms, noise_spec, 1.0e3, 1.0, out_dir, save, show_plots)
    elif mode == "iquv":
        plot_stokes(fname, out_dir, corr_dspec, iquvt, freq_mhz, time_ms, save, figsize, show_plots)
    elif mode == "lvpa":
        plot_ilv_pa_ds(corr_dspec, freq_mhz, time_ms, save, fname, out_dir, ts_data, noise_stokes, figsize, scatter_ms, show_plots)
    elif mode == "dpa":
        plot_dpa(fname, out_dir, noise_stokes, ts_data, time_ms, 5, save, figsize, show_plots)
    elif mode == "rm":
        estimate_rm(corr_dspec, freq_mhz, time_ms, noise_spec, 1.0e3, 1.0, out_dir, save, show_plots)
    else:
        print(f"Invalid mode: {mode} \n")