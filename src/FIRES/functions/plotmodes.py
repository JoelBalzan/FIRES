# -----------------------------------------------------------------------------
# plotmodes.py
# FIRES: The Fast, Intense Radio Emission Simulator
#
# This module defines plot modes and their associated processing and plotting
# functions for visualizing FRB simulation results. It includes classes and
# functions for plotting Stokes parameters, polarization angle variance,
# linear fraction, and other derived quantities as a function of scattering
# and simulation parameters.
#
# Author: JB
# Date: 2025-05-20
# -----------------------------------------------------------------------------

#	--------------------------	Import modules	---------------------------
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt

from FIRES.functions.basicfns import process_dynspec, median_percentiles, weight_dict
from FIRES.functions.plotfns import plot_stokes, plot_ilv_pa_ds, plot_dpa, estimate_rm


class PlotMode:
	def __init__(self, name, process_func, plot_func, requires_multiple_tau=False):
		"""
		Represents a plot mode with its associated processing and plotting functions.

		Args:
			name (str): Name of the plot mode.
			process_func (callable): Function to process data for this plot mode.
			plot_func (callable): Function to generate the plot.
			requires_multiple_tau (bool): Whether this plot mode requires `plot_var=True`.
		"""
		self.name = name
		self.process_func = process_func
		self.plot_func = plot_func
		self.requires_multiple_tau = requires_multiple_tau
		
def basic_plots(fname, frb_data, mode, rm, out_dir, save, figsize, scatter_ms, show_plots):

	ds_data = frb_data

	ts_data, corr_dspec, noise_spec, noise_stokes = process_dynspec(
		ds_data.dynamic_spectrum, ds_data.freq_mhz, ds_data.time_ms, rm
	)

	iquvt = ts_data.iquvt
	time_ms = ds_data.time_ms
	freq_mhz = ds_data.freq_mhz

	if mode == "all":
		plot_ilv_pa_ds(corr_dspec, freq_mhz, time_ms, save, fname, out_dir, ts_data, figsize, scatter_ms, show_plots)
		plot_stokes(fname, out_dir, corr_dspec, iquvt, freq_mhz, time_ms, save, figsize, show_plots)
		plot_dpa(fname, out_dir, noise_stokes, ts_data, time_ms, 5, save, figsize, show_plots)
		estimate_rm(corr_dspec, freq_mhz, time_ms, noise_spec, 1.0e3, 1.0, out_dir, save, show_plots)
	elif mode == "iquv":
		plot_stokes(fname, out_dir, corr_dspec, iquvt, freq_mhz, time_ms, save, figsize, show_plots)
	elif mode == "lvpa":
		plot_ilv_pa_ds(corr_dspec, freq_mhz, time_ms, save, fname, out_dir, ts_data, figsize, scatter_ms, show_plots)
	elif mode == "dpa":
		plot_dpa(fname, out_dir, noise_stokes, ts_data, time_ms, 5, save, figsize, show_plots)
	elif mode == "rm":
		estimate_rm(corr_dspec, freq_mhz, time_ms, noise_spec, 1.0e3, 1.0, out_dir, save, show_plots)
	else:
		print(f"Invalid mode: {mode} \n")


def get_freq_window_indices(freq_mhz, freq_window):
	q = int(len(freq_mhz) / 4)
	windows = {
		"1q": slice(0, q),
		"2q": slice(q, 2*q),
		"3q": slice(2*q, 3*q),
		"4q": slice(3*q, None),
		"all": slice(None)
	}
	return windows.get(freq_window, None)

def get_phase_window_indices(phase_window, peak_index):
	"""
	Returns a slice object for the desired phase window.
	"""
	phase_slices = {
		"first": slice(0, peak_index),
		"last": slice(peak_index, None),
		"all": slice(None)
	}
	return phase_slices.get(phase_window, None)

# Processing function for pa_var
def process_pa_var(dspec, freq_mhz, time_ms, rm, phase_window, freq_window):
	
	slc = get_freq_window_indices(freq_mhz, freq_window)
	freq_mhz = freq_mhz[slc]
	dspec = dspec[:, slc, :]
		
	ts_data, _, _, _ = process_dynspec(dspec, freq_mhz, time_ms, rm)
	
	peak_index = np.argmax(ts_data.iquvt[0])
	phase_slc = get_phase_window_indices(phase_window, peak_index)

	phits = ts_data.phits[phase_slc]
	dphits = ts_data.dphits[phase_slc]
		
   
	pa_var = np.nanvar(phits)
	pa_var_err = np.sqrt(np.nansum((phits * dphits)**2)) / (pa_var * len(phits))
	return pa_var, pa_var_err


def plot_pa_var(scatter_ms, vals, save, fname, out_dir, figsize, show_plots, width_ms, var_PA_microshots):
	"""
	Plot the var of the polarization angle (PA) and its error bars vs the scattering timescale.
	"""
	
	# weight PA_var by microshot var
	vals = weight_dict(scatter_ms, vals, var_PA_microshots)
 
	med_vals, percentile_errs = median_percentiles(vals, scatter_ms)
 
	fig, ax = plt.subplots(figsize=figsize)

	# weight the scattering timescale by initial Gaussian width
	tau_weighted = scatter_ms / width_ms

	lower_errors = [median - lower for (lower, upper), median in zip(percentile_errs, med_vals)]
	upper_errors = [upper - median for (lower, upper), median in zip(percentile_errs, med_vals)]
	
	ax.errorbar(tau_weighted, med_vals, 
				yerr=(lower_errors, upper_errors), 
				fmt='o', capsize=1, color='black', label=r'\psi$_{var}$', markersize=2)
 
	ax.set_xlabel(r"$\tau_{ms} / \sigma_{ms}$")
	ax.set_ylabel(r"Var(\psi) / Var(\psi$_{microshots}$)")
	ax.grid(True, linestyle='--', alpha=0.6)

	if show_plots:
		plt.show()

	if save:
		fig.savefig(os.path.join(out_dir, fname + "_pa_var_vs_scatter.pdf"), bbox_inches='tight', dpi=600)
		print(f"Saved figure to {os.path.join(out_dir, fname + '_pa_var_vs_scatter.pdf')}  \n")


def process_lfrac(dspec, freq_mhz, time_ms, rm, phase_window, freq_window):
	
	freq_slc = get_freq_window_indices(freq_mhz, freq_window)
	
	peak_index = np.argmax(np.mean(dspec, axis=(0, 1)))
	phase_slc = get_phase_window_indices(phase_window, peak_index)

	freq_mhz = freq_mhz[freq_slc]
	time_ms = time_ms[phase_slc]
	dspec = dspec[:, freq_slc, phase_slc]
	
	ts_data, _, _, _ = process_dynspec(dspec, freq_mhz, time_ms, rm)
 
	iquvt = ts_data.iquvt
	I = ts_data.iquvt[0]

	threshold = 0.05 * np.nanmax(I)
	mask = I <= threshold
 
	I_masked = np.where(mask, np.nan, iquvt[0])
	Q_masked = np.where(mask, np.nan, iquvt[1])
	U_masked = np.where(mask, np.nan, iquvt[2])
	V_masked = np.where(mask, np.nan, iquvt[3])
	
	L = np.sqrt(Q_masked**2 + U_masked**2)
 
	integrated_I = np.nansum(I_masked)
	integrated_L = np.nansum(L)
	lfrac = integrated_L / integrated_I
 
	mask = I > threshold
	noise_I = np.nanstd(I[mask])
	noise_L = np.nanstd(L[mask])
	lfrac_err = np.sqrt((noise_L / integrated_I)**2 + (integrated_L * noise_I / integrated_I**2)**2)

	return lfrac, lfrac_err


def plot_lfrac_var(scatter_ms, vals, save, fname, out_dir, figsize, show_plots):
	
	med_vals, percentile_errs = median_percentiles(vals, scatter_ms)
	
	fig, ax = plt.subplots(figsize=figsize)

	lower_errors = [median - lower for (lower, upper), median in zip(percentile_errs, med_vals)]
	upper_errors = [upper - median for (lower, upper), median in zip(percentile_errs, med_vals)]
	
	ax.errorbar(scatter_ms, med_vals, 
				yerr=(lower_errors, upper_errors), 
				fmt='o', capsize=1, color='black', label=r'\psi$_{var}$', markersize=2)
 
	ax.set_xlabel(r"L/I")
	ax.set_ylabel(r"Var(\psi) / Var(\psi$_{microshots}$)")
	ax.grid(True, linestyle='--', alpha=0.6)

	if show_plots:
		plt.show()

	if save:
		fig.savefig(os.path.join(out_dir, fname + "_lfrac_vs_scatter.pdf"), bbox_inches='tight', dpi=600)
		print(f"Saved figure to {os.path.join(out_dir, fname + '_lfrac_vs_scatter.pdf')}  \n")


pa_var = PlotMode(
	name="pa_var",
	process_func=process_pa_var,
	plot_func=plot_pa_var,
	requires_multiple_tau=True  
)

lfrac = PlotMode(
	name="lfrac",
	process_func=process_lfrac,
	plot_func=plot_lfrac_var,
	requires_multiple_tau=True  
)

iquv = PlotMode(
	name="iquv",
	process_func=None, 
	plot_func=basic_plots,
	requires_multiple_tau=False  
)

lvpa = PlotMode(
	name="lvpa",
	process_func=None, 
	plot_func=basic_plots,
	requires_multiple_tau=False
)

dpa = PlotMode(
	name="dpa",
	process_func=None, 
	plot_func=basic_plots,
	requires_multiple_tau=False
)

rm = PlotMode(
	name="rm",
	process_func=None,  
	plot_func=basic_plots,
	requires_multiple_tau=False
)

plot_modes = {
	"pa_var": pa_var,
	"iquv": iquv,
	"lvpa": lvpa,
	"dpa": dpa,
	"rm": rm,
	"lfrac": lfrac,
}