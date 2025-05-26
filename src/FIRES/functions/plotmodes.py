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



#	--------------------------	Set plot parameters	---------------------------
plt.rcParams['pdf.fonttype']	= 42
plt.rcParams['ps.fonttype'] 	= 42
plt.rcParams['savefig.dpi'] 	= 600
plt.rcParams['font.size'] 		= 14  
plt.rcParams['font.family']		= 'sans-serif'  
plt.rcParams['axes.labelsize']  = 16    
plt.rcParams['axes.titlesize']  = 18    
plt.rcParams['legend.fontsize'] = 12   
plt.rcParams['xtick.labelsize'] = 12   
plt.rcParams['ytick.labelsize'] = 12   
plt.rcParams['text.usetex'] 	= True



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


def set_scale_and_labels(ax, scale, xvar, yvar):
    """
    Set axis scales and labels for the plot based on the scale argument.
    """
    if scale == "linear":
        ax.set_yscale('linear')
        ax.set_xlabel(rf"${xvar}$")
        ax.set_ylabel(rf"${yvar}$")
    elif scale == "logx":
        ax.set_xscale('log')
        ax.set_xlabel(rf"$\log_{{10}}\left({yvar}\right)$")
        ax.set_ylabel(rf"${yvar}$")
    elif scale == "logy":
        ax.set_yscale('log')
        ax.set_xlabel(rf"${xvar}$")
        ax.set_ylabel(rf"$\log_{{10}}\left({yvar}\right)$")
    elif scale == "loglog":
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(rf"$\log_{{10}}\left({yvar}\right)$")
        ax.set_ylabel(rf"$\log_{{10}}\left({yvar}\right)$")


def make_plot_fname(plot_type, scale, fname, freq_window="all", phase_window="all"):
	"""
	Generate a plot filename with freq/phase window at the front if not 'all'.
	"""
	parts = []
	if freq_window != "all":
		parts.append(f"freq_{freq_window}")
	if phase_window != "all":
		parts.append(f"phase_{phase_window}")
	parts.extend([plot_type, scale, fname])
	return "_".join(parts)


def is_multi_run_dict(frb_dict):
	"""
	Returns True if frb_dict contains multiple run dictionaries (i.e., is a dict of dicts with 'scatter_ms' keys).
	"""
	return all(isinstance(v, dict) and "scatter_ms" in v for v in frb_dict.values())


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


def plot_pa_var(frb_dict, save, fname, out_dir, figsize, show_plots, scale, phase_window, freq_window):
	"""
	Plot the var of the polarization angle (PA) and its error bars vs the scattering timescale.
	Supports plotting multiple run groups for comparison.
	"""
	# If frb_dict contains multiple runs, plot each on the same axes
	if is_multi_run_dict(frb_dict):
		fig, ax = plt.subplots(figsize=figsize)
		cmap = plt.get_cmap('Set1')
		#linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]
		for idx, (run, subdict) in enumerate(frb_dict.items()):
			color = cmap(idx % cmap.N)
			#linestyle = linestyles[idx % len(linestyles)]
			
			scatter_ms = np.array(subdict["scatter_ms"])
			vals = subdict["vals"]
			var_PA_microshots = subdict["var_PA_microshots"]
			width_ms = np.array(subdict["width_ms"])[0]
			
			vals = weight_dict(scatter_ms, vals, var_PA_microshots)
			med_vals, percentile_errs = median_percentiles(vals, scatter_ms)
			tau_weighted = scatter_ms / width_ms
			
			lower = np.array([lower for (lower, upper) in percentile_errs])
			upper = np.array([upper for (lower, upper) in percentile_errs])
			
			ax.plot(tau_weighted, med_vals, label=run, color=color)#, linestyle=linestyle)
			ax.set_xlim(tau_weighted[0], tau_weighted[-1])
			ax.fill_between(tau_weighted, lower, upper, color=color, alpha=0.1)
		ax.grid(True, linestyle='--', alpha=0.6)
		set_scale_and_labels(ax, scale, xvar=r"\tau_\mathrm{ms} / \sigma_\mathrm{ms}", yvar=r"\frac{\mathrm{Var}(\psi)}{\mathrm{Var}(\psi_\mathrm{micro})}")
		ax.legend()
		if show_plots:
			plt.show()
		if save:
			name = make_plot_fname("pa_var", scale, fname, freq_window, phase_window)
			name = os.path.join(out_dir, name + ".pdf")
			fig.savefig(name, bbox_inches='tight', dpi=600)
			print(f"Saved figure to {name}  \n")
		return
	
	# Otherwise, plot as usual (single job)
	scatter_ms = frb_dict["scatter_ms"]
	vals = frb_dict["vals"]
	var_PA_microshots = frb_dict["var_PA_microshots"]
	width_ms = frb_dict["width_ms"]
 
	vals = weight_dict(scatter_ms, vals, var_PA_microshots)
	med_vals, percentile_errs = median_percentiles(vals, scatter_ms)
	tau_weighted = scatter_ms / width_ms
 
	lower = np.array([lower for (lower, upper) in percentile_errs])
	upper = np.array([upper for (lower, upper) in percentile_errs])
 
	fig, ax = plt.subplots(figsize=figsize)
	ax.plot(tau_weighted, med_vals, color='black', label=r'\psi$_{var}$')
	ax.fill_between(tau_weighted, lower, upper, color='black', alpha=0.2)
	ax.grid(True, linestyle='--', alpha=0.6)
	ax.set_xlim(tau_weighted[0], tau_weighted[-1])
	set_scale_and_labels(ax, scale, xvar=r"\tau_{ms} / \sigma_{ms}", yvar=r"\frac{\mathrm{Var}(\psi_\mathrm{env})}{\mathrm{Var}(\psi_\mathrm{micro})}")
	if show_plots:
		plt.show()
	if save:
		name = make_plot_fname("pa_var", scale, fname, freq_window, phase_window)
		name = os.path.join(out_dir, name + ".pdf")
		fig.savefig(name, bbox_inches='tight', dpi=600)
		print(f"Saved figure to {name}  \n")


def process_lfrac(dspec, freq_mhz, time_ms, rm, phase_window, freq_window):
	
	freq_slc = get_freq_window_indices(freq_mhz, freq_window)
	
	peak_index = np.argmax(np.nansum(dspec, axis=(0, 1)))
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


def plot_lfrac_var(frb_dict, save, fname, out_dir, figsize, show_plots, scale, phase_window, freq_window):
	# If frb_dict contains multiple job IDs, plot each on the same axes
	if is_multi_run_dict(frb_dict):
		fig, ax = plt.subplots(figsize=figsize)
		cmap = plt.get_cmap('Set1')
		#linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]
		for idx, (run, subdict) in frb_dict.items():
			color = cmap(idx % cmap.N)
			#linestyle = linestyles[idx % len(linestyles)]
   
			scatter_ms = np.array(subdict["scatter_ms"])
			vals = subdict["vals"]
			errs = subdict["errs"]
			width_ms = np.array(subdict["width_ms"])[0]
			
			med_vals, percentile_errs = median_percentiles(vals, scatter_ms)
			tau_weighted = scatter_ms / width_ms
			
			lower = np.array([lower for (lower, upper) in percentile_errs])
			upper = np.array([upper for (lower, upper) in percentile_errs])
			
			ax.plot(tau_weighted, med_vals, label=run, color=color)#, linestyle=linestyle)
			ax.fill_between(tau_weighted, lower, upper, alpha=0.2, color=color)
		ax.grid(True, linestyle='--', alpha=0.6)
		set_scale_and_labels(ax, scale, xvar=r"$\tau_{ms} / \sigma_{ms}$", yvar=r"L/I")
		ax.legend()
		if show_plots:
			plt.show()
		if save:
			name = make_plot_fname("lfrac", scale, fname, freq_window, phase_window)
			name = os.path.join(out_dir, name + ".pdf")
			fig.savefig(name, bbox_inches='tight', dpi=600)
			print(f"Saved figure to {name}  \n")

		return

	# Otherwise, plot as usual (single job)
	scatter_ms = frb_dict["scatter_ms"]
	vals = frb_dict["vals"]
	errs = frb_dict["errs"]
	width_ms = frb_dict["width_ms"]
 
	med_vals, percentile_errs = median_percentiles(vals, scatter_ms)
	tau_weighted = scatter_ms / width_ms
 
	lower = np.array([lower for (lower, upper) in percentile_errs])
	upper = np.array([upper for (lower, upper) in percentile_errs])
 
	fig, ax = plt.subplots(figsize=figsize)
	ax.plot(tau_weighted, med_vals, color='black', label=r'\psi$_{var}$')
	ax.fill_between(tau_weighted, lower, upper, color='black', alpha=0.2)
	ax.grid(True, linestyle='--', alpha=0.6)
	set_scale_and_labels(ax, scale, xvar=r"$\tau_{ms} / \sigma_{ms}$", yvar=r"L/I")
	if show_plots:
		plt.show()
	if save:
		name = make_plot_fname("lfrac", scale, fname, freq_window, phase_window)
		name = os.path.join(out_dir, name + ".pdf")
		fig.savefig(name, bbox_inches='tight', dpi=600)
		print(f"Saved figure to {name}  \n")




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