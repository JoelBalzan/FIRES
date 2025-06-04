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
import numpy as np
from scipy.optimize import curve_fit

from FIRES.functions.basicfns import process_dynspec, median_percentiles, weight_dict
from FIRES.functions.plotfns import plot_stokes, plot_ilv_pa_ds, plot_dpa, estimate_rm



#	--------------------------	Set plot parameters	---------------------------
plt.rcParams['pdf.fonttype']	= 42
plt.rcParams['ps.fonttype'] 	= 42
plt.rcParams['savefig.dpi'] 	= 600
plt.rcParams['font.size'] 		= 18
plt.rcParams['font.family']		= 'sans-serif'  
plt.rcParams['axes.labelsize']  = 20
plt.rcParams['axes.titlesize']  = 18
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['text.usetex'] 	= True

#colour blind friendly: https://gist.github.com/thriveth/8560036
colours = {
	'purple':  '#984ea3',
	'red':     '#e41a1c',
	'blue':    '#377eb8', 
	'green':   '#4daf4a',
	'pink':    '#f781bf',
	'brown':   '#a65628',
	'orange':  '#ff7f00',
	'gray':    '#999999',
	'yellow':  '#dede00'
} 


class PlotMode:
	def __init__(self, name, process_func, plot_func, requires_multiple_frb=False):
		"""
		Represents a plot mode with its associated processing and plotting functions.

		Args:
			name (str): Name of the plot mode.
			process_func (callable): Function to process data for this plot mode.
			plot_func (callable): Function to generate the plot.
			requires_multiple_frb (bool): Whether this plot mode requires `plot_var=True`.
		"""
		self.name = name
		self.process_func = process_func
		self.plot_func = plot_func
		self.requires_multiple_frb = requires_multiple_frb
		
def basic_plots(fname, frb_data, mode, gdict, out_dir, save, figsize, tau_ms, show_plots):

	ds_data = frb_data

	ts_data, corr_dspec, noise_spec, noise_stokes = process_dynspec(
		ds_data.dynamic_spectrum, ds_data.freq_mhz, ds_data.time_ms, gdict, tau_ms
	)

	iquvt = ts_data.iquvt
	time_ms = ds_data.time_ms
	freq_mhz = ds_data.freq_mhz

	if mode == "all":
		plot_ilv_pa_ds(corr_dspec, freq_mhz, time_ms, save, fname, out_dir, ts_data, figsize, tau_ms, show_plots)
		plot_stokes(fname, out_dir, corr_dspec, iquvt, freq_mhz, time_ms, save, figsize, show_plots)
		plot_dpa(fname, out_dir, noise_stokes, ts_data, time_ms, 5, save, figsize, show_plots)
		estimate_rm(ds_data.dynamic_spectrum, freq_mhz, time_ms, noise_spec, 1.0e3, 1.0, out_dir, save, show_plots)
	elif mode == "iquv":
		plot_stokes(fname, out_dir, corr_dspec, iquvt, freq_mhz, time_ms, save, figsize, show_plots)
	elif mode == "lvpa":
		plot_ilv_pa_ds(corr_dspec, freq_mhz, time_ms, save, fname, out_dir, ts_data, figsize, tau_ms, show_plots)
	elif mode == "dpa":
		plot_dpa(fname, out_dir, noise_stokes, ts_data, time_ms, 5, save, figsize, show_plots)
	elif mode == "RM":
		estimate_rm(ds_data.dynamic_spectrum, freq_mhz, time_ms, noise_spec, 1.0e3, 1.0, out_dir, save, show_plots)
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


def set_scale_and_labels(ax, scale, xvar, yvar, x=None):
	"""
	Set axis scales and labels for the plot based on the scale argument.
	Optionally set x-limits using the provided x array.
	"""
	if scale == "linear":
		ax.set_yscale('linear')
		ax.set_xlabel(rf"${xvar}$")
		ax.set_ylabel(rf"${yvar}$")
		if x is not None:
			ax.set_xlim(x[0], x[-1])
	elif scale == "logx":
		ax.set_xscale('log')
		ax.set_xlabel(rf"$\log_{{10}}({xvar})$")
		ax.set_ylabel(rf"${yvar}$")
		if x is not None:
			x_positive = x[x > 0]
			if len(x_positive) > 0:
				ax.set_xlim(x_positive[0], x_positive[-1])
	elif scale == "logy":
		ax.set_yscale('log')
		ax.set_xlabel(rf"${xvar}$")
		ax.set_ylabel(rf"$\log_{{10}}({yvar})$")
		if x is not None:
			ax.set_xlim(x[0], x[-1])
	elif scale == "loglog":
		ax.set_xscale('log')
		ax.set_yscale('log')
		ax.set_xlabel(rf"$\log_{{10}}({xvar})$")
		ax.set_ylabel(rf"$\log_{{10}}({yvar})$")
		if x is not None:
			x_positive = x[x > 0]
			if len(x_positive) > 0:
				ax.set_xlim(x_positive[0], x_positive[-1])


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
	Returns True if frb_dict contains multiple run dictionaries (i.e., is a dict of dicts with 'tau_ms' keys).
	"""
	return all(isinstance(v, dict) and "xvals" in v for v in frb_dict.values())


def fit_and_plot(ax, x, y, fit, label=None, color='black'):
	"""
	Fit the data (x, y) with the specified function and plot the fit on ax.
	fit: list or tuple, e.g. ['power'], ['exp'], or ['power', '3']
	"""
	if fit is None:
		return
	
	x = np.asarray(x)
	y = np.asarray(y)

	# Parse fit argument
	if isinstance(fit, str):
		fit = [fit]
	fit_type = fit[0].lower()
	fit_degree = int(fit[1]) if len(fit) > 1 and fit_type == "power" else None

	# Define fit functions
	def power_law(x, a, n):
		return a * x**n

	def power_fixed_n(x, a):
		return a * x**fit_degree

	def exponential(x, a, b):
		return a * np.exp(b * x)

	# Remove NaNs and non-positive x for log fits
	mask = np.isfinite(x) & np.isfinite(y)
	if fit_type in ("power",) and fit_degree is None:
		mask &= (x > 0)
	x_fit = x[mask]
	y_fit = y[mask]

	# Choose and fit the model
	try:
		if fit_type == "power" and fit_degree is not None:
			popt, _ = curve_fit(power_fixed_n, x_fit, y_fit, p0=[np.max(y_fit)])
			y_model = power_fixed_n(x_fit, *popt)
			fit_label = f"Fit: $a x^{fit_degree}$"
		elif fit_type == "power":
			popt, _ = curve_fit(power_law, x_fit, y_fit, p0=[np.max(y_fit), 1])
			y_model = power_law(x_fit, *popt)
			fit_label = f"Fit: $a x^n$\n($n$={popt[1]:.2f})"
		elif fit_type == "exp":
			popt, _ = curve_fit(exponential, x_fit, y_fit, p0=[np.max(y_fit), -1])
			y_model = exponential(x_fit, *popt)
			fit_label = f"Fit: $a e^{{b x}}$\n($b$={popt[1]:.2f})"
		else:
			print(f"Unknown fit type: {fit_type}")
			return
		# Plot the fit
		ax.plot(x_fit, y_model, '--', color = color, label=fit_label if label is None else label)
	except Exception as e:
		print(f"Fit failed: {e}")


def get_x_and_xvar(frb_dict, width_ms, plot_type="pa_var"):
	"""
	Extracts the x values and variable name for the x-axis based on the xname of frb_dict.
	"""
	if plot_type == "pa_var":
		if frb_dict["xname"] == "tau_ms":
			x = np.array(frb_dict["xvals"]) / width_ms
			xvar = r"\tau_\mathrm{ms} / \sigma_\mathrm{ms}"
		elif frb_dict["xname"] == "PA_var":
			x = np.array(frb_dict["xvals"])
			xvar = r"\Delta\psi_\mathrm{micro}"
		else:
			raise ValueError(f"Unknown xname: {frb_dict['xname']}")
	elif plot_type == "lfrac":
		if frb_dict["xname"] == "tau_ms":
			x = np.array(frb_dict["xvals"]) / width_ms
			xvar = r"\tau_\mathrm{ms} / \sigma_\mathrm{ms}"
		elif frb_dict["xname"] == "PA_var":
			x = np.array(frb_dict["xvals"])
			xvar = r"\Delta\psi_\mathrm{micro}"
		else:
			raise ValueError(f"Unknown xname: {frb_dict['xname']}")
	return x, xvar


def process_pa_var(dspec, freq_mhz, time_ms, gdict, phase_window, freq_window, tau_ms):
	
	slc = get_freq_window_indices(freq_mhz, freq_window)
	freq_mhz = freq_mhz[slc]
	dspec = dspec[:, slc, :]
		
	ts_data, _, _, _ = process_dynspec(dspec, freq_mhz, time_ms, gdict, tau_ms)
	
	peak_index = np.argmax(ts_data.iquvt[0])
	phase_slc = get_phase_window_indices(phase_window, peak_index)

	phits = ts_data.phits[phase_slc]
	dphits = ts_data.dphits[phase_slc]
		
   
	pa_var = np.nanvar(phits)
	pa_var_err = np.sqrt(np.nansum((phits * dphits)**2)) / (pa_var * len(phits))
	return pa_var, pa_var_err


def plot_pa_var(frb_dict, save, fname, out_dir, figsize, show_plots, scale, phase_window, freq_window, fit):
	"""
	Plot the var of the polarization angle (PA) and its error bars vs the scattering timescale.
	Supports plotting multiple run groups for comparison.
	"""
	# If frb_dict contains multiple runs, plot each on the same axes
	yvar = r"\mathcal{R}_\psi"
	if is_multi_run_dict(frb_dict):
		fig, ax = plt.subplots(figsize=figsize)
		#linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]
		colour_list = list(colours.values())
		for idx, (run, subdict) in enumerate(frb_dict.items()):
			colour = colour_list[idx % len(colour_list)]
			#linestyle = linestyles[idx % len(linestyles)]
			
			xvals = np.array(subdict["xvals"])
			yvals = subdict["yvals"]
			var_PA_microshots = subdict["var_PA_microshots"]
			dspec_params = subdict["dspec_params"]
			if isinstance(dspec_params, dict):
				width_ms = np.array(dspec_params["gdict"]["width_ms"])[0]
			elif isinstance(dspec_params, tuple):
				width_ms = np.array(dspec_params[0]["width_ms"])[0]
			else:
				raise TypeError("dspec_params is not a dict or tuple")
			
			yvals = weight_dict(xvals, yvals, var_PA_microshots)
			med_vals, percentile_errs = median_percentiles(yvals, xvals)

			x, xvar = get_x_and_xvar(subdict, width_ms)
	
			lower = np.array([lower for (lower, upper) in percentile_errs])
			upper = np.array([upper for (lower, upper) in percentile_errs])
			
			ax.plot(x, med_vals, label=run, color=colour, linewidth=2)#, linestyle=linestyle)
			ax.fill_between(x, lower, upper, color=colour, alpha=0.08)
			if fit is not None:
				fit_and_plot(ax, x, med_vals, fit, label=None, color=colour)
		ax.grid(True, linestyle='--', alpha=0.6)
		set_scale_and_labels(ax, scale, xvar=xvar, yvar=yvar, x=x)
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
	xvals = frb_dict["xvals"]
	yvals = frb_dict["yvals"]
	var_PA_microshots = frb_dict["var_PA_microshots"]
	width_ms = frb_dict["dspec_params"]["gdict"]["width_ms"]
 
	yvals = weight_dict(xvals, yvals, var_PA_microshots)
	med_vals, percentile_errs = median_percentiles(yvals, xvals)
 
	x, xvar = get_x_and_xvar(frb_dict, width_ms)
 
	lower = np.array([lower for (lower, upper) in percentile_errs])
	upper = np.array([upper for (lower, upper) in percentile_errs])
 
	fig, ax = plt.subplots(figsize=figsize)
	ax.plot(x, med_vals, color='black', label=r'\psi$_{var}$', linewidth=2)
	ax.fill_between(x, lower, upper, color='black', alpha=0.2)
	ax.grid(True, linestyle='--', alpha=0.6)
	if fit is not None:
		fit_and_plot(ax, x, med_vals, fit, label=None)
		ax.legend()
	set_scale_and_labels(ax, scale, xvar=xvar, yvar=yvar, x=x)
	if show_plots:
		plt.show()
	if save:
		name = make_plot_fname("pa_var", scale, fname, freq_window, phase_window)
		name = os.path.join(out_dir, name + ".pdf")
		fig.savefig(name, bbox_inches='tight', dpi=600)
		print(f"Saved figure to {name}  \n")


def process_lfrac(dspec, freq_mhz, time_ms, gdict, phase_window, freq_window, tau_ms):
	
	freq_slc = get_freq_window_indices(freq_mhz, freq_window)
	
	peak_index = np.argmax(np.nansum(dspec, axis=(0, 1)))
	phase_slc = get_phase_window_indices(phase_window, peak_index)

	freq_mhz = freq_mhz[freq_slc]
	time_ms = time_ms[phase_slc]
	dspec = dspec[:, freq_slc, phase_slc]
	
	ts_data, _, _, _ = process_dynspec(dspec, freq_mhz, time_ms, gdict, tau_ms)
 
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


def plot_lfrac_var(frb_dict, save, fname, out_dir, figsize, show_plots, scale, phase_window, freq_window, fit):
	yvar = r"L/I"
	# If frb_dict contains multiple job IDs, plot each on the same axes
	if is_multi_run_dict(frb_dict):
		fig, ax = plt.subplots(figsize=figsize)
		#linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]
		colour_list = list(colours.values())
		for idx, (run, subdict) in enumerate(frb_dict.items()):
			colour = colour_list[idx % len(colour_list)]
			#linestyle = linestyles[idx % len(linestyles)]
   
			tau_ms = np.array(subdict["xvals"])
			yvals = subdict["yvals"]
			errs = subdict["errs"]
			dspec_params = subdict["dspec_params"]
			if isinstance(dspec_params, dict):
				width_ms = np.array(dspec_params["gdict"]["width_ms"])[0]
			elif isinstance(dspec_params, tuple):
				width_ms = np.array(dspec_params[0]["width_ms"])[0]
			else:
				raise TypeError("dspec_params is not a dict or tuple")
			
			med_vals, percentile_errs = median_percentiles(yvals, tau_ms)
			x, xvar = get_x_and_xvar(subdict, width_ms, plot_type="lfrac")
			
			lower = np.array([lower for (lower, upper) in percentile_errs])
			upper = np.array([upper for (lower, upper) in percentile_errs])
			
			ax.plot(x, med_vals, label=run, color=colour)#, linestyle=linestyle)
			ax.fill_between(x, lower, upper, alpha=0.2, color=colour)
			if fit is not None:
				fit_and_plot(ax, x, med_vals, fit, label=None)
		ax.grid(True, linestyle='--', alpha=0.6)
		set_scale_and_labels(ax, scale, xvar=xvar, yvar=yvar, x=x)
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
	tau_ms = frb_dict["tau_ms"]
	yvals = frb_dict["yvals"]
	errs = frb_dict["errs"]
	width_ms = frb_dict["dspec_params"]["gdict"]["width_ms"]
 
	med_vals, percentile_errs = median_percentiles(yvals, tau_ms)
	x, xvar = get_x_and_xvar(frb_dict, width_ms, plot_type="lfrac")
 
	lower = np.array([lower for (lower, upper) in percentile_errs])
	upper = np.array([upper for (lower, upper) in percentile_errs])
 
	fig, ax = plt.subplots(figsize=figsize)
	ax.plot(x, med_vals, color='black', label=r'\psi$_{var}$')
	ax.fill_between(x, lower, upper, color='black', alpha=0.2)
	ax.grid(True, linestyle='--', alpha=0.6)
	if fit is not None:
		fit_and_plot(ax, x, med_vals, fit, label=None)
		ax.legend()
	set_scale_and_labels(ax, scale, xvar=xvar, yvar=yvar, x=x)
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
	requires_multiple_frb=True  
)

lfrac = PlotMode(
	name="lfrac",
	process_func=process_lfrac,
	plot_func=plot_lfrac_var,
	requires_multiple_frb=True  
)

iquv = PlotMode(
	name="iquv",
	process_func=None, 
	plot_func=basic_plots,
	requires_multiple_frb=False  
)

lvpa = PlotMode(
	name="lvpa",
	process_func=None, 
	plot_func=basic_plots,
	requires_multiple_frb=False
)

dpa = PlotMode(
	name="dpa",
	process_func=None, 
	plot_func=basic_plots,
	requires_multiple_frb=False
)

RM = PlotMode(
	name="RM",
	process_func=None,  
	plot_func=basic_plots,
	requires_multiple_frb=False
)

plot_modes = {
	"pa_var": pa_var,
	"iquv": iquv,
	"lvpa": lvpa,
	"dpa": dpa,
	"RM": RM,
	"lfrac": lfrac,
}