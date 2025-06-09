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
plt.rcParams['font.size'] 		= 14
plt.rcParams['font.family']		= 'sans-serif'  
plt.rcParams['axes.labelsize']  = 14
plt.rcParams['axes.titlesize']  = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
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
		"lowest-quarter": slice(0, q),
		"lower-mid-quarter": slice(q, 2*q),
		"upper-mid-quarter": slice(2*q, 3*q),
		"highest-quarter": slice(3*q, None),
		"full-band": slice(None)
	}
	return windows.get(freq_window, None)

def get_phase_window_indices(phase_window, peak_index):
	"""
	Returns a slice object for the desired phase window.
	"""
	phase_slices = {
		"leading": slice(0, peak_index),
		"trailing": slice(peak_index, None),
		"total": slice(None)
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


def fit_and_plot(ax, x, y, fit_type, fit_degree=None, label=None, color='black'):
	"""
	Fit the data (x, y) with the specified function and plot the fit on ax.
	fit: list or tuple, e.g. ['power'], ['exp'], or ['power', '3']
	"""

	x = np.asarray(x)
	y = np.asarray(y)
	
	
	def power_law(x, a, n): 
		return a * x**n
	def power_fixed_n(x, a): 
		return a * x**fit_degree
	def broken_power_law(x, a, n1, n2, x_break): 
		return np.where(x < x_break, a * x**n1, a * x_break**(n1-n2) * x**n2)
	def exponential(x, a, b): 
		return a * np.exp(b * x)

	def fit_power_fixed_n(x_fit, y_fit):
		popt, _ = curve_fit(power_fixed_n, x_fit, y_fit, p0=[np.max(y_fit)])
		y_model = power_fixed_n(x_fit, *popt)
		return y_model, f"Fit: $a x^{fit_degree}$"

	def fit_power_law(x_fit, y_fit):
		popt, _ = curve_fit(power_law, x_fit, y_fit, p0=[np.max(y_fit), 1])
		y_model = power_law(x_fit, *popt)
		return y_model, f"Fit: $a x^n$\n($n$={popt[1]:.2f})"

	def fit_exponential(x_fit, y_fit):
		popt, _ = curve_fit(exponential, x_fit, y_fit, p0=[np.max(y_fit), -1])
		y_model = exponential(x_fit, *popt)
		return y_model, f"Fit: $a e^{{b x}}$\n($b$={popt[1]:.2f})"

	def fit_linear(x_fit, y_fit):
		popt = np.polyfit(x_fit, y_fit, 1)
		y_model = np.polyval(popt, x_fit)
		return y_model, f"Fit: $y = mx + c$\n($m$={popt[0]:.2f}, $c$={popt[1]:.2f})"

	def fit_constant(x_fit, y_fit):
		popt = np.mean(y_fit)
		y_model = np.full_like(x_fit, popt)
		return y_model, f"Fit: $y = {popt:.2f}$"

	def fit_poly(x_fit, y_fit):
		popt = np.polyfit(x_fit, y_fit, fit_degree)
		y_model = np.polyval(popt, x_fit)
		terms = ' + '.join([f'{coef:.2f}x^{i}' for i, coef in enumerate(popt)])
		return y_model, f"Fit: $y = {terms}$"

	def fit_log(x_fit, y_fit):
		if np.any(x_fit <= 0):
			print("Log fit requires positive x values. Skipping fit.")
			return None, None
		popt = np.polyfit(np.log10(x_fit), y_fit, 1)
		y_model = np.polyval(popt, np.log10(x_fit))
		return y_model, f"Fit: $y = {popt[0]:.2f} \log_{{10}}(x) + {popt[1]:.2f}$"

	def fit_broken_power_law(x_fit, y_fit):
		# Initial guess: a=max(y), n1=1, n2=0, x_break=median(x)
		p0 = [np.max(y_fit), 1, 0, np.median(x_fit)]
		bounds = ([0, -np.inf, -np.inf, np.min(x_fit)], [np.inf, np.inf, np.inf, np.max(x_fit)])
		popt, _ = curve_fit(broken_power_law, x_fit, y_fit, p0=p0, bounds=bounds)
		y_model = broken_power_law(x_fit, *popt)
		return y_model, (r"Broken power: $a x^{n_1}$ ($x<x_b$), $a x_b^{n_1-n_2} x^{n_2}$ ($x\geq x_b$)"
					 	f"\n($n_1$={popt[1]:.2f}, $n_2$={popt[2]:.2f}, $x_b$={popt[3]:.2f})")

	fit_handlers = {
		"power_fixed_n": fit_power_fixed_n,
		"power": fit_power_law if fit_degree is None else fit_power_fixed_n,
		"exp": fit_exponential,
		"linear": fit_linear,
		"constant": fit_constant,
		"poly": fit_poly,
		"log": fit_log,
		"broken-power": fit_broken_power_law
	}

	# Remove NaNs and non-positive x for log fits
	mask = np.isfinite(x) & np.isfinite(y)
	if fit_type in ("power", "broken-power", "power_fixed_n") and (fit_degree is None or fit_type == "broken-power"):
		mask &= (x > 0)
	if fit_type == "log":
		mask &= (x > 0)
	x_fit = x[mask]
	y_fit = y[mask]

	try:
		handler = fit_handlers.get(fit_type)
		if handler is None:
			print(f"Unknown fit type: {fit_type}")
			return
		y_model, fit_label = handler(x_fit, y_fit)
		if y_model is not None:
			ax.plot(x_fit, y_model, '--', color=color, label=fit_label if label is None else label)
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


def parse_fit_arg(fit_item):
	"""
	Parse a fit argument string like 'poly,2' or 'poly' into (fit_type, fit_degree).
	"""
	if isinstance(fit_item, (list, tuple)):
		fit_type = fit_item[0]
		fit_degree = None
		if len(fit_item) > 1 and fit_item[1] is not None and str(fit_item[1]).strip() != "":
			try:
				fit_degree = int(fit_item[1])
			except Exception:
				fit_degree = None
		if fit_type == "poly" and fit_degree is None:
			print("Warning: 'poly' fit requires a degree (e.g., 'poly,2'). Skipping fit for this run.")
		return fit_type, fit_degree
	elif isinstance(fit_item, str):
		if ',' in fit_item:
			parts = fit_item.split(',', 1)
			fit_type = parts[0]
			fit_degree = None
			if len(parts) > 1 and parts[1] is not None and parts[1].strip() != "":
				try:
					fit_degree = int(parts[1])
				except Exception:
					fit_degree = None
			if fit_type == "poly" and fit_degree is None:
				print("Warning: 'poly' fit requires a degree (e.g., 'poly,2'). Skipping fit for this run.")
			return fit_type, fit_degree
		else:
			if fit_item == "poly":
				print("Warning: 'poly' fit requires a degree (e.g., 'poly,2'). Skipping fit for this run.")
			return fit_item, None
	else:
		return None, None


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
				# Accept fit as a list of strings like ['poly,1', 'poly,2', 'poly,3'] or just ['poly', 'poly,2', 'poly']
				if isinstance(fit, (list, tuple)) and len(fit) == len(frb_dict):
					fit_type, fit_degree = parse_fit_arg(fit[idx])
				else:
					fit_type, fit_degree = parse_fit_arg(fit)
				fit_and_plot(ax, x, med_vals, fit_type, fit_degree, label=None, color=colour)
			else:
				print("No fit provided, skipping fit plotting.")
		
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
	dspec_params = frb_dict["dspec_params"]
	width_ms = np.array(dspec_params[0]["width_ms"])[0]
 
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
				# Accept fit as a list of strings like ['poly,1', 'poly,2', 'poly,3'] or just ['poly', 'poly,2', 'poly']
				if isinstance(fit, (list, tuple)) and len(fit) == len(frb_dict):
					fit_type, fit_degree = parse_fit_arg(fit[idx])
				else:
					fit_type, fit_degree = parse_fit_arg(fit)
				fit_and_plot(ax, x, med_vals, fit_type, fit_degree, label=None, color=colour)
			else:
				print("No fit provided, skipping fit plotting.")
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
	dspec_params = frb_dict["dspec_params"]
	width_ms = np.array(dspec_params[0]["width_ms"])[0]
 
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