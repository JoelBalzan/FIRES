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

from FIRES.functions.basicfns import process_dynspec, boxcar_width
from FIRES.functions.plotfns import plot_stokes, plot_ilv_pa_ds, plot_dpa, estimate_rm



#	--------------------------	Set plot parameters	---------------------------
plt.rcParams['pdf.fonttype']	= 42
plt.rcParams['ps.fonttype'] 	= 42
plt.rcParams['savefig.dpi'] 	= 600
plt.rcParams['font.size'] 		= 14
plt.rcParams['font.family']		= 'sans-serif'  
plt.rcParams['axes.labelsize']  = 20
plt.rcParams['axes.titlesize']  = 20
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['text.usetex'] 	= True

#colour blind friendly: https://gist.github.com/thriveth/8560036

colours = {
	'red'   : '#e41a1c',
	'blue'  : '#377eb8',
	'purple': '#984ea3',
	'orange': '#ff7f00',
	'green' : '#4daf4a',
	'pink'  : '#f781bf',
	'brown' : '#a65628',
	'gray'  : '#999999',
	'yellow': '#dede00'
} 

colour_map = {
	'lowest-quarter'   : '#e41a1c',
	'highest-quarter'  : '#377eb8',
	'full-band'        : '#984ea3',
	'leading'          : '#ff7f00',
	'trailing'         : '#4daf4a',
	'total'            : '#984ea3',
	'lower-mid-quarter': '#a65628',
	'upper-mid-quarter': '#999999',
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
		
def basic_plots(fname, frb_data, mode, gdict, out_dir, save, figsize, tau_ms, show_plots, extension):

	dspec_params = frb_data.dspec_params
	freq_mhz = dspec_params.freq_mhz
	time_ms = dspec_params.time_ms
	

	ts_data, corr_dspec, noise_spec, noise_stokes = process_dynspec(
		frb_data.dynamic_spectrum, freq_mhz, time_ms, gdict
	)

	iquvt = ts_data.iquvt
	snr = frb_data.snr
	
	if mode == "all":
		plot_ilv_pa_ds(corr_dspec, freq_mhz, time_ms, save, fname, out_dir, ts_data, figsize, tau_ms, show_plots, snr, extension)
		plot_stokes(fname, out_dir, corr_dspec, iquvt, freq_mhz, time_ms, save, figsize, show_plots, extension)
		plot_dpa(fname, out_dir, noise_stokes, ts_data, time_ms, 5, save, figsize, show_plots, extension)
		estimate_rm(frb_data.dynamic_spectrum, freq_mhz, time_ms, noise_spec, 1.0e3, 1.0, out_dir, save, show_plots)
	elif mode == "iquv":
		plot_stokes(fname, out_dir, corr_dspec, iquvt, freq_mhz, time_ms, save, figsize, show_plots, extension)
	elif mode == "lvpa":
		plot_ilv_pa_ds(corr_dspec, freq_mhz, time_ms, save, fname, out_dir, ts_data, figsize, tau_ms, show_plots, snr, extension)
	elif mode == "dpa":
		plot_dpa(fname, out_dir, noise_stokes, ts_data, time_ms, 5, save, figsize, show_plots, extension)
	elif mode == "RM":
		estimate_rm(frb_data.dynamic_spectrum, freq_mhz, time_ms, noise_spec, 1.0e3, 1.0, out_dir, save, show_plots)
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



def median_percentiles(yvals, x, ndigits=3):
	"""
	Calculate median values and percentile-based error bars from grouped data.

	Parameters:
	-----------
	yvals : dict
		Dictionary where keys are parameter values and values are lists/arrays of measurements
	x : array_like
		Array of parameter values for which to compute statistics  
	ndigits : int, optional
		Number of decimal places for rounding keys during lookup (default: 3)
		
	Returns:
	--------
	tuple
		Two-element tuple containing:
		- med_vals: List of median values for each x value
		- percentile_errs: List of (lower_percentile, upper_percentile) tuples
	"""
 
	med_vals = []
	percentile_errs = []
	# Round all keys in yvals for consistent lookup
	vals_rounded = {round(float(k), ndigits): v for k, v in yvals.items()}
	for var in x:
		key = round(float(var), ndigits)
		v = vals_rounded.get(key, None)
		if v is not None and isinstance(v, (list, np.ndarray)) and len(v) > 0:
			median_val = np.median(v)
			lower_percentile = np.percentile(v, 16)
			upper_percentile = np.percentile(v, 84)
			med_vals.append(median_val)
			percentile_errs.append((lower_percentile, upper_percentile))
		else:
			med_vals.append(np.nan)
			percentile_errs.append((np.nan, np.nan))
	return med_vals, percentile_errs


def weight_dict(xvals, yvals, weight, var_name=None):
	"""
	Normalize values in yvals by weights from weights for a specific variable or a single weight value.

	Parameters:
	-----------
	xvals : array_like
		Array of parameter values for which to compute normalized values.
	yvals : dict
		Dictionary where keys are parameter values and values are lists/arrays of measurements.
	weights : dict or float
		If var_name is provided, a dictionary where keys are parameter values and values are dictionaries of weight factors.
		If var_name is None, a single weight value to normalize all values.
	var_name : str, optional
		Name of the variable in weights to use for normalization. If None, weights is treated as a single value.

	Returns:
	--------
	dict
		Dictionary with normalized values for each key in xvals.
	"""
	normalized_vals = {}

	if var_name is not None:
		# Case where var_name is provided
		# First check if var_name exists in any of the weight dictionaries
		var_name_exists = any(var_name in weight.get(var, {}) for var in xvals)
		if not var_name_exists:
			print(f"Warning: var_name '{var_name}' not found in weight dictionaries. Returning empty result.")
			return {}
			
		for var in xvals:
			y_values = yvals.get(var, [])
			var_weights = weight.get(var, {})
			
			weights = var_weights.get(var_name, [])

			if y_values and weights and len(y_values) == len(weights):
				normalized_vals[var] = [
					val / weights if weights != 0 else 0
					for val, weights in zip(y_values, weights)
				]
			else:
				normalized_vals[var] = None  # Handle missing or invalid data
	else:
		# Case where weights_dict is a single value
		for var in xvals:
			y_values = yvals.get(var, [])
			normalized_vals[var] = [
				val / weight if weight != 0 else 0
				for val, weight in zip(y_values, weight)
			]

	return normalized_vals
	

def set_scale_and_labels(ax, scale, xname, yname, x=None):
    # Set labels (same for all scales now)
    ax.set_xlabel(rf"${xname}$")
    ax.set_ylabel(rf"${yname}$")
    
    # Set scales
    if scale == "logx" or scale == "loglog":
        ax.set_xscale('log')
    if scale == "logy" or scale == "loglog":
        ax.set_yscale('log')
    
    # Set limits
    if x is not None:
        if scale in ["logx", "loglog"]:
            x_positive = x[x > 0]
            if len(x_positive) > 0:
                ax.set_xlim(x_positive[0], x_positive[-1])
        else:
            ax.set_xlim(x[0], x[-1])


def make_plot_fname(plot_type, scale, fname, freq_window="all", phase_window="all"):
	"""
	Generate a plot filename with freq/phase window at the front if not 'all'.
	"""
	parts = []
	if freq_window != "full-band":
		parts.append(f"freq_{freq_window}")
	if phase_window != "total":
		parts.append(f"phase_{phase_window}")
	parts.extend([fname, scale, plot_type])
	return "_".join(parts)


def is_multi_run_dict(frb_dict):
	"""
	Returns True if frb_dict contains multiple run dictionaries (i.e., is a dict of dicts with 'tau_ms' keys).
	"""
	return all(isinstance(v, dict) and "xvals" in v for v in frb_dict.values())


def fit_and_plot(ax, x, y, fit_type, fit_degree=None, label=None, color='black'):
	"""
	Fit the data (x, y) with the specified function and plot the fit on ax.
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
		return y_model, f"Power (n={fit_degree})"
	
	def fit_power_law(x_fit, y_fit):
		popt, _ = curve_fit(power_law, x_fit, y_fit, p0=[np.max(y_fit), 1])
		y_model = power_law(x_fit, *popt)
		return y_model, f"Power (n={popt[1]:.2f})"
	
	def fit_exponential(x_fit, y_fit):
		popt, _ = curve_fit(exponential, x_fit, y_fit, p0=[np.max(y_fit), -1])
		y_model = exponential(x_fit, *popt)
		return y_model, f"Exp (b={popt[1]:.2f})"
	
	def fit_linear(x_fit, y_fit):
		popt = np.polyfit(x_fit, y_fit, 1)
		y_model = np.polyval(popt, x_fit)
		return y_model, f"Linear (m={popt[0]:.2f})"
	
	def fit_constant(x_fit, y_fit):
		popt = np.mean(y_fit)
		y_model = np.full_like(x_fit, popt)
		return y_model, f"Const ({popt:.2f})"
	
	def fit_poly(x_fit, y_fit):
		popt = np.polyfit(x_fit, y_fit, fit_degree)
		y_model = np.polyval(popt, x_fit)
		return y_model, f"Poly (deg={fit_degree})"
	
	def fit_log(x_fit, y_fit):
		if np.any(x_fit <= 0):
			print("Log fit requires positive x values. Skipping fit.")
			return None, None
		popt = np.polyfit(np.log10(x_fit), y_fit, 1)
		y_model = np.polyval(popt, np.log10(x_fit))
		return y_model, f"Log (m={popt[0]:.2f})"
	
	def fit_broken_power_law(x_fit, y_fit):
		p0 = [np.max(y_fit), 1, 0, np.median(x_fit)]
		bounds = ([0, -np.inf, -np.inf, np.min(x_fit)], [np.inf, np.inf, np.inf, np.max(x_fit)])
		popt, _ = curve_fit(broken_power_law, x_fit, y_fit, p0=p0, bounds=bounds)
		y_model = broken_power_law(x_fit, *popt)
		return y_model, (f"Broken power\n"
				  		 r"$(n_1={:.2f},\ n_2={:.2f},\ x_b={:.2f})$".format(popt[1], popt[2], popt[3]))

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



def weight_x_get_xname(frb_dict, width_ms, plot_type="pa_var"):
	"""
	Extracts the x values and variable name for the x-axis based on the xname of frb_dict.
	"""
	if plot_type == "pa_var":
		if frb_dict["xname"] == "tau_ms":
			x = np.array(frb_dict["xvals"]) / width_ms
			xname = r"\tau / W"
		elif frb_dict["xname"] == "PA_var":
			x = np.array(frb_dict["xvals"])
			xname = r"\Delta\psi_\mathrm{micro}"
		else:
			raise ValueError(f"Unknown xname: {frb_dict['xname']}")
	elif plot_type == "lfrac":
		if frb_dict["xname"] == "tau_ms":
			x = np.array(frb_dict["xvals"]) / width_ms
			xname = r"\tau_\mathrm{ms} / W"
		elif frb_dict["xname"] == "PA_var":
			x = np.array(frb_dict["xvals"])
			xname = r"\Delta\psi_\mathrm{micro}"
		else:
			raise ValueError(f"Unknown xname: {frb_dict['xname']}")
	return x, xname


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


def print_avg_snrs(subdict):
	snrs = subdict.get("snrs", [])
	# Handle dict or list
	if isinstance(snrs, dict):
		snr_values = list(snrs.values())
	else:
		snr_values = snrs
	# Skip noiseless cases
	if not snr_values or all(s is None or (isinstance(s, list) and all(v is None for v in s)) for s in snr_values):
		return
	# Get S/N at lowest and highest xvals
	if isinstance(snrs, dict):
		keys_sorted = sorted(snrs.keys())
		lowest = snrs[keys_sorted[0]]
		highest = snrs[keys_sorted[-1]]
	else:
		lowest = snrs[0]
		highest = snrs[-1]
	def avg(val):
		if isinstance(val, list):
			vals = [v for v in val if v is not None]
			if not vals:
				return None
			return np.nanmean(vals)
		return val if val is not None else None
	avg_low = np.round(avg(lowest), 2)
	avg_high = np.round(avg(highest), 2)
	# Only print if at least one is not None
	if avg_low is not None or avg_high is not None:
		print(f"Avg S/N at:\n lowest x: S/N = {avg_low if avg_low is not None else 'nan'}, \nhighest x: S/N = {avg_high if avg_high is not None else 'nan'}\n")
		


def plot_multirun(frb_dict, ax, plot_type, fit, scale, yname, colour_map, colours, weight_key):
	"""
	Common plotting logic for plot_pa_var and plot_lfrac_var.

	Parameters:
	-----------
	frb_dict : dict
		Dictionary containing FRB simulation data.
	ax : matplotlib.axes.Axes
		Axis object for plotting.
	plot_type : str
		Type of plot ("pa_var" or "lfrac").
	fit : str or list
		Fit type or list of fit types.
	scale : str
		Scale type ("linear", "logx", "logy", "loglog").
	yname : str
		Y-axis variable name.
	colour_map : dict
		Mapping of run names to colors.
	colours : dict
		Default color palette.
	weight_key : str
		Key to use for weighting ("PA" or "lfrac").
	"""
	colour_list = list(colours.values())
	for idx, (run, subdict) in enumerate(frb_dict.items()):
		print(run+":")
		colour = colour_map[run] if run in colour_map else colour_list[idx % len(colour_list)]

		xvals = np.array(subdict["xvals"])
		yvals = subdict["yvals"]
		var_params = subdict["var_params"]
		dspec_params = subdict["dspec_params"]
		width_ms = np.array(dspec_params[0]["width_ms"])[0]
  
		y = weight_dict(xvals, yvals, var_params, weight_key)
		med_vals, percentile_errs = median_percentiles(y, xvals)

		x, xname = weight_x_get_xname(subdict, width_ms, plot_type=plot_type)
		lower = np.array([lower for (lower, upper) in percentile_errs])
		upper = np.array([upper for (lower, upper) in percentile_errs])

		ax.plot(x, med_vals, label=run, color=colour, linewidth=2)
		ax.fill_between(x, lower, upper, color=colour, alpha=0.08)
		if fit is not None:
			if isinstance(fit, (list, tuple)) and len(fit) == len(frb_dict):
				fit_type, fit_degree = parse_fit_arg(fit[idx])
			else:
				fit_type, fit_degree = parse_fit_arg(fit)
			fit_and_plot(ax, x, med_vals, fit_type, fit_degree, label=None, color=colour)
		else:
			print("No fit provided, skipping fit plotting.")
		print_avg_snrs(subdict)

	ax.grid(True, linestyle='--', alpha=0.6)
	ax.legend()
	set_scale_and_labels(ax, scale, xname=xname, yname=yname, x=x)



def process_pa_var(dspec, freq_mhz, time_ms, gdict, phase_window, freq_window, tau_ms):
	
	slc = get_freq_window_indices(freq_mhz, freq_window)
	freq_mhz = freq_mhz[slc]
	dspec = dspec[:, slc, :]
		
	ts_data, _, _, _ = process_dynspec(dspec, freq_mhz, time_ms, gdict)
	
	peak_index = np.argmax(ts_data.iquvt[0])
	phase_slc = get_phase_window_indices(phase_window, peak_index)

	phits = ts_data.phits[phase_slc]
	dphits = ts_data.dphits[phase_slc]
		
   
	pa_var = np.nanvar(phits)
	pa_var_err = np.sqrt(np.nansum((phits * dphits)**2)) / (pa_var * len(phits))
	return pa_var, pa_var_err


def plot_pa_var(frb_dict, save, fname, out_dir, figsize, show_plots, scale, phase_window, freq_window, fit, extension):
	"""
	Plot the var of the polarization angle (PA) and its error bars vs the scattering timescale.
	Supports plotting multiple run groups for comparison.
	"""
	# If frb_dict contains multiple runs, plot each on the same axes
	yname = r"\mathcal{R}_{\mathrm{\psi}}"
 
	if figsize is None:
		figsize = (10, 9)
	if is_multi_run_dict(frb_dict):
		fig, ax = plt.subplots(figsize=figsize)
		plot_multirun(frb_dict, ax, plot_type="pa_var", fit=fit, scale=scale, yname=yname, colour_map=colour_map, colours=colours, weight_key="PA")
		if show_plots:
			plt.show()
		if save:
			name = make_plot_fname("pa_var", scale, fname, freq_window, phase_window)
			name = os.path.join(out_dir, name + "." + extension)
			fig.savefig(name, bbox_inches='tight', dpi=600)
			print(f"\nSaved figure to {name}  \n")
		return
	
	# Otherwise, plot as usual (single job)
	xvals = frb_dict["xvals"]
	yvals = frb_dict["yvals"]
	var_params = frb_dict["var_params"]
	dspec_params = frb_dict["dspec_params"]
	width_ms = np.array(dspec_params[0]["width_ms"])[0]
 
	y = weight_dict(xvals, yvals, var_params, "PA")
	med_vals, percentile_errs = median_percentiles(y, xvals)
 
	x, xname = weight_x_get_xname(frb_dict, width_ms)
 
	lower = np.array([lower for (lower, upper) in percentile_errs])
	upper = np.array([upper for (lower, upper) in percentile_errs])
 
	fig, ax = plt.subplots(figsize=figsize)
	ax.plot(x, med_vals, color='black', label=r'\psi$_{var}$', linewidth=2)
	ax.fill_between(x, lower, upper, color='black', alpha=0.2)
	ax.grid(True, linestyle='--', alpha=0.6)
	if fit is not None:
		fit_and_plot(ax, x, med_vals, fit, label=None)
		ax.legend()
	set_scale_and_labels(ax, scale, xname=xname, yname=yname, x=x)
	if show_plots:
		plt.show()
	if save:
		name = make_plot_fname("pa_var", scale, fname, freq_window, phase_window)
		name = os.path.join(out_dir, name + ".pdf")
		fig.savefig(name, bbox_inches='tight', dpi=600)
		print(f"Saved figure to {name}  \n")


def process_lfrac(dspec, freq_mhz, time_ms, gdict, phase_window, freq_window, tau_ms):
	
	if freq_window != "full-band":
		freq_slc = get_freq_window_indices(freq_mhz, freq_window)
		freq_mhz = freq_mhz[freq_slc]
		dspec = dspec[:, freq_slc, :]
	if phase_window != "total":
		peak_index = np.argmax(np.nansum(dspec, axis=(0, 1)))
		phase_slc = get_phase_window_indices(phase_window, peak_index)
		time_ms = time_ms[phase_slc]
		dspec = dspec[:, :, phase_slc]

 
	ts_data, _, _, _ = process_dynspec(dspec, freq_mhz, time_ms, gdict)
 
	iquvt = ts_data.iquvt
	I = iquvt[0]
	Q = iquvt[1]
	U = iquvt[2]
	V = iquvt[3]
 
	_, left, right = boxcar_width(I, time_ms, frac=0.95)
	onpulse_mask = np.zeros(I.shape, dtype=bool)
	onpulse_mask[left:right+1] = True  # Include on-pulse region

	I_masked = np.where(onpulse_mask, I, np.nan)
	Q_masked = np.where(onpulse_mask, Q, np.nan)
	U_masked = np.where(onpulse_mask, U, np.nan)
	V_masked = np.where(onpulse_mask, V, np.nan)

	L = np.sqrt(Q_masked**2 + U_masked**2)
	
	integrated_I = np.nansum(I_masked)
	integrated_L = np.nansum(L)

	
	lfrac = integrated_L / integrated_I
 
	noise_I = np.nanstd(I[~onpulse_mask])
	noise_L = np.nanstd(L[~onpulse_mask])
	lfrac_err = np.sqrt((noise_L / integrated_I)**2 + (integrated_L * noise_I / integrated_I**2)**2)

	return lfrac, lfrac_err


def plot_lfrac_var(frb_dict, save, fname, out_dir, figsize, show_plots, scale, phase_window, freq_window, fit, extension):
	yname = r"L/I"
	# If frb_dict contains multiple job IDs, plot each on the same axes
	if figsize is None:
		figsize = (10, 9)
	if is_multi_run_dict(frb_dict):
		fig, ax = plt.subplots(figsize=figsize)
		#linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]
		colour_list = list(colours.values())
		for idx, (run, subdict) in enumerate(frb_dict.items()):
			colour = colour_map[run] if run in colour_map else colour_list[idx % len(colour_list)]
			#linestyle = linestyles[idx % len(linestyles)]
   
			xvals = np.array(subdict["xvals"])
			yvals = subdict["yvals"]
			var_params = subdict["var_params"]
			dspec_params = subdict["dspec_params"]
			width_ms = np.array(dspec_params["gdict"]["width_ms"])[0]

			
			y = weight_dict(xvals, yvals, var_params, "lfrac")
			med_vals, percentile_errs = median_percentiles(y, xvals)
   
			x, xname = weight_x_get_xname(subdict, width_ms, plot_type="lfrac")
			lower = np.array([lower for (lower, upper) in percentile_errs])
			upper = np.array([upper for (lower, upper) in percentile_errs])
			
			ax.plot(x, med_vals, label=run, color=colour)#, linestyle=linestyle)
			ax.fill_between(x, lower, upper, alpha=0.2, color=colour)
			if fit is not None:
				# Accept fit as a list of strings like ['poly,1', 'poly,2', 'poly,3'] or something like ['poly', 'poly,2', 'poly']
				if isinstance(fit, (list, tuple)) and len(fit) == len(frb_dict):
					fit_type, fit_degree = parse_fit_arg(fit[idx])
				else:
					fit_type, fit_degree = parse_fit_arg(fit)
				fit_and_plot(ax, x, med_vals, fit_type, fit_degree, label=None, color=colour)
			else:
				print("No fit provided, skipping fit plotting.")
		ax.grid(True, linestyle='--', alpha=0.6)
		set_scale_and_labels(ax, scale, xname=xname, yname=yname, x=x)
		ax.legend()
		if show_plots:
			plt.show()
		if save:
			name = make_plot_fname("lfrac", scale, fname, freq_window, phase_window)
			name = os.path.join(out_dir, name + "." + extension)
			fig.savefig(name, bbox_inches='tight', dpi=600)
			print(f"Saved figure to {name}  \n")

		return

	# Otherwise, plot as usual (single job)
	xvals = frb_dict["xvals"]
	yvals = frb_dict["yvals"]
	var_params = frb_dict["var_params"]
	errs = frb_dict["errs"]
 
	dspec_params = frb_dict["dspec_params"]
	width_ms = np.array(dspec_params[0]["width_ms"])[0]
 
	y = weight_dict(xvals, yvals, dspec_params[0]["lfrac"])
	med_vals, percentile_errs = median_percentiles(y, xvals)
 
	x, xname = weight_x_get_xname(frb_dict, width_ms, plot_type="lfrac")
	lower = np.array([lower for (lower, upper) in percentile_errs])
	upper = np.array([upper for (lower, upper) in percentile_errs])
 
	fig, ax = plt.subplots(figsize=figsize)
	ax.plot(x, med_vals, color='black', label=r'\psi$_{var}$')
	ax.fill_between(x, lower, upper, color='black', alpha=0.2)
	ax.grid(True, linestyle='--', alpha=0.6)
	if fit is not None:
		fit_and_plot(ax, x, med_vals, fit, label=None)
		ax.legend()
	set_scale_and_labels(ax, scale, xname=xname, yname=yname, x=x)
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