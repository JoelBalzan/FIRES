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
import logging
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import circvar

from fires.core.basicfns import (estimate_rm, on_off_pulse_masks_from_profile,
                                 process_dspec)
from fires.plotting.plotfns import plot_dpa, plot_ilv_pa_ds, plot_stokes

logging.basicConfig(level=logging.INFO)
# Suppress noisy fontTools subset INFO messages
for _name in ("fontTools", "fontTools.subset"):
	_lg = logging.getLogger(_name)
	_lg.setLevel(logging.WARNING)   
	_lg.propagate = False

#	--------------------------	Set plot parameters	---------------------------


def configure_matplotlib(use_latex=False):
	"""
	Configure global Matplotlib style once (call after parsing CLI flags).
	"""
	rc = {
		'pdf.fonttype'    : 42,
		'ps.fonttype'     : 42,
		'savefig.dpi'     : 600,
		'font.size'       : 22,
		'axes.labelsize'  : 22,
		'axes.titlesize'  : 22,
		'legend.fontsize' : 20,
		'xtick.labelsize' : 22,
		'ytick.labelsize' : 22,
		'text.usetex'     : bool(use_latex),
		'font.family'     : 'sans-serif'
	}
	for k, v in rc.items():
		plt.rcParams[k] = v

	if use_latex:
		try:
			plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{siunitx}'
		except Exception as e:
			warnings.warn(f"LaTeX setup failed ({e}); falling back to non-LaTeX text rendering.")
			plt.rcParams['text.usetex'] = False

configure_matplotlib(use_latex=bool(int(os.environ.get("FIRES_USE_LATEX", "0"))))

#	--------------------------	Colour maps	---------------------------
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
	'lowest-quarter, total'   : '#e41a1c',
	'highest-quarter, total'  : '#377eb8',
	'full-band, total'        : '#984ea3',
	'full-band, leading'      : '#ff7f00',
	'full-band, trailing'     : '#4daf4a',
	'lower-mid-quarter, total': '#a65628',
	'upper-mid-quarter, total': '#999999',
}

#	--------------------------	Parameter mappings	---------------------------
param_map = {
	# Intrinsic parameters
	"tau_ms"         : r"\tau_0",
	"width_ms"       : r"W_0",
	"A"       : r"A_0",
	"spec_idx"       : r"\alpha_0",
	"DM"             : r"\mathrm{DM}_0",
	"RM"             : r"\mathrm{RM}_0",
	"PA"             : r"\psi_0",
	"lfrac"          : r"\Pi_{L,0}",
	"vfrac"          : r"\Pi_{V,0}",
	"dPA"            : r"\Delta\psi_0",
	"band_centre_mhz": r"\nu_{\mathrm{c},0}",
	"band_width_mhz" : r"\Delta \nu_0",
	"ngauss"         : r"N_{\mathrm{gauss},0}",
	"mg_width_low"   : r"W_{\mathrm{low},0}",
	"mg_width_high"  : r"W_{\mathrm{high},0}",
	# Std deviation sweep parameters
	"t0_i"             : r"\sigma_{t_0}",
	"width_ms_i"       : r"\sigma_W",
	"A_i"       : r"\sigma_A",
	"spec_idx_i"       : r"\sigma_\alpha",
	"DM_i"             : r"\sigma_{\mathrm{DM}}",
	"RM_i"             : r"\sigma_{\mathrm{RM}}",
	"PA_i"             : r"\sigma_{\psi}",
	"lfrac_i"          : r"\sigma_{\Pi_L}",
	"vfrac_i"          : r"\sigma_{\Pi_V}",
	"dPA_i"            : r"\sigma_{\Delta\psi}",
	"band_centre_mhz_i": r"\sigma_{\nu_c}",
	"band_width_mhz_i" : r"\sigma_{\Delta \nu}",
}

#	--------------------------	PlotMode class	---------------------------
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
		

# --------------------------	Plot modes definitions	---------------------------
def basic_plots(fname, frb_data, mode, gdict, out_dir, save, figsize, show_plots, extension, 
				legend, info, buffer_frac, show_onpulse, show_offpulse):
	"""
	Call basic plot functions
	"""
	dspec_params = frb_data.dspec_params
	freq_mhz = dspec_params.freq_mhz
	time_ms = dspec_params.time_ms

	tau_ms = dspec_params.gdict['tau_ms']

	ts_data, corr_dspec, noise_spec, noise_stokes = process_dspec(
		frb_data.dynamic_spectrum, freq_mhz, gdict, buffer_frac
	)

	iquvt = ts_data.iquvt
	snr = frb_data.snr
	
	if mode == "all":
		plot_ilv_pa_ds(corr_dspec, freq_mhz, time_ms, save, fname, out_dir, ts_data, figsize, tau_ms, show_plots, snr, extension, 
		legend, info, buffer_frac, show_onpulse, show_offpulse)
		plot_stokes(fname, out_dir, corr_dspec, iquvt, freq_mhz, time_ms, save, figsize, show_plots, extension)
		plot_dpa(fname, out_dir, noise_stokes, ts_data, time_ms, 5, save, figsize, show_plots, extension)
		estimate_rm(frb_data.dynamic_spectrum, freq_mhz, time_ms, noise_spec, 1.0e3, 1.0, out_dir, save, show_plots)
	elif mode == "iquv":
		plot_stokes(fname, out_dir, corr_dspec, iquvt, freq_mhz, time_ms, save, figsize, show_plots, extension)
	elif mode == "lvpa":
		plot_ilv_pa_ds(corr_dspec, freq_mhz, time_ms, save, fname, out_dir, ts_data, figsize, tau_ms, show_plots, snr, extension, 
		legend, info, buffer_frac, show_onpulse, show_offpulse)
	elif mode == "dpa":
		plot_dpa(fname, out_dir, noise_stokes, ts_data, time_ms, 5, save, figsize, show_plots, extension)
	elif mode == "RM":
		estimate_rm(frb_data.dynamic_spectrum, freq_mhz, time_ms, noise_spec, 1.0e3, 1.0, out_dir, save, show_plots)
	else:
		logging.warning(f"Invalid mode: {mode} \n")


def _get_freq_window_indices(freq_window, freq_mhz):
	q = int(len(freq_mhz) / 4)
	windows = {
		"lowest-quarter": slice(0, q),
		"lower-mid-quarter": slice(q, 2*q),
		"upper-mid-quarter": slice(2*q, 3*q),
		"highest-quarter": slice(3*q, None),
		"full-band": slice(None)
	}
	sl = windows.get(freq_window)
	if sl is None:
		raise ValueError(f"Unknown freq_window '{freq_window}'. Valid: {list(windows.keys())}")
	return sl

def _get_phase_window_indices(phase_window, peak_index):
	"""
	Returns a slice object for the desired phase window.
	"""
	phase_slices = {
		"leading": slice(0, peak_index),
		"trailing": slice(peak_index, None),
		"total": slice(None)
	}
	sl = phase_slices.get(phase_window)
	if sl is None:
		raise ValueError(f"Unknown phase_window '{phase_window}'. Valid: {list(phase_slices.keys())}")
	return sl



def _median_percentiles(yvals, x, ndigits=3, atol=1e-12, rtol=1e-9):
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
	atol : float, optional
		Absolute tolerance for matching float keys (default: 1e-12)
	rtol : float, optional
		Relative tolerance for matching float keys (default: 1e-9)
		
	Returns:
	--------
	tuple
		Two-element tuple containing:
		- med_vals: List of median values for each x value
		- percentile_errs: List of (lower_percentile, upper_percentile) tuples
	"""
 
	med_vals = []
	percentile_errs = []

	# Prepare keys for fuzzy matching
	if len(yvals) == 0:
		return [np.nan] * len(x), [(np.nan, np.nan)] * len(x)

	key_list = list(yvals.keys())
	keys = np.array(key_list, dtype=float)

	for var in np.asarray(x, dtype=float):
		v = None
		# Fast path: exact key match
		if var in yvals:
			v = yvals[var]
		else:
			# Find keys close to var within tolerance
			close = np.isclose(keys, var, rtol=rtol, atol=atol)
			if np.any(close):
				idxs = np.where(close)[0]
				# If multiple, pick nearest
				if idxs.size > 1:
					idx = idxs[np.argmin(np.abs(keys[idxs] - var))]
				else:
					idx = idxs[0]
				v = yvals[key_list[idx]]
			elif ndigits is not None:
				# Optional coarse rounding fallback
				diffs = np.abs(np.round(keys, ndigits) - np.round(var, ndigits))
				idx = int(np.argmin(diffs))
				v = yvals[key_list[idx]]

		if isinstance(v, (list, np.ndarray)) and len(v) > 0:
			v_arr = np.asarray(v, dtype=float)
			med_vals.append(np.nanmedian(v_arr))
			lower = np.nanpercentile(v_arr, 16)
			upper = np.nanpercentile(v_arr, 84)
			percentile_errs.append((lower, upper))
		else:
			med_vals.append(np.nan)
			percentile_errs.append((np.nan, np.nan))

	return med_vals, percentile_errs


def _weight_dict(xvals, yvals, weight_params, weight_by=None, return_status=False):
	"""
	Normalise values in yvals by weights from weight_params for a specific variable or by any parameter.

	Parameters:
	-----------
	xvals : array_like
		Array of parameter values for which to compute normalised values.
	yvals : dict
		Dictionary where keys are parameter values and values are lists/arrays of measurements.
	weight_params : dict or list
		Parameter dictionaries containing weight factors. Can be V_params, dspec_params, or combined.
	weight_by : str, optional
		Parameter name to use for weighting/normalization. Can be any parameter in weight_params.
		Takes precedence over var_name if both are provided.
	return_status : bool, optional
		If True, returns a tuple (normalised_vals, applied) where 'applied' is True when
		weighting was actually performed; otherwise returns only the dict.

	Returns:
	--------
	dict or (dict, bool)
		Dictionary with normalised/weighted values for each key in xvals, and optionally
		a boolean indicating if weighting was applied.
	"""
	normalised_vals = {}
	applied = False
	
	if weight_by is None:
		# No weighting requested - return original values
		for var in xvals:
			normalised_vals[var] = yvals.get(var, [])
		return (normalised_vals, applied) if return_status else normalised_vals
	
	# Handle both dict of dicts (multi-parameter case) and list of dicts (single parameter case)
	if isinstance(weight_params, dict) and any(isinstance(v, dict) for v in weight_params.values()):
		# Multi-parameter case: weight_params is like {param_value: {param_name: [values], ...}, ...}
		weighting_param_exists = any(weight_by in weight_params.get(var, {}) for var in xvals)
		if not weighting_param_exists:
			logging.warning(f"Weighting parameter '{weight_by}' not found in weight dictionaries. Returning unweighted values.")
			for var in xvals:
				normalised_vals[var] = yvals.get(var, [])
			return (normalised_vals, applied) if return_status else normalised_vals
			
		for var in xvals:
			y_values = yvals.get(var, [])
			var__weight_dict = weight_params.get(var, {})
			weights = var__weight_dict.get(weight_by, [])

			if y_values and weights and len(y_values) == len(weights):
				out = []
				for val, weight in zip(y_values, weights):
					if weight != 0 and np.isfinite(weight):
						out.append(val / weight)
						applied = True
					else:
						out.append(0)
				normalised_vals[var] = out
			else:
				logging.warning(f"Mismatched lengths or missing data for parameter {var}. Skipping normalization.")
				normalised_vals[var] = y_values
				
	elif isinstance(weight_params, (list, tuple)) and len(weight_params) > 0:
		# Single parameter case: weight_params is like [{param_name: value, ...}, ...]
		# Extract the weighting parameter value
		if isinstance(weight_params[0], dict) and (weight_by in weight_params[0]):
			weight_value = weight_params[0][weight_by]
			if isinstance(weight_value, (list, np.ndarray)) and len(weight_value) > 0:
				weight_value = weight_value[0]  # Take first value if it's an array
			
			for var in xvals:
				y_values = yvals.get(var, [])
				if y_values and weight_value is not None and weight_value != 0 and np.isfinite(weight_value):
					normalised_vals[var] = [val / weight_value for val in y_values]
					applied = True
				else:
					normalised_vals[var] = list(y_values) if y_values else []
		else:
			logging.warning(f"Weighting parameter '{weight_by}' not found in weight_params. Returning unweighted values.")
			for var in xvals:
				normalised_vals[var] = yvals.get(var, [])
	else:
		logging.warning(f"Unsupported weight_params format. Returning unweighted values.")
		for var in xvals:
			normalised_vals[var] = yvals.get(var, [])

	return (normalised_vals, applied) if return_status else normalised_vals


def _apply_log_decade_ticks(ax, axis='y', base=10, show_minor=True):
	"""
	Show only decade ticks (10^k) as labels on a log axis and keep unlabeled minor ticks.
	- Major: 10^k labeled as 10^{k}
	- Minor: 2..9 within each decade, no labels
	"""
	major_loc = mticker.LogLocator(base=base, subs=(1.0,))
	major_fmt = mticker.LogFormatterMathtext(base=base)

	# Minor ticks at 2..9 within each decade
	minor_loc = mticker.LogLocator(base=base, subs=np.arange(2, base) / base)
	minor_fmt = mticker.NullFormatter()

	if axis == 'y':
		ax.yaxis.set_major_locator(major_loc)
		ax.yaxis.set_major_formatter(major_fmt)
		if show_minor:
			ax.yaxis.set_minor_locator(minor_loc)
			ax.yaxis.set_minor_formatter(minor_fmt)
		else:
			ax.yaxis.set_minor_locator(mticker.NullLocator())
	else:
		ax.xaxis.set_major_locator(major_loc)
		ax.xaxis.set_major_formatter(major_fmt)
		if show_minor:
			ax.xaxis.set_minor_locator(minor_loc)
			ax.xaxis.set_minor_formatter(minor_fmt)
		else:
			ax.xaxis.set_minor_locator(mticker.NullLocator())
	

def _set_scale_and_labels(ax, scale, xname, yname, x=None):
	# Set labels (same for all scales now)
	ax.set_xlabel(rf"${xname}$")
	ax.set_ylabel(rf"${yname}$")
	
	# Set scales
	if scale == "logx" or scale == "loglog":
		ax.set_xscale('log')
	if scale == "logy" or scale == "loglog":
		ax.set_yscale('log')
		_apply_log_decade_ticks(ax, axis='y', base=10)  # enforce 10^k labels
	# Set limits
	if x is not None:
		if scale in ["logx", "loglog"]:
			x_positive = x[x > 0]
			if len(x_positive) > 0:
				ax.set_xlim(x_positive[0], x_positive[-1])
		else:
			# Check if x range is valid before setting limits
			if len(x) > 1 and x[0] != x[-1]:
				ax.set_xlim(x[0], x[-1])
			elif len(x) == 1:
				# For single point, set reasonable range around it
				center = x[0]
				if center == 0:
					ax.set_xlim(-1, 1)
				else:
					margin = abs(center) * 0.1
					ax.set_xlim(center - margin, center + margin)


def _make_plot_fname(plot_type, scale, fname, freq_window="all", phase_window="all"):
	"""
	Generate a plot filename with freq/phase window at the front if not 'all'.
	"""
	parts = [fname, scale]
	parts.append(f"freq_{freq_window}")
	parts.append(f"phase_{phase_window}")
	parts.append(plot_type)
	return "_".join(parts)


def _is_multi_run_dict(frb_dict):
	"""
	Returns True if frb_dict contains multiple run dictionaries (i.e., is a dict of dicts with 'tau_ms' keys).
	"""
	return all(isinstance(v, dict) and "xvals" in v for v in frb_dict.values())


def _fit_and_plot(ax, x, y, fit_type, fit_degree=None, label=None, color='black'):
	"""
	Fit the data (x, y) with the specified function and plot the fit on ax.
	"""

	x = np.asarray(x)
	y = np.asarray(y)
	
	
	def _power_law(x, a, n): 
		return a * x**n
	def _power_fixed_n(x, a): 
		return a * x**fit_degree
	def _broken_power_law(x, a, n1, n2, x_break): 
		return np.where(x < x_break, a * x**n1, a * x_break**(n1-n2) * x**n2)
	def _exponential(x, a, b): 
		return a * np.exp(b * x)

	def _fit_power_fixed_n(x_fit, y_fit):
		popt, _ = curve_fit(_power_fixed_n, x_fit, y_fit, p0=[np.max(y_fit)])
		y_model = _power_fixed_n(x_fit, *popt)
		return y_model, f"Power (n={fit_degree})"
	
	def _fit_power_law(x_fit, y_fit):
		popt, _ = curve_fit(_power_law, x_fit, y_fit, p0=[np.max(y_fit), 1])
		y_model = _power_law(x_fit, *popt)
		return y_model, f"Power (n={popt[1]:.2f})"
	
	def _fit_exponential(x_fit, y_fit):
		popt, _ = curve_fit(_exponential, x_fit, y_fit, p0=[np.max(y_fit), -1])
		y_model = _exponential(x_fit, *popt)
		return y_model, f"Exp (b={popt[1]:.2f})"
	
	def _fit_linear(x_fit, y_fit):
		popt = np.polyfit(x_fit, y_fit, 1)
		y_model = np.polyval(popt, x_fit)
		return y_model, f"Linear (m={popt[0]:.2f})"
	
	def _fit_constant(x_fit, y_fit):
		popt = np.mean(y_fit)
		y_model = np.full_like(x_fit, popt)
		return y_model, f"Const ({popt:.2f})"
	
	def _fit_poly(x_fit, y_fit):
		popt = np.polyfit(x_fit, y_fit, fit_degree)
		y_model = np.polyval(popt, x_fit)
		return y_model, f"Poly (deg={fit_degree})"
	
	def _fit_log(x_fit, y_fit):
		if np.any(x_fit <= 0):
			logging.warning("Log fit requires positive x values. Skipping fit.")
			return None, None
		popt = np.polyfit(np.log10(x_fit), y_fit, 1)
		y_model = np.polyval(popt, np.log10(x_fit))
		return y_model, f"Log (m={popt[0]:.2f})"
	
	def _fit_broken_power_law(x_fit, y_fit):
		p0 = [np.max(y_fit), 1, 0, np.median(x_fit)]
		bounds = ([0, -np.inf, -np.inf, np.min(x_fit)], [np.inf, np.inf, np.inf, np.max(x_fit)])
		popt, _ = curve_fit(_broken_power_law, x_fit, y_fit, p0=p0, bounds=bounds)
		y_model = _broken_power_law(x_fit, *popt)
		return y_model, (f"Broken power\n"
				  		 r"$(n_1={:.2f},\ n_2={:.2f},\ x_b={:.2f})$".format(popt[1], popt[2], popt[3]))

	fit_handlers = {
		"power_fixed_n": _fit_power_fixed_n,
		"power": _fit_power_law if fit_degree is None else _fit_power_fixed_n,
		"exp": _fit_exponential,
		"linear": _fit_linear,
		"constant": _fit_constant,
		"poly": _fit_poly,
		"log": _fit_log,
		"broken-power": _fit_broken_power_law
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
			logging.warning(f"Unknown fit type: {fit_type}")
			return
		y_model, fit_label = handler(x_fit, y_fit)
		if y_model is not None:
			ax.plot(x_fit, y_model, '--', color=color, label=fit_label if label is None else label)
	except Exception as e:
		logging.error(f"Fit failed: {e}")



def _weight_x_get_xname(frb_dict, weight_x_by=None):
	"""
	Extracts the x values and variable name for the x-axis based on the xname of frb_dict.
	Now supports any intrinsic parameter or variation parameter with flexible weighting.

	Behavior:
	- sweep_mode == "sd": x label = SD(xname), no weighting applied
	- sweep_mode == "mean": x label = xname/weight_x_by (if provided), else raw xname
	"""
	xname_raw = frb_dict["xname"]
	xvals_raw = np.array(frb_dict["xvals"])

	dspec_params = frb_dict.get("dspec_params", None)
	V_params = frb_dict.get("V_params", None)

	# Resolve sweep_mode robustly from dspec_params
	sweep_mode = None
	if dspec_params is not None:
		# Namedtuple-like with attribute
		sweep_mode = getattr(dspec_params, "sweep_mode", None)
		# Dict case
		if sweep_mode is None and isinstance(dspec_params, dict):
			sweep_mode = dspec_params.get("sweep_mode")
		# List/tuple container case
		if sweep_mode is None and isinstance(dspec_params, (list, tuple)) and len(dspec_params) > 0:
			first = dspec_params[0]
			sweep_mode = getattr(first, "sweep_mode", None) if not isinstance(first, dict) else first.get("sweep_mode")

	# Base LaTeX name
	base_name = param_map.get(xname_raw, xname_raw)

	# Variance sweep: SD(xname), no weighting
	if sweep_mode == "sd":
		x = xvals_raw
		base_core = base_name.replace(",0", "").replace("_0", "")
		xname = param_map.get(f"sd_{xname_raw}", rf"\sigma_{{{base_core}}}")
		return x, xname

	# Mean sweep or default: allow optional normalisation by weight_x_by
	weight = None
	if weight_x_by is not None:
		# Check in dspec_params
		if isinstance(dspec_params, (list, tuple)) and len(dspec_params) > 0:
			if isinstance(dspec_params[0], dict) and weight_x_by in dspec_params[0]:
				weight = np.array(dspec_params[0][weight_x_by])[0]
		elif isinstance(dspec_params, dict) and weight_x_by in dspec_params:
			weight = np.array(dspec_params[weight_x_by])[0]
		# Check in V_params if not found in dspec_params
		if weight is None and V_params is not None:
			if isinstance(V_params, (list, tuple)) and len(V_params) > 0:
				if isinstance(V_params[0], dict) and weight_x_by in V_params[0]:
					weight = np.array(V_params[0][weight_x_by])[0]
			elif isinstance(V_params, dict) and weight_x_by in V_params:
				weight = np.array(V_params[weight_x_by])[0]
		if weight is None:
			logging.warning(f"'{weight_x_by}' not found in parameters. Using raw values.")

	# Apply normalisation if available
	if weight is None:
		x = xvals_raw
		xname = base_name
	else:
		x = xvals_raw / weight
		weight_symbol = param_map.get(weight_x_by, weight_x_by)
		xname = base_name + r" / " + weight_symbol

	return x, xname


def _get_weighted_y_name(yname, weight_y_by):
	"""
	Get LaTeX formatted y-axis name based on the weighting parameter and plot type.
	"""
	# No weighting: keep the original label
	if weight_y_by is None:
		return yname

	# Special case for PA variance ratio
	if yname == r"Var($\psi$)" and weight_y_by == "PA_i":
		return r"\mathcal{R}_{\mathrm{\psi}}"

	w_name = param_map.get(weight_y_by.removesuffix("_i"), weight_y_by.removesuffix("_i"))
	if "/" in w_name:
		return "(" + yname + ")/" + w_name
	return yname + '/' + w_name


def _parse_fit_arg(fit_item):
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
			logging.warning("'poly' fit requires a degree (e.g., 'poly,2'). Skipping fit for this run.")
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
				logging.warning("'poly' fit requires a degree (e.g., 'poly,2'). Skipping fit for this run.")
			return fit_type, fit_degree
		else:
			if fit_item == "poly":
				logging.warning("'poly' fit requires a degree (e.g., 'poly,2'). Skipping fit for this run.")
			return fit_item, None
	else:
		return None, None


def _print_avg_snrs(subdict):
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
		logging.info(f"Avg S/N at:\n lowest x: S/N = {avg_low if avg_low is not None else 'nan'}, \nhighest x: S/N = {avg_high if avg_high is not None else 'nan'}\n")
	
	med_low = np.round(np.nanmedian(lowest), 2)
	med_high = np.round(np.nanmedian(highest), 2)
	if med_low is not None or med_high is not None:
		logging.info(f"Median S/N at:\n lowest x: S/N = {med_low if med_low is not None else 'nan'}, \nhighest x: S/N = {med_high if med_high is not None else 'nan'}\n")
		

def _extract_expected_curves(exp_vars, V_params, xvals, param_key='exp_var_PA', weight_y_by=None):
	"""
	Extract expected series for each x in xvals from exp_vars.
	Handles:
	 - exp_vars[x][param_key] as list over realizations of either scalar or [regular, basic]
	 - optional normalization by a variance parameter (e.g., 'PA_i') from V_params

	Returns:
	 (exp_primary, exp_secondary) as np.ndarray (second can be None if not present)
	"""
	def first_non_none(seq):
		if isinstance(seq, (list, tuple, np.ndarray)):
			for v in seq:
				if v is None:
					continue
				return v
			return None
		return seq

	exp1, exp2 = [], []
	wts = []

	for xv in xvals:
		x_dict = exp_vars.get(xv, {})
		vals = x_dict.get(param_key, None)
		v = first_non_none(vals)

		if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
			e1, e2 = v[0], v[1]
		else:
			e1, e2 = v, None

		exp1.append(np.nan if e1 is None else float(e1))
		exp2.append(np.nan if e2 is None else (float(e2) if e2 is not None else np.nan))

		# Optional weighting by variance parameter (e.g., 'PA_i')
		if weight_y_by is not None:
			wdict = V_params.get(xv, {})
			wlist = wdict.get(weight_y_by, None)
			w = first_non_none(wlist)
			wts.append(np.nan if w is None else float(w))

	exp1 = np.asarray(exp1, dtype=float)
	exp2 = np.asarray(exp2, dtype=float)
	if weight_y_by is not None:
		wts = np.asarray(wts, dtype=float)
		with np.errstate(invalid='ignore', divide='ignore'):
			exp1 = exp1 / wts
			if np.any(np.isfinite(exp2)):
				exp2 = exp2 / wts
			else:
				exp2 = None
	else:
		if not np.any(np.isfinite(exp2)):
			exp2 = None

	return exp1, exp2


def _plot_expected(x, frb_dict, ax, V_params, xvals, param_key='exp_var_PA', weight_y_by=None):
	exp1, exp2 = _extract_expected_curves(
		frb_dict["exp_vars"], V_params, xvals, param_key=param_key, weight_y_by=weight_y_by
	)
	# Only plot if there are finite expected values
	has_exp1 = np.any(np.isfinite(exp1))
	has_exp2 = exp2 is not None and np.any(np.isfinite(exp2))
	if has_exp1:
		ax.plot(x, exp1, 'k--', linewidth=2.0, label='Expected')
	if has_exp2:
		ax.plot(x, exp2, 'k:', linewidth=2.0, label='Expected (basic)')


def _plot_single_job_common(
	frb_dict,
	yname_base,
	weight_y_by,
	x_weight_by,
	figsize,
	fit,
	scale,
	series_label,
	series_color='black',
	expected_param_key=None,
	ax=None,
	embed=False,
	plot_expected=True
):
	"""
	Common single-job plotting helper used by plot_pa_var and plot_lfrac_var.

	When embed=True, draws onto the provided Axes 'ax' without setting axis labels/scales
	or plotting expected curves (unless plot_expected=True). Returns (None, ax, meta)
	where meta = {'applied': bool, 'x': np.ndarray, 'xname': str}.

	When embed=False (default), creates a new Figure/Axes, sets labels/scales, and returns (fig, ax).
	"""
	xvals = frb_dict["xvals"]
	yvals = frb_dict["yvals"]
	V_params = frb_dict.get("V_params", {})
	dspec_params = frb_dict.get("dspec_params", {})

	# Select weight source for y
	if isinstance(weight_y_by, str) and weight_y_by.endswith("_i"):
		weight_source = V_params
	else:
		weight_source = dspec_params

	# Weight measured values (track if applied)
	y, applied = _weight_dict(xvals, yvals, weight_source, weight_by=weight_y_by, return_status=True)
	med_vals, percentile_errs = _median_percentiles(y, xvals)

	# X weighting
	x, xname = _weight_x_get_xname(frb_dict, weight_x_by=x_weight_by)
	lower = np.array([lower for (lower, upper) in percentile_errs])
	upper = np.array([upper for (lower, upper) in percentile_errs])

	# Axes setup
	created_fig = None
	if ax is None:
		created_fig, ax = plt.subplots(figsize=figsize)
		ax.grid(True, linestyle='--', alpha=0.6)
		fig = created_fig
	else:
		fig = None  # embedding on existing axes
		if not embed:
			ax.grid(True, linestyle='--', alpha=0.6)

	# Draw measured series
	ax.plot(x, med_vals, color=series_color, label=series_label, linewidth=2)
	ax.fill_between(x, lower, upper, color=series_color, alpha=0.2)

	# Expected curves (optional)
	if plot_expected and (expected_param_key is not None) and ("exp_vars" in frb_dict):
		exp_weight = weight_y_by if applied else None
		_plot_expected(x, frb_dict, ax, V_params, xvals, param_key=expected_param_key, weight_y_by=exp_weight)

	# Optional fit
	if fit is not None:
		logging.info(f"Applying fit: {fit}")
		fit_type, fit_degree = _parse_fit_arg(fit)
		_fit_and_plot(ax, x, med_vals, fit_type, fit_degree, label=None)
		if not embed:
			ax.legend()

	# Labels/scales only in standalone mode
	if not embed:
		final_yname = _get_weighted_y_name(yname_base, weight_y_by) if (weight_y_by is not None and applied) else yname_base
		_set_scale_and_labels(ax, scale, xname=xname, yname=final_yname, x=x)

	if embed:
		meta = {'applied': bool(applied), 'x': x, 'xname': xname}
		return None, ax, meta

	return fig, ax


def _plot_multirun(frb_dict, ax, fit, scale, yname=None, weight_y_by=None, weight_x_by=None, legend=True):
	"""
	Common plotting logic for plot_pa_var and plot_lfrac_var.

	Parameters:
	-----------
	frb_dict : dict
		Dictionary containing FRB simulation data.
	ax : matplotlib.axes.Axes
		Axis object for plotting.
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
		Key to use for weighting y-values ("PA", "l_frac", or any parameter name).
	weight_x_by : str or None, optional
		Parameter to weight/normalise x-axis by.
	"""

	 # Determine y-axis name if not provided
	base_yname = yname

	colour_list = list(colours.values())

	# Prefer a specific run for expected curves if present
	preferred_run = None
	for run in frb_dict.keys():
		if "full-band" in run and "total" in run:
			preferred_run = run
			break

	# Track whether weighting succeeded in all runs
	weight_applied_all = True
	first_run_key = next(iter(frb_dict))

	# Keep info for plotting expected after we know if weighting applied
	exp_ref_run = preferred_run if preferred_run is not None else first_run_key
	exp_ref_subdict = frb_dict[exp_ref_run]

	for idx, (run, subdict) in enumerate(frb_dict.items()):
		logging.info(f"Processing {run}:")
		colour = colour_map[run] if run in colour_map else colour_list[idx % len(colour_list)]

		# Per-run fit handling (can be a single string or a list per run)
		run_fit = None
		if fit is not None:
			if isinstance(fit, (list, tuple)) and len(fit) == len(frb_dict):
				run_fit = fit[idx]
			else:
				run_fit = fit

		# Embed single-job helper for each run (no expected curves per-run)
		_, ax, meta = _plot_single_job_common(
			frb_dict=subdict,
			yname_base=base_yname,
			weight_y_by=weight_y_by,
			x_weight_by=weight_x_by,
			figsize=None,        # unused when embed=True
			fit=run_fit,
			scale=scale,           # scales/labels set after loop
			series_label=run,
			series_color=colour,
			expected_param_key=None,
			ax=ax,
			embed=True,
			plot_expected=False
		)

		if weight_y_by is not None and not meta['applied']:
			logging.warning(f"Requested weighting by '{weight_y_by}' for run '{run}' but it could not be applied. Using unweighted values.")
		weight_applied_all &= meta['applied']

		_print_avg_snrs(subdict)

		# Track last x and xname for axis labeling
		x_last = meta['x']
		xname = meta['xname']

	# Plot expected curves once (from preferred run if available), matching weighting decision
	param_key = 'exp_var_' + (weight_y_by.removesuffix('_i'))
	weight_for_expected = weight_y_by if (weight_y_by is not None and weight_applied_all) else None
	_plot_expected(x_last, exp_ref_subdict, ax, exp_ref_subdict["V_params"], np.array(exp_ref_subdict["xvals"]),
				   param_key=param_key, weight_y_by=weight_for_expected)

	ax.grid(True, linestyle='--', alpha=0.6)
	if legend:
		ax.legend()

	# Decide y-axis label after knowing if weighting applied
	final_yname = _get_weighted_y_name(base_yname, weight_y_by) if (weight_y_by is not None and weight_applied_all) else base_yname
	_set_scale_and_labels(ax, scale, xname=xname, yname=final_yname, x=x_last)


def _process_pa_var(dspec, freq_mhz, time_ms, gdict, phase_window, freq_window, buffer_frac):
	freq_slc = slice(None)
	phase_slc = slice(None)

	if freq_window != "full-band":
		freq_slc = _get_freq_window_indices(freq_window, freq_mhz)
		freq_mhz = freq_mhz[freq_slc]
		dspec = dspec[:, freq_slc, :]

	if phase_window != "total":
		# Collapse to time profile and find peak robustly
		Its = np.nansum(dspec, axis=(0, 1))
		peak_index = int(np.nanargmax(Its))
		phase_slc = _get_phase_window_indices(phase_window, peak_index)
		# Avoid zero-length windows
		if isinstance(phase_slc, slice):
			start = 0 if phase_slc.start is None else phase_slc.start
			stop = Its.size if phase_slc.stop is None else phase_slc.stop
			if stop - start <= 0:
				# Fallback to at least a single-sample window around the peak
				start = max(0, peak_index - 1)
				stop = min(Its.size, peak_index + 1)
				phase_slc = slice(start, stop)
		time_ms = time_ms[phase_slc]
		#dspec = dspec_fslc[:, :, phase_slc]

	ts_data, _, _, _ = process_dspec(dspec, freq_mhz, gdict, buffer_frac)

	phits = ts_data.phits[phase_slc]
	ephits = ts_data.ephits[phase_slc]

	if phits is None or len(phits) == 0:
		return np.nan, np.nan

	# Use circular variance for polarization angles (in radians)
	# PA has a pi-radian ambiguity, so we work with 2*PA
	valid_phits = phits[np.isfinite(phits)]
	if len(valid_phits) == 0:
		return np.nan, np.nan
	
	pa_var = circvar(2 * valid_phits) / 4.0
	pa_var_deg2 = np.rad2deg(np.sqrt(pa_var))**2

	if not np.isfinite(pa_var) or pa_var == 0:
		return pa_var, np.nan
	with np.errstate(divide='ignore', invalid='ignore'):
		pa_var_err_deg2 = np.sqrt(np.nansum((np.rad2deg(phits) * np.rad2deg(ephits))**2)) / (pa_var_deg2 * len(phits))
	
	logging.info(f"Var(psi) = {pa_var_deg2:.3f} +/- {pa_var_err_deg2:.3f}")
	return pa_var_deg2, pa_var_err_deg2


def plot_pa_var(frb_dict, save, fname, out_dir, figsize, show_plots, scale, phase_window, freq_window, fit, extension, legend):
	"""
	Plot the variance of the polarization angle (PA) as a function of scattering parameters.
	
	This function creates a plot showing how the polarization angle variance (R_ψ) changes
	with scattering timescale or PA variance parameters. It supports both single-run and 
	multi-run plotting for parameter comparison studies.
	
	Parameters:
	-----------
	frb_dict : dict
		Dictionary containing FRB simulation results. For single runs, contains keys like
		'xvals', 'yvals', 'V_params', 'dspec_params'. For multi-run comparisons, contains
		nested dictionaries with run names as keys.
	save : bool
		Whether to save the plot to disk.
	fname : str
		Base filename for the saved plot (without extension).
	out_dir : str
		Output directory path for saving the plot.
	figsize : tuple or None
		Figure size as (width, height) in inches. If None, defaults to (10, 9).
	show_plots : bool
		Whether to display the plot interactively.
	scale : str
		Axis scaling type. Options: 'linear', 'logx', 'logy', 'loglog'.
	phase_window : str
		Phase window for analysis. Options: 'total', 'leading', 'trailing'.
	freq_window : str
		Frequency window for analysis. Options: 'full-band', 'lowest-quarter', 
		'lower-mid-quarter', 'upper-mid-quarter', 'highest-quarter'.
	fit : str, list, or None
		Fitting function(s) to overlay. Can be single fit type (applied to all runs)
		or list of fit types (one per run). Options include 'linear', 'power', 
		'poly,N' (polynomial of degree N), 'exp', 'log', etc.
	extension : str
		File extension for saved plots (e.g., 'pdf', 'png').
		
	Notes:
	------
	- For multi-run plots, each run is plotted with different colors and includes
	  median values with 16th-84th percentile error bands
	- X-axis shows τ/W (scattering time normalised by pulse width) 
	- Y-axis shows R_ψ, the variance ratio of polarisation angles
	- Automatic color mapping is applied for predefined run types
	"""

	yname = r"Var($\psi$)"
	if figsize is None:
		figsize = (10, 9)
	# If frb_dict contains multiple runs, plot each on the same axes
	if _is_multi_run_dict(frb_dict):
		fig, ax = plt.subplots(figsize=figsize)
		fig.subplots_adjust(left=0.18, right=0.98, bottom=0.16, top=0.98)
		# Weight measured and expected by PA_i to show ratio R_psi
		_plot_multirun(frb_dict, ax, fit=fit, scale=scale, weight_y_by="PA_i", weight_x_by="width_ms", yname=yname, legend=legend)

	else:	
		# Single job
		fig, ax = _plot_single_job_common(
			frb_dict=frb_dict,
			yname_base=r"Var($\psi$)",
			weight_y_by="PA_i",
			x_weight_by="width_ms",
			figsize=figsize,
			fit=fit,
			scale=scale,
			series_label=r'\psi$_{var}$',
			series_color='black',
			expected_param_key='exp_var_PA'
		)
	if show_plots:
		plt.show()
	if save:
		name = _make_plot_fname("pa_var", scale, fname, freq_window, phase_window)
		name = os.path.join(out_dir, name + f".{extension}")
		fig.savefig(name, dpi=600)
		logging.info(f"Saved figure to {name}  \n")


def _process_lfrac(dspec, freq_mhz, time_ms, gdict, phase_window, freq_window, buffer_frac):
	freq_slc = slice(None)
	phase_slc = slice(None)

	if freq_window != "full-band":
		freq_slc = _get_freq_window_indices(freq_window, freq_mhz)
		freq_mhz = freq_mhz[freq_slc]
		dspec = dspec[:, freq_slc, :]

	if phase_window != "total":
		# Collapse to time profile and find peak robustly
		Its = np.nansum(dspec, axis=(0, 1))
		peak_index = int(np.nanargmax(Its))
		phase_slc = _get_phase_window_indices(phase_window, peak_index)
		# Avoid zero-length windows
		if isinstance(phase_slc, slice):
			start = 0 if phase_slc.start is None else phase_slc.start
			stop = Its.size if phase_slc.stop is None else phase_slc.stop
			if stop - start <= 0:
				# Fallback to at least a single-sample window around the peak
				start = max(0, peak_index - 1)
				stop = min(Its.size, peak_index + 1)
				phase_slc = slice(start, stop)
		time_ms = time_ms[phase_slc]
		dspec = dspec[:, :, phase_slc]

	ts_data, _, _, _ = process_dspec(dspec, freq_mhz, gdict, buffer_frac)

	I, Q, U, V = ts_data.iquvt
	buffer_frac = buffer_frac
	on_mask, off_mask, (left, right) = on_off_pulse_masks_from_profile(
		I, frac=0.95, buffer_frac=buffer_frac
	)

	I_masked = np.where(on_mask, I, np.nan)
	Q_masked = np.where(on_mask, Q, np.nan)
	U_masked = np.where(on_mask, U, np.nan)

	L = np.sqrt(Q_masked**2 + U_masked**2)

	integrated_I = np.nansum(I_masked)
	integrated_L = np.nansum(L)

	with np.errstate(divide='ignore', invalid='ignore'):
		lfrac = integrated_L / integrated_I
		noise_I = np.nanstd(I[off_mask]) if np.any(off_mask) else np.nan
		noise_L = np.nanstd(L[off_mask]) if np.any(off_mask) else np.nan
		lfrac_err = np.sqrt((noise_L / integrated_I)**2 + (integrated_L * noise_I / integrated_I**2)**2)

	return lfrac, lfrac_err


def plot_lfrac_var(frb_dict, save, fname, out_dir, figsize, show_plots, scale, phase_window, freq_window, fit, extension, legend):
	"""
	Plot the linear polarization fraction (L/I) as a function of scattering parameters.
	
	This function visualises how the degree of linear polarization changes with varying
	scattering conditions or other simulation parameters. It computes the integrated
	linear polarization fraction over specified frequency and phase windows.
	
	Parameters:
	-----------
	frb_dict : dict
		Dictionary containing FRB simulation results. For single runs, contains keys like
		'xvals', 'yvals', 'V_params', 'dspec_params'. For multi-run comparisons, contains
		nested dictionaries with run names as keys.
	save : bool
		Whether to save the plot to disk.
	fname : str
		Base filename for the saved plot (without extension).
	out_dir : str
		Output directory path for saving the plot.
	figsize : tuple or None
		Figure size as (width, height) in inches. If None, defaults to (10, 9).
	show_plots : bool
		Whether to display the plot interactively.
	scale : str
		Axis scaling type. Options: 'linear', 'logx', 'logy', 'loglog'.
	phase_window : str
		Phase window for integration. Options: 'total', 'leading', 'trailing'.
		Determines which part of the pulse profile to include in L/I calculation.
	freq_window : str
		Frequency window for integration. Options: 'full-band', 'lowest-quarter', 
		'lower-mid-quarter', 'upper-mid-quarter', 'highest-quarter'.
	fit : str, list, or None
		Fitting function(s) to overlay. Can be single fit type (applied to all runs)
		or list of fit types (one per run). Options include 'linear', 'power', 
		'poly,N' (polynomial of degree N), 'exp', 'log', etc.
	extension : str
		File extension for saved plots (e.g., 'pdf', 'png').
		
	Notes:
	------
	- Linear fraction is calculated as L/I = √(Q² + U²)/I integrated over the 
	  specified windows and on-pulse region (95% width)
	- For multi-run plots, median values with 16th-84th percentile error bands are shown
	- X-axis shows τ/W (scattering time normalised by pulse width)
	- Error bars include both statistical noise and systematic uncertainties
	- Useful for studying depolarisation effects due to scattering
	"""

	# If frb_dict contains multiple job IDs, plot each on the same axes
	yname = r"\Pi_L"

	if figsize is None:
		figsize = (10, 9)
	if _is_multi_run_dict(frb_dict):
		fig, ax = plt.subplots(figsize=figsize)
		fig.subplots_adjust(left=0.18, right=0.98, bottom=0.16, top=0.98)
		_plot_multirun(frb_dict, ax, fit=fit, scale=scale, weight_y_by="lfrac", weight_x_by=None, yname=yname, legend=legend)
	else:
		# Single job via helper
		fig, ax = _plot_single_job_common(
			frb_dict=frb_dict,
			yname_base=r"\Pi_L",
			weight_y_by="lfrac",              # keep raw L/I by default
			x_weight_by="width_ms",
			figsize=figsize,
			fit=fit,
			scale=scale,
			series_label='L/I',
			series_color='black',
			expected_param_key=None
		)
	if show_plots:
		plt.show()
	if save:
		name = _make_plot_fname("l_frac", scale, fname, freq_window, phase_window)
		name = os.path.join(out_dir, name + f".{extension}")
		fig.savefig(name, dpi=600)
		logging.info(f"Saved figure to {name}  \n")




pa_var = PlotMode(
	name="pa_var",
	process_func=_process_pa_var,
	plot_func=plot_pa_var,
	requires_multiple_frb=True  
)

l_frac = PlotMode(
	name="l_frac",
	process_func=_process_lfrac,
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
	"l_frac": l_frac,
	"iquv": iquv,
	"lvpa": lvpa,
	"dpa": dpa,
	"RM": RM,
}