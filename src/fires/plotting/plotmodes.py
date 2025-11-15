# -----------------------------------------------------------------------------
# plotmodes.py
# FIRES: The Fast, Intense Radio Emission Simulator
#
# This module defines plot modes and their associated processing and plotting
# functions for visualizing FRB simulation results. It includes classes and
# functions for plotting Stokes parameters, polarisation angle variance,
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
import matplotlib.transforms as mtransforms
import numpy as np
from scipy.optimize import curve_fit

from fires.core.basicfns import (compute_segments, estimate_rm,
                                 on_off_pulse_masks_from_profile,
                                 pa_variance_deg2, process_dspec)
from fires.plotting.plotfns import (plot_dpa, plot_ilv_pa_ds, plot_pa_profile,
                                    plot_stokes)
from fires.utils.loaders import load_data
from fires.utils.params import base_param_name, is_measured_key, param_info
from fires.utils.utils import normalise_freq_window, normalise_phase_window

logging.basicConfig(level=logging.INFO)
for _name in ("fontTools", "fontTools.subset"):
	_lg = logging.getLogger(_name)
	_lg.setLevel(logging.WARNING)   
	_lg.propagate = False

#	--------------------------	Set plot parameters	---------------------------
def configure_matplotlib_from_config(plot_config=None, use_latex=None):
	"""
	Configure global Matplotlib style from plot configuration.
	
	Parameters:
	-----------
	plot_config : dict or None
		Plot configuration dictionary loaded from plotparams.toml
	use_latex : bool or None
		Override for LaTeX usage. If None, uses config value.
	"""
	if plot_config is None:
		plot_config = {}
	
	styling = plot_config.get('styling', {})
	general = plot_config.get('general', {})
	latex_config = plot_config.get('latex', {})
	
	# Build rcParams dict from config
	rc = {}
	
	# Map config keys to matplotlib rcParams
	if 'pdf_fonttype' in styling:
		rc['pdf.fonttype'] = styling['pdf_fonttype']
	if 'ps_fonttype' in styling:
		rc['ps.fonttype'] = styling['ps_fonttype']
	if 'savefig_dpi' in styling:
		rc['savefig.dpi'] = styling['savefig_dpi']
	if 'font_size' in styling:
		rc['font.size'] = styling['font_size']
	if 'axes_labelsize' in styling:
		rc['axes.labelsize'] = styling['axes_labelsize']
	if 'axes_titlesize' in styling:
		rc['axes.titlesize'] = styling['axes_titlesize']
	if 'legend_fontsize' in styling:
		rc['legend.fontsize'] = styling['legend_fontsize']
	if 'xtick_labelsize' in styling:
		rc['xtick.labelsize'] = styling['xtick_labelsize']
	if 'ytick_labelsize' in styling:
		rc['ytick.labelsize'] = styling['ytick_labelsize']
	if 'font_family' in styling:
		rc['font.family'] = styling['font_family']
	if 'color_cycle' in styling:
		rc['axes.prop_cycle'] = plt.cycler(color=styling['color_cycle'])
	if 'line_width' in styling:
		rc['lines.linewidth'] = styling['line_width']
	if 'marker_size' in styling:
		rc['lines.markersize'] = styling['marker_size']
	
	# Determine LaTeX usage
	latex_enabled = False
	if use_latex is not None:
		latex_enabled = bool(use_latex)
	elif 'use_latex' in general:
		latex_enabled = bool(general['use_latex'])
	else:
		# Fallback to environment variable
		latex_enabled = bool(int(os.environ.get("FIRES_USE_LATEX", "0")))
	
	rc['text.usetex'] = latex_enabled
	
	# Apply all rcParams
	for k, v in rc.items():
		plt.rcParams[k] = v
	
	# Handle LaTeX preamble
	if latex_enabled:
		try:
			preamble = latex_config.get('preamble', 
				r'\usepackage{amsmath}\usepackage{amssymb}\usepackage{siunitx}')
			plt.rcParams['text.latex.preamble'] = preamble
		except Exception as e:
			fallback = latex_config.get('fallback_on_error', True)
			if fallback:
				warnings.warn(f"LaTeX setup failed ({e}); falling back to non-LaTeX text rendering.")
				plt.rcParams['text.usetex'] = False
			else:
				raise


def get_plot_param(plot_config, section, key, default=None):
	"""Helper to safely get plotting parameters"""
	if plot_config is None:
		return default
	sec = plot_config.get(section)
	if key is None:
		return sec if sec is not None else default
	if isinstance(sec, dict):
		return sec.get(key, default)
	return default


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
	# Intrinsic parameters - format: (LaTeX_symbol, unit)
	"tau"         : (r"\tau_0", r"\mathrm{ms}"),
	"width"          : (r"W_0", r"\mathrm{ms}"),
	"A"              : (r"A_0", r"\mathrm{Jy}"),
	"spec_idx"       : (r"\alpha_0", ""),
	"DM"             : (r"\mathrm{DM}_0", r"\mathrm{pc\,cm^{-3}}"),
	"RM"             : (r"\mathrm{RM}_0", r"\mathrm{rad\,m^{-2}}"),
	"PA"             : (r"\psi_0", r"\mathrm{deg}"),
	"lfrac"          : (r"\Pi_{L,0}", ""),
	"vfrac"          : (r"\Pi_{V,0}", ""),
	"dPA"            : (r"\Delta\psi_0", r"\mathrm{deg}"),
	"band_centre_mhz": (r"\nu_{\mathrm{c},0}", r"\mathrm{MHz}"),
	"band_width_mhz" : (r"\Delta \nu_0", r"\mathrm{MHz}"),
	"N"              : (r"N", ""),
	"mg_width_low"   : (r"w_{\mathrm{low},0}", r"\%"),
	"mg_width_high"  : (r"w_{\mathrm{high},0}", r"\%"),
	# sd_<param> 
	"sd_t0"             : (r"\sigma_{t_0}", r"\mathrm{ms}"),
	"sd_A"              : (r"\sigma_A", ""),
	"sd_spec_idx"       : (r"\sigma_\alpha", ""),
	"sd_DM"             : (r"\sigma_{\mathrm{DM}}", r"\mathrm{pc\,cm^{-3}}"),
	"sd_RM"             : (r"\sigma_{\mathrm{RM}}", r"\mathrm{rad\,m^{-2}}"),
	"sd_PA"             : (r"\sigma_{\psi}", r"\mathrm{deg}"),
	"sd_lfrac"          : (r"\sigma_{\Pi_L}", ""),
	"sd_vfrac"          : (r"\sigma_{\Pi_V}", ""),
	"sd_dPA"            : (r"\sigma_{\Delta\psi}", r"\mathrm{deg}"),
	"sd_band_centre_mhz": (r"\sigma_{\nu_c}", r"\mathrm{MHz}"),
	"sd_band_width_mhz" : (r"\sigma_{\Delta \nu}", r"\mathrm{MHz}"),
}

def _param_info_or_dynamic(name: str) -> tuple[str, str]:
	"""
	Get (symbol, unit) for a parameter key.
	- First, try param_map (explicit overrides).
	- Otherwise, build from canonical rules in fires.utils.params.
	"""
	if name in param_map:
		val = param_map[name]
		return val if isinstance(val, tuple) else (val, "")
	return param_info(name)

def _base_of(key: str | None) -> str | None:
	if key is None:
		return None
	return base_param_name(key)

#	--------------------------	PlotMode class	---------------------------
class PlotMode:
	def __init__(self, name, plot_func, requires_multiple_frb=False):
		"""
		Represents a plot mode with its associated processing and plotting functions.

		Args:
			name (str): Name of the plot mode.
Z			plot_func (callable): Function to generate the plot.
			requires_multiple_frb (bool): Whether this plot mode requires `plot_var=True`.
		"""
		self.name = name
		self.plot_func = plot_func
		self.requires_multiple_frb = requires_multiple_frb
		

# --------------------------	Plot modes definitions	---------------------------
def basic_plots(fname, frb_data, mode, out_dir, plot_config=None, buffer_frac=None, **kwargs):
	"""
	Generate basic plots using configuration from plot_config.
	"""
	# Extract what we need from plot_config
	save = get_plot_param(plot_config, 'general', 'save_plots', False)
	figsize = get_plot_param(plot_config, 'general', 'figsize', [10, 9])
	show_plots = get_plot_param(plot_config, 'general', 'show_plots', True)
	extension = get_plot_param(plot_config, 'general', 'extension', 'pdf')
	legend = get_plot_param(plot_config, 'general', 'legend', True)
	xlim = get_plot_param(plot_config, 'general', 'xlim', None)
	ylim = get_plot_param(plot_config, 'general', 'ylim', None)
	show_onpulse = get_plot_param(plot_config, 'windows', 'show_onpulse', False)
	show_offpulse = get_plot_param(plot_config, 'windows', 'show_offpulse', False)
	dspec_params = frb_data.dspec_params
	freq_mhz = dspec_params.freq_mhz
	time_ms = dspec_params.time_ms

	tau = dspec_params.gdict['tau']

	ts_data, corr_dspec, noise_spec, noise_stokes = process_dspec(
		frb_data.dynamic_spectrum, freq_mhz, dspec_params, buffer_frac, skip_rm=True
	)

	iquvt = ts_data.iquvt
	
	if mode == "all":
		plot_ilv_pa_ds(corr_dspec, dspec_params, freq_mhz, time_ms, save, fname, out_dir, 
				ts_data, figsize, tau, show_plots, extension, 
				legend, buffer_frac, show_onpulse, show_offpulse)
		plot_stokes(fname, out_dir, corr_dspec, iquvt, freq_mhz, time_ms, save, figsize, show_plots, extension)
		plot_pa_profile(fname, out_dir, ts_data, time_ms, save, figsize, show_plots, extension, xlim, ylim)
		plot_dpa(fname, out_dir, noise_stokes, ts_data, time_ms, 5, save, figsize, show_plots, extension)
		estimate_rm(frb_data.dynamic_spectrum, freq_mhz, time_ms, noise_spec, 1.0e3, 1.0, out_dir, save, show_plots)
	elif mode == "iquv":
		plot_stokes(fname, out_dir, corr_dspec, iquvt, freq_mhz, time_ms, save, figsize, show_plots, extension)
	elif mode == "lvpa":
		plot_ilv_pa_ds(corr_dspec, dspec_params, freq_mhz, time_ms, save, fname, out_dir, 
				ts_data, figsize, tau, show_plots, extension, 
				legend, buffer_frac, show_onpulse, show_offpulse)
	elif mode == "pa":
			plot_pa_profile(fname, out_dir, ts_data, time_ms, save, figsize, show_plots, extension, xlim, ylim)
	elif mode == "dpa":
		plot_dpa(fname, out_dir, noise_stokes, ts_data, time_ms, 5, save, figsize, show_plots, extension)
	elif mode == "RM":
		estimate_rm(frb_data.dynamic_spectrum, freq_mhz, time_ms, noise_spec, 1.0e3, 1.0, out_dir, save, show_plots)
	else:
		logging.warning(f"Invalid mode: {mode} \n")


def _get_freq_window_indices(freq_window, freq_mhz):
	"""
	Map a user-provided frequency window (alias-friendly) to a channel slice.

	Accepts both 'dspec' aliases ('full-band', 'lowest-quarter', ...) and
	'segments' aliases ('all', '1q'..'4q').
	"""
	# Canonicalise to dspec labels first (so '1q' -> 'lowest-quarter', 'all' -> 'full-band')
	fw = normalise_freq_window(freq_window, target='dspec')

	q = int(len(freq_mhz) / 4)
	windows = {
		"lowest-quarter": slice(0, q),
		"lower-mid-quarter": slice(q, 2*q),
		"upper-mid-quarter": slice(2*q, 3*q),
		"highest-quarter": slice(3*q, None),
		"full-band": slice(None)
	}
	sl = windows.get(fw)
	if sl is None:
		raise ValueError(f"Unknown freq_window '{freq_window}' (canonical='{fw}'). "
						 f"Valid: {list(windows.keys())} or '1q'..'4q'/'all'")
	return sl


def _get_phase_window_indices(phase_window, peak_index):
	"""
	Map a user-provided phase window (alias-friendly) to a time slice.

	Accepts both 'dspec' aliases ('leading', 'trailing', 'total') and
	'segments' aliases ('first', 'last', 'total').
	"""
	# Canonicalise to dspec labels first (so 'first' -> 'leading', 'last' -> 'trailing')
	pw = normalise_phase_window(phase_window, target='dspec')

	phase_slices = {
		"leading": slice(0, peak_index),
		"trailing": slice(peak_index, None),
		"total": slice(None)
	}
	sl = phase_slices.get(pw)
	if sl is None:
		raise ValueError(f"Unknown phase_window '{phase_window}' (canonical='{pw}'). "
						 f"Valid: {list(phase_slices.keys())} or 'first'/'last'/'total'")
	return sl


def _canonical_window_keys(phase_window: str, freq_window: str) -> tuple[str, str]:
	"""
	Return ('segments' phase key, 'segments' freq key) from flexible inputs.
	- phase: 'leading'/'trailing'/'total' or 'first'/'last'/'total' -> 'first'/'last'/'total'
	- freq : 'full-band' or 'lowest-quarter'..'highest-quarter' or 'all'/'1q'..'4q' -> 'all'/'1q'..'4q'
	"""
	phase_key = normalise_phase_window(phase_window, target='segments')
	freq_key = normalise_freq_window(freq_window, target='segments')
	return phase_key, freq_key


def _extract_value_from_segments(seg_dict, quantity: str, phase_window: str, freq_window: str):
	"""
	Extract a measured quantity from compute_segments output using canonical keys.

	quantity: 'Vpsi', 'Lfrac', or 'Vfrac'
	phase_window: flexible alias ('leading'/'trailing'/'total' or 'first'/'last'/'total')
	freq_window : flexible alias ('full-band' or quarter names, or 'all'/'1q'..'4q')
	"""
	if not isinstance(seg_dict, dict):
		return np.nan

	phase_key, freq_key = _canonical_window_keys(phase_window, freq_window)

	# Simple rule:
	# - If a sub-phase is requested ('first' or 'last'), use the phase split.
	# - Otherwise ('total'), use the frequency split.
	try:
		if phase_key != 'total':
			return seg_dict['phase'][phase_key].get(quantity, np.nan)
		return seg_dict['freq'][freq_key].get(quantity, np.nan)
	except Exception:
		return np.nan


def _get_obs_cfg(plot_config):
	cfg = get_plot_param(plot_config, 'observational', None)
	if not isinstance(cfg, dict):
		return {}
	if not cfg.get('enabled', True):
		return {}
	return {
		'marker'          : cfg.get('marker', '*'),
		'size'            : cfg.get('size', 200),
		'colour'          : cfg.get('colour', 'magenta'),
		'edgecolor'       : cfg.get('edgecolor', 'black'),
		'linewidth'       : cfg.get('linewidth', 1.0),
		'alpha'           : cfg.get('alpha', 1.0),
		'error_style'     : cfg.get('error_style', 'spans'),
		'span_alpha'      : cfg.get('span_alpha', 0.1),
		'bar_alpha'       : cfg.get('bar_alpha', 0.7),
		'bar_capsize'     : cfg.get('bar_capsize', 5),
		'use_series_colour': cfg.get('use_series_colour', False),
		'label_prefix'    : cfg.get('label_prefix', "")
	}


def _legend_if_any(ax, loc='best'):
	"""Only draw legend if there are labeled artists."""
	try:
		handles, labels = ax.get_legend_handles_labels()
		labels = [str(l).strip() for l in labels if l and not str(l).startswith('_')]
		if not labels:
			return
		# Support a few convenient presets
		if isinstance(loc, str):
			lc = loc.strip().lower()
			if lc in ('outside right', 'right outside'):
				ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
				return
			if lc in ('outside top', 'top outside'):
				ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), borderaxespad=0.)
				return
		# Default: pass loc through to Matplotlib
		ax.legend(loc=loc)
	except Exception:
		pass


def _median_percentiles(yvals, x, ndigits=3, atol=1e-12, rtol=1e-9, p_low=16, p_high=84):
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
	p_low : float, optional
		Lower percentile to compute (default: 16)
	p_high : float, optional
		Upper percentile to compute (default: 84)
		
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
			lower = np.nanpercentile(v_arr, p_low)
			upper = np.nanpercentile(v_arr, p_high)
			percentile_errs.append((lower, upper))
		else:
			med_vals.append(np.nan)
			percentile_errs.append((np.nan, np.nan))

	return med_vals, percentile_errs


def _x_percentiles_measured(frb_dict, x_measured: str, p_low=16, p_high=84,
							phase_window='total', freq_window='all'):
	"""
	Compute per-parameter-step percentiles of a measured x quantity for the SAME
	(freq_window, phase_window) used on the x-axis, returning (x_med, x_lo, x_hi).
	"""
	xvals_sweep = np.array(frb_dict["xvals"])
	measures = frb_dict.get("measures", {})
	x_med = []
	x_lo = []
	x_hi = []
	for xv in xvals_sweep:
		seg_list = measures.get(xv, [])
		vals = []
		for seg in seg_list:
			if not isinstance(seg, dict):
				continue
			val = _extract_value_from_segments(seg, x_measured, phase_window, freq_window)
			if np.isfinite(val):
				vals.append(val)
		if len(vals) == 0:
			x_med.append(np.nan); x_lo.append(np.nan); x_hi.append(np.nan)
		else:
			arr = np.asarray(vals, dtype=float)
			x_med.append(np.nanmedian(arr))
			x_lo.append(np.nanpercentile(arr, p_low))
			x_hi.append(np.nanpercentile(arr, p_high))
	return (np.asarray(x_med, dtype=float),
			np.asarray(x_lo, dtype=float),
			np.asarray(x_hi, dtype=float))


def _find_matching_key(key, candidates, atol=1e-10, rtol=1e-8, ndigits=6):
	"""Find a key in candidates that matches key within tolerance or rounding."""
	candidates_arr = np.array(list(candidates), dtype=float)
	# Try isclose first
	close = np.isclose(candidates_arr, key, atol=atol, rtol=rtol)
	if np.any(close):
		return list(candidates)[np.where(close)[0][0]]
	# Fallback: try rounding
	rounded = np.round(candidates_arr, ndigits)
	key_rounded = round(key, ndigits)
	idxs = np.where(rounded == key_rounded)[0]
	if len(idxs) > 0:
		return list(candidates)[idxs[0]]
	return None


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
		Parameter name to use for weighting/normalisation. Can be any parameter in weight_params.
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
		for var in xvals:
			normalised_vals[var] = yvals.get(var, [])
		return (normalised_vals, applied) if return_status else normalised_vals
	
	if isinstance(weight_params, dict) and any(isinstance(v, dict) for v in weight_params.values()):
		weighting_param_exists = any(weight_by in weight_params.get(var, {}) for var in xvals)
		if not weighting_param_exists:
			logging.warning(f"Weighting parameter '{weight_by}' not found in weight dictionaries. Returning unweighted values.")
			for var in xvals:
				normalised_vals[var] = yvals.get(var, [])
			return (normalised_vals, applied) if return_status else normalised_vals
			
		for var in xvals:
			y_key = _find_matching_key(var, yvals.keys())
			w_key = _find_matching_key(var, weight_params.keys())
			y_values = yvals.get(y_key, [])
			var__weight_dict = weight_params.get(w_key, {})
			weights = var__weight_dict.get(weight_by, [])
		
			# Pairwise robust normalisation: align by index, drop non-finite or zero weights
			out = []
			if y_values and weights:
				n = min(len(y_values), len(weights))
				for val, wt in zip(y_values[:n], weights[:n]):
					if np.isfinite(val) and np.isfinite(wt) and wt != 0:
						out.append(val / wt)
						applied = True
				# if nothing usable, fall back to unweighted values
				normalised_vals[var] = out if out else list(y_values)
			else:
				normalised_vals[var] = list(y_values) if y_values else []
				
	elif isinstance(weight_params, (list, tuple)) and len(weight_params) > 0:
		if isinstance(weight_params[0], dict) and (weight_by in weight_params[0]):
			weight_value = weight_params[0][weight_by]
			if isinstance(weight_value, (list, np.ndarray)) and len(weight_value) > 0:
				weight_value = weight_value[0]
			
			for var in xvals:
				y_values = yvals.get(var, [])
				if y_values and weight_value is not None and weight_value != 0 and np.isfinite(weight_value):
					out = [val / weight_value for val in y_values if np.isfinite(val)]
					applied = applied or bool(out)
					normalised_vals[var] = out if out else list(y_values)
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


def _apply_axis_limits(ax, xlim=None, ylim=None):
	"""Safely apply axis limits; respects log scales and ignores invalid bounds."""
	def _clean(lim, scale):
		if lim is None:
			return None
		if not (isinstance(lim, (list, tuple)) and len(lim) == 2):
			return None
		try:
			lo = float(lim[0]); hi = float(lim[1])
		except Exception:
			return None
		if not (np.isfinite(lo) and np.isfinite(hi)):
			return None
		if lo == hi:
			return None
		if scale == 'log' and (lo <= 0 or hi <= 0):
			return None
		return (lo, hi)

	xs = 'log' if ax.get_xscale() == 'log' else 'linear'
	ys = 'log' if ax.get_yscale() == 'log' else 'linear'
	xok = _clean(xlim, xs)
	yok = _clean(ylim, ys)
	if xok:
		ax.set_xlim(*xok)
	if yok:
		ax.set_ylim(*yok)


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


def _text_with_offset(ax, x, y, s, dx_pts=0, dy_pts=0, ha='left', va='bottom',
					  transform='data', color=None, fontsize=None, alpha=None,
					  bbox=None, zorder=None, rotation=None):
	"""
	Draw text at (x, y) with an offset in points (dx_pts, dy_pts).
	- transform='data' or 'axes' selects the base transform.
	"""
	base = ax.transData if transform == 'data' else ax.transAxes
	offset = mtransforms.ScaledTranslation(dx_pts/72.0, dy_pts/72.0, ax.figure.dpi_scale_trans)
	tr = base + offset
	return ax.text(x, y, s, transform=tr, ha=ha, va=va, color=color, fontsize=fontsize,
				   alpha=alpha, bbox=bbox, zorder=zorder, rotation=rotation)


def _set_scale_and_labels(ax, scale, xname, yname, x=None, x_unit="", y_unit=""):
	# Format labels with units in square brackets if non-empty
	# Units containing LaTeX commands must be kept inside math mode
	xlabel = rf"${xname}$" + (rf" $[{x_unit}]$" if x_unit else "")
	ylabel = rf"${yname}$" + (rf" $[{y_unit}]$" if y_unit else "")
	
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	
	# Set scales
	if scale == "logx" or scale == "log":
		ax.set_xscale('log')
	if scale == "logy" or scale == "log":
		ax.set_yscale('log')
		_apply_log_decade_ticks(ax, axis='y', base=10)  # enforce 10^k labels
	# Set limits
	if x is not None:
		if scale in ["logx", "log"]:
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
	Returns True if frb_dict contains multiple run dictionaries (i.e., is a dict of dicts with 'xvals' keys).
	"""
	return isinstance(frb_dict, dict) and all(isinstance(v, dict) and "xvals" in v for v in frb_dict.values())


def _yvals_from_measures_dict(xvals, measures, plot_type: str, phase_window: str, freq_window: str) -> dict:
	"""
	Build yvals dict {xv: [values...]} from a measures dict produced by compute_segments.
	Does not mutate inputs.
	"""
	yvals = {}
	
	if plot_type == 'pa_var':
		quantity = 'Vpsi'
	elif plot_type == 'l_frac':
		quantity = 'Lfrac'
	else:
		quantity = plot_type
	
	for xv in xvals:
		vals = []
		for seg in measures.get(xv, []):
			val = _extract_value_from_segments(seg, quantity, phase_window, freq_window)
			if np.isfinite(val):  
				vals.append(val)
		yvals[xv] = vals
		
		if not vals:
			logging.debug(f"No valid {quantity} measurements for xval={xv}, phase={phase_window}, freq={freq_window}")
	
	return yvals


def _fit_and_plot(ax, x, y, fit_type, fit_degree=None, label=None, colour='black'):
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
			ax.plot(x_fit, y_model, '--', color=colour, label=fit_label if label is None else label)
	except Exception as e:
		logging.error(f"Fit failed: {e}")


def _weight_x_get_xname(
		frb_dict,
		weight_x_by=None,
		x_measured=None,
		phase_window=None,
		freq_window=None
	):
	"""
	Extract x-axis values.
	When x_measured is set, now respects phase_window/freq_window instead of
	forcing ('total','all'). Falls back to ('total','all') if not provided.
	"""
	if x_measured is not None:
		xvals_sweep = np.array(frb_dict["xvals"])
		measures = frb_dict.get("measures", {})
		
		# Use provided windows or defaults
		phase_key = normalise_phase_window(
			phase_window if phase_window is not None else 'total',
			target='segments'
		)
		freq_key = normalise_freq_window(
			freq_window if freq_window is not None else 'all',
			target='segments'
		)
		# Map back to dspec form when extracting
		phase_key_dspec = normalise_phase_window(phase_key, target='dspec')
		freq_key_dspec = normalise_freq_window(freq_key, target='dspec')

		measured_x = []
		for xv in xvals_sweep:
			seg_list = measures.get(xv, [])
			if not seg_list:
				measured_x.append(np.nan)
				continue
			vals = []
			for seg in seg_list:
				if not isinstance(seg, dict):
					continue
				val = _extract_value_from_segments(seg, x_measured, phase_key_dspec, freq_key_dspec)
				if np.isfinite(val):
					vals.append(val)
			measured_x.append(np.nanmedian(vals) if vals else np.nan)
		
		x = np.array(measured_x, dtype=float)
		
		if x_measured == 'Vpsi':
			xname = r"\mathbb{V}(\psi)"
			x_unit = r"\mathrm{deg}^2"
		elif x_measured == 'Lfrac':
			xname = r"\Pi_{L}"
			x_unit = ""
		elif x_measured == 'Vfrac':
			xname = r"\Pi_{V}"
			x_unit = ""
		else:
			raise ValueError(f"Unknown x_measured: {x_measured}")
		
		return x, xname, x_unit
	
	# Original behavior: use input parameter sweep
	xname_raw = frb_dict["xname"]
	xvals_raw = np.array(frb_dict["xvals"])

	dspec_params = frb_dict.get("dspec_params", None)
	V_params = frb_dict.get("V_params", None)

	sweep_mode = None
	if dspec_params is not None:
		sweep_mode = getattr(dspec_params, "sweep_mode", None)
		if sweep_mode is None and isinstance(dspec_params, dict):
			sweep_mode = dspec_params.get("sweep_mode")
		if sweep_mode is None and isinstance(dspec_params, (list, tuple)) and len(dspec_params) > 0:
			first = dspec_params[0]
			sweep_mode = getattr(first, "sweep_mode", None) if not isinstance(first, dict) else first.get("sweep_mode")

	param_info_entry = param_map.get(xname_raw, (xname_raw, ""))
	base_name = param_info_entry[0] if isinstance(param_info_entry, tuple) else param_info_entry
	base_unit = param_info_entry[1] if isinstance(param_info_entry, tuple) else ""

	if sweep_mode == "sd":
		x = xvals_raw
		sym, unit = _param_info_or_dynamic(f"sd_{xname_raw}")
		return x, sym, unit

	# Try to find weights
	weight = None  # scalar or per-x array
	weight_unit = ""
	if weight_x_by is not None:
		# 1) dspec_params.gdict
		if hasattr(dspec_params, "gdict") and isinstance(dspec_params.gdict, dict) and weight_x_by in dspec_params.gdict:
			weight = np.asarray(dspec_params.gdict[weight_x_by], dtype=float)
		elif isinstance(dspec_params, dict):
			# dict-like with embedded gdict
			if "gdict" in dspec_params and isinstance(dspec_params["gdict"], dict) and weight_x_by in dspec_params["gdict"]:
				weight = np.asarray(dspec_params["gdict"][weight_x_by], dtype=float)
			elif weight_x_by in dspec_params:
				weight = np.asarray(dspec_params[weight_x_by], dtype=float)
		# 2) Fall back to V_params per-x
		if weight is None and isinstance(V_params, dict) and len(V_params) > 0:
			ws = []
			for xv in xvals_raw:
				key = _find_matching_key(float(xv), V_params.keys())
				wd = V_params.get(key, {}) if key in V_params else {}
				wv = wd.get(weight_x_by, None)
				if isinstance(wv, (list, np.ndarray)):
					# take first non-None
					wv = next((vv for vv in wv if vv is not None and np.isfinite(vv)), None)
				ws.append(float(wv) if (wv is not None and np.isfinite(wv)) else np.nan)
			if np.any(np.isfinite(ws)):
				weight = np.asarray(ws, dtype=float)

	# If we found a weight, normalise x by it (scalar or per-x)
	if weight is None or (np.size(weight) == 0) or not np.any(np.isfinite(np.atleast_1d(weight))):
		# No usable weights
		if weight_x_by is not None:
			logging.warning(f"'{weight_x_by}' not found in parameters. Using raw values.")
		x = xvals_raw
		xname = base_name
		x_unit = base_unit
		return x, xname, x_unit

	# Make element-wise or scalar division robustly
	w_arr = np.atleast_1d(weight).astype(float)
	if w_arr.size == 1:
		w = float(w_arr[0])
		if not (np.isfinite(w) and w != 0):
			logging.warning(f"Weight '{weight_x_by}' is non-finite or zero. Using raw values.")
			x = xvals_raw
		else:
			x = xvals_raw / w
	elif w_arr.size == xvals_raw.size:
		# element-wise
		x = np.full_like(xvals_raw, np.nan, dtype=float)
		m = np.isfinite(xvals_raw) & np.isfinite(w_arr) & (w_arr != 0)
		x[m] = xvals_raw[m] / w_arr[m]
	else:
		# length mismatch: fall back to first value
		w = float(w_arr[0])
		if not (np.isfinite(w) and w != 0):
			x = xvals_raw
		else:
			x = xvals_raw / w

	# Build labels/units for weighted x
	weight_symbol, weight_unit = _param_info_or_dynamic(weight_x_by)
	xname = base_name + r" / " + weight_symbol
	x_unit = "" if base_unit == weight_unit else (f"{base_unit}/{weight_unit}" if weight_unit else base_unit)
	return x, xname, x_unit


def _get_weighted_y_name(yname, weight_y_by):
	"""
	Get LaTeX formatted y-axis name based on the weighting parameter and plot type.
	
	Returns:
	--------
	tuple
		(formatted_yname, y_unit) where y_unit is empty string if units cancel
	"""
	# Get base units for the y quantity
	y_base_unit = ""
	if yname == r"\mathbb{V}(\psi)":
		y_base_unit = r"\mathrm{deg}^2"
	elif yname == r"\Pi_L":
		y_base_unit = ""

	if weight_y_by is None:
		return yname, y_base_unit

	w_sym, w_unit = _param_info_or_dynamic(weight_y_by)

	# Special case: PA variance ratio
	if yname == r"\mathbb{V}(\psi)" and _base_of(weight_y_by) == "PA_i":
		return r"\mathcal{R}_{\mathrm{\psi}}", ""

	# Build name and units
	formatted_name = f"{yname}/{w_sym}" if "/" not in w_sym else f"({yname})/{w_sym}"
	result_unit = "" if y_base_unit == w_unit else (f"{y_base_unit}/{w_unit}" if w_unit else y_base_unit)
	return formatted_name, result_unit


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


def _print_med_snrs(subdict):
	snrs = subdict.get("snrs", [])
	# Handle dict or list
	if isinstance(snrs, dict):
		snr_values = list(snrs.values())
	else:
		snr_values = snrs
	# Flatten if nested
	if isinstance(snr_values, list) and snr_values and isinstance(snr_values[0], list):
		snr_values = [item for sublist in snr_values for item in (sublist if isinstance(sublist, list) else [sublist])]
	# Remove None and non-finite
	snr_values = [s for s in snr_values if s is not None and np.isfinite(s)]
	if not snr_values:
		logging.info("No valid S/N values found for this run.")
		return
	# Get S/N at lowest and highest xvals
	if isinstance(snrs, dict):
		keys_sorted = sorted(snrs.keys())
		lowest = snrs[keys_sorted[0]]
		highest = snrs[keys_sorted[-1]]
	else:
		lowest = snrs[0]
		highest = snrs[-1]
	def med(val):
		if isinstance(val, list):
			vals = [v for v in val if v is not None and np.isfinite(v)]
			if not vals:
				return None
			return np.nanmedian(vals)
		return val if val is not None and np.isfinite(val) else None
	med_low = np.max(lowest) #med(lowest)
	med_high = np.max(highest) #med(highest)
	if med_low is not None or med_high is not None:
		logging.info(f"Median S/N at:\n lowest x: S/N = {med_low if med_low is not None else 'nan'}, \nhighest x: S/N = {med_high if med_high is not None else 'nan'}\n")
	else:
		logging.info("No valid S/N values found for this run.")
		

def _extract_expected_curves(exp_vars, V_params, xvals, param_key='exp_var_PA', weight_y_by=None):
	"""
	Extract expected series for each x in xvals from exp_vars.
	Handles:
	 - exp_vars[x][param_key] as list over realisations of either scalar or [regular, basic]
	 - optional normalisation by a variance parameter (e.g., 'meas_var_PA') from V_params

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

		# Optional weighting by variance parameter (e.g., 'meas_var_PA')
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
	#if has_exp1:
		#ax.plot(x, exp1, 'k--', linewidth=2.0, label='Expected')
	if has_exp2:
		ax.plot(x, exp2, 'k:', linewidth=2.0, label='Expected')


def _parse_override_dict(override_str) -> dict:
	"""
	Parse an override string into a dict of param -> value.
	"""
	od = {}
	if not override_str:
		return od
	if ('=' in override_str) or ('+' in override_str):
		for tok in override_str.split('+'):
			if not tok or '=' not in tok:
				continue
			k, v = tok.split('=', 1)
			k = k.strip().replace('.', '_')
			try:
				od[k] = float(v)
			except Exception:
				continue
		return od
	import re
	pattern = r'(sd_?)?([A-Za-z][A-Za-z0-9_]*?)(-?(?:\d+(?:\.\d*)?|\.\d+))'
	for m in re.finditer(pattern, override_str):
		sd_prefix, key, value = m.groups()
		key = key.strip()
		if sd_prefix:
			key = f"sd_{key}"
		try:
			od[key] = float(value)
		except Exception:
			continue
	return od

def _format_legend_label(od: dict, legend_params=None, gdict=None) -> str:
	"""
	Build a label 'sym=value, sym=value' from a dict, only for keys in legend_params.
	If not found in od, look in gdict.
	"""
	if not legend_params:
		return ""
	parts = []
	for k in legend_params:
		val = None
		if k in od:
			val = od[k]
		elif gdict is not None and k in gdict and len(gdict[k]) > 0:
			val = gdict[k][0]
		if val is not None:
			sym, _ = _param_info_or_dynamic(k)
			try:
				s = str(int(val)) if float(val).is_integer() else f"{float(val):g}"
			except Exception:
				s = str(val)
			parts.append(rf"{sym}={s}")
	return ", ".join(parts)


def _plot_equal_value_lines(ax, frb_dict, target_param, weight_x_by=None, weight_y_by=None,
							target_values=None, n_lines=5, linestyle='-', alpha=1,
							show_labels=True, zorder=0, plot_type='pa_var', phase_window='total',
							freq_window='full-band', x_measured=None, y_measured=None, 
							interpolate=False, interp_kind='quadratic', interp_points=100, fit=None, fit_points=200,
							colour_mode='shades', base_colour='#005bff', cmap_name='viridis',
							extend_outside=False, extend_mode='linear',
							label_position='start', label_offset_pts=(0, 0), label_ha='left', label_va='top'):
	"""
	Plot background lines showing constant values of a swept parameter.
	Added dynamic colouring:
	 - colour_mode='shades': dark→light shades of base_colour
	 - colour_mode='cmap': sampled from matplotlib colormap cmap_name
	Parameters added:
		colour_mode : 'shades' | 'cmap'
		base_colour : hex (used if colour_mode='shades')
		cmap_name   : matplotlib colormap name (used if colour_mode='cmap')
		label_position: 'start' | 'end' | 'max-y' | 'mid'
		label_offset_pts: (dx, dy) points
	"""
	# Collect all xvals across runs
	all_xvals = set()
	for run_key, run_data in frb_dict.items():
		for xv in run_data["xvals"]:
			if np.isfinite(xv):
				all_xvals.add(float(xv))
	if not all_xvals:
		logging.warning(f"No valid xvals found. Cannot plot equal-value lines.")
		return

	# Determine axis span early
	xlim = ax.get_xlim()
	x_plot_min, x_plot_max = (min(xlim), max(xlim))
	x_is_log = (ax.get_xscale() == 'log')
	if x_is_log:
		eps = 1e-300
		x_plot_min = max(x_plot_min, eps)
		x_plot_max = max(x_plot_max, x_plot_min * (1 + 1e-6))
	def _x_grid(n):
		n = int(max(10, n))
		return np.geomspace(x_plot_min, x_plot_max, n) if x_is_log else np.linspace(x_plot_min, x_plot_max, n)

	# Select parameter values
	sorted_values = sorted(all_xvals)

	def _select_nearest(sorted_vals, desired_vals):
		"""Map each desired value to the nearest available sweep value (stable, unique)."""
		if not sorted_vals or not desired_vals:
			return []
		avail = np.asarray(sorted_vals, dtype=float)
		out = []
		used = set()
		for dv in desired_vals:
			idx = int(np.nanargmin(np.abs(avail - float(dv))))
			chosen = float(avail[idx])
			# avoid duplicates if possible by searching neighbors
			if chosen in used and avail.size > 1:
				lo, hi = idx - 1, idx + 1
				best = None
				best_dist = np.inf
				if lo >= 0 and float(avail[lo]) not in used:
					best = float(avail[lo]); best_dist = abs(avail[lo] - dv)
				if hi < avail.size and float(avail[hi]) not in used:
					d = abs(avail[hi] - dv)
					if d < best_dist:
						best = float(avail[hi]); best_dist = d
				if best is not None:
					chosen = best
			out.append(chosen)
			used.add(chosen)
		# preserve input order while removing exact repeats
		seen = set()
		return [v for v in out if not (v in seen or seen.add(v))]

	if target_values is None:
		if len(sorted_values) <= n_lines:
			selected_values = sorted_values
		else:
			indices = np.linspace(0, len(sorted_values) - 1, n_lines, dtype=int)
			selected_values = [sorted_values[i] for i in indices]
	else:
		# Map user-provided targets to nearest available sweep values
		selected_values = _select_nearest(sorted_values, list(target_values))
		if not selected_values:
			logging.warning("No selectable values for equal-value lines from target_values; skipping.")
			return

	# Build colour sequence
	def _generate_colours(vals, mode, base, cmap_nm):
		import matplotlib.cm as cm
		import matplotlib.colors as mcolors
		n = len(vals)
		if n == 0:
			return []
		if mode == 'cmap':
			cmap = cm.get_cmap(cmap_nm)
			# Normalise by actual numeric range of vals for semantic colour mapping
			vmin, vmax = min(vals), max(vals)
			if vmax == vmin:
				return [cmap(0.5)] * n
			return [cmap((v - vmin) / (vmax - vmin)) for v in vals]
		# shades mode
		base_rgb = np.array(mcolors.to_rgb(base))
		# factors from darker (0.55) to lighter (1.25)
		factors = np.linspace(0.55, 1.25, n)
		out = []
		for f in factors:
			if f <= 1:
				col = base_rgb * f
			else:
				blend = f - 1.0  # move toward white
				col = base_rgb + (1 - base_rgb) * blend
			col = np.clip(col, 0, 1)
			out.append(col)
		return out

	colours_for_lines = _generate_colours(selected_values, colour_mode, base_colour, cmap_name)

	logging.info(f"Plotting {len(selected_values)} equal-{target_param} lines with colour_mode='{colour_mode}'\n")

	# (Existing model helpers unchanged)
	def _safe_exp(z): return np.exp(np.clip(z, -700.0, 700.0))
	def _model_power(x, a, n): return a * np.power(x, n)
	def _model_power_fixed(x, a, n_fixed): return a * np.power(x, n_fixed)
	def _model_exp(x, a, b): return a * _safe_exp(b * x)
	def _model_broken_power(x, a, n1, n2, x_break): return np.where(x < x_break, a * x**n1, a * x_break**(n1-n2) * x**n2)

	for (xval, line_colour) in zip(selected_values, colours_for_lines):
		x_points = []
		y_points = []
		for run_key, run_data in frb_dict.items():
			xvals = run_data["xvals"]
			measures = run_data["measures"]
			V_params = run_data.get("V_params", {})
			dspec_params = run_data.get("dspec_params", {})

			xval_indices = np.where(np.isclose(xvals, xval, rtol=1e-6))[0]
			if len(xval_indices) == 0:
				continue

			# x extraction
			if x_measured is not None:
				x_vals, _, _ = _weight_x_get_xname(
					run_data,
					weight_x_by=weight_x_by,
					x_measured=x_measured,
					phase_window=phase_window,
					freq_window=freq_window
				)
			else:
				x_vals, _, _ = _weight_x_get_xname(
					run_data,
					weight_x_by=weight_x_by,
					x_measured=None,
					phase_window=phase_window,
					freq_window=freq_window
				)

			y_quantity = 'Vpsi' if plot_type == 'pa_var' else 'Lfrac'
			yvals_dict = _yvals_from_measures_dict(xvals, measures, plot_type, phase_window, freq_window)

			if weight_y_by is not None:
				weight_source = V_params if is_measured_key(weight_y_by) else dspec_params
				yvals_weighted, _ = _weight_dict(xvals, yvals_dict, weight_source, weight_by=weight_y_by, return_status=True)
			else:
				yvals_weighted = yvals_dict

			for idx in xval_indices:
				x_coord = x_vals[idx] if idx < len(x_vals) else np.nan
				xv_key = xvals[idx]
				y_vals = yvals_weighted.get(xv_key, [])
				y_coord = np.nanmedian(y_vals) if y_vals else np.nan
				if np.isfinite(x_coord) and np.isfinite(y_coord):
					x_points.append(x_coord)
					y_points.append(y_coord)

		if len(x_points) < 2:
			continue

		sorted_indices = np.argsort(x_points)
		x_sorted = np.array(x_points)[sorted_indices]
		y_sorted = np.array(y_points)[sorted_indices]

		# Fit or interpolate (unchanged logic condensed)
		def _attempt_fit():
			if fit is None:
				return None
			fit_type, fit_degree = _parse_fit_arg(fit)
			x_fit = _x_grid(fit_points)
			xd = x_sorted
			yd = y_sorted
			if xd.size < 2:
				return None
			try:
				if fit_type == 'linear':
					p = np.polyfit(xd, yd, 1); return x_fit, np.polyval(p, x_fit)
				if fit_type == 'poly':
					deg = fit_degree if (fit_degree and fit_degree >= 1) else 2
					deg = min(deg, max(1, xd.size - 1))
					p = np.polyfit(xd, yd, deg); return x_fit, np.polyval(p, x_fit)
				if fit_type == 'exp':
					m = yd > 0
					xf, yf = xd[m], yd[m]
					if xf.size >= 2:
						p = np.polyfit(xf, np.log(yf), 1)
						return x_fit, _safe_exp(p[0] * x_fit + p[1])
				if fit_type in ('power', 'power_fixed_n'):
					mpos = xd > 0
					xf, yf = xd[mpos], yd[mpos]
					if xf.size >= 2:
						if fit_degree is not None:
							nf = float(fit_degree)
							if np.all(yf > 0):
								ln_a = np.nanmean(np.log(yf) - nf * np.log(xf))
								a_hat = np.exp(ln_a)
								return x_fit, _model_power_fixed(x_fit, a_hat, nf)
						else:
							if np.all(yf > 0):
								p = np.polyfit(np.log(xf), np.log(yf), 1)
								n_hat, ln_a_hat = p[0], p[1]
								a_hat = np.exp(ln_a_hat)
								return x_fit, _model_power(x_fit, a_hat, n_hat)
				if fit_type == 'log':
					mpos = xd > 0
					xf, yf = xd[mpos], yd[mpos]
					if xf.size >= 2:
						p = np.polyfit(np.log10(xf), yf, 1)
						return x_fit, np.polyval(p, np.log10(np.clip(x_fit, 1e-300, None)))
				if fit_type == 'broken-power':
					mpos = xd > 0
					xf, yf = xd[mpos], yd[mpos]
					if xf.size >= 4:
						p0 = [np.nanmax(yf), 1.0, 0.0, np.median(xf)]
						bounds = ([0, -np.inf, -np.inf, np.min(xf)], [np.inf, np.inf, np.inf, np.max(xf)])
						from scipy.optimize import curve_fit
						popt, _ = curve_fit(_model_broken_power, xf, yf, p0=p0, bounds=bounds, maxfev=20000)
						return x_fit, _model_broken_power(x_fit, *popt)
			except Exception as e:
				logging.debug(f"Equal-line fit failed ({fit_type}): {e}")
			return None

		fit_results = _attempt_fit()
		if fit_results is not None:
			x_line, y_line = fit_results
			ax.plot(x_line, y_line, linestyle=linestyle, alpha=alpha, color=line_colour, linewidth=1.8, zorder=zorder)
		else:
			if interpolate and len(x_sorted) >= 3:
				try:
					actual_kind = interp_kind
					if interp_kind == 'cubic' and len(x_sorted) < 4:
						actual_kind = 'quadratic'
					if interp_kind == 'quadratic' and len(x_sorted) < 3:
						actual_kind = 'linear'
					from scipy.interpolate import interp1d
					f = interp1d(x_sorted, y_sorted, kind=actual_kind, bounds_error=False, fill_value='extrapolate')
					x_smooth = _x_grid(interp_points)
					y_smooth = f(x_smooth)
					ax.plot(x_smooth, y_smooth, linestyle=linestyle, alpha=alpha, color=line_colour, linewidth=1.5, zorder=zorder)
				except Exception:
					ax.plot(x_sorted, y_sorted, linestyle=linestyle, alpha=alpha, color=line_colour, linewidth=1.5, zorder=zorder)
			else:
				# No smoothing interpolation: optionally extend to axis limits
				if extend_outside and len(x_sorted) >= 2:
					left_extend  = x_plot_min < x_sorted[0]
					right_extend = x_plot_max > x_sorted[-1]

					x_ext = []
					y_ext = []

					# Left extension point
					if left_extend:
						if extend_mode == 'linear':
							dx = x_sorted[1] - x_sorted[0]
							if dx != 0 and np.isfinite(dx):
								m = (y_sorted[1] - y_sorted[0]) / dx
								y_left = y_sorted[0] + m * (x_plot_min - x_sorted[0])
							else:
								y_left = y_sorted[0]
						else:  # 'flat'
							y_left = y_sorted[0]
						x_ext.append(x_plot_min)
						y_ext.append(y_left)

					# Core polyline
					x_ext.extend(x_sorted.tolist())
					y_ext.extend(y_sorted.tolist())

					# Right extension point
					if right_extend:
						if extend_mode == 'linear':
							dx = x_sorted[-1] - x_sorted[-2]
							if dx != 0 and np.isfinite(dx):
								m = (y_sorted[-1] - y_sorted[-2]) / dx
								y_right = y_sorted[-1] + m * (x_plot_max - x_sorted[-1])
							else:
								y_right = y_sorted[-1]
						else:  # 'flat'
							y_right = y_sorted[-1]
						x_ext.append(x_plot_max)
						y_ext.append(y_right)

					ax.plot(np.asarray(x_ext), np.asarray(y_ext), linestyle=linestyle, alpha=alpha, color=line_colour, linewidth=1.5, zorder=zorder)
				else:
					ax.plot(x_sorted, y_sorted, linestyle=linestyle, alpha=alpha, color=line_colour, linewidth=1.5, zorder=zorder)

		if show_labels:
			param_symbol, _ = _param_info_or_dynamic(target_param)
			val_str = f"{int(xval)}" if xval == int(xval) else f"{xval:.2g}"
			# choose label anchor on the polyline
			lbl_pos = (label_position or 'start').lower()
			if lbl_pos == 'end':
				ix = -1
			elif lbl_pos == 'max-y':
				ix = int(np.nanargmax(y_sorted))
			elif lbl_pos == 'mid':
				ix = int(len(x_sorted) // 2)
			else:  # 'start'
				ix = 0
			_text_with_offset(
				ax, float(x_sorted[ix]), float(y_sorted[ix]),
				rf"${param_symbol} = {val_str}$",
				dx_pts=float(label_offset_pts[0]) if label_offset_pts else 0.0,
				dy_pts=float(label_offset_pts[1]) if label_offset_pts else 0.0,
				ha=label_ha, va=label_va,
				color=line_colour, fontsize=18, alpha=min(alpha + 0.2, 1.0),
				zorder=10
			)


def plot_constant_param_lines(
	ax,
	param_data,
	x_key='vpsi',
	y_key='lfrac',
	color='gray',
	alpha=0.5,
	linestyle='--',
	label_fmt=r"${param} = {val}$",
	param_name=r"\sigma_\psi"
):
	"""
	Overlay lines of constant parameter on a 2D plot.

	param_data: dict
		Keys are parameter values, values are dicts with keys for x and y arrays.
		Example:
			{
				1.0: {'vpsi': [...], 'lfrac': [...]},
				2.0: {'vpsi': [...], 'lfrac': [...]},
				...
			}
	x_key: str
		Key for x-axis data in each dict (default: 'vpsi')
	y_key: str
		Key for y-axis data in each dict (default: 'lfrac')
	color: str
		Line color
	alpha: float
		Line transparency
	linestyle: str
		Line style
	label_fmt: str
		Format string for legend label. Use {param} and {val} for substitution.
	param_name: str
		LaTeX or string for parameter symbol in label.
	"""
	for param_val, vals in param_data.items():
		x = np.asarray(vals.get(x_key, []))
		y = np.asarray(vals.get(y_key, []))
		mask = np.isfinite(x) & np.isfinite(y)
		if np.sum(mask) < 2:
			continue
		# Sort by x so the line is monotonic in x
		x_valid = x[mask]
		y_valid = y[mask]
		order = np.argsort(x_valid)
		x_sorted = x_valid[order]
		y_sorted = y_valid[order]
		# Plot the line
		ax.plot(x_sorted, y_sorted, color=color, alpha=alpha, linestyle=linestyle, label=None)
		# Add text label at the rightmost point
		val_str = f"{int(param_val)}" if param_val == int(param_val) else f"{param_val:.2g}"
		label_text = label_fmt.format(param=param_name, val=val_str)
		ax.text(x_sorted[-1], y_sorted[-1], label_text,
				fontsize=15, alpha=alpha+0.2, color=color, zorder=0,
				verticalalignment='bottom', horizontalalignment='left',
				bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
						   alpha=0.7, edgecolor='none'))


def _label_series(ax, frb_dict, params_to_label, weight_x_by=None, weight_y_by=None,
				  plot_type='pa_var', phase_window='total', freq_window='full-band',
				  x_measured=None, position='max-y', fontsize=16, alpha=0.95,
				  offset_pts=(0, 0), ha='left', va='bottom', collect_texts=None):
	"""
	Place labels for each run with the values of params_to_label.
	Position options:
	 - 'max-y': at point of maximum y in the series
	 - 'end'  : at maximum x
	 - 'start': at minimum x
	 - 'max-x': at maximum x
	 - 'min-x': at minimum x
	 - 'min-y': at minimum y
	 - 'mid'  : at the middle index of valid points
	offset_pts: (dx, dy) in points to nudge label from anchor
	ha/va: horizontal/vertical alignment
	"""
	if not params_to_label:
		return
	for run_key, run_data in frb_dict.items():
		try:
			# x values
			x_vals, _, _ = _weight_x_get_xname(
				run_data,
				weight_x_by=weight_x_by,
				x_measured=x_measured,
				phase_window=phase_window,
				freq_window=freq_window
			)
			x_vals = np.asarray(x_vals, dtype=float)

			# y values
			xvals = run_data["xvals"]
			measures = run_data["measures"]
			V_params = run_data.get("V_params", {})
			dspec_params = run_data.get("dspec_params", {})

			yvals_dict = _yvals_from_measures_dict(xvals, measures, plot_type, phase_window, freq_window)
			if weight_y_by is not None:
				weight_source = V_params if is_measured_key(weight_y_by) else dspec_params
				yvals_weighted, _ = _weight_dict(xvals, yvals_dict, weight_source, weight_by=weight_y_by, return_status=True)
			else:
				yvals_weighted = yvals_dict

			# Align y with x via xvals keys (median per x)
			y_med = []
			for i, xv in enumerate(xvals):
				y_i = yvals_weighted.get(xv, [])
				y_med.append(np.nanmedian(y_i) if y_i else np.nan)
			y_med = np.asarray(y_med, dtype=float)

			# finite mask
			m = np.isfinite(x_vals) & np.isfinite(y_med)
			if not np.any(m):
				continue
			xf, yf = x_vals[m], y_med[m]

			# choose anchor index
			pos = (position or 'max-y').lower()
			if pos == 'start' or pos == 'min-x':
				idx = int(np.nanargmin(xf))
			elif pos == 'end' or pos == 'max-x':
				idx = int(np.nanargmax(xf))
			elif pos == 'min-y':
				idx = int(np.nanargmin(yf))
			elif pos == 'mid':
				idx = int(len(xf) // 2)
			else:  # 'max-y' (default)
				idx = int(np.nanargmax(yf))

			x_anchor, y_anchor = float(xf[idx]), float(yf[idx])

			# Build label text from params_to_label for this run
			label_parts = []
			for pname in params_to_label:
				sym, _ = _param_info_or_dynamic(pname)
				# Look in run_data param stores
				val = None
				# Try dspec_params.gdict
				gd = None
				dsp = run_data.get("dspec_params")
				if dsp is not None and hasattr(dsp, "gdict"):
					gd = dsp.gdict
				if gd and pname in gd and gd[pname]:
					try:
						v0 = gd[pname]
						val = float(v0[0] if isinstance(v0, (list, tuple, np.ndarray)) else v0)
					except Exception:
						pass
				# Try V_params
				if val is None:
					vp = run_data.get("V_params", {})
					if pname in vp and vp[pname] is not None:
						v0 = vp[pname]
						try:
							val = float(v0[0] if isinstance(v0, (list, tuple, np.ndarray)) else v0)
						except Exception:
							pass
				# Fallback: unknown
				if val is None:
					continue
				val_str = f"{int(val)}" if val == int(val) else f"{val:g}"
				label_parts.append(rf"{sym}={val_str}")
			if not label_parts:
				continue

			label_text = r"$" + ",\; ".join(label_parts) + r"$"
			txt = _text_with_offset(
				ax, x_anchor, y_anchor, label_text,
				dx_pts=float(offset_pts[0]) if offset_pts else 0.0,
				dy_pts=float(offset_pts[1]) if offset_pts else 0.0,
				ha=ha, va=va, color='black', fontsize=fontsize, alpha=alpha, zorder=15
			)
			if isinstance(collect_texts, list):
				collect_texts.append(txt)
		except Exception as e:
			logging.debug(f"Series labeling failed for run '{run_key}': {e}")


def _bin_xy(x, y, nbins=15, strategy='equal-count'):
	"""
	Bin (x, y) and compute median and 16–84 percentiles per bin.
	- strategy='equal-count' uses quantile edges to keep similar counts per bin.
	"""
	x = np.asarray(x, dtype=float)
	y = np.asarray(y, dtype=float)
	m = np.isfinite(x) & np.isfinite(y)
	x = x[m]; y = y[m]
	if x.size == 0:
		return x, y, y, y

	if strategy == 'equal-count':
		q = np.linspace(0, 1, nbins + 1)
		edges = np.quantile(x, q)
		edges[0]  = np.min(x) - 1e-12
		edges[-1] = np.max(x) + 1e-12
	else:
		edges = np.linspace(np.min(x), np.max(x), nbins + 1)

	xb, ymed, ylo, yhi = [], [], [], []
	for i in range(nbins):
		mask = (x >= edges[i]) & (x < edges[i+1])
		if not np.any(mask):
			continue
		xb.append(np.nanmedian(x[mask]))
		ymed.append(np.nanmedian(y[mask]))
		ylo.append(np.nanpercentile(y[mask], 16))
		yhi.append(np.nanpercentile(y[mask], 84))
	return np.asarray(xb), np.asarray(ymed), np.asarray(ylo), np.asarray(yhi)


def _plot_single_run_multi_window(
	frb_dict,
	ax,
	plot_type,
	gauss_file=None,
	sim_file=None,
	window_pairs=None,  
	weight_y_by=None,
	weight_x_by=None,
	x_measured=None,
	fit=None,
	scale="linear",
	legend=True,
	obs_data=None,
	obs_params=None,
	buffer_frac=1,
	draw_style='line-param',
	nbins=15,
	colour_by_sweep=False,
	plot_config=None
):
	"""
	Plot multiple freq/phase window combinations from a SINGLE run on the same axes.
	
	Parameters:
	-----------
	frb_dict : dict
		Single-run dict with 'measures', 'xvals', 'V_params', etc.
	ax : matplotlib.axes.Axes
		Axes to plot on
	plot_type : str
		'pa_var' or 'l_frac'
	window_pairs : list of tuples or None
		List of (freq_window, phase_window) pairs to plot.
		e.g., [('1q', 'total'), ('4q', 'total'), ('all', 'leading')]
		If None, defaults to [('all', 'total')].
	weight_y_by : str or None
	weight_x_by : str or None
	fit : str or None
	scale : str
	legend : bool
	obs_data : str or None
	obs_params : str or None
	buffer_frac : float
	"""
	if window_pairs is None:
		window_pairs = [('all', 'total')]
	
	xvals = frb_dict["xvals"]
	measures = frb_dict["measures"]
	V_params = frb_dict.get("V_params", {})
	dspec_params = frb_dict.get("dspec_params", {})
	
	if isinstance(weight_y_by, str) and is_measured_key(weight_y_by):
		weight_source = V_params
	else:
		weight_source = dspec_params
	
	n_combos = len(window_pairs)
	
	freq_windows = [pair[0] for pair in window_pairs]
	phase_windows = [pair[1] for pair in window_pairs]
	
	varying_freq = len(set(freq_windows)) > 1
	varying_phase = len(set(phase_windows)) > 1
	
	if varying_freq and not varying_phase:
		def get_colour(freq_win, phase_win):
			freq_label = normalise_freq_window(freq_win, target='dspec')
			phase_label = normalise_phase_window(phase_win, target='dspec')
			key = f"{freq_label}, {phase_label}"
			return colour_map.get(key, colours['blue'])
	
	elif varying_phase and not varying_freq:
		phase_colours = {
			'leading': colours['orange'],
			'trailing': colours['green'],
			'total': colours['purple']
		}
		def get_colour(freq_win, phase_win):
			phase_label = normalise_phase_window(phase_win, target='dspec')
			return phase_colours.get(phase_label, colours['blue'])
	
	elif varying_freq and varying_phase:
		# Both varying: use combination from colour_map or cycle through colours
		def get_colour(freq_win, phase_win):
			freq_label = normalise_freq_window(freq_win, target='dspec')
			phase_label = normalise_phase_window(phase_win, target='dspec')
			key = f"{freq_label}, {phase_label}"
			if key in colour_map:
				return colour_map[key]
			# Fallback: cycle through base colours
			idx = window_pairs.index((freq_win, phase_win))
			if n_combos <= len(colours):
				return list(colours.values())[idx % len(colours)]
			import matplotlib.cm as cm
			cmap = cm.get_cmap('tab20', n_combos)
			return cmap(idx)
	
	else:
		def get_colour(freq_win, phase_win):
			freq_label = normalise_freq_window(freq_win, target='dspec')
			phase_label = normalise_phase_window(phase_win, target='dspec')
			key = f"{freq_label}, {phase_label}"
			return colour_map.get(key, colours['purple']) 
	
	pct_cfg = get_plot_param(plot_config, 'analytical', 'percentiles', {}) or {}
	pct_enabled = bool(pct_cfg.get('enabled', True))
	p_low = float(pct_cfg.get('p_low', 16))
	p_high = float(pct_cfg.get('p_high', 84))
	y_cfg = pct_cfg.get('y', {}) or {}
	y_shade = bool(y_cfg.get('shade', True)) and pct_enabled
	y_alpha = float(y_cfg.get('alpha', 0.2))
	x_cfg = pct_cfg.get('x', {}) or {}
	x_pct_enabled = bool(x_cfg.get('enabled', False)) and pct_enabled and (x_measured is not None)
	x_style = str(x_cfg.get('style', 'bars')).lower()
	x_alpha = float(x_cfg.get('alpha', 0.12))

	x_last = None
	xname = None
	x_unit = None
	
	for idx, (freq_win, phase_win) in enumerate(window_pairs):
		yvals = _yvals_from_measures_dict(xvals, measures, plot_type, phase_win, freq_win)
		
		y_weighted, applied = _weight_dict(
			xvals, yvals, weight_source, 
			weight_by=weight_y_by, 
			return_status=True
		)
		
		med_vals, percentile_errs = _median_percentiles(y_weighted, xvals, p_low=p_low, p_high=p_high)
		lower = np.array([lo for (lo, hi) in percentile_errs])
		upper = np.array([hi for (lo, hi) in percentile_errs])
		
		if x_last is None:
			x, xname, x_unit = _weight_x_get_xname(
				frb_dict,
				weight_x_by=weight_x_by,
				x_measured=x_measured,
				phase_window=phase_win,
				freq_window=freq_win
			)
			x_last = x
		else:
			x = x_last

		x_lo = x_hi = None
		if x_pct_enabled:
			x_med_arr, x_lo_arr, x_hi_arr = _x_percentiles_measured(
				frb_dict, x_measured, p_low=p_low, p_high=p_high,
				phase_window=phase_win, freq_window=freq_win
			)
			x_lo = x_lo_arr
			x_hi = x_hi_arr
		# Sorting reorders x; keep percentile arrays aligned
		if draw_style == 'line-x':
			order = np.argsort(plot_x)
			plot_x   = plot_x[order]
			plot_med = plot_med[order]
			plot_lo  = plot_lo[order]
			plot_hi  = plot_hi[order]
			if x_lo is not None and x_hi is not None:
				x_lo = x_lo[order]
				x_hi = x_hi[order]
				
		elif draw_style == 'binned':
			bx, by, blo, bhi = _bin_xy(plot_x, plot_med, nbins=nbins, strategy='equal-count')
			plot_x, plot_med, plot_lo, plot_hi = bx, by, blo, bhi

		# labels/colours as before
		freq_label = normalise_freq_window(freq_win, target='dspec')
		phase_label = normalise_phase_window(phase_win, target='dspec')
		if varying_freq and varying_phase:
			series_label = f"{freq_label}, {phase_label}"
		elif varying_freq:
			series_label = freq_label
		elif varying_phase:
			series_label = phase_label
		else:
			series_label = f"{freq_label}, {phase_label}"
		colour = get_colour(freq_win, phase_win)

		if draw_style == 'scatter':
			yerr = np.vstack([(plot_med - plot_lo), (plot_hi - plot_med)])
			xerr = None
			if x_pct_enabled and x_lo is not None and x_hi is not None:
				# Ensure non-negative magnitudes for errorbar
				dx_low  = np.maximum(plot_x - x_lo, 0)
				dx_high = np.maximum(x_hi - plot_x, 0)
				xerr = np.vstack([dx_low, dx_high])
			if colour_by_sweep:
				cvals = np.arange(len(plot_x))
				ax.errorbar(
					plot_x, plot_med, yerr=yerr, xerr=xerr, fmt='none',
					ecolor='gray', alpha=0.6, elinewidth=1.2, capsize=2, zorder=1
				)
				ax.scatter(plot_x, plot_med, c=cvals, cmap='viridis', s=25, alpha=0.8, label=series_label, zorder=2)
			else:
				ax.errorbar(
					plot_x, plot_med, yerr=yerr, xerr=xerr, fmt='none',
					ecolor=colour, alpha=0.6, elinewidth=1.2, capsize=2, zorder=1
				)
				ax.scatter(plot_x, plot_med, color=colour, s=25, alpha=0.8, label=series_label, zorder=2)
		else:
			ax.plot(plot_x, plot_med, color=colour, label=series_label, linewidth=2)
			# Y-shaded band (16–84 by default)
			if y_shade:
				ax.fill_between(plot_x, plot_lo, plot_hi, color=colour, alpha=y_alpha)
			# Optional X-shaded band when using measured x
			if x_pct_enabled and x_style == 'shade':
				m = np.isfinite(plot_med) & np.isfinite(plot_x)
				m &= np.isfinite(x_lo) & np.isfinite(x_hi)
				if np.any(m):
					# sort by y for fill_betweenx stability
					order = np.argsort(plot_med[m])
					ax.fill_betweenx(plot_med[m][order], x_lo[m][order], x_hi[m][order],
						color=colour, alpha=x_alpha)

		
		if fit is not None and draw_style != 'binned' and draw_style != 'scatter':
			fit_type, fit_degree = _parse_fit_arg(fit)
			_fit_and_plot(ax, plot_x, plot_med, fit_type, fit_degree, label=None, colour=colour)

		
		if fit is not None:
			fit_type, fit_degree = _parse_fit_arg(fit)
			_fit_and_plot(ax, x, med_vals, fit_type, fit_degree, label=None, color=colour)

	if "exp_vars" in frb_dict:
		V_params = frb_dict.get("V_params", {})
		xvals = frb_dict["xvals"]
		if plot_type == 'pa_var':
			param_key = 'exp_var_PA'
		elif plot_type == 'l_frac':
			param_key = 'exp_var_lfrac'
		else:
			param_key = None
		if param_key is not None:
			x, _, _ = _weight_x_get_xname(frb_dict, weight_x_by=weight_x_by)
			if get_plot_param(plot_config, 'analytical', 'plot_expected', True):
				_plot_expected(x, frb_dict, ax, V_params, xvals, param_key=param_key, weight_y_by=weight_y_by)
			
	# Observational overlay
	if obs_data is not None:
		obs_cfg = _get_obs_cfg(plot_config) 
		try:
			plot_mode_obj = plot_modes.get(plot_type)
			if plot_mode_obj is None:
				logging.warning(f"Unknown plot_type '{plot_type}' for observational overlay")
			else:
				for idx, (freq_win, phase_win) in enumerate(window_pairs):
					freq_canonical = normalise_freq_window(freq_win, target='dspec')
					phase_canonical = normalise_phase_window(phase_win, target='dspec')
					
					obs_result = _process_observational_data(
						obs_data, obs_params, gauss_file, sim_file,
						phase_canonical,
						freq_canonical,
						buffer_frac=buffer_frac, 
						plot_mode=plot_mode_obj,
						x_measured=x_measured,
					)
					
					colour = get_colour(freq_win, phase_win)
					
					freq_label = freq_canonical
					phase_label = phase_canonical
					if varying_freq and varying_phase:
						obs_label = f"{obs_result['label']} ({freq_label}, {phase_label})"
					elif varying_freq:
						obs_label = f"{obs_result['label']} ({freq_label})"
					elif varying_phase:
						obs_label = f"{obs_result['label']} ({phase_label})"
					else:
						obs_label = obs_result['label']
					
					obs_result['label'] = obs_label
					
					use_series_colour = obs_cfg.get('use_series_colour', False)
					obs_colour = colour if use_series_colour else obs_cfg.get('colour', colour)
					_plot_observational_overlay(
						ax, obs_result,
						weight_x_by=weight_x_by,
						weight_y_by=weight_y_by,
						colour=obs_colour,
						marker=obs_cfg.get('marker', '*'),
						size=obs_cfg.get('size', 200),
						edgecolor=obs_cfg.get('edgecolor', 'black'),
						linewidth=obs_cfg.get('linewidth', 1.0),
						alpha=obs_cfg.get('alpha', 1.0),
						error_style=obs_cfg.get('error_style', 'spans'),
						span_alpha=obs_cfg.get('span_alpha', 0.1),
						bar_alpha=obs_cfg.get('bar_alpha', 0.7),
						bar_capsize=obs_cfg.get('bar_capsize', 5),
						label_prefix=obs_cfg.get('label_prefix', "")
					)
		except Exception as e:
			logging.error(f"Failed to overlay observational data: {e}")
	
	# Set axis labels and scales
	base_yname = r"\mathbb{V}(\psi)" if plot_type == 'pa_var' else r"\Pi_L"
	final_yname, y_unit = _get_weighted_y_name(base_yname, weight_y_by) if weight_y_by else (base_yname, "")
	_set_scale_and_labels(ax, scale, xname=xname, yname=final_yname, x=x_last, x_unit=x_unit, y_unit=y_unit)
	
	if legend:
		legend_loc = get_plot_param(plot_config, 'general', 'legend_loc', 'best')
		_legend_if_any(ax, loc=legend_loc)


def _plot_single_job_common(
	frb_dict,
	yname_base,
	weight_y_by,
	weight_x_by,
	x_measured,
	figsize,
	fit,
	scale,
	series_label,
	series_colour='black',
	expected_param_key=None,
	ax=None,
	embed=False,
	plot_expected=True,
	yvals_override=None,
	draw_style='line-param',
	nbins=15,
	colour_by_sweep=False,
	legend_loc='best',
	plot_config=None,
	phase_window=None,
	freq_window=None
):
	"""
	Common single-job plotting helper used by plot_pa_var and plot_lfrac_var.
	"""
	xvals = frb_dict["xvals"]
	yvals = yvals_override if yvals_override is not None else frb_dict["yvals"]
	V_params = frb_dict.get("V_params", {})
	dspec_params = frb_dict.get("dspec_params", {})

	if isinstance(weight_y_by, str) and is_measured_key(weight_y_by):  
		weight_source = V_params
	else:
		weight_source = dspec_params
	pct_cfg = get_plot_param(plot_config, 'analytical', 'percentiles', {}) or {}
	pct_enabled = bool(pct_cfg.get('enabled', True))
	p_low = float(pct_cfg.get('p_low', 16))
	p_high = float(pct_cfg.get('p_high', 84))
	y_cfg = pct_cfg.get('y', {}) or {}
	y_shade = bool(y_cfg.get('shade', True)) and pct_enabled
	y_alpha = float(y_cfg.get('alpha', 0.2))
	x_cfg = pct_cfg.get('x', {}) or {}
	x_pct_enabled = bool(x_cfg.get('enabled', False)) and pct_enabled and (x_measured is not None)
	x_style = str(x_cfg.get('style', 'bars')).lower()
	x_alpha = float(x_cfg.get('alpha', 0.12))

	y, applied = _weight_dict(xvals, yvals, weight_source, weight_by=weight_y_by, return_status=True)
	med_vals, percentile_errs = _median_percentiles(y, xvals, p_low=p_low, p_high=p_high)

	x, xname, x_unit = _weight_x_get_xname(
		frb_dict,
		weight_x_by=weight_x_by,
		x_measured=x_measured,
		phase_window=phase_window,
		freq_window=freq_window
	)
	lower = np.array([lower for (lower, upper) in percentile_errs])
	upper = np.array([upper for (lower, upper) in percentile_errs])

	# Optional x-percentiles (only when using x_measured)
	x_lo = x_hi = None
	if x_pct_enabled:
		x_med_raw, x_lo_arr, x_hi_arr = _x_percentiles_measured(
			frb_dict, x_measured, p_low=p_low, p_high=p_high,
			phase_window=phase_window if phase_window else 'total',
			freq_window=freq_window if freq_window else 'all'
		)
		x_lo = x_lo_arr
		x_hi = x_hi_arr

	created_fig = None
	if ax is None:
		created_fig, ax = plt.subplots(figsize=figsize)
		ax.grid(True, linestyle='--', alpha=0.6)
		fig = created_fig
	else:
		fig = None
		if not embed:
			ax.grid(True, linestyle='--', alpha=0.6)

	plot_x = x
	plot_med = np.asarray(med_vals, dtype=float)
	plot_lo  = lower
	plot_hi  = upper

	if draw_style == 'line-x':
		order = np.argsort(plot_x)
		plot_x   = plot_x[order]
		plot_med = plot_med[order]
		plot_lo  = plot_lo[order]
		plot_hi  = plot_hi[order]
		if x_lo is not None and x_hi is not None:
			x_lo = x_lo[order]
			x_hi = x_hi[order]
	elif draw_style == 'binned':
		bx, by, blo, bhi = _bin_xy(plot_x, plot_med, nbins=nbins, strategy='equal-count')
		plot_x, plot_med, plot_lo, plot_hi = bx, by, blo, bhi
		# After binning we cannot derive x-error bars from original percentiles safely -> drop xerr
		x_lo = None; x_hi = None

	if draw_style == 'scatter':
		yerr = np.vstack([(plot_med - plot_lo), (plot_hi - plot_med)])
		xerr = None
		if x_pct_enabled and x_lo is not None and x_hi is not None:
			dx_low  = np.maximum(plot_x - x_lo, 0)
			dx_high = np.maximum(x_hi - plot_x, 0)
			xerr = np.vstack([dx_low, dx_high])
		if colour_by_sweep:
			cvals = np.arange(len(plot_x))
			ax.errorbar(
				plot_x, plot_med, yerr=yerr, xerr=xerr, fmt='none',
				ecolor='gray', alpha=0.6, elinewidth=1.2, capsize=2, zorder=1
			)
			ax.scatter(plot_x, plot_med, c=cvals, cmap='viridis', s=25, alpha=0.8, label=series_label, zorder=2)
		else:
			ax.errorbar(
				plot_x, plot_med, yerr=yerr, xerr=xerr, fmt='none',
				ecolor=series_colour, alpha=0.6, elinewidth=1.2, capsize=2, zorder=1
			)
			ax.scatter(plot_x, plot_med, color=series_colour, s=25, alpha=0.8, label=series_label, zorder=2)
	else:
		ax.plot(plot_x, plot_med, color=series_colour, label=series_label, linewidth=2)
		if y_shade:
			ax.fill_between(plot_x, plot_lo, plot_hi, color=series_colour, alpha=y_alpha)
		if x_pct_enabled and x_style == 'shade':
			m = np.isfinite(plot_med) & np.isfinite(plot_x)
			m &= np.isfinite(x_lo) & np.isfinite(x_hi)
			if np.any(m):
				order = np.argsort(plot_med[m])
				ax.fill_betweenx(plot_med[m][order], x_lo[m][order], x_hi[m][order],
					color=series_colour, alpha=x_alpha)

	if plot_expected and (expected_param_key is not None) and ("exp_vars" in frb_dict) and draw_style != 'binned':
		# respect config override if provided
		_plot_expected_cfg = get_plot_param(plot_config, 'analytical', 'plot_expected', True)
		if _plot_expected_cfg:
			exp_weight = weight_y_by if applied else None
			_plot_expected(plot_x, frb_dict, ax, V_params, xvals, param_key=expected_param_key, weight_y_by=exp_weight)

	if fit is not None and draw_style not in ('binned', 'scatter'):
		logging.info(f"Applying fit: {fit}")
		fit_type, fit_degree = _parse_fit_arg(fit)
		_fit_and_plot(ax, plot_x, plot_med, fit_type, fit_degree, label=None)
		if not embed:
			_legend_if_any(ax, loc=legend_loc)

	if plot_expected and (expected_param_key is not None) and ("exp_vars" in frb_dict):
		# second expected plotting (older call) — respect config as well
		_plot_expected_cfg = get_plot_param(plot_config, 'analytical', 'plot_expected', True)
		if _plot_expected_cfg:
			exp_weight = weight_y_by if applied else None
			_plot_expected(x, frb_dict, ax, V_params, xvals, param_key=expected_param_key, weight_y_by=exp_weight)

	if fit is not None:
		logging.info(f"Applying fit: {fit}")
		fit_type, fit_degree = _parse_fit_arg(fit)
		_fit_and_plot(ax, x, med_vals, fit_type, fit_degree, label=None)
		if not embed:
			_legend_if_any(ax, loc=legend_loc)

	if not embed:
		final_yname, y_unit = _get_weighted_y_name(yname_base, weight_y_by) if (weight_y_by is not None and applied) else (yname_base, param_map.get(yname_base, ""))
		_set_scale_and_labels(ax, scale, xname=xname, yname=final_yname, x=plot_x, x_unit=x_unit, y_unit=y_unit)

	if embed:
		meta = {'applied': bool(applied), 'x': plot_x, 'xname': xname, 'x_unit': x_unit}
		return None, ax, meta

	return fig, ax


def _plot_multirun(frb_dict, ax, fit, scale, yname=None, weight_y_by=None, weight_x_by=None, x_measured=None,
				   legend=True, equal_value_lines=None, equal_lines_cfg=None, plot_type='pa_var',
				   phase_window='total', freq_window='full-band',
				   draw_style='line-param', nbins=15, colour_by_sweep=False,
				   legend_params=None, plot_text=None, plot_config=None):
	"""
	Common plotting logic for plot_pa_var and plot_lfrac_var.
	"""
	if equal_value_lines is not None and x_measured is None:
		logging.warning("--equal-value-lines requires --x-measured. Ignoring equal_value_lines.")
		equal_value_lines = None
	
	base_yname = yname

	curr_freq_label = normalise_freq_window(freq_window, target='dspec')
	curr_phase_label = normalise_phase_window(phase_window, target='dspec')
	base_label_for_all = f"{curr_freq_label}, {curr_phase_label}"

	preferred_run = None
	for run in frb_dict.keys():
		if "full-band" in run and "total" in run:
			preferred_run = run
			break

	weight_applied_all = True
	first_run_key = next(iter(frb_dict))

	exp_ref_run = preferred_run if preferred_run is not None else first_run_key
	exp_ref_subdict = frb_dict[exp_ref_run]

	x_last = None
	xname = None
	x_unit = None
	swept_param = None

	from collections import defaultdict
	base_groups = defaultdict(list)
	for freq_phase_key in frb_dict.keys():
		base_groups[freq_phase_key].append(freq_phase_key)
	
	def get_colour_shades(base_colour, n_shades):
		"""
		Generate n_shades of a base colour, from darker to lighter.
		Returns list of hex colours.
		"""
		import matplotlib.colors as mcolors
		
		if n_shades == 1:
			return [base_colour]

		rgb = mcolors.hex2color(base_colour)
		shades = []
		for i in range(n_shades):
			if n_shades == 2:
				factors = [0.7, 1.0]
			elif n_shades == 3:
				factors = [0.6, 1.0, 1.3]
			else:
				t = i / (n_shades - 1)
				factors = [0.5 + 0.8 * t]
			
			factor = factors[i] if i < len(factors) else 0.5 + 0.8 * (i / (n_shades - 1))
			
			if factor <= 1.0:
				new_rgb = tuple(max(0, min(1, c * factor)) for c in rgb)
			else:
				blend = factor - 1.0  
				new_rgb = tuple(max(0, min(1, c + (1 - c) * blend)) for c in rgb)
			shades.append(mcolors.rgb2hex(new_rgb))
		
		return shades

	base_colour = colour_map.get(base_label_for_all, colours['purple'])
	
	n_runs = len(frb_dict)
	if n_runs > 1:
		colour_shades = get_colour_shades(base_colour, n_runs)
	else:
		colour_shades = [base_colour]

	for idx, (override_key, run_data) in enumerate(frb_dict.items()):
		logging.info(f"Processing {override_key}:")
		od = _parse_override_dict(override_key)
		gdict = None
		if "dspec_params" in run_data and hasattr(run_data["dspec_params"], "gdict"):
			gdict = run_data["dspec_params"].gdict
		elif "gdict" in run_data:
			gdict = run_data["gdict"]
		series_label = _format_legend_label(od, legend_params, gdict)
		
		
		colour = colour_shades[idx]

		run_fit = None
		if fit is not None:
			if isinstance(fit, (list, tuple)) and len(fit) == len(frb_dict):
				run_fit = fit[idx]
			else:
				run_fit = fit

		if "measures" not in run_data:
			raise KeyError(f"Run '{override_key}' missing 'measures'.")
		yvals_run = _yvals_from_measures_dict(run_data["xvals"], run_data["measures"], plot_type, phase_window, freq_window)

		_, ax, meta = _plot_single_job_common(
			frb_dict=run_data,
			yname_base=base_yname,
			weight_y_by=weight_y_by,
			weight_x_by=weight_x_by,
			x_measured=x_measured, 
			figsize=None,
			fit=run_fit,
			scale=scale,
			series_label=series_label,
			series_colour=colour,
			expected_param_key=None,
			ax=ax,
			embed=True,
			plot_expected=False,
			yvals_override=yvals_run,
			draw_style=draw_style,
			nbins=nbins,
			colour_by_sweep=colour_by_sweep,
			plot_config=plot_config,
			phase_window=phase_window,
			freq_window=freq_window
		)
		if weight_y_by is not None and not meta['applied']:
			logging.warning(f"Requested weighting by '{weight_y_by}' for run '{override_key}' but it could not be applied. Using unweighted values.")
		weight_applied_all &= meta['applied']

		_print_med_snrs(run_data)

		x_last = meta['x']
		xname = meta['xname']
		x_unit = meta['x_unit']

		if swept_param is None:
			dspec_params = run_data.get("dspec_params", None)
			sweep_mode = None
			if dspec_params is not None:
				sweep_mode = getattr(dspec_params, "sweep_mode", None)
				if sweep_mode is None and isinstance(dspec_params, dict):
					sweep_mode = dspec_params.get("sweep_mode")
				if sweep_mode is None and isinstance(dspec_params, (list, tuple)) and len(dspec_params) > 0:
					first = dspec_params[0]
					sweep_mode = getattr(first, "sweep_mode", None) if not isinstance(first, dict) else first.get("sweep_mode")
			base_param = run_data.get("xname")
			if sweep_mode == "sd":
				swept_param = f"sd_{base_param}"
			else:
				swept_param = base_param

	if weight_y_by is not None:
		base = _base_of(weight_y_by)
		param_key = 'exp_var_' + (base if base else 'PA')
	elif base_yname == r"\mathbb{V}(\psi)":
		param_key = 'exp_var_PA'
	elif base_yname == r"\Pi_L":
		param_key = 'exp_var_lfrac'
	else:
		param_key = 'exp_var_PA'
		logging.warning(f"Could not determine expected parameter key for yname='{base_yname}', using default '{param_key}'")
	
	weight_for_expected = weight_y_by if (weight_y_by is not None and weight_applied_all) else None
	# only plot expected if config allows
	if get_plot_param(plot_config, 'analytical', 'plot_expected', True):
		_plot_expected(x_last, exp_ref_subdict, ax, exp_ref_subdict["V_params"], np.array(exp_ref_subdict["xvals"]),
					   param_key=param_key, weight_y_by=weight_for_expected)

	if legend:
		legend_loc = get_plot_param(plot_config, 'general', 'legend_loc', 'best')
		_legend_if_any(ax, loc=legend_loc)

	if plot_text:
		# If plot_text is a list of param names, look up and format their values
		first_run = next(iter(frb_dict.values()))
		gdict = None
		if "dspec_params" in first_run and hasattr(first_run["dspec_params"], "gdict"):
			gdict = first_run["dspec_params"].gdict
		elif "gdict" in first_run:
			gdict = first_run["gdict"]
		# Build label
		label_parts = []
		for item in plot_text:
			if gdict and item in gdict and len(gdict[item]) > 0:
				val = gdict[item][0]
				sym, unit = _param_info_or_dynamic(item)
				val_str = str(int(val)) if float(val).is_integer() else f"{float(val):g}"
				label = rf"{sym} = {val_str}" + (rf"~[{unit}]" if unit else "")
				label_parts.append(label)
			else:
				# Not a param, treat as literal text
				label_parts.append(str(item))
		display_text = r",\; ".join(label_parts)
		ax.text(
			0.98, 0.01, f"${display_text}$",
			transform=ax.transAxes, color='gray',
			va='bottom', ha='right', zorder=5,
			bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.5, edgecolor='none')
		)

	final_yname, y_unit = _get_weighted_y_name(base_yname, weight_y_by) if (weight_y_by is not None and weight_applied_all) else (base_yname, param_map.get(base_yname, ""))
	
	# Don't pass x to _set_scale_and_labels - let matplotlib auto-scale based on plotted data
	# Only set the axis labels and scale type
	xlabel = rf"${xname}$" + (rf" $[{x_unit}]$" if x_unit else "")
	ylabel = rf"${final_yname}$" + (rf" $[{y_unit}]$" if y_unit else "")
	
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	
	# Set scales
	if scale == "logx" or scale == "log":
		ax.set_xscale('log')
	if scale == "logy" or scale == "log":
		ax.set_yscale('log')
		_apply_log_decade_ticks(ax, axis='y', base=10)
	
	ax.autoscale(enable=True, axis='both', tight=False)

	if equal_value_lines is not None and x_measured is not None and swept_param is not None:
		_xlim0, _ylim0 = ax.get_xlim(), ax.get_ylim()
		y_measured = 'Vpsi' if plot_type == 'pa_var' else 'Lfrac'

		# Style parameters (fallback to defaults if cfg missing)
		if equal_lines_cfg:
			linestyle    = equal_lines_cfg.get('linestyle', '--')
			alpha        = float(equal_lines_cfg.get('alpha', 0.85))
			colour_mode  = equal_lines_cfg.get('colour_mode', 'shades')
			base_colour  = equal_lines_cfg.get('base_colour', '#005bff')
			cmap_name    = equal_lines_cfg.get('cmap_name', 'viridis')
			show_labels  = bool(equal_lines_cfg.get('show_labels', True))
			extend_out   = bool(equal_lines_cfg.get('extend_outside', False))
			extend_mode  = equal_lines_cfg.get('extend_mode', 'linear')
			lbl_pos      = equal_lines_cfg.get('label_position', 'start')
			lbl_off      = equal_lines_cfg.get('label_offset_pts', [0, 0]) or [0, 0]
			lbl_ha       = equal_lines_cfg.get('label_ha', 'left')
			lbl_va       = equal_lines_cfg.get('label_va', 'top')
		else:
			linestyle, alpha, colour_mode, base_colour, cmap_name, show_labels = '--', 0.85, 'shades', 'black', 'viridis', True
			extend_out, extend_mode = False, 'linear'
			lbl_pos, lbl_off, lbl_ha, lbl_va = 'start', [0, 0], 'left', 'top'

		if isinstance(equal_value_lines, int):
			logging.info(f"Using swept parameter '{swept_param}' for equal-value lines")
			_plot_equal_value_lines(
				ax, frb_dict, target_param=swept_param,
				weight_x_by=weight_x_by, weight_y_by=weight_y_by,
				x_measured=x_measured, y_measured=y_measured,
				target_values=None, n_lines=equal_value_lines,
				linestyle=linestyle, alpha=alpha, show_labels=show_labels, zorder=0,
				plot_type=plot_type, phase_window=phase_window, freq_window=freq_window,
				colour_mode=colour_mode, base_colour=base_colour, cmap_name=cmap_name,
				extend_outside=extend_out, extend_mode=extend_mode,
				label_position=lbl_pos, label_offset_pts=lbl_off, label_ha=lbl_ha, label_va=lbl_va
			)
		elif isinstance(equal_value_lines, (list, tuple, np.ndarray)):
			vals = [float(v) for v in equal_value_lines]
			logging.info(f"Plotting equal-value lines at explicit values: {vals}")
			_plot_equal_value_lines(
				ax, frb_dict, target_param=swept_param,
				weight_x_by=weight_x_by, weight_y_by=weight_y_by,
				x_measured=x_measured, y_measured=y_measured,
				target_values=vals, n_lines=len(vals),
				linestyle=linestyle, alpha=alpha, show_labels=show_labels, zorder=0,
				plot_type=plot_type, phase_window=phase_window, freq_window=freq_window,
				colour_mode=colour_mode, base_colour=base_colour, cmap_name=cmap_name,
				extend_outside=extend_out, extend_mode=extend_mode,
				label_position=lbl_pos, label_offset_pts=lbl_off, label_ha=lbl_ha, label_va=lbl_va
			)
		elif isinstance(equal_value_lines, str) and os.path.isdir(equal_value_lines):
			from fires.utils.loaders import load_multiple_data_grouped
			param_dict = load_multiple_data_grouped(equal_value_lines)
			param_data = {}
			import re
			for key, subdict in param_dict.items() if isinstance(param_dict, dict) else [(None, param_dict)]:
				param_val = None
				param_name = None
				try:
					if key is not None:
						m = re.search(r'([A-Za-z0-9_]+?)(?:_sd)?([0-9.]+)', key)
						if m:
							param_name = m.group(1)
							param_val = float(m.group(2))
					if param_val is None and 'dspec_params' in subdict and param_name is not None:
						param_val = float(subdict['dspec_params'].gdict.get(param_name, [np.nan])[0])
				except Exception:
					param_val = None
				if param_val is None or not np.isfinite(param_val):
					continue
				xvals = subdict['xvals']
				measures = subdict['measures']
				vpsi_vals = []
				lfrac_vals = []
				for xv in xvals:
					segs = measures.get(xv, [])
					vpsi = [_extract_value_from_segments(seg, 'Vpsi', phase_window, freq_window) for seg in segs]
					lfrac = [_extract_value_from_segments(seg, 'Lfrac', phase_window, freq_window) for seg in segs]
					vpsi_vals.append(np.nanmedian(vpsi))
					lfrac_vals.append(np.nanmedian(lfrac))
				param_data[param_val] = {'vpsi': vpsi_vals, 'lfrac': lfrac_vals}
			if param_name in param_map:
				param_symbol = param_map[param_name][0]
			else:
				param_symbol = param_name
			plot_constant_param_lines(
				ax, param_data, x_key='vpsi', y_key='lfrac',
				color='gray', alpha=0.5, param_name=rf"{param_symbol}"
			)
		else:
			logging.warning(f"equal_value_lines must be int, list, or directory path, got {type(equal_value_lines)}")
	
		ax.set_xlim(_xlim0)
		ax.set_ylim(_ylim0)
		
	series_labels_cfg = get_plot_param(plot_config, 'analytical', 'series_labels', None) if plot_config else None
	if isinstance(series_labels_cfg, dict) and series_labels_cfg.get('enabled', True):
		params_to_label = series_labels_cfg.get('params', None)
		position       = series_labels_cfg.get('position', 'max-y')
		fs             = int(series_labels_cfg.get('fontsize', 16))
		a              = float(series_labels_cfg.get('alpha', 0.95))
		dxdy          = series_labels_cfg.get('offset_pts', [0, 0]) or [0, 0]
		ha            = series_labels_cfg.get('ha', 'left')
		va            = series_labels_cfg.get('va', 'bottom')
		avoid_overlap = bool(series_labels_cfg.get('avoid_overlap', False))
		_texts = []
		try:
			_label_series(
				ax, frb_dict, params_to_label=params_to_label,
				weight_x_by=weight_x_by, weight_y_by=weight_y_by,
				plot_type=plot_type, phase_window=phase_window, freq_window=freq_window,
				x_measured=x_measured, position=position, fontsize=fs, alpha=a,
				offset_pts=dxdy, ha=ha, va=va, collect_texts=_texts
			)
			if avoid_overlap and _texts:
				try:
					from adjustText import adjust_text
					adjust_text(_texts, ax=ax, only_move={'points':'y', 'texts':'y'},
								autoalign='y', expand_text=(1.05, 1.15),
								arrowprops=dict(arrowstyle='-', color='0.3', lw=0.8, alpha=0.6))
				except Exception:
					# adjustText not available or failed; continue without adjustment
					pass
		except Exception as e:
			logging.debug(f"series_labels failed: {e}")


def plot_pa_var(
	frb_dict, 
	fname, 
	out_dir, 
	phase_window, 
	freq_window, 
	gauss_file=None,
	sim_file=None,
	obs_data=None,
	obs_params=None,
	compare_windows=None,
	buffer_frac=1,
	plot_config=None,
	**kwargs
	):
	"""
	Plot the variance of the polarisation angle (PA) as a function of scattering parameters.
	
	This function creates a plot showing how the polarisation angle variance (R_ψ) changes
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
		Axis scaling type. Options: 'linear', 'logx', 'logy', 'log'.
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
	compare_windows : dict or None
		If provided, enables multi-window comparison from a SINGLE run.
		Format: {'freq': ['1q', '4q', 'all'], 'phase': ['first', 'last', 'total']}
		Plots all combinations of freq x phase windows on same axes.
		Overrides phase_window/freq_window parameters.
		equal_value_lines : int or str
		If int, number of equal-value lines to plot for reference.
		If str, path to directory containing parameter data for custom lines.
	draw_style : str
		Style of data representation. Options: 'line-param', 'line-x', 'scatter', 'binned'.
		- 'line-param': Line plot ordered by simulation parameter (default).
		- 'line-x': Line plot ordered by x-values.
		- 'scatter': Scatter plot of individual points.
		- 'binned': Binned averages with error bands.
	nbins : int
		Number of bins to use if draw_style is 'binned'.
	colour_by_sweep : bool
		If True and draw_style is 'scatter', colours points by sweep order.
		
	Notes:
	------
	- For multi-run plots, each run is plotted with different colours and includes
	  median values with 16th-84th percentile error bands
	- X-axis shows τ/W (scattering time normalised by pulse width) 
	- Y-axis shows R_ψ, the variance ratio of polarisation angles
	- Automatic colour mapping is applied for predefined run types
	"""

	# Extract analytical config
	scale = get_plot_param(plot_config, 'analytical', 'plot_scale', 'linear')
	draw_style = get_plot_param(plot_config, 'analytical', 'draw_style', 'line-param')
	fit = get_plot_param(plot_config, 'analytical', 'fit_functions', None)
	weight_x_by = get_plot_param(plot_config, 'analytical', 'weight_x_by', None)
	weight_y_by = get_plot_param(plot_config, 'analytical', 'weight_y_by', None)
	x_measured = get_plot_param(plot_config, 'analytical', 'x_measured', None)
	equal_value_lines = get_plot_param(plot_config, 'analytical', 'equal_value_lines', None)
	if equal_value_lines is not None:
		try:
			equal_value_lines = int(equal_value_lines)
		except (ValueError, TypeError):
			# allow list passthrough
			pass
	equal_lines_cfg = get_plot_param(plot_config, 'analytical', 'equal_lines', None)
	if isinstance(equal_lines_cfg, dict):
		if not equal_lines_cfg.get('enabled', True):
			equal_lines_cfg = None
		else:
			if 'n_lines' in equal_lines_cfg:
				nv = equal_lines_cfg['n_lines']
				if isinstance(nv, (list, tuple)):
					equal_value_lines = [float(v) for v in nv]
				else:
					equal_value_lines = int(nv)
	legend_params = get_plot_param(plot_config, 'analytical', 'legend_params', [])
	plot_text = get_plot_param(plot_config, 'analytical', 'plot_text', [])
	nbins = get_plot_param(plot_config, 'analytical', 'nbins', 15)
	colour_by_sweep = get_plot_param(plot_config, 'analytical', 'colour_by_sweep', False)
	xlim_cfg = get_plot_param(plot_config, 'analytical', 'xlim', None)
	ylim_cfg = get_plot_param(plot_config, 'analytical', 'ylim', None)

	# Extract general config
	figsize = get_plot_param(plot_config, 'general', 'figsize', [10, 9])
	show = get_plot_param(plot_config, 'general', 'show_plots', True)
	save = get_plot_param(plot_config, 'general', 'save_plots', False)
	extension = get_plot_param(plot_config, 'general', 'extension', 'pdf')
	legend = get_plot_param(plot_config, 'general', 'legend', True)
	legend_loc = get_plot_param(plot_config, 'general', 'legend_loc', 'best')
	

	yname = r"\mathbb{V}(\psi)"
	if figsize is None:
		figsize = (10, 9)

	if compare_windows is not None:
		if _is_multi_run_dict(frb_dict):
			logging.warning("compare_windows only works with single-run data; ignoring.")
		else:
			fig, ax = plt.subplots(figsize=figsize)
			fig.subplots_adjust(left=0.18, right=0.98, bottom=0.16, top=0.98)
			_plot_single_run_multi_window(
				frb_dict=frb_dict,
				ax=ax,
				plot_type='pa_var',
				window_pairs=compare_windows,
				weight_y_by=weight_y_by,
				weight_x_by=weight_x_by,
				fit=fit,
				scale=scale,
				legend=legend,
				obs_data=obs_data,
				obs_params=obs_params,
				buffer_frac=buffer_frac,
				draw_style=draw_style,
				nbins=nbins,
				colour_by_sweep=colour_by_sweep,
				plot_config=plot_config
			)
			# Save/show
			_apply_axis_limits(ax, xlim_cfg, ylim_cfg)
			if show:
				plt.show()
			if save:
				name = f"{fname}_{scale}_pa_var_window_comparison.{extension}"
				name = os.path.join(out_dir, name)
				fig.savefig(name, dpi=600)
				logging.info(f"Saved figure to {name}\n")
			return

	# Multi-run
	if _is_multi_run_dict(frb_dict):
		fig, ax = plt.subplots(figsize=figsize)
		fig.subplots_adjust(left=0.18, right=0.98, bottom=0.16, top=0.98)
		_plot_multirun(
			frb_dict, ax, fit=fit, scale=scale, yname=yname,
			weight_y_by=weight_y_by, weight_x_by=weight_x_by, x_measured=x_measured,
			legend=legend, equal_value_lines=equal_value_lines, equal_lines_cfg=equal_lines_cfg,
			plot_type='pa_var', phase_window=phase_window, freq_window=freq_window,
			draw_style=draw_style, nbins=nbins, colour_by_sweep=colour_by_sweep,
			legend_params=legend_params, plot_text=plot_text, plot_config=plot_config
		)
		_apply_axis_limits(ax, xlim_cfg, ylim_cfg)
	else:
		yvals = _yvals_from_measures_dict(frb_dict["xvals"], frb_dict["measures"], 'pa_var', phase_window, freq_window)

		freq_label = normalise_freq_window(freq_window, target='dspec')
		phase_label = normalise_phase_window(phase_window, target='dspec')
		key = f"{freq_label}, {phase_label}"
		series_colour = colour_map.get(key, colours['purple']) 

		fig, ax = _plot_single_job_common(
			frb_dict=frb_dict,
			yname_base=yname,
			weight_y_by=weight_y_by,
			weight_x_by=weight_x_by,
			x_measured=x_measured,
			figsize=figsize,
			fit=fit,
			scale=scale,
			series_label=r'$\mathbb{V}(\psi)$',
			series_colour=series_colour,
			expected_param_key='exp_var_PA',
			yvals_override=yvals,
			draw_style=draw_style,
			nbins=nbins,
			colour_by_sweep=colour_by_sweep,
			legend_loc=legend_loc,
			plot_config=plot_config,
			phase_window=phase_window,
			freq_window=freq_window
		)
		_apply_axis_limits(ax, xlim_cfg, ylim_cfg)
		
	# Overlay observational data if provided
	if obs_data is not None:
		obs_cfg = _get_obs_cfg(plot_config)
		if obs_cfg:
			try:
				obs_result = _process_observational_data(
					obs_data, obs_params, gauss_file, sim_file, phase_window, freq_window,
					buffer_frac=buffer_frac, plot_mode=pa_var, x_measured=x_measured
				)
				_plot_observational_overlay(
					ax, obs_result,
					weight_x_by=weight_x_by,
					weight_y_by=weight_y_by,
					colour=obs_cfg.get('colour', 'red'),
					marker=obs_cfg.get('marker', '*'),
					size=obs_cfg.get('size', 200),
					edgecolor=obs_cfg.get('edgecolor', 'black'),
					linewidth=obs_cfg.get('linewidth', 1.0),
					alpha=obs_cfg.get('alpha', 1.0),
					error_style=obs_cfg.get('error_style', 'spans'),
					span_alpha=obs_cfg.get('span_alpha', 0.1),
					bar_alpha=obs_cfg.get('bar_alpha', 0.7),
					bar_capsize=obs_cfg.get('bar_capsize', 5),
					label_prefix=obs_cfg.get('label_prefix', "")
				)
				if legend:
					_legend_if_any(ax, loc=legend_loc)
			except Exception as e:
				logging.error(f"Failed to overlay observational data: {e}")
	
	if show:
		plt.show()
	if save:
		name = _make_plot_fname("pa_var", scale, fname, freq_window, phase_window)
		name = os.path.join(out_dir, name + f".{extension}")
		fig.savefig(name, dpi=600, bbox_inches='tight')
		logging.info(f"Saved figure to {name}  \n")


def plot_lfrac(
	frb_dict, 
	fname, 
	out_dir, 
	phase_window, 
	freq_window, 
	gauss_file=None,
	sim_file=None,
	obs_data=None,
	obs_params=None,
	compare_windows=None,
	buffer_frac=1,
	plot_config=None,
	**kwargs
	):
	"""
	Plot the linear polarisation fraction (L/I) as a function of scattering parameters.
	
	This function visualises how the degree of linear polarisation changes with varying
	scattering conditions or other simulation parameters. It computes the integrated
	linear polarisation fraction over specified frequency and phase windows.
	
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
		Axis scaling type. Options: 'linear', 'logx', 'logy', 'log'.
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
	compare_windows : dict or None
		If provided, enables multi-window comparison from a SINGLE run.
		Format: {'freq': ['1q', '4q', 'all'], 'phase': ['first', 'last', 'total']}
		Plots all combinations of freq x phase windows on same axes.
		Overrides phase_window/freq_window parameters.
	equal_value_lines : int or str
		If int, number of equal-value lines to plot for reference.
		If str, path to directory containing parameter data for custom lines.
	draw_style : str
		Style of data representation. Options: 'line-param', 'line-x', 'scatter', 'binned'.
		- 'line-param': Line plot ordered by simulation parameter (default).
		- 'line-x': Line plot ordered by x-values.
		- 'scatter': Scatter plot of individual points.
		- 'binned': Binned averages with error bands.
	nbins : int
		Number of bins to use if draw_style is 'binned'.
	colour_by_sweep : bool
		If True and draw_style is 'scatter', colours points by sweep order.
	"""

	# Extract analytical config
	scale = get_plot_param(plot_config, 'analytical', 'plot_scale', 'linear')
	draw_style = get_plot_param(plot_config, 'analytical', 'draw_style', 'line-param')
	fit = get_plot_param(plot_config, 'analytical', 'fit_functions', None)
	weight_x_by = get_plot_param(plot_config, 'analytical', 'weight_x_by', None)
	weight_y_by = get_plot_param(plot_config, 'analytical', 'weight_y_by', None)
	x_measured = get_plot_param(plot_config, 'analytical', 'x_measured', None)
	equal_value_lines = get_plot_param(plot_config, 'analytical', 'equal_value_lines', None)
	if equal_value_lines is not None:
		try:
			equal_value_lines = int(equal_value_lines)
		except (ValueError, TypeError):
			# allow list passthrough
			pass
	equal_lines_cfg = get_plot_param(plot_config, 'analytical', 'equal_lines', None)
	if isinstance(equal_lines_cfg, dict):
		if not equal_lines_cfg.get('enabled', True):
			equal_lines_cfg = None
		else:
			if 'n_lines' in equal_lines_cfg:
				nv = equal_lines_cfg['n_lines']
				if isinstance(nv, (list, tuple)):
					equal_value_lines = [float(v) for v in nv]
				else:
					equal_value_lines = int(nv)
	legend_params = get_plot_param(plot_config, 'analytical', 'legend_params', [])
	plot_text = get_plot_param(plot_config, 'analytical', 'plot_text', [])
	nbins = get_plot_param(plot_config, 'analytical', 'nbins', 15)
	colour_by_sweep = get_plot_param(plot_config, 'analytical', 'colour_by_sweep', False)
	xlim_cfg = get_plot_param(plot_config, 'analytical', 'xlim', None)
	ylim_cfg = get_plot_param(plot_config, 'analytical', 'ylim', None)
	
	# Extract general config
	figsize = get_plot_param(plot_config, 'general', 'figsize', [10, 9])
	show = get_plot_param(plot_config, 'general', 'show_plots', True)
	save = get_plot_param(plot_config, 'general', 'save_plots', False)
	extension = get_plot_param(plot_config, 'general', 'extension', 'pdf')
	legend = get_plot_param(plot_config, 'general', 'legend', True)
	legend_loc = get_plot_param(plot_config, 'general', 'legend_loc', 'best')

	yname = r"\Pi_L"

	if compare_windows is not None:
		if _is_multi_run_dict(frb_dict):
			logging.warning("compare_windows only works with single-run data; ignoring.")
		else:
			fig, ax = plt.subplots(figsize=figsize)
			fig.subplots_adjust(left=0.18, right=0.98, bottom=0.16, top=0.98)
			_plot_single_run_multi_window(
				frb_dict=frb_dict,
				ax=ax,
				plot_type='l_frac',
				window_pairs=compare_windows,
				weight_y_by=weight_y_by,
				weight_x_by=weight_x_by,
				fit=fit,
				scale=scale,
				legend=legend,
				obs_data=obs_data,
				obs_params=obs_params,
				buffer_frac=buffer_frac,
				draw_style=draw_style,
				nbins=nbins,
				colour_by_sweep=colour_by_sweep
			)
			_apply_axis_limits(ax, xlim_cfg, ylim_cfg)
			if show:
				plt.show()
			if save:
				name = f"{fname}_{scale}_l_frac_window_comparison.{extension}"
				name = os.path.join(out_dir, name)
				fig.savefig(name, dpi=600)
				logging.info(f"Saved figure to {name}\n")
			return

	if _is_multi_run_dict(frb_dict):
		fig, ax = plt.subplots(figsize=figsize)
		fig.subplots_adjust(left=0.18, right=0.98, bottom=0.16, top=0.98)
		_plot_multirun(
			frb_dict, ax, fit=fit, scale=scale, yname=yname,
			weight_y_by=weight_y_by, weight_x_by=weight_x_by, x_measured=x_measured,
			legend=legend, equal_value_lines=equal_value_lines, equal_lines_cfg=equal_lines_cfg,
			plot_type='l_frac', phase_window=phase_window, freq_window=freq_window,
			draw_style=draw_style, nbins=nbins, colour_by_sweep=colour_by_sweep,
			legend_params=legend_params, plot_text=plot_text, plot_config=plot_config
		)
		_apply_axis_limits(ax, xlim_cfg, ylim_cfg)
	else:
		yvals = _yvals_from_measures_dict(frb_dict["xvals"], frb_dict["measures"], 'l_frac', phase_window, freq_window)

		freq_label = normalise_freq_window(freq_window, target='dspec')
		phase_label = normalise_phase_window(phase_window, target='dspec')
		key = f"{freq_label}, {phase_label}"
		series_colour = colour_map.get(key, colours['purple']) 
		
		fig, ax = _plot_single_job_common(
			frb_dict=frb_dict,
			yname_base=r"\Pi_L",
			weight_y_by=weight_y_by,
			weight_x_by=weight_x_by,
			x_measured=x_measured,
			figsize=figsize,
			fit=fit,
			scale=scale,
			series_label='L/I',
			series_colour=series_colour,
			expected_param_key=None,
			yvals_override=yvals,
			draw_style=draw_style,
			nbins=nbins,
			colour_by_sweep=colour_by_sweep,
			legend_loc=legend_loc,
			plot_config=plot_config,
			phase_window=phase_window,
			freq_window=freq_window
		)
		_apply_axis_limits(ax, xlim_cfg, ylim_cfg)

	if obs_data is not None:
		obs_cfg = _get_obs_cfg(plot_config)
		if obs_cfg:
			try:
				obs_result = _process_observational_data(
					obs_data, obs_params, gauss_file, sim_file, phase_window, freq_window,
					buffer_frac=buffer_frac, plot_mode=l_frac, x_measured=x_measured
				)
				_plot_observational_overlay(
					ax, obs_result,
					weight_x_by=weight_x_by,
					weight_y_by=weight_y_by,
					colour=obs_cfg.get('colour', 'red'),
					marker=obs_cfg.get('marker', '*'),
					size=obs_cfg.get('size', 200),
					edgecolor=obs_cfg.get('edgecolor', 'black'),
					linewidth=obs_cfg.get('linewidth', 1.0),
					alpha=obs_cfg.get('alpha', 1.0),
					error_style=obs_cfg.get('error_style', 'spans'),
					span_alpha=obs_cfg.get('span_alpha', 0.1),
					bar_alpha=obs_cfg.get('bar_alpha', 0.7),
					bar_capsize=obs_cfg.get('bar_capsize', 5),
					label_prefix=obs_cfg.get('label_prefix', "")
				)
				if legend:
					_legend_if_any(ax, loc=legend_loc)
			except Exception as e:
				logging.error(f"Failed to overlay observational data: {e}")
	
	if show:
		plt.show()
	if save:
		name = _make_plot_fname("l_frac", scale, fname, freq_window, phase_window)
		name = os.path.join(out_dir, name + f".{extension}")
		fig.savefig(name, dpi=600, bbox_inches='tight')
		logging.info(f"Saved figure to {name}  \n")


def _process_observational_data(obs_data_path, obs_params_path, gauss_file, sim_file, phase_window, freq_window, buffer_frac, plot_mode, x_measured=None):
	"""
	Load and process observational FRB data to extract measurements for overlay plotting.
	
	Uses the full basicfns processing pipeline to ensure consistency with simulated data.
	
	Parameters:
	-----------
	obs_data_path : str
		Path to observational data directory or file
	obs_params_path : str or None
		Path to parameters file (optional)
	phase_window : str
		Phase window to use for analysis ('total', 'leading', 'trailing')
	freq_window : str
		Frequency window to use for analysis
	buffer_frac : float
		Buffer fraction for on/off pulse regions
	plot_mode : PlotMode
		Plot mode object (determines what to measure)
	x_measured : str or None
		If set, use measured quantity on x-axis (e.g., 'Vpsi')
		
	Returns:
	--------
	dict
		Dictionary with measurement results and metadata
	"""

	if os.path.isdir(obs_data_path) or (os.path.isfile(obs_data_path) and obs_data_path.endswith('.npy')):
		dspec, freq_mhz, time_ms, dspec_params = load_data(obs_data_path, obs_params_path, gauss_file, sim_file)
		gdict = dspec_params.gdict
	else:
		raise ValueError(f"Unsupported file format: {obs_data_path}")

	# Ensure 4-stokes shape
	if dspec.ndim == 2:
		dspec = dspec[np.newaxis, :, :]
		zeros = np.zeros_like(dspec[0:1])
		dspec = np.concatenate([dspec, zeros, zeros, zeros], axis=0)
	elif dspec.ndim == 3 and dspec.shape[0] < 4:
		n_missing = 4 - dspec.shape[0]
		zeros = np.zeros((n_missing, dspec.shape[1], dspec.shape[2]))
		dspec = np.concatenate([dspec, zeros], axis=0)

	# Process full-band once (used for metadata and some fallbacks)
	ts_data_full, corr_dspec, noisespec_full, noise_stokes_full = process_dspec(dspec, freq_mhz, dspec_params, buffer_frac, skip_rm=True)

	I_profile_full = ts_data_full.iquvt[0]
	L_profile_full = ts_data_full.Lts
	peak_index_full = int(np.nanargmax(I_profile_full)) if I_profile_full.size > 0 else 0

	# Compute per-window measures once
	segments = compute_segments(dspec, freq_mhz, time_ms, dspec_params, buffer_frac=buffer_frac)

	try:
		lfrac_win = _extract_value_from_segments(segments, 'Lfrac', phase_window, freq_window)
	except Exception:
		lfrac_win = np.nan
	try:
		vfrac_win = _extract_value_from_segments(segments, 'Vfrac', phase_window, freq_window)
	except Exception:
		vfrac_win = np.nan
	try:
		vpsi_win = _extract_value_from_segments(segments, 'Vpsi', phase_window, freq_window)
	except Exception:
		vpsi_win = np.nan

	# Log measured values for requested window
	logging.info(
		f"Observational window [{normalise_freq_window(freq_window, 'dspec')}, "
		f"{normalise_phase_window(phase_window, 'dspec')}]: "
		f"L/I={lfrac_win:.4g}, V/I={vfrac_win:.4g}, V(psi)={vpsi_win:.4g} deg^2"
	)
	
	# X-axis value
	x_value = None
	x_err = None

	if x_measured == 'Vpsi':
		x_value = _extract_value_from_segments(segments, 'Vpsi', phase_window, freq_window)
	elif x_measured == 'Lfrac':
		x_value = _extract_value_from_segments(segments, 'Lfrac', phase_window, freq_window)
	else:
		phits = ts_data_full.phits
		ephits = ts_data_full.ephits
		valid_mask = np.isfinite(phits) & np.isfinite(ephits)
		valid_phits = phits[valid_mask]
		valid_ephits = ephits[valid_mask]
		pa_var_deg2 = pa_variance_deg2(valid_phits) if valid_phits.size > 0 else np.nan
		pa_std_deg = np.sqrt(pa_var_deg2) if np.isfinite(pa_var_deg2) else np.nan
		pa_var_err = pa_variance_deg2(valid_ephits) if valid_ephits.size > 0 else 0.0
		x_value = pa_std_deg
		x_err = pa_var_err

	# Y-axis value from segments
	if plot_mode.name == 'pa_var':
		y_value = _extract_value_from_segments(segments, 'Vpsi', phase_window, freq_window)
		y_err = None
	elif plot_mode.name == 'l_frac':
		y_value = _extract_value_from_segments(segments, 'Lfrac', phase_window, freq_window)
		y_err = None
	else:
		raise ValueError(f"Observational overlay not supported for plot mode '{plot_mode.name}'")

	# Error estimates in the requested dspec window
	freq_slc = _get_freq_window_indices(freq_window, freq_mhz)
	dspec_win = dspec[:, freq_slc, :]
	freq_win = freq_mhz[freq_slc] if isinstance(freq_slc, slice) else freq_mhz

	ts_data_win, corr_dspec_win, noisespec_win, noise_stokes_win = process_dspec(dspec_win, freq_win, dspec_params, buffer_frac, skip_rm=True)

	I_profile = ts_data_win.iquvt[0]
	L_profile = ts_data_win.Lts

	peak_index = int(np.nanargmax(I_profile)) if I_profile.size > 0 else 0
	phase_slc = _get_phase_window_indices(phase_window, peak_index)

	on_mask, off_mask, (left, right) = on_off_pulse_masks_from_profile(
		I_profile, gdict["width"][0]/dspec_params.time_res_ms, frac=0.95, buffer_frac=buffer_frac
	)

	# Restrict to phase window
	on_mask_windowed = on_mask.copy()
	if isinstance(phase_slc, slice):
		tmp = np.zeros_like(on_mask, dtype=bool)
		start = 0 if phase_slc.start is None else phase_slc.start
		stop = I_profile.size if phase_slc.stop is None else phase_slc.stop
		if stop > start:
			tmp[start:stop] = True
		on_mask_windowed &= tmp

	# Errors
	if plot_mode.name == 'pa_var':
		phits = ts_data_win.phits[phase_slc]
		ephits = ts_data_win.ephits[phase_slc]
		valid_e = np.isfinite(ephits)
		y_err = pa_variance_deg2(ephits[valid_e]) if np.any(valid_e) else 0.0

	def _li_with_error(Its, Lts, on_mask_local):
		I_masked = np.where(on_mask_local, Its, np.nan)
		L_masked = np.where(on_mask_local, Lts, np.nan)
		integrated_I = np.nansum(I_masked)
		integrated_L = np.nansum(L_masked)
		if not np.isfinite(integrated_I) or integrated_I <= 0:
			return np.nan, np.nan
		val = integrated_L / integrated_I

		I_off = Its[off_mask]
		L_off = Lts[off_mask]
		# Count valid (non-NaN) samples
		nI = int(np.sum(np.isfinite(I_off)))
		nL = int(np.sum(np.isfinite(L_off)))

		if nI >= 1 and nL >= 1:
			# Use ddof=1 only if at least 2 valid samples
			ddof_I = 1 if nI > 1 else 0
			ddof_L = 1 if nL > 1 else 0
			with np.errstate(invalid='ignore'):
				noise_I = np.nanstd(I_off, ddof=ddof_I)
				noise_L = np.nanstd(L_off, ddof=ddof_L)

			n_on = int(np.sum(on_mask_local))
			if n_on > 0 and np.isfinite(noise_I) and np.isfinite(noise_L):
				sigma_I_int = noise_I * np.sqrt(n_on)
				sigma_L_int = noise_L * np.sqrt(n_on)
				err = val * np.sqrt(
					(sigma_L_int / max(integrated_L, 1e-12))**2 +
					(sigma_I_int / max(integrated_I, 1e-12))**2
				)
			else:
				err = 0.1 * val
		else:
			err = 0.1 * val

		return val, err

	if plot_mode.name == 'l_frac':
		_, li_err = _li_with_error(I_profile, L_profile, on_mask_windowed)
		y_err = li_err

	if x_measured == 'Lfrac':
		_, x_err_est = _li_with_error(I_profile, L_profile, on_mask_windowed)
		x_err = x_err_est

	if x_measured == 'Vpsi':
		phits = ts_data_win.phits[phase_slc]
		ephits = ts_data_win.ephits[phase_slc]
		valid_e = np.isfinite(ephits)
		x_err = pa_variance_deg2(ephits[valid_e]) if np.any(valid_e) else 0.0

	label = gdict.get('label', os.path.splitext(os.path.basename(obs_data_path))[0])

	result = {
		'x_value': x_value,
		'x_err': x_err,
		'y_value': y_value,
		'y_err': y_err,
		'label': label,
		'dspec': corr_dspec,
		'freq_mhz': freq_mhz,
		'time_ms': time_ms,
		'gdict': gdict,
		'ts_data': ts_data_full,
		'noisespec': noisespec_full
	}

	if (x_value is not None and np.isfinite(x_value)) and (y_value is not None and np.isfinite(y_value)):
		logging.info(f"Processed observational data")
	else:
		logging.warning("Failed to extract valid measurements from observational data")

	return result


def _plot_observational_overlay(ax, obs_result, weight_x_by=None, weight_y_by=None,
								colour='magenta', marker='*', size=200,
								edgecolor='black', linewidth=1.0, alpha=1.0,
								error_style='spans', span_alpha=0.1, bar_alpha=0.7,
								bar_capsize=5, label_prefix=""):
	"""
	Add observational data point with error bars as crosshairs on existing plot.
	
	Parameters:
	-----------
	ax : matplotlib.axes.Axes
		Axes to plot on
	obs_result : dict
		Result from _process_observational_data
	weight_x_by : str or None
		X-axis weighting parameter (for normalisation)
	weight_y_by : str or None
		Y-axis weighting parameter (for normalisation)
	colour : str
		colour for the observational marker
	marker : str
		Marker style
	size : float
		Marker size
	"""
	x = obs_result['x_value']
	x_err = obs_result['x_err']
	y = obs_result['y_value']
	y_err = obs_result['y_err']
	label = (label_prefix + obs_result['label']).strip()

	if x is None or y is None:
		logging.warning("Cannot plot observational data: missing x or y value")
		return

	# Apply weighting if specified
	if weight_x_by is not None and weight_x_by in obs_result['gdict']:
		x_weight = float(np.nanmean(obs_result['gdict'][weight_x_by]))
		if x_weight > 0 and np.isfinite(x_weight):
			x = x / x_weight
			if x_err is not None:
				x_err = x_err / x_weight
	if weight_y_by is not None and weight_y_by in obs_result['gdict']:
		y_weight = float(np.nanmean(obs_result['gdict'][weight_y_by]))
		if y_weight > 0 and np.isfinite(y_weight):
			y = y / y_weight
			if y_err is not None:
				y_err = y_err / y_weight

	# Plot central point
	ax.scatter(x, y, marker=marker, s=size, color=colour,
			   edgecolors=edgecolor, linewidths=linewidth,
			   zorder=100, alpha=alpha, label=label)

	# Error representations
	show_bars = error_style in ('bars', 'both')
	show_spans = error_style in ('spans', 'both')

	if show_bars:
		if x_err is not None and x_err > 0:
			ax.errorbar(x, y, xerr=x_err, fmt='none',
						ecolor=colour, elinewidth=1.8, capsize=bar_capsize,
						capthick=1.2, alpha=bar_alpha, zorder=99)
		if y_err is not None and y_err > 0:
			ax.errorbar(x, y, yerr=y_err, fmt='none',
						ecolor=colour, elinewidth=1.8, capsize=bar_capsize,
						capthick=1.2, alpha=bar_alpha, zorder=99)

	if show_spans:
		if x_err is not None and x_err > 0:
			ax.axvspan(x - x_err, x + x_err, alpha=span_alpha, color=colour, zorder=1)
		if y_err is not None and y_err > 0:
			ax.axhspan(y - y_err, y + y_err, alpha=span_alpha, color=colour, zorder=1)



# Define PlotMode instances for each plot type
pa_var = PlotMode(
	name="pa_var",
	plot_func=plot_pa_var,
	requires_multiple_frb=True  
)

l_frac = PlotMode(
	name="l_frac",
	plot_func=plot_lfrac,
	requires_multiple_frb=True  
)

iquv = PlotMode(
	name="iquv",
	plot_func=basic_plots,
	requires_multiple_frb=False  
)

lvpa = PlotMode(
	name="lvpa",
	plot_func=basic_plots,
	requires_multiple_frb=False
)

dpa = PlotMode(
	name="dpa",
	plot_func=basic_plots,
	requires_multiple_frb=False
)

RM = PlotMode(
	name="RM",
	plot_func=basic_plots,
	requires_multiple_frb=False
)

pa = PlotMode(
    name="pa",
    plot_func=basic_plots,
    requires_multiple_frb=False
)

plot_modes = {
    "pa_var": pa_var,
    "l_frac": l_frac,
    "iquv"  : iquv,
    "lvpa"  : lvpa,
    "dpa"   : dpa,
    "RM"    : RM,
    "pa"    : pa,
}