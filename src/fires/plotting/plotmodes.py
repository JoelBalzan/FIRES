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
import ast
import logging
import os
import re
import warnings
from types import SimpleNamespace

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.optimize import curve_fit

from fires.core.basicfns import (compute_segments, estimate_rm,
                                 on_off_pulse_masks_from_profile,
                                 pa_variance_deg2, process_dspec)
from fires.plotting.plot_helper import (build_plot_text_string, colour_map,
                                        colours, draw_plot_text,
                                        get_plot_param, get_pub_col,
                                        param_info_or_dynamic, param_map,
                                        pub_figsize, savefig, set_pub_style,
                                        text_with_offset)
from fires.plotting.plotfns import (plot_dpa, plot_ilv_pa_ds, plot_lv,
                                    plot_pa_li_scatter, plot_pa_profile,
                                    plot_pads, plot_qu, plot_stokes)
from fires.utils.loaders import (load_data, load_multiple_data_grouped,
                                 parse_override_key, scrunch_dspec)
from fires.utils.params import base_param_name, is_measured_key
from fires.utils.utils import normalise_freq_window, normalise_phase_window

logging.basicConfig(level=logging.INFO)
for _name in ("fontTools", "fontTools.subset"):
	_lg = logging.getLogger(_name)
	_lg.setLevel(logging.WARNING)   
	_lg.propagate = False

#	--------------------------	Set plot parameters	---------------------------
# [styling] key in plotparams.toml -> matplotlib rcParams key
_STYLING_RC_KEYS = {
	'pdf_fonttype': 'pdf.fonttype',
	'ps_fonttype': 'ps.fonttype',
	'savefig_dpi': 'savefig.dpi',
	'figure_dpi': 'figure.dpi',
	'font_size': 'font.size',
	'axes_labelsize': 'axes.labelsize',
	'axes_titlesize': 'axes.titlesize',
	'legend_fontsize': 'legend.fontsize',
	'xtick_labelsize': 'xtick.labelsize',
	'ytick_labelsize': 'ytick.labelsize',
	'font_family': 'font.family',
	'font_serif': 'font.serif',
	'mathtext_fontset': 'mathtext.fontset',
	'color_cycle': 'axes.prop_cycle',
	'line_width': 'lines.linewidth',
	'marker_size': 'lines.markersize',
	'axes_linewidth': 'axes.linewidth',
	'xtick_major_width': 'xtick.major.width',
	'ytick_major_width': 'ytick.major.width',
	'xtick_direction': 'xtick.direction',
	'ytick_direction': 'ytick.direction',
	'xtick_top': 'xtick.top',
	'ytick_right': 'ytick.right',
	'xtick_minor_visible': 'xtick.minor.visible',
	'ytick_minor_visible': 'ytick.minor.visible',
	'axes_axisbelow': 'axes.axisbelow',
	'constrained_layout': 'figure.constrained_layout.use',
	'legend_frameon': 'legend.frameon',
	'legend_handlelength': 'legend.handlelength',
}


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
	
	# Determine LaTeX usage
	latex_enabled = False
	if use_latex is not None:
		latex_enabled = bool(use_latex)
	elif 'use_latex' in general:
		latex_enabled = bool(general['use_latex'])
	else:
		# Fallback to environment variable
		latex_enabled = bool(int(os.environ.get("FIRES_USE_LATEX", "0")))

	# Start from publication defaults, then apply overrides from config.
	set_pub_style(use_latex=latex_enabled)

	# Apply [styling] overrides on top of the publication defaults
	for key, rc_key in _STYLING_RC_KEYS.items():
		if key in styling:
			value = styling[key]
			plt.rcParams[rc_key] = plt.cycler(color=value) if key == 'color_cycle' else value

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
			plot_func (callable): Function to generate the plot.
			requires_multiple_frb (bool): Whether this plot mode requires `plot_var=True`.
		"""
		self.name = name
		self.plot_func = plot_func
		self.requires_multiple_frb = requires_multiple_frb
		

# --------------------------	Plot modes definitions	---------------------------
def basic_plots(fname, frb_data, mode, out_dir, plot_config=None, buffer_frac=None, segments=None,**kwargs):
	"""
	Generate basic plots using configuration from plot_config.
	"""
	# Extract what we need from plot_config
	save = get_plot_param(plot_config, 'general', 'save_plots', False)
	figsize = get_plot_param(plot_config, 'general', 'figsize', None)
	show_plots = get_plot_param(plot_config, 'general', 'show_plots', True)
	extension = get_plot_param(plot_config, 'general', 'extension', 'pdf')
	legend = get_plot_param(plot_config, 'general', 'legend', True)
	xlim = get_plot_param(plot_config, 'general', 'xlim', None)
	ylim = get_plot_param(plot_config, 'general', 'ylim', None)
	inset = get_plot_param(plot_config, 'general', 'inset', False)
	show_spectrum = get_plot_param(plot_config, 'general', 'show_spectrum', True)
	t_scrunch = get_plot_param(plot_config, 'general', 't_scrunch', 1)
	f_scrunch = get_plot_param(plot_config, 'general', 'f_scrunch', 1)
	if t_scrunch is None:
		t_scrunch = get_plot_param(plot_config, 'general', 'tscrunch', 1)
	if f_scrunch is None:
		f_scrunch = get_plot_param(plot_config, 'general', 'fscrunch', 1)

	show_onpulse = get_plot_param(plot_config, 'windows', 'show_onpulse', False)
	show_offpulse = get_plot_param(plot_config, 'windows', 'show_offpulse', False)
	dspec_params = frb_data.dspec_params
	freq_mhz = dspec_params.freq_mhz
	time_ms = dspec_params.time_ms

	tau = dspec_params.gdict['tau']

	text = get_plot_param(plot_config, 'general', 'text', [])
	if text!= "None":
		display_text = build_plot_text_string(text, dspec_params.gdict)


	plot_dspec = frb_data.dynamic_spectrum
	if mode in ("all", "lv", "lvpa", "iquv", "stokes", "qu"):
		plot_dspec, freq_mhz, time_ms, dspec_params = scrunch_dspec(
			plot_dspec, freq_mhz, time_ms, dspec_params,
			t_scrunch=t_scrunch, f_scrunch=f_scrunch,
		)

	ts_data, corr_dspec, noise_spec, noise_stokes = process_dspec(
		plot_dspec, freq_mhz, dspec_params, buffer_frac, skip_rm=True, remove_pa_trend=True
	)

	iquvt = ts_data.iquvt
	
	if mode == "all":
		plot_ilv_pa_ds(corr_dspec, dspec_params, plot_config, freq_mhz, time_ms, save, fname, out_dir, 
				ts_data, figsize, tau, show_plots, extension, 
				legend, buffer_frac, show_onpulse, show_offpulse, segments=segments, display_text=display_text,
				inset=inset, show_spectrum=show_spectrum, t_scrunch=t_scrunch, f_scrunch=f_scrunch, noise_spec=noise_spec)
		plot_stokes(fname, out_dir, corr_dspec, iquvt, freq_mhz, time_ms, save, figsize, show_plots, extension,
		            dspec_params=dspec_params, plot_config=plot_config, display_text=display_text,
		            t_scrunch=t_scrunch, f_scrunch=f_scrunch, noise_spec=noise_spec)
		plot_pa_profile(fname, out_dir, ts_data, time_ms, save, figsize, show_plots, extension, xlim, ylim)
		plot_dpa(fname, out_dir, noise_stokes, ts_data, time_ms, 5, save, figsize, show_plots, extension)
		estimate_rm(frb_data.dynamic_spectrum, freq_mhz, time_ms, noise_spec, 1.0e3, 1.0, out_dir, save, show_plots)
		plot_pa_li_scatter(fname,out_dir,ts_data,time_ms,noise_stokes,save,figsize,show_plots,extension)
		plot_pads(corr_dspec, freq_mhz, time_ms, save, fname, out_dir, figsize, show_plots, extension,
		          plot_config=plot_config, display_text=display_text)
	elif mode in ("iquv", "stokes"):
		plot_stokes(
			fname, out_dir, corr_dspec, iquvt, freq_mhz, time_ms, save, figsize,
			show_plots, extension, dspec_params=dspec_params, plot_config=plot_config,
			display_text=display_text, t_scrunch=t_scrunch, f_scrunch=f_scrunch,
			noise_spec=noise_spec,
		)
	elif mode == "qu":
		plot_qu(
			corr_dspec, dspec_params, plot_config, freq_mhz, time_ms, save, fname, out_dir,
			ts_data, figsize, tau, show_plots, extension, legend, buffer_frac,
			show_onpulse, show_offpulse, segments=segments, display_text=display_text,
			show_spectrum=show_spectrum, t_scrunch=t_scrunch, f_scrunch=f_scrunch,
		)
	elif mode == "lv":
		plot_lv(corr_dspec, dspec_params, plot_config, freq_mhz, time_ms, save, fname, out_dir,
				ts_data, figsize, tau, show_plots, extension,
				legend, buffer_frac, show_onpulse, show_offpulse, segments=segments, display_text=display_text,
				inset=inset, show_spectrum=show_spectrum, t_scrunch=t_scrunch, f_scrunch=f_scrunch, noise_spec=noise_spec)
	elif mode == "lvpa":
		plot_ilv_pa_ds(corr_dspec, dspec_params, plot_config, freq_mhz, time_ms, save, fname, out_dir, 
				ts_data, figsize, tau, show_plots, extension, 
				legend, buffer_frac, show_onpulse, show_offpulse, segments=segments, display_text=display_text,
				inset=inset, show_spectrum=show_spectrum, t_scrunch=t_scrunch, f_scrunch=f_scrunch, noise_spec=noise_spec)
	elif mode == "pa":
			plot_pa_profile(fname, out_dir, ts_data, time_ms, save, figsize, show_plots, extension, xlim, ylim)
	elif mode == "dpa":
		plot_dpa(fname, out_dir, noise_stokes, ts_data, time_ms, 5, save, figsize, show_plots, extension)
	elif mode == "RM":
		estimate_rm(frb_data.dynamic_spectrum, freq_mhz, time_ms, noise_spec, 1.0e3, 1.0, out_dir, save, show_plots)
	elif mode == "pali":
		plot_pa_li_scatter(fname,out_dir,ts_data,time_ms,noise_stokes,save,figsize,show_plots,extension)
	elif mode == "pads":
		plot_pads(corr_dspec, freq_mhz, time_ms, save, fname, out_dir, figsize, show_plots, extension,
		          plot_config=plot_config, display_text=display_text)
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


_OVERLAY_STYLE_KEYS = ('marker', 'size', 'edgecolor', 'linewidth', 'alpha', 'error_style',
					   'span_alpha', 'bar_alpha', 'bar_capsize', 'label_prefix')


def _overlay_style(obs_cfg):
	"""Keyword arguments for _plot_observational_overlay taken from an observational config."""
	return {k: obs_cfg[k] for k in _OVERLAY_STYLE_KEYS if k in obs_cfg}


def _gdict_of(frb_dict):
	"""The run's gdict: from its dspec_params if present, else a top-level 'gdict' entry."""
	dsp = frb_dict.get("dspec_params")
	if dsp is not None and hasattr(dsp, "gdict"):
		return dsp.gdict
	return frb_dict.get("gdict")


def _draw_text_items(ax, items, gdict, kind, plot_config):
	"""Draw the parameter text for ``items`` ('general' text or 'analytical' param_text)."""
	if items:
		draw_plot_text(ax, build_plot_text_string(items, gdict), kind, plot_config)


def _window_colour(freq_window, phase_window, default=colours['purple']):
	"""Series colour for a (freq, phase) window from colour_map."""
	key = f"{normalise_freq_window(freq_window, target='dspec')}, {normalise_phase_window(phase_window, target='dspec')}"
	return colour_map.get(key, default)


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
			#print("med:", np.nanmedian(v_arr), "avg:", np.nanmean(v_arr), "min:", np.nanmin(v_arr), "max:", np.nanmax(v_arr))
			med_vals.append(np.nanmedian(v_arr))
			lower = np.nanpercentile(v_arr, p_low)
			upper = np.nanpercentile(v_arr, p_high)
			percentile_errs.append((lower, upper))
		else:
			med_vals.append(np.nan)
			percentile_errs.append((np.nan, np.nan))

	return med_vals, percentile_errs


def _segment_values(seg_list, quantity, phase_window, freq_window):
	"""Finite values of a measured quantity across the realisations (segment dicts) at one sweep step."""
	vals = []
	for seg in seg_list:
		val = _extract_value_from_segments(seg, quantity, phase_window, freq_window)
		if np.isfinite(val):
			vals.append(val)
	return vals


def _x_percentiles_measured(frb_dict, x_measured: str, p_low=16, p_high=84,
							phase_window='total', freq_window='all'):
	"""
	Compute per-parameter-step percentiles of a measured x quantity for the SAME
	(freq_window, phase_window) used on the x-axis, returning (x_med, x_lo, x_hi).
	"""
	measures = frb_dict.get("measures", {})
	x_med = []
	x_lo = []
	x_hi = []
	for xv in np.array(frb_dict["xvals"]):
		vals = _segment_values(measures.get(xv, []), x_measured, phase_window, freq_window)
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
	def result(vals, applied):
		return (vals, applied) if return_status else vals

	def unweighted():
		return {var: yvals.get(var, []) for var in xvals}

	normalised_vals = {}
	applied = False

	if weight_by is None:
		return result(unweighted(), False)

	if isinstance(weight_params, dict) and any(isinstance(v, dict) for v in weight_params.values()):
		if not any(weight_by in weight_params.get(var, {}) for var in xvals):
			logging.warning(f"Weighting parameter '{weight_by}' not found in weight dictionaries. Returning unweighted values.")
			return result(unweighted(), False)

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
			normalised_vals = unweighted()
	else:
		logging.warning("Unsupported weight_params format. Returning unweighted values.")
		normalised_vals = unweighted()

	return result(normalised_vals, applied)


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


def _parse_xshade_ranges(xshade):
	"""
	Parse x-shading ranges from config.

	Accepted formats:
	- list/tuple of 2-item ranges: [[x0, x1], [x2, x3]]
	- list of dict entries: [{range=[x0, x1], colour="...", alpha=0.2, zorder=0}, ...]
	- TOML array-of-arrays: [[0, 1], [2, 3]]
	- string representation of either form.
	"""
	if xshade is None:
		return []

	parsed = xshade
	if isinstance(xshade, str):
		s = xshade.strip()
		if not s:
			return []
		try:
			parsed = ast.literal_eval(s)
		except Exception:
			logging.warning("analytical.xshade could not be parsed from string; ignoring.")
			return []

	if not isinstance(parsed, (list, tuple)):
		logging.warning("analytical.xshade must be a list/tuple of [xmin, xmax] ranges.")
		return []

	ranges = []
	for idx, item in enumerate(parsed):
		entry_colour = None
		entry_alpha = None
		entry_zorder = None
		rng = item

		if isinstance(item, dict):
			rng = item.get('range', item.get('xrange', None))
			entry_colour = item.get('colour', item.get('color', None))
			entry_alpha = item.get('alpha', None)
			entry_zorder = item.get('zorder', None)

		if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
			logging.warning(f"Skipping invalid xshade entry at index {idx}: {item}")
			continue
		try:
			x0 = float(rng[0])
			x1 = float(rng[1])
		except Exception:
			logging.warning(f"Skipping non-numeric xshade entry at index {idx}: {item}")
			continue
		if not (np.isfinite(x0) and np.isfinite(x1)):
			logging.warning(f"Skipping non-finite xshade entry at index {idx}: {item}")
			continue
		if x0 == x1:
			continue

		if entry_alpha is not None:
			try:
				entry_alpha = float(entry_alpha)
			except Exception:
				entry_alpha = None
		if entry_zorder is not None:
			try:
				entry_zorder = int(entry_zorder)
			except Exception:
				entry_zorder = None

		ranges.append({
			'lo': min(x0, x1),
			'hi': max(x0, x1),
			'colour': entry_colour,
			'alpha': entry_alpha,
			'zorder': entry_zorder,
		})

	return ranges


def _apply_xshade_regions(ax, xshade, colour='lightgray', alpha=0.25, zorder=0):
	"""Apply vertical shaded x-ranges to an axis."""
	ranges = _parse_xshade_ranges(xshade)
	if not ranges:
		return

	xscale = ax.get_xscale()
	for item in ranges:
		lo = item['lo']
		hi = item['hi']
		item_colour = item.get('colour') if item.get('colour') is not None else colour
		item_alpha = item.get('alpha') if item.get('alpha') is not None else alpha
		item_zorder = item.get('zorder') if item.get('zorder') is not None else zorder
		if xscale == 'log':
			if hi <= 0:
				continue
			if lo <= 0:
				lo = np.nextafter(0.0, 1.0)
		if lo < hi:
			ax.axvspan(lo, hi, color=item_colour, alpha=item_alpha, zorder=item_zorder, linewidth=0)


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
	quantity = {'pa_var': 'Vpsi', 'l_frac': 'Lfrac'}.get(plot_type, plot_type)
	yvals = {}
	for xv in xvals:
		vals = _segment_values(measures.get(xv, []), quantity, phase_window, freq_window)
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
		return y_model, ("Broken power\n"
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


# Axis symbol and unit for each measured quantity that can go on the x-axis
_MEASURED_X_LABELS = {
	'Vpsi': (r"\mathbb{V}(\psi)", r"\mathrm{deg}^2"),
	'Lfrac': (r"\Pi_{L}", ""),
	'Vfrac': (r"\Pi_{V}", ""),
}


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
		measures = frb_dict.get("measures", {})
		phase_window = phase_window if phase_window is not None else 'total'
		freq_window = freq_window if freq_window is not None else 'all'
		x = np.array([_median_or_nan(_segment_values(measures.get(xv, []), x_measured, phase_window, freq_window))
					  for xv in np.array(frb_dict["xvals"])], dtype=float)
		if x_measured not in _MEASURED_X_LABELS:
			raise ValueError(f"Unknown x_measured: {x_measured}")
		xname, x_unit = _MEASURED_X_LABELS[x_measured]
		return x, xname, x_unit
	
	# Original behavior: use input parameter sweep
	xname_raw = frb_dict["xname"]
	xvals_raw = np.array(frb_dict["xvals"])

	dspec_params = frb_dict.get("dspec_params", None)
	V_params = frb_dict.get("V_params", None)

	sweep_mode = _sweep_mode(dspec_params)

	param_info_entry = param_map.get(xname_raw, (xname_raw, ""))
	base_name = param_info_entry[0] if isinstance(param_info_entry, tuple) else param_info_entry
	base_unit = param_info_entry[1] if isinstance(param_info_entry, tuple) else ""

	if sweep_mode == "sd":
		x = xvals_raw
		sym, unit = param_info_or_dynamic(f"sd_{xname_raw}")
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
	weight_symbol, weight_unit = param_info_or_dynamic(weight_x_by)
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

	w_sym, w_unit = param_info_or_dynamic(weight_y_by)

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
	if isinstance(fit_item, str):
		fit_item = fit_item.split(',', 1)
	if not isinstance(fit_item, (list, tuple)):
		return None, None
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
	med_low = med(lowest)
	med_high = med(highest)
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
		exp2.append(np.nan if e2 is None else float(e2))

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
	# Only the second ("basic") expected series is drawn
	if exp2 is not None and np.any(np.isfinite(exp2)):
		ax.plot(x, exp2, 'k:', linewidth=2.0, label='Expected')
	else:
		logging.info(f"No valid expected values found for '{param_key}' to plot.")


def _parse_override_dict(override_str) -> dict:
	"""
	Parse an override string into a dict of param -> value.
	"""
	od = {}
	if not override_str:
		return od
	if ('=' in override_str) or ('+' in override_str):
		return parse_override_key(override_str)
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
			sym, _ = param_info_or_dynamic(k)
			try:
				s = str(int(val)) if float(val).is_integer() else f"{float(val):g}"
			except Exception:
				s = str(val)
			parts.append(rf"{sym}={s}")
	return ", ".join(parts)


def _weight_source(run_data, weight_y_by):
	"""Where y weights live: V_params for measured keys (meas_var_*), otherwise dspec_params."""
	if isinstance(weight_y_by, str) and is_measured_key(weight_y_by):
		return run_data.get("V_params", {})
	return run_data.get("dspec_params", {})


def _run_x_and_yvals(run_data, weight_x_by, weight_y_by, x_measured, plot_type, phase_window, freq_window):
	"""For one run: the plotted x values, and the (weighted) y samples keyed by sweep value."""
	x_vals, _, _ = _weight_x_get_xname(
		run_data, weight_x_by=weight_x_by, x_measured=x_measured,
		phase_window=phase_window, freq_window=freq_window
	)
	xvals = run_data["xvals"]
	yvals = _yvals_from_measures_dict(xvals, run_data["measures"], plot_type, phase_window, freq_window)
	if weight_y_by is not None:
		yvals, _ = _weight_dict(xvals, yvals, _weight_source(run_data, weight_y_by), weight_by=weight_y_by, return_status=True)
	return x_vals, yvals


def _x_grid(ax, n):
	"""At least 10 points spanning the current x-limits, log-spaced on a log axis."""
	xlim = ax.get_xlim()
	x_min, x_max = min(xlim), max(xlim)
	n = int(max(10, n))
	if ax.get_xscale() == 'log':
		x_min = max(x_min, 1e-300)
		x_max = max(x_max, x_min * (1 + 1e-6))
		return np.geomspace(x_min, x_max, n)
	return np.linspace(x_min, x_max, n)


def _extend_piecewise(x, xs, ys, mode='linear'):
	"""Piecewise-linear interpolation through (xs, ys), extrapolated linearly or flat beyond the ends."""
	x = np.asarray(x, dtype=float)
	y = np.interp(x, xs, ys)
	if xs.size >= 2:
		if mode == 'linear':
			mL = (ys[1] - ys[0]) / (xs[1] - xs[0])
			mR = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
			left_mask = (x < xs[0])
			right_mask = (x > xs[-1])
			y[left_mask] = ys[0] + mL * (x[left_mask] - xs[0])
			y[right_mask] = ys[-1] + mR * (x[right_mask] - xs[-1])
		else:
			y[x < xs[0]] = ys[0]
			y[x > xs[-1]] = ys[-1]
	return y


def _label_index(xa, ya, position, fallback):
	"""Index along a drawn line at which to anchor its label ('start', 'end', 'max-y', 'mid')."""
	if position == 'start':
		return 0
	if position == 'end':
		return -1
	if position == 'max-y':
		return int(np.nanargmax(ya)) if ya.size > 0 else -1
	if position == 'mid':
		return int(len(xa) // 2) if xa.size > 0 else -1
	return fallback


def _first_float(v):
	"""A scalar, or the first element of a sequence, as float."""
	return float(v[0] if isinstance(v, (list, tuple, np.ndarray)) else v)


def _median_or_nan(vals):
	return np.nanmedian(vals) if vals else np.nan


def _select_nearest(sorted_vals, desired_vals):
	"""For each desired value pick the nearest available one, avoiding repeats where a neighbour is free."""
	if not sorted_vals or not desired_vals:
		return []
	avail = np.asarray(sorted_vals, dtype=float)
	out = []
	used = set()
	for dv in desired_vals:
		idx = int(np.nanargmin(np.abs(avail - float(dv))))
		chosen = float(avail[idx])
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
	seen = set()
	return [v for v in out if not (v in seen or seen.add(v))]


def _equal_line_colours(vals, mode, base, cmap_nm):
	"""One colour per line: sampled from a colormap ('cmap') or darker-to-lighter shades of base."""
	n = len(vals)
	if n == 0:
		return []
	if mode == 'cmap':
		cmap = cm.get_cmap(cmap_nm)
		vmin, vmax = min(vals), max(vals)
		if vmax == vmin:
			return [cmap(0.5)] * n
		return [cmap((v - vmin) / (vmax - vmin)) for v in vals]
	base_rgb = np.array(mcolors.to_rgb(base))
	out = []
	for f in np.linspace(0.55, 1.25, n):
		col = base_rgb * f if f <= 1 else base_rgb + (1 - base_rgb) * (f - 1.0)
		out.append(np.clip(col, 0, 1))
	return out


def _safe_exp(z):
	return np.exp(np.clip(z, -700.0, 700.0))


def _model_power(x, a, n):
	return a * np.power(x, n)


def _model_broken_power(x, a, n1, n2, x_break):
	return np.where(x < x_break, a * x**n1, a * x_break**(n1-n2) * x**n2)


def _equal_line_curve(xd, yd, fit, interpolate, interp_kind, extend_mode):
	"""
	Callable g(x) through the sorted points (xd, yd) of one equal-value line, plus a short
	description. Uses the requested fit when it applies and succeeds, else interpolation
	when asked, else a polyline extended linearly or flat past the ends.
	"""
	g = None
	desc = None
	if fit is not None:
		try:
			fit_type, fit_degree = _parse_fit_arg(fit)
			if fit_type == 'linear' and xd.size >= 2:
				p = np.polyfit(xd, yd, 1)
				g = lambda x: np.polyval(p, x)
				desc = 'linear'
			elif fit_type == 'poly' and xd.size >= max(2, (fit_degree or 2) + 1):
				deg = fit_degree if (fit_degree and fit_degree >= 1) else 2
				deg = min(deg, max(1, xd.size - 1))
				p = np.polyfit(xd, yd, deg)
				g = lambda x, p=p: np.polyval(p, x)
				desc = f'poly-{deg}'
			elif fit_type == 'exp' and xd.size >= 2 and np.all(yd > 0):
				p = np.polyfit(xd, np.log(yd), 1)
				g = lambda x, p=p: _safe_exp(p[0] * x + p[1])
				desc = 'exp'
			elif fit_type in ('power', 'power_fixed_n') and xd.size >= 2 and np.all(yd > 0) and np.all(xd > 0):
				if fit_degree is not None:
					nf = float(fit_degree)
					a_hat = np.exp(np.nanmean(np.log(yd) - nf * np.log(xd)))
					g = lambda x, a=a_hat, nf=nf: _model_power(np.clip(x, 1e-300, None), a, nf)
					desc = f'power-n{nf}'
				else:
					n_hat, ln_a_hat = np.polyfit(np.log(xd), np.log(yd), 1)
					g = lambda x, a=np.exp(ln_a_hat), n=n_hat: _model_power(np.clip(x, 1e-300, None), a, n)
					desc = 'power'
			elif fit_type == 'log' and xd.size >= 2 and np.all(xd > 0):
				p = np.polyfit(np.log10(xd), yd, 1)
				g = lambda x, p=p: np.polyval(p, np.log10(np.clip(x, 1e-300, None)))
				desc = 'log'
			elif fit_type == 'broken-power' and xd.size >= 4 and np.all(xd > 0):
				p0 = [np.nanmax(yd), 1.0, 0.0, np.median(xd)]
				bounds = ([0, -np.inf, -np.inf, np.min(xd)], [np.inf, np.inf, np.inf, np.max(xd)])
				popt, _ = curve_fit(_model_broken_power, xd, yd, p0=p0, bounds=bounds, maxfev=20000)
				g = lambda x, p=popt: _model_broken_power(np.clip(x, 1e-300, None), *p)
				desc = 'broken-power'
		except Exception as e:
			logging.debug(f"Equal-line fit callable failed: {e}")
			g = None

	if g is None and interpolate and len(xd) >= 2:
		try:
			from scipy.interpolate import interp1d
			kind = interp_kind
			if kind == 'cubic' and len(xd) < 4:
				kind = 'quadratic'
			if kind == 'quadratic' and len(xd) < 3:
				kind = 'linear'
			f = interp1d(xd, yd, kind=kind, bounds_error=False, fill_value='extrapolate', assume_sorted=True)
			g = lambda x, f=f: f(x)
			desc = f'interp-{kind}'
		except Exception as e:
			logging.debug(f"Equal-line interpolation failed: {e}")
			g = None

	if g is None:
		g = lambda x, xs=xd, ys=yd: _extend_piecewise(x, xs, ys, extend_mode)
		desc = f'polyline-{extend_mode}'
	return g, desc


def _plot_equal_value_lines(ax, frb_dict, target_param, weight_x_by=None, weight_y_by=None,
							target_values=None, n_lines=5, linestyle='-', alpha=1,
							show_labels=True, zorder=0, plot_type='pa_var', phase_window='total',
							freq_window='full-band', x_measured=None,
							interpolate=False, interp_kind='quadratic', interp_points=100, fit=None, fit_points=200,
							colour_mode='shades', base_colour='#005bff', cmap_name='viridis',
							extend_mode='linear',
							label_position='start', label_offset_pts=(0, 0), label_ha='left', label_va='top',
							dynamic_update=True, max_draw_points=512):
	"""
	Across a multi-run dict, join the points that share one value of the swept parameter
	(e.g. all runs at N=10) and draw them as background lines labelled with that value.

	Lines are drawn across the full visible x-range. With dynamic_update=True (default) an
	axes callback recomputes them whenever the x-limits change (zoom/pan/set_xlim).
	"""
	all_xvals = set()
	for run_data in frb_dict.values():
		for xv in run_data["xvals"]:
			if np.isfinite(xv):
				all_xvals.add(float(xv))
	if not all_xvals:
		logging.warning("No valid xvals found. Cannot plot equal-value lines.")
		return

	sorted_values = sorted(all_xvals)
	if target_values is None:
		if len(sorted_values) <= n_lines:
			selected_values = sorted_values
		else:
			indices = np.linspace(0, len(sorted_values) - 1, n_lines, dtype=int)
			selected_values = [sorted_values[i] for i in indices]
	else:
		selected_values = _select_nearest(sorted_values, list(target_values))
		if not selected_values:
			logging.warning("No selectable values for equal-value lines from target_values; skipping.")
			return

	colours_for_lines = _equal_line_colours(selected_values, colour_mode, base_colour, cmap_name)
	logging.info(f"Plotting {len(selected_values)} equal-{target_param} lines with colour_mode='{colour_mode}'\n")

	n_draw = min(max_draw_points, fit_points if fit else (interp_points if interpolate else max_draw_points))
	label_pos = (label_position or 'start').lower()

	# Keep handles and updaters alive (avoid GC)
	if not hasattr(ax, '_equal_lines_dynamic'):
		ax._equal_lines_dynamic = []

	for (xval, line_colour) in zip(selected_values, colours_for_lines):
		# Median of each run at this constant parameter value
		x_points = []
		y_points = []
		for run_data in frb_dict.values():
			xvals = run_data["xvals"]
			xval_indices = np.where(np.isclose(xvals, xval, rtol=1e-6))[0]
			if len(xval_indices) == 0:
				continue
			x_vals, yvals_weighted = _run_x_and_yvals(run_data, weight_x_by, weight_y_by, x_measured,
													  plot_type, phase_window, freq_window)
			for idx in xval_indices:
				x_coord = x_vals[idx] if idx < len(x_vals) else np.nan
				y_coord = _median_or_nan(yvals_weighted.get(xvals[idx], []))
				if np.isfinite(x_coord) and np.isfinite(y_coord):
					x_points.append(x_coord)
					y_points.append(y_coord)

		if len(x_points) < 2:
			continue

		sorted_indices = np.argsort(x_points)
		g, fit_desc = _equal_line_curve(np.array(x_points)[sorted_indices], np.array(y_points)[sorted_indices],
										fit, interpolate, interp_kind, extend_mode)

		x_draw = _x_grid(ax, n_draw)
		y_draw = np.asarray(g(x_draw), dtype=float)
		line, = ax.plot(x_draw, y_draw, linestyle=linestyle, alpha=alpha, color=line_colour, linewidth=1.8, zorder=zorder)

		txt = None
		if show_labels:
			param_symbol, _ = param_info_or_dynamic(target_param)
			val_str = f"{int(xval)}" if xval == int(xval) else f"{xval:.2g}"
			ix = _label_index(x_draw, y_draw, label_pos, 0)
			txt = text_with_offset(
				ax, float(x_draw[ix]), float(y_draw[ix]),
				rf"${param_symbol} = {val_str}$",
				dx_pts=float(label_offset_pts[0]) if label_offset_pts else 0.0,
				dy_pts=float(label_offset_pts[1]) if label_offset_pts else 0.0,
				ha=label_ha, va=label_va,
				color=line_colour, fontsize=18, alpha=min(alpha + 0.2, 1.0),
				zorder=max(zorder + 1, 10)
			)

		def _update_equal_line(event_ax=None, _line=line, _text=txt, _g=g):
			try:
				x_new = _x_grid(ax, n_draw)
				y_new = np.asarray(_g(x_new), dtype=float)
				_line.set_data(x_new, y_new)
				if _text is not None and y_new.size > 0 and np.all(np.isfinite(y_new)):
					ii = _label_index(x_new, y_new, label_pos, 0)
					_text.set_position((float(x_new[ii]), float(y_new[ii])))
			except Exception as e:
				logging.debug(f"Equal-line update failed: {e}")

		if dynamic_update:
			cid = ax.callbacks.connect('xlim_changed', _update_equal_line)
			_update_equal_line()  # match any autoscale done after plotting
			ax._equal_lines_dynamic.append({'cid': cid, 'update': _update_equal_line, 'line': line, 'text': txt, 'desc': fit_desc})


def plot_constant_param_lines(
	ax,
	param_data,
	x_key='vpsi',
	y_key='lfrac',
	color='gray',
	alpha=0.5,
	linestyle='--',
	label_fmt=r"${param} = {val}$",
	param_name=r"\sigma_\psi",
	extend_mode='linear',          # 'linear' or 'flat' extrapolation
	label_position='end',          # 'start' | 'end' | 'max-y' | 'mid'
	label_offset_pts=(0, 0),
	dynamic_update=True,
	linewidth=1.5,
	max_draw_points=512
):
	"""
	Overlay lines of constant parameter on a 2D plot and keep them extended across
	the visible axes even after xlim/ylim changes (via callbacks).

	param_data: dict
		{ param_value: {x_key: [...], y_key: [...]}, ... }
	"""
	label_pos = (label_position or 'end').lower()

	# Keep handles and callbacks alive (avoid GC)
	if not hasattr(ax, '_const_param_lines_dynamic'):
		ax._const_param_lines_dynamic = []

	# Deterministic order
	for param_val in sorted(param_data.keys(), key=lambda v: (isinstance(v, str), v)):
		vals = param_data[param_val]
		x = np.asarray(vals.get(x_key, []), dtype=float)
		y = np.asarray(vals.get(y_key, []), dtype=float)
		m = np.isfinite(x) & np.isfinite(y)
		if np.sum(m) < 2:
			continue

		order = np.argsort(x[m])
		x_ordered = x[m][order]
		y_ordered = y[m][order]
		x_sorted = np.unique(x_ordered)
		if x_sorted.size != x_ordered.size:
			# Duplicate x: take the median y at each x
			inv = np.unique(x_ordered, return_inverse=True)[1]
			y_accum = [[] for _ in range(x_sorted.size)]
			for xi, yi in zip(inv, y_ordered):
				y_accum[xi].append(yi)
			y_sorted = np.array([np.nanmedian(v) if v else np.nan for v in y_accum], dtype=float)
		else:
			y_sorted = y_ordered

		if x_sorted.size < 2:
			continue

		def _g(xq, xs=x_sorted, ys=y_sorted):
			return _extend_piecewise(xq, xs, ys, mode=extend_mode)

		x_draw = _x_grid(ax, max_draw_points)
		y_draw = np.asarray(_g(x_draw), dtype=float)
		line, = ax.plot(x_draw, y_draw, color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth, label=None)

		if isinstance(param_val, (int, float)):
			val_str = f"{int(param_val)}" if float(param_val).is_integer() else f"{param_val:.2g}"
		else:
			val_str = str(param_val)
		ix = _label_index(x_draw, y_draw, label_pos, -1)
		txt = text_with_offset(
			ax, float(x_draw[ix]), float(y_draw[ix]), label_fmt.format(param=param_name, val=val_str),
			dx_pts=float(label_offset_pts[0]) if label_offset_pts else 0.0,
			dy_pts=float(label_offset_pts[1]) if label_offset_pts else 0.0,
			ha='left', va='bottom',
			color=color, fontsize=15, alpha=min(alpha + 0.2, 1.0),
			zorder=max(line.get_zorder() + 1, 10),
		)

		def _update_const_line(event_ax=None, _line=line, _text=txt, _g=_g):
			try:
				x_new = _x_grid(ax, max_draw_points)
				y_new = np.asarray(_g(x_new), dtype=float)
				_line.set_data(x_new, y_new)
				if _text is not None and y_new.size > 0 and np.all(np.isfinite(y_new)):
					ii = _label_index(x_new, y_new, label_pos, -1)
					_text.set_position((float(x_new[ii]), float(y_new[ii])))
			except Exception:
				pass

		if dynamic_update:
			cid1 = ax.callbacks.connect('xlim_changed', _update_const_line)
			cid2 = ax.callbacks.connect('ylim_changed', _update_const_line)  # reposition label if needed
			_update_const_line()
			ax._const_param_lines_dynamic.append({'cids': (cid1, cid2), 'update': _update_const_line, 'line': line, 'text': txt, 'param': param_val})


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
			x_vals, yvals_weighted = _run_x_and_yvals(run_data, weight_x_by, weight_y_by, x_measured,
													  plot_type, phase_window, freq_window)
			x_vals = np.asarray(x_vals, dtype=float)
			y_med = np.asarray([_median_or_nan(yvals_weighted.get(xv, [])) for xv in run_data["xvals"]], dtype=float)

			m = np.isfinite(x_vals) & np.isfinite(y_med)
			if not np.any(m):
				continue
			xf, yf = x_vals[m], y_med[m]

			pos = (position or 'max-y').lower()
			if pos in ('start', 'min-x'):
				idx = int(np.nanargmin(xf))
			elif pos in ('end', 'max-x'):
				idx = int(np.nanargmax(xf))
			elif pos == 'min-y':
				idx = int(np.nanargmin(yf))
			elif pos == 'mid':
				idx = int(len(xf) // 2)
			else:  # 'max-y' (default)
				idx = int(np.nanargmax(yf))

			# Label text from the run's gdict, else its V_params
			gd = None
			dsp = run_data.get("dspec_params")
			if dsp is not None and hasattr(dsp, "gdict"):
				gd = dsp.gdict
			vp = run_data.get("V_params", {})
			label_parts = []
			for pname in params_to_label:
				sym, _ = param_info_or_dynamic(pname)
				val = None
				if gd and pname in gd and gd[pname]:
					try:
						val = _first_float(gd[pname])
					except Exception:
						pass
				if val is None and pname in vp and vp[pname] is not None:
					try:
						val = _first_float(vp[pname])
					except Exception:
						pass
				if val is None:
					continue
				val_str = f"{int(val)}" if val == int(val) else f"{val:g}"
				label_parts.append(rf"{sym}={val_str}")
			if not label_parts:
				continue

			txt = text_with_offset(
				ax, float(xf[idx]), float(yf[idx]), r"$" + r",\; ".join(label_parts) + r"$",
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


def _percentile_cfg(plot_config, x_measured):
	"""Settings from [analytical.percentiles]: percentile range, y band, and optional x band."""
	cfg = get_plot_param(plot_config, 'analytical', 'percentiles', {}) or {}
	enabled = bool(cfg.get('enabled', True))
	y_cfg = cfg.get('y', {}) or {}
	x_cfg = cfg.get('x', {}) or {}
	return SimpleNamespace(
		p_low=float(cfg.get('p_low', 16)),
		p_high=float(cfg.get('p_high', 84)),
		y_shade=bool(y_cfg.get('shade', True)) and enabled,
		y_alpha=float(y_cfg.get('alpha', 0.2)),
		x_enabled=bool(x_cfg.get('enabled', False)) and enabled and (x_measured is not None),
		x_style=str(x_cfg.get('style', 'bars')).lower(),
		x_alpha=float(x_cfg.get('alpha', 0.12)),
	)


def _median_band(yvals, xvals, pct):
	"""Median per x and the lower/upper percentile arrays."""
	med_vals, percentile_errs = _median_percentiles(yvals, xvals, p_low=pct.p_low, p_high=pct.p_high)
	lower = np.array([lo for (lo, hi) in percentile_errs])
	upper = np.array([hi for (lo, hi) in percentile_errs])
	return med_vals, lower, upper


def _draw_series(ax, x, med, lo, hi, x_lo, x_hi, pct, draw_style, nbins, colour, label, colour_by_sweep):
	"""
	Draw one median curve with its percentile band(s) in the requested draw_style
	('line-param', 'line-x' sorted by x, 'scatter' with error bars, or 'binned').

	Returns the (x, median) arrays as drawn, i.e. after sorting or binning.
	"""
	med = np.asarray(med, dtype=float)
	if draw_style == 'line-x':
		order = np.argsort(x)
		x, med, lo, hi = x[order], med[order], lo[order], hi[order]
		if x_lo is not None and x_hi is not None:
			x_lo, x_hi = x_lo[order], x_hi[order]
	elif draw_style == 'binned':
		x, med, lo, hi = _bin_xy(x, med, nbins=nbins, strategy='equal-count')
		x_lo = x_hi = None  # x percentiles cannot be carried through the binning

	if draw_style == 'scatter':
		yerr = np.vstack([(med - lo), (hi - med)])
		xerr = None
		if pct.x_enabled and x_lo is not None and x_hi is not None:
			xerr = np.vstack([np.maximum(x - x_lo, 0), np.maximum(x_hi - x, 0)])
		ax.errorbar(x, med, yerr=yerr, xerr=xerr, fmt='none', ecolor='gray' if colour_by_sweep else colour,
					alpha=0.6, elinewidth=1.2, capsize=2, zorder=1)
		if colour_by_sweep:
			ax.scatter(x, med, c=np.arange(len(x)), cmap='viridis', s=25, alpha=0.8, label=label, zorder=2)
		else:
			ax.scatter(x, med, color=colour, s=25, alpha=0.8, label=label, zorder=2)
	else:
		ax.plot(x, med, color=colour, label=label, linewidth=2)
		if pct.y_shade:
			ax.fill_between(x, lo, hi, color=colour, alpha=pct.y_alpha)
		if pct.x_enabled and pct.x_style == 'shade':
			m = np.isfinite(med) & np.isfinite(x)
			m &= np.isfinite(x_lo) & np.isfinite(x_hi)
			if np.any(m):
				order = np.argsort(med[m])  # sort by y for fill_betweenx
				ax.fill_betweenx(med[m][order], x_lo[m][order], x_hi[m][order], color=colour, alpha=pct.x_alpha)
	return x, med


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
	plot_config=None,
	param_text=None,
	text=None
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
	_print_med_snrs(frb_dict)
	weight_source = _weight_source(frb_dict, weight_y_by)

	n_combos = len(window_pairs)
	
	freq_windows = [pair[0] for pair in window_pairs]
	phase_windows = [pair[1] for pair in window_pairs]
	
	varying_freq = len(set(freq_windows)) > 1
	varying_phase = len(set(phase_windows)) > 1
	
	phase_colours = {'leading': colours['orange'], 'trailing': colours['green'], 'total': colours['purple']}

	def get_colour(freq_win, phase_win):
		phase_label = normalise_phase_window(phase_win, target='dspec')
		if varying_phase and not varying_freq:
			return phase_colours.get(phase_label, colours['blue'])
		key = f"{normalise_freq_window(freq_win, target='dspec')}, {phase_label}"
		if key in colour_map:
			return colour_map[key]
		if not (varying_freq and varying_phase):
			return colours['blue'] if varying_freq else colours['purple']
		# Both varying and not in colour_map: cycle through the base colours
		idx = window_pairs.index((freq_win, phase_win))
		if n_combos <= len(colours):
			return list(colours.values())[idx % len(colours)]
		return cm.get_cmap('tab20', n_combos)(idx)

	def window_label(freq_win, phase_win):
		"""'Freq, Phase' with only the parts that vary between the compared windows."""
		freq_label = normalise_freq_window(freq_win, target='dspec').capitalize()
		phase_label = normalise_phase_window(phase_win, target='dspec').capitalize()
		if varying_freq and not varying_phase:
			return freq_label
		if varying_phase and not varying_freq:
			return phase_label
		return f"{freq_label}, {phase_label}"

	pct = _percentile_cfg(plot_config, x_measured)

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
		
		med_vals, lower, upper = _median_band(y_weighted, xvals, pct)

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
		if pct.x_enabled:
			_, x_lo, x_hi = _x_percentiles_measured(
				frb_dict, x_measured, p_low=pct.p_low, p_high=pct.p_high,
				phase_window=phase_win, freq_window=freq_win
			)

		colour = get_colour(freq_win, phase_win)
		_draw_series(ax, x, med_vals, lower, upper, x_lo, x_hi, pct, draw_style, nbins,
					 colour, window_label(freq_win, phase_win), colour_by_sweep)

		if fit is not None:
			fit_type, fit_degree = _parse_fit_arg(fit)
			_fit_and_plot(ax, x, med_vals, fit_type, fit_degree, label=None, colour=colour)

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
				print(f"Plotting expected {plot_type} from {param_key}...")
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
					if varying_freq or varying_phase:
						obs_result['label'] = f"{obs_result['label']} ({window_label(freq_win, phase_win)})"
					obs_colour = colour if obs_cfg.get('use_series_colour', False) else obs_cfg.get('colour', colour)
					_plot_observational_overlay(ax, obs_result, weight_x_by=weight_x_by, weight_y_by=weight_y_by,
												colour=obs_colour, **_overlay_style(obs_cfg))
		except Exception as e:
			logging.error(f"Failed to overlay observational data: {e}")

	# Set axis labels and scales
	base_yname = _ANALYTIC_PLOTS[plot_type]['yname']
	final_yname, y_unit = _get_weighted_y_name(base_yname, weight_y_by) if weight_y_by else (base_yname, "")
	_set_scale_and_labels(ax, scale, xname=xname, yname=final_yname, x=x_last, x_unit=x_unit, y_unit=y_unit)

	gdict = _gdict_of(frb_dict)
	_draw_text_items(ax, param_text, gdict, 'analytical', plot_config)
	_draw_text_items(ax, text, gdict, 'general', plot_config)

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
	weight_source = _weight_source(frb_dict, weight_y_by)
	pct =_percentile_cfg(plot_config, x_measured)
	y, applied = _weight_dict(xvals, yvals, weight_source, weight_by=weight_y_by, return_status=True)
	med_vals, lower, upper = _median_band(y, xvals, pct)

	x, xname, x_unit = _weight_x_get_xname(
		frb_dict,
		weight_x_by=weight_x_by,
		x_measured=x_measured,
		phase_window=phase_window,
		freq_window=freq_window
	)

	# Optional x-percentiles (only when using x_measured)
	x_lo = x_hi = None
	if pct.x_enabled:
		_, x_lo, x_hi = _x_percentiles_measured(
			frb_dict, x_measured, p_low=pct.p_low, p_high=pct.p_high,
			phase_window=phase_window if phase_window else 'total',
			freq_window=freq_window if freq_window else 'all'
		)

	fig = None
	if ax is None:
		fig, ax = plt.subplots(figsize=figsize)
		ax.grid(True, linestyle='--', alpha=0.6)
	elif not embed:
		ax.grid(True, linestyle='--', alpha=0.6)

	plot_x, _ = _draw_series(ax, x, med_vals, lower, upper, x_lo, x_hi, pct, draw_style, nbins,
									series_colour, series_label, colour_by_sweep)

	# Expected curve and fit use the unsorted, unbinned x so each value stays paired with its sweep step
	if plot_expected and (expected_param_key is not None) and ("exp_vars" in frb_dict):
		if get_plot_param(plot_config, 'analytical', 'plot_expected', True):
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


def _sweep_mode(dspec_params):
	"""sweep_mode from a dspecParams, a dict, or the first entry of a list of either."""
	if dspec_params is None:
		return None
	sweep_mode = getattr(dspec_params, "sweep_mode", None)
	if sweep_mode is None and isinstance(dspec_params, dict):
		sweep_mode = dspec_params.get("sweep_mode")
	if sweep_mode is None and isinstance(dspec_params, (list, tuple)) and len(dspec_params) > 0:
		first = dspec_params[0]
		sweep_mode = getattr(first, "sweep_mode", None) if not isinstance(first, dict) else first.get("sweep_mode")
	return sweep_mode


def _colour_shades(base_colour, n_shades):
	"""n_shades hex colours from darker to lighter around base_colour."""

	if n_shades == 1:
		return [base_colour]
	if n_shades == 2:
		factors = [0.7, 1.0]
	elif n_shades == 3:
		factors = [0.6, 1.0, 1.3]
	else:
		factors = [0.5 + 0.8 * (i / (n_shades - 1)) for i in range(n_shades)]

	rgb = mcolors.hex2color(base_colour)
	shades = []
	for factor in factors:
		if factor <= 1.0:
			new_rgb = tuple(max(0, min(1, c * factor)) for c in rgb)
		else:
			new_rgb = tuple(max(0, min(1, c + (1 - c) * (factor - 1.0))) for c in rgb)
		shades.append(mcolors.rgb2hex(new_rgb))
	return shades


def _equal_lines_style(equal_lines_cfg):
	"""Styling keyword arguments for _plot_equal_value_lines from [analytical.equal_lines]."""
	cfg = equal_lines_cfg or {}
	return dict(
		linestyle=cfg.get('linestyle', '--'),
		alpha=float(cfg.get('alpha', 0.85)),
		colour_mode=cfg.get('colour_mode', 'shades'),
		base_colour=cfg.get('base_colour', '#005bff' if equal_lines_cfg else 'black'),
		cmap_name=cfg.get('cmap_name', 'viridis'),
		show_labels=bool(cfg.get('show_labels', True)),
		extend_mode=cfg.get('extend_mode', 'linear'),
		label_position=cfg.get('label_position', 'start'),
		label_offset_pts=cfg.get('label_offset_pts', [0, 0]) or [0, 0],
		label_ha=cfg.get('label_ha', 'left'),
		label_va=cfg.get('label_va', 'top'),
	)


def _constant_param_lines_from_dir(ax, data_dir, phase_window, freq_window):
	"""Draw V(psi)-vs-L/I lines of constant parameter from the grouped sweep pickles in data_dir."""
	param_dict = load_multiple_data_grouped(data_dir)
	param_data = {}
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
		measures = subdict['measures']
		vpsi_vals = []
		lfrac_vals = []
		for xv in subdict['xvals']:
			segs = measures.get(xv, [])
			vpsi_vals.append(np.nanmedian([_extract_value_from_segments(seg, 'Vpsi', phase_window, freq_window) for seg in segs]))
			lfrac_vals.append(np.nanmedian([_extract_value_from_segments(seg, 'Lfrac', phase_window, freq_window) for seg in segs]))
		param_data[param_val] = {'vpsi': vpsi_vals, 'lfrac': lfrac_vals}
	param_symbol = param_map[param_name][0] if param_name in param_map else param_name
	plot_constant_param_lines(ax, param_data, x_key='vpsi', y_key='lfrac', color='gray', alpha=0.5,
							  param_name=rf"{param_symbol}")


def _draw_series_labels(ax, frb_dict, plot_config, weight_x_by, weight_y_by, plot_type,
						phase_window, freq_window, x_measured):
	"""Per-run text labels from [analytical.series_labels], optionally de-overlapped with adjustText."""
	cfg = get_plot_param(plot_config, 'analytical', 'series_labels', None) if plot_config else None
	if not (isinstance(cfg, dict) and cfg.get('enabled', True)):
		return
	texts = []
	try:
		_label_series(
			ax, frb_dict, params_to_label=cfg.get('params', None),
			weight_x_by=weight_x_by, weight_y_by=weight_y_by,
			plot_type=plot_type, phase_window=phase_window, freq_window=freq_window,
			x_measured=x_measured, position=cfg.get('position', 'max-y'),
			fontsize=int(cfg.get('fontsize', 16)), alpha=float(cfg.get('alpha', 0.95)),
			offset_pts=cfg.get('offset_pts', [0, 0]) or [0, 0],
			ha=cfg.get('ha', 'left'), va=cfg.get('va', 'bottom'), collect_texts=texts
		)
	except Exception as e:
		logging.debug(f"series_labels failed: {e}")

	if bool(cfg.get('avoid_overlap', False)) and texts:
		try:
			from adjustText import adjust_text
			adjust_text(texts, ax=ax)
		except ImportError:
			logging.debug(
				"series_labels: avoid_overlap=True but 'adjustText' is not installed; "
				"run 'pip install adjustText' to enable automatic label adjustment."
			)


def _plot_multirun(frb_dict, ax, fit, scale, yname=None, weight_y_by=None, weight_x_by=None, x_measured=None,
				   legend=True, equal_value_lines=None, equal_lines_cfg=None, plot_type='pa_var',
				   phase_window='total', freq_window='full-band',
				   draw_style='line-param', nbins=15, colour_by_sweep=False,
				   legend_params=None, param_text=None, text=None, plot_config=None):
	"""
	Draw every run of a multi-run dict on one axes (pa_var or l_frac), in shades of the
	window colour, plus the expected curve, text, equal-value lines and series labels.
	"""
	if equal_value_lines is not None and x_measured is None:
		logging.warning("--equal-value-lines requires --x-measured. Ignoring equal_value_lines.")
		equal_value_lines = None

	base_yname = yname

	# Run used for the expected curve
	exp_ref_run = next((run for run in frb_dict if "full-band" in run and "total" in run), next(iter(frb_dict)))
	exp_ref_subdict = frb_dict[exp_ref_run]

	weight_applied_all = True
	x_last = None
	xname = None
	x_unit = None
	swept_param = None

	colour_shades = _colour_shades(_window_colour(freq_window, phase_window), len(frb_dict))

	for idx, (override_key, run_data) in enumerate(frb_dict.items()):
		logging.info(f"Processing {override_key}:")
		series_label = _format_legend_label(_parse_override_dict(override_key), legend_params, _gdict_of(run_data))

		run_fit = None
		if fit is not None:
			run_fit = fit[idx] if isinstance(fit, (list, tuple)) and len(fit) == len(frb_dict) else fit

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
			series_colour=colour_shades[idx],
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
			base_param = run_data.get("xname")
			swept_param = f"sd_{base_param}" if _sweep_mode(run_data.get("dspec_params", None)) == "sd" else base_param

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
	if get_plot_param(plot_config, 'analytical', 'plot_expected', True):
		_plot_expected(x_last, exp_ref_subdict, ax, exp_ref_subdict["V_params"], np.array(exp_ref_subdict["xvals"]),
					   param_key=param_key, weight_y_by=weight_for_expected)

	if legend:
		_legend_if_any(ax, loc=get_plot_param(plot_config, 'general', 'legend_loc', 'best'))

	_draw_text_items(ax, param_text, _gdict_of(next(iter(frb_dict.values()))), 'analytical', plot_config)
	_draw_text_items(ax, text, _gdict_of(frb_dict), 'general', plot_config)

	if weight_y_by is not None and weight_applied_all:
		final_yname, y_unit = _get_weighted_y_name(base_yname, weight_y_by)
	else:
		final_yname, y_unit = base_yname, param_map.get(base_yname, "")
	# No x passed: leave the limits to matplotlib's autoscaling of the plotted data
	_set_scale_and_labels(ax, scale, xname=xname, yname=final_yname, x_unit=x_unit, y_unit=y_unit)
	ax.autoscale(enable=True, axis='both', tight=False)

	if equal_value_lines is not None and x_measured is not None and swept_param is not None:
		xlim0, ylim0 = ax.get_xlim(), ax.get_ylim()
		line_kw = dict(weight_x_by=weight_x_by, weight_y_by=weight_y_by, x_measured=x_measured, zorder=0,
					   plot_type=plot_type, phase_window=phase_window, freq_window=freq_window,
					   **_equal_lines_style(equal_lines_cfg))
		if isinstance(equal_value_lines, int):
			logging.info(f"Using swept parameter '{swept_param}' for equal-value lines")
			_plot_equal_value_lines(ax, frb_dict, target_param=swept_param, target_values=None,
									n_lines=equal_value_lines, **line_kw)
		elif isinstance(equal_value_lines, (list, tuple, np.ndarray)):
			vals = [float(v) for v in equal_value_lines]
			logging.info(f"Plotting equal-value lines at explicit values: {vals}")
			_plot_equal_value_lines(ax, frb_dict, target_param=swept_param, target_values=vals,
									n_lines=len(vals), **line_kw)
		elif isinstance(equal_value_lines, str) and os.path.isdir(equal_value_lines):
			_constant_param_lines_from_dir(ax, equal_value_lines, phase_window, freq_window)
		else:
			logging.warning(f"equal_value_lines must be int, list, or directory path, got {type(equal_value_lines)}")
		# Keep the data-driven limits; the lines span them rather than extend them
		ax.set_xlim(xlim0)
		ax.set_ylim(ylim0)

	_draw_series_labels(ax, frb_dict, plot_config, weight_x_by, weight_y_by, plot_type,
						phase_window, freq_window, x_measured)


def _read_analytic_config(plot_config):
	"""Collect the [analytical] and [general] settings used by the pa_var / l_frac plots."""
	def analytical(key, default=None):
		return get_plot_param(plot_config, 'analytical', key, default)

	def general(key, default=None):
		return get_plot_param(plot_config, 'general', key, default)

	equal_value_lines = analytical('equal_value_lines')
	if equal_value_lines is not None:
		try:
			equal_value_lines = int(equal_value_lines)
		except (ValueError, TypeError):
			pass  # allow list passthrough
	equal_lines_cfg = analytical('equal_lines')
	if isinstance(equal_lines_cfg, dict):
		if not equal_lines_cfg.get('enabled', True):
			equal_lines_cfg = None
		elif 'n_lines' in equal_lines_cfg:
			nv = equal_lines_cfg['n_lines']
			equal_value_lines = [float(v) for v in nv] if isinstance(nv, (list, tuple)) else int(nv)

	return SimpleNamespace(
		scale=analytical('plot_scale', 'linear'),
		draw_style=analytical('draw_style', 'line-param'),
		fit=analytical('fit_functions'),
		weight_x_by=analytical('weight_x_by'),
		weight_y_by=analytical('weight_y_by'),
		x_measured=analytical('x_measured'),
		equal_value_lines=equal_value_lines,
		equal_lines_cfg=equal_lines_cfg,
		legend_params=analytical('legend_params', []),
		param_text=analytical('param_text', []),
		text=general('text'),
		nbins=analytical('nbins', 15),
		colour_by_sweep=analytical('colour_by_sweep', False),
		xlim=analytical('xlim'),
		ylim=analytical('ylim'),
		xshade=analytical('xshade'),
		xshade_colour=analytical('xshade_colour', 'lightgray'),
		xshade_alpha=float(analytical('xshade_alpha', 0.25)),
		xshade_zorder=int(analytical('xshade_zorder', 0)),
		figsize=general('figsize'),
		show=general('show_plots', True),
		save=general('save_plots', False),
		extension=general('extension', 'pdf'),
		legend=general('legend', True),
		legend_loc=general('legend_loc', 'best'),
	)


# Per-quantity settings for the analytic plots (y symbol, single-run series label,
# expected-curve key, and whether a publication figsize is the default).
_ANALYTIC_PLOTS = {
	'pa_var': dict(yname=r"\mathbb{V}(\psi)", series_label=r'$\mathbb{V}(\psi)$',
				   expected_key='exp_var_PA', pub_figsize_default=True),
	'l_frac': dict(yname=r"\Pi_L", series_label='L/I',
				   expected_key=None, pub_figsize_default=False),
}


def _plot_analytic(plot_type, frb_dict, fname, out_dir, phase_window, freq_window, gauss_file, sim_file,
				   obs_data, obs_params, compare_windows, buffer_frac, plot_config):
	"""Shared body of plot_pa_var and plot_lfrac."""
	spec = _ANALYTIC_PLOTS[plot_type]
	cfg = _read_analytic_config(plot_config)
	yname = spec['yname']
	figsize = cfg.figsize
	if figsize is None and spec['pub_figsize_default']:
		figsize = pub_figsize(ncol=get_pub_col())

	def finish_axes(ax):
		_apply_axis_limits(ax, cfg.xlim, cfg.ylim)

	def shade_x(ax):
		_apply_xshade_regions(ax, cfg.xshade, colour=cfg.xshade_colour, alpha=cfg.xshade_alpha, zorder=cfg.xshade_zorder)

	if compare_windows is not None:
		if _is_multi_run_dict(frb_dict):
			logging.warning("compare_windows only works with single-run data; ignoring.")
		else:
			fig, ax = plt.subplots(figsize=figsize)
			fig.subplots_adjust(left=0.18, right=0.98, bottom=0.16, top=0.98)
			_plot_single_run_multi_window(
				frb_dict=frb_dict, ax=ax, plot_type=plot_type, window_pairs=compare_windows,
				gauss_file=gauss_file, sim_file=sim_file, x_measured=cfg.x_measured,
				weight_y_by=cfg.weight_y_by, weight_x_by=cfg.weight_x_by, fit=cfg.fit, scale=cfg.scale,
				legend=cfg.legend, obs_data=obs_data, obs_params=obs_params, buffer_frac=buffer_frac,
				draw_style=cfg.draw_style, nbins=cfg.nbins, colour_by_sweep=cfg.colour_by_sweep,
				plot_config=plot_config, param_text=cfg.param_text, text=cfg.text
			)
			finish_axes(ax)
			shade_x(ax)
			if cfg.show:
				plt.show()
			if cfg.save:
				name = os.path.join(out_dir, f"{fname}_{cfg.scale}_{plot_type}_window_comparison.{cfg.extension}")
				savefig(name, dpi=600, fig=fig)
				logging.info(f"Saved figure to {name}\n")
			return

	if _is_multi_run_dict(frb_dict):
		fig, ax = plt.subplots(figsize=figsize)
		fig.subplots_adjust(left=0.18, right=0.98, bottom=0.16, top=0.98)
		_plot_multirun(
			frb_dict, ax, fit=cfg.fit, scale=cfg.scale, yname=yname,
			weight_y_by=cfg.weight_y_by, weight_x_by=cfg.weight_x_by, x_measured=cfg.x_measured,
			legend=cfg.legend, equal_value_lines=cfg.equal_value_lines, equal_lines_cfg=cfg.equal_lines_cfg,
			plot_type=plot_type, phase_window=phase_window, freq_window=freq_window,
			draw_style=cfg.draw_style, nbins=cfg.nbins, colour_by_sweep=cfg.colour_by_sweep,
			legend_params=cfg.legend_params, param_text=cfg.param_text, text=cfg.text, plot_config=plot_config
		)
		finish_axes(ax)
	else:
		yvals = _yvals_from_measures_dict(frb_dict["xvals"], frb_dict["measures"], plot_type, phase_window, freq_window)
		fig, ax = _plot_single_job_common(
			frb_dict=frb_dict,
			yname_base=yname,
			weight_y_by=cfg.weight_y_by,
			weight_x_by=cfg.weight_x_by,
			x_measured=cfg.x_measured,
			figsize=figsize,
			fit=cfg.fit,
			scale=cfg.scale,
			series_label=spec['series_label'],
			series_colour=_window_colour(freq_window, phase_window),
			expected_param_key=spec['expected_key'],
			yvals_override=yvals,
			draw_style=cfg.draw_style,
			nbins=cfg.nbins,
			colour_by_sweep=cfg.colour_by_sweep,
			legend_loc=cfg.legend_loc,
			plot_config=plot_config,
			phase_window=phase_window,
			freq_window=freq_window
		)
		finish_axes(ax)
		gdict = _gdict_of(frb_dict)
		_draw_text_items(ax, cfg.param_text, gdict, 'analytical', plot_config)
		_draw_text_items(ax, cfg.text, gdict, 'general', plot_config)

	shade_x(ax)

	if obs_data is not None:
		obs_cfg = _get_obs_cfg(plot_config)
		if obs_cfg:
			try:
				obs_result = _process_observational_data(
					obs_data, obs_params, gauss_file, sim_file, phase_window, freq_window,
					buffer_frac=buffer_frac, plot_mode=plot_modes[plot_type], x_measured=cfg.x_measured
				)
				_plot_observational_overlay(ax, obs_result, weight_x_by=cfg.weight_x_by, weight_y_by=cfg.weight_y_by,
											colour=obs_cfg['colour'], **_overlay_style(obs_cfg))
				if cfg.legend:
					_legend_if_any(ax, loc=cfg.legend_loc)
			except Exception as e:
				logging.error(f"Failed to overlay observational data: {e}")

	if cfg.show:
		plt.show()
	if cfg.save:
		name = _make_plot_fname(plot_type, cfg.scale, fname, freq_window, phase_window)
		name = os.path.join(out_dir, name + f".{cfg.extension}")
		savefig(name, dpi=600, bbox_inches='tight', fig=fig)
		logging.info(f"Saved figure to {name}  \n")


def plot_pa_var(frb_dict, fname, out_dir, phase_window, freq_window, gauss_file=None, sim_file=None,
				obs_data=None, obs_params=None, compare_windows=None, buffer_frac=1, plot_config=None, **kwargs):
	"""
	Plot the PA variance V(psi) against the swept parameter (or a measured quantity).

	frb_dict is a single-run dict ('xvals', 'measures', 'V_params', 'dspec_params', ...) or a
	dict of such runs keyed by override label. All styling comes from plot_config: the
	[analytical] section (plot_scale, draw_style 'line-param'|'line-x'|'scatter'|'binned',
	fit_functions, weight_x_by/weight_y_by, x_measured, equal_lines, nbins, colour_by_sweep,
	xlim/ylim, xshade, ...) and [general] (figsize, show_plots, save_plots, extension, legend).

	compare_windows, a list of (freq_window, phase_window) pairs, overlays several windows
	of a single run on one axes instead of using phase_window/freq_window. obs_data adds an
	observed FRB as a marker with error spans or bars.
	"""
	_plot_analytic('pa_var', frb_dict, fname, out_dir, phase_window, freq_window, gauss_file, sim_file,
				   obs_data, obs_params, compare_windows, buffer_frac, plot_config)


def plot_lfrac(frb_dict, fname, out_dir, phase_window, freq_window, gauss_file=None, sim_file=None,
			   obs_data=None, obs_params=None, compare_windows=None, buffer_frac=1, plot_config=None, **kwargs):
	"""
	Plot the integrated linear polarisation fraction L/I against the swept parameter.

	Takes the same inputs and plot_config settings as plot_pa_var.
	"""
	_plot_analytic('l_frac', frb_dict, fname, out_dir, phase_window, freq_window, gauss_file, sim_file,
				   obs_data, obs_params, compare_windows, buffer_frac, plot_config)


def _integrated_lfrac_with_error(Its, Lts, on_mask, off_mask):
	"""On-pulse integrated L/I and its error from the off-pulse noise (10% if that can't be estimated)."""
	integrated_I = np.nansum(np.where(on_mask, Its, np.nan))
	integrated_L = np.nansum(np.where(on_mask, Lts, np.nan))
	if not np.isfinite(integrated_I) or integrated_I <= 0:
		return np.nan, np.nan
	val = integrated_L / integrated_I

	I_off = Its[off_mask]
	L_off = Lts[off_mask]
	nI = int(np.sum(np.isfinite(I_off)))
	nL = int(np.sum(np.isfinite(L_off)))
	if nI >= 1 and nL >= 1:
		# ddof=1 only with at least 2 valid samples
		with np.errstate(invalid='ignore'):
			noise_I = np.nanstd(I_off, ddof=1 if nI > 1 else 0)
			noise_L = np.nanstd(L_off, ddof=1 if nL > 1 else 0)
		n_on = int(np.sum(on_mask))
		if n_on > 0 and np.isfinite(noise_I) and np.isfinite(noise_L):
			sigma_I_int = noise_I * np.sqrt(n_on)
			sigma_L_int = noise_L * np.sqrt(n_on)
			return val, val * np.sqrt(
				(sigma_L_int / max(integrated_L, 1e-12))**2 +
				(sigma_I_int / max(integrated_I, 1e-12))**2
			)
	return val, 0.1 * val


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

	if os.path.isdir(obs_data_path) or (os.path.isfile(obs_data_path) and obs_data_path.endswith(('.npy', '.npz'))):
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
	ts_data_full, corr_dspec, noisespec_full, _ = process_dspec(dspec, freq_mhz, dspec_params, buffer_frac, skip_rm=True, remove_pa_trend=True)

	# Compute per-window measures once
	segments = compute_segments(dspec, freq_mhz, time_ms, dspec_params, buffer_frac=buffer_frac, skip_rm=True, remove_pa_trend=True)

	measured = {q: _extract_value_from_segments(segments, q, phase_window, freq_window) for q in ('Lfrac', 'Vfrac', 'Vpsi')}
	logging.info(
		f"Observational window [{normalise_freq_window(freq_window, 'dspec')}, "
		f"{normalise_phase_window(phase_window, 'dspec')}]: "
		f"L/I={measured['Lfrac']:.4g}, V/I={measured['Vfrac']:.4g}, V(psi)={measured['Vpsi']:.4g} deg^2"
	)

	# X-axis value: a measured quantity, or else the full-band PA standard deviation
	x_err = None
	if x_measured in ('Vpsi', 'Lfrac'):
		x_value = measured[x_measured]
	else:
		phits = ts_data_full.phits
		ephits = ts_data_full.ephits
		valid_mask = np.isfinite(phits) & np.isfinite(ephits)
		valid_phits = phits[valid_mask]
		valid_ephits = ephits[valid_mask]
		pa_var_deg2 = pa_variance_deg2(valid_phits) if valid_phits.size > 0 else np.nan
		x_value = np.sqrt(pa_var_deg2) if np.isfinite(pa_var_deg2) else np.nan
		x_err = pa_variance_deg2(valid_ephits) if valid_ephits.size > 0 else 0.0

	# Y-axis value from segments
	y_quantity = {'pa_var': 'Vpsi', 'l_frac': 'Lfrac'}.get(plot_mode.name)
	if y_quantity is None:
		raise ValueError(f"Observational overlay not supported for plot mode '{plot_mode.name}'")
	y_value = measured[y_quantity]

	# Error estimates in the requested dspec window
	freq_slc = _get_freq_window_indices(freq_window, freq_mhz)
	freq_win = freq_mhz[freq_slc] if isinstance(freq_slc, slice) else freq_mhz
	ts_data_win, _, _, _ = process_dspec(dspec[:, freq_slc, :], freq_win, dspec_params, buffer_frac, skip_rm=True, remove_pa_trend=True)

	I_profile = ts_data_win.iquvt[0]
	L_profile = ts_data_win.Lts

	peak_index = int(np.nanargmax(I_profile)) if I_profile.size > 0 else 0
	phase_slc = _get_phase_window_indices(phase_window, peak_index)

	on_mask, off_mask, _ = on_off_pulse_masks_from_profile(
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

	def pa_var_error():
		ephits = ts_data_win.ephits[phase_slc]
		valid_e = np.isfinite(ephits)
		return pa_variance_deg2(ephits[valid_e]) if np.any(valid_e) else 0.0

	def lfrac_error():
		return _integrated_lfrac_with_error(I_profile, L_profile, on_mask_windowed, off_mask)[1]

	y_err = pa_var_error() if plot_mode.name == 'pa_var' else lfrac_error()
	if x_measured == 'Lfrac':
		x_err = lfrac_error()
	if x_measured == 'Vpsi':
		x_err = pa_var_error()

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
		logging.info("Processed observational data")
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



# Plot modes. pa_var and l_frac need a parameter sweep (many FRBs); the rest plot one FRB
# through basic_plots. Order matters only for listing.
pa_var = PlotMode(name="pa_var", plot_func=plot_pa_var, requires_multiple_frb=True)
l_frac = PlotMode(name="l_frac", plot_func=plot_lfrac, requires_multiple_frb=True)

plot_modes = {
	"pa_var": pa_var,
	"l_frac": l_frac,
	**{name: PlotMode(name=name, plot_func=basic_plots, requires_multiple_frb=False)
	   for name in ("iquv", "lv", "lvpa", "stokes", "qu", "dpa", "RM", "pa", "pali", "pads")},
}
