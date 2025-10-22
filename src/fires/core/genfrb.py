# -----------------------------------------------------------------------------
# genfrb.py
# FIRES: The Fast, Intense Radio Emission Simulator
#
# This module contains functions for generating Fast Radio Burst (FRB) dynamic
# spectra, handling baseline subtraction, off-pulse window selection, scattering,
# data loading, and parallelised simulation and aggregation of FRB realizations.
# It is a core part of the FIRES simulation pipeline.
#
# Author: JB
# Date: 2025-05-20
# -----------------------------------------------------------------------------

# -------------------------- Import modules ---------------------------
import functools
import inspect
import logging
import os
import pickle as pkl
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
from tqdm import tqdm

from fires.core.basicfns import (add_noise, process_dspec, scatter_dspec,
								 snr_onpulse)
from fires.core.genfns import psn_dspec
from fires.utils.config import get_parameters, load_params
from fires.utils.utils import dspecParams, simulated_frb

logging.basicConfig(level=logging.INFO)


def _scatter_loaded_dspec(dspec, freq_mhz, time_ms, tau_ms, sc_idx, ref_freq_mhz):
	"""
	Scatter all Stokes channels of a loaded dynamic spectrum.
	Args:
		dspec: 3D array [4, nchan, ntime] (Stokes I, Q, U, V)
		freq_mhz: Frequency array (nchan,)
		time_ms: Time array (ntime,)
		tau_ms: Scattering timescale (float)
		sc_idx: Scattering index (float)
		ref_freq_mhz: Reference frequency (float)
	Returns:
		dspec_scattered: Scattered dynamic spectrum (same shape as input)
	"""
	dspec_scattered = dspec.copy()
	time_res_ms = np.median(np.diff(time_ms))
	tau_cms = tau_ms * (freq_mhz / ref_freq_mhz) ** (-sc_idx)
	for stokes_idx in range(dspec.shape[0]):  # Loop over I, Q, U, V
		dspec_scattered[stokes_idx] = scatter_dspec(
			dspec[stokes_idx], time_res_ms, tau_cms
		)
	return dspec_scattered


def load_data(obs_data_path, obs_params_path=None):
	"""
	Load real FRB data from individual Stokes I, Q, U, V files and parameter file.
	
	Expected file structure:
	- Individual Stokes files: out_I.npy, out_Q.npy, out_U.npy, out_V.npy
	  OR: {label}_htr_dsI.npy, {label}_htr_dsQ.npy, etc.
	- Frequency array: {label}_htr_freq.npy or zoom_x_0.npy
	- Time array: {label}_htr_time.npy or zoom_y_0.npy
	- Parameters: parameters.txt
	
	Parameters:
	-----------
	obs_data_path : str
		Path to directory containing Stokes files, or path to a single Stokes file
	obs_params_path : str or None
		Path to parameters.txt file. If None, will search in obs_data_path directory
		
	Returns:
	--------
	tuple
		(dspec, freq_mhz, time_ms, gdict) where dspec has shape [4, nfreq, ntime]
	"""
	# Determine base directory and pattern
	if os.path.isdir(obs_data_path):
		data_dir = obs_data_path
		base_pattern = ""
		# Try to infer base pattern from directory contents
		import glob
		htr_files = glob.glob(os.path.join(data_dir, "*_htr_dsI.npy"))
		if htr_files:
			basename = os.path.basename(htr_files[0])
			base_pattern = basename.replace("_htr_dsI.npy", "")
	else:
		data_dir = os.path.dirname(obs_data_path)
		# Extract base name pattern from filename
		basename = os.path.basename(obs_data_path)
		
		# Handle various naming patterns
		if '_htr_dsI.npy' in basename:
			base_pattern = basename.replace('_htr_dsI.npy', '')
		elif '_htr_dsQ.npy' in basename:
			base_pattern = basename.replace('_htr_dsQ.npy', '')
		elif '_htr_dsU.npy' in basename:
			base_pattern = basename.replace('_htr_dsU.npy', '')
		elif '_htr_dsV.npy' in basename:
			base_pattern = basename.replace('_htr_dsV.npy', '')
		elif 'out_I.npy' in basename or 'out_Q.npy' in basename:
			base_pattern = ""
		else:
			base_pattern = ""
	
	logging.info(f"Loading real FRB data from {data_dir}")
	if base_pattern:
		logging.info(f"Using base pattern: {base_pattern}")
	
	# Find Stokes files - try multiple naming conventions
	stokes_files = {}
	for stokes in ['I', 'Q', 'U', 'V']:
		candidates = []
		
		# Pattern 1: {label}_htr_dsI.npy (e.g., 191001_htr_dsI.npy)
		if base_pattern:
			candidates.append(os.path.join(data_dir, f"{base_pattern}_htr_ds{stokes}.npy"))
		
		# Pattern 2: out_I.npy
		candidates.append(os.path.join(data_dir, f"out_{stokes}.npy"))
		
		# Pattern 3: out_I_noflagged.npy
		candidates.append(os.path.join(data_dir, f"out_{stokes}_noflagged.npy"))
		
		# Pattern 4: Generic patterns
		if base_pattern:
			candidates.extend([
				os.path.join(data_dir, f"{base_pattern}_{stokes}.npy"),
				os.path.join(data_dir, f"{base_pattern}_ds{stokes}.npy"),
			])
		
		candidates.extend([
			os.path.join(data_dir, f"{stokes}.npy"),
			os.path.join(data_dir, f"stokes_{stokes}.npy"),
		])
		
		found = False
		for candidate in candidates:
			if os.path.exists(candidate):
				stokes_files[stokes] = candidate
				found = True
				logging.info(f"Found Stokes {stokes}: {os.path.basename(candidate)}")
				break
		
		if not found:
			logging.warning(f"Stokes {stokes} file not found in {data_dir}")
			stokes_files[stokes] = None
	
	# Load Stokes data
	stokes_arrays = []
	for stokes in ['I', 'Q', 'U', 'V']:
		if stokes_files[stokes] is not None:
			arr = np.load(stokes_files[stokes])
			stokes_arrays.append(arr)
		else:
			# Create zeros if file doesn't exist
			if len(stokes_arrays) > 0:
				stokes_arrays.append(np.zeros_like(stokes_arrays[0]))
			else:
				# Can't proceed without at least Stokes I
				if stokes == 'I':
					raise FileNotFoundError(
						f"Stokes I file required but not found in {data_dir}\n"
						f"Tried patterns: out_I.npy, {base_pattern}_htr_dsI.npy, etc."
					)
	
	# Stack into [4, nfreq, ntime] array
	dspec = np.array(stokes_arrays)
	logging.info(f"Loaded dynamic spectrum with shape: {dspec.shape} [Stokes, freq, time]")
	
	# Load frequency array
	freq_mhz = None
	freq_candidates = []
	
	# Pattern 1: {label}_htr_freq.npy
	if base_pattern:
		freq_candidates.append(os.path.join(data_dir, f"{base_pattern}_htr_freq.npy"))
	
	# Pattern 2: zoom_x_0.npy (frequency axis)
	freq_candidates.append(os.path.join(data_dir, "zoom_x_0.npy"))
	
	# Generic patterns
	freq_candidates.extend([
		os.path.join(data_dir, "freq.npy"),
		os.path.join(data_dir, "frequency.npy"),
		os.path.join(data_dir, "freqs.npy"),
	])
	
	# Try with glob for wildcards
	import glob
	freq_candidates.extend(glob.glob(os.path.join(data_dir, "*_freq.npy")))
	freq_candidates.extend(glob.glob(os.path.join(data_dir, "*_htr_freq.npy")))
	
	for candidate in freq_candidates:
		if os.path.exists(candidate):
			freq_mhz = np.load(candidate)
			logging.info(f"Loaded frequency array from {os.path.basename(candidate)}: {len(freq_mhz)} channels")
			break
	
	if freq_mhz is None:
		logging.warning("No frequency array found, creating default MHz array")
		freq_mhz = np.arange(dspec.shape[1], dtype=float)
	
	# Ensure freq_mhz matches dspec dimensions
	if len(freq_mhz) != dspec.shape[1]:
		logging.warning(
			f"Frequency array length ({len(freq_mhz)}) doesn't match dspec freq axis ({dspec.shape[1]}). "
			"Creating default array."
		)
		freq_mhz = np.arange(dspec.shape[1], dtype=float)
	
	# Load time array
	time_ms = None
	time_candidates = []
	
	# Pattern 1: {label}_htr_time.npy
	if base_pattern:
		time_candidates.append(os.path.join(data_dir, f"{base_pattern}_htr_time.npy"))
	
	# Pattern 2: zoom_y_0.npy (time axis)
	time_candidates.append(os.path.join(data_dir, "zoom_y_0.npy"))
	
	# Generic patterns
	time_candidates.extend([
		os.path.join(data_dir, "time.npy"),
		os.path.join(data_dir, "times.npy"),
	])
	
	# Try with glob
	time_candidates.extend(glob.glob(os.path.join(data_dir, "*_time.npy")))
	time_candidates.extend(glob.glob(os.path.join(data_dir, "*_htr_time.npy")))
	
	for candidate in time_candidates:
		if os.path.exists(candidate):
			time_ms = np.load(candidate)
			logging.info(f"Loaded time array from {os.path.basename(candidate)}: {len(time_ms)} samples")
			break
	
	if time_ms is None:
		logging.warning("No time array found, creating default ms array")
		time_ms = np.arange(dspec.shape[2], dtype=float)
	
	# Ensure time_ms matches dspec dimensions
	if len(time_ms) != dspec.shape[2]:
		logging.warning(
			f"Time array length ({len(time_ms)}) doesn't match dspec time axis ({dspec.shape[2]}). "
			"Creating default array."
		)
		time_ms = np.arange(dspec.shape[2], dtype=float)
	
	# Convert time to milliseconds if needed (check if values are too small)
	if np.median(time_ms) < 1.0 and np.median(time_ms) > 0:
		logging.info("Time array appears to be in seconds, converting to milliseconds")
		time_ms = time_ms * 1000.0
	
	# Load parameters file
	gdict = {}
	if obs_params_path is None:
		obs_params_path = os.path.join(data_dir, "parameters.txt")
	
	if os.path.exists(obs_params_path):
		logging.info(f"Loading parameters from {os.path.basename(obs_params_path)}")
		gdict = get_parameters(obs_params_path)
	else:
		logging.warning(f"Parameters file not found: {obs_params_path}")
		# Create minimal gdict with defaults
		gdict = {
			'tau_ms': np.array([0.0]),
			'width_ms': np.array([np.median(np.diff(time_ms)) * 10 if len(time_ms) > 1 else 1.0]),
			'DM': np.array([0.0]),
			'RM': np.array([0.0]),
			'band_centre_mhz': np.array([np.median(freq_mhz) if len(freq_mhz) > 0 else 1000.0]),
			'band_width_mhz': np.array([np.ptp(freq_mhz) if len(freq_mhz) > 0 else 336.0])
		}
	
	logging.info(
		f"Final data shape: {dspec.shape}, "
		f"freq range: {freq_mhz.min():.1f}-{freq_mhz.max():.1f} MHz, "
		f"time range: {time_ms.min():.3f}-{time_ms.max():.3f} ms\n"
	)
	
	return dspec, freq_mhz, time_ms, gdict


def _load_multiple_data_grouped(data):
	"""
	Group simulation outputs by freq and phase info (everything after freq_ and phase_).
	Loads ALL files per group and merges their xvals/yvals together.
	Returns a dictionary: {freq_phase_key: {'xname': ..., 'xvals': ..., 'yvals': ..., ...}, ...}
	Dictionary is sorted by override parameter values (e.g., N=10, N=100, N=1000).
	"""
	import re
	from collections import defaultdict
	
	logging.info(f"Loading grouped data from {data}...")
  
	def normalise_override_value(value_str):
		"""Convert '10.0', '10', '100.0' to normalised form like '10', '100'"""
		try:
			val = float(value_str)
			if val.is_integer():
				return str(int(val))
			else:
				return f"{val:.2f}"
		except ValueError:
			return value_str
	
	def extract_override_params(freq_phase_key):
		"""
		Extract override parameters from freq_phase_key for sorting.
		Returns dict of {param_name: numeric_value}.
		Example: "full-band, leading, N10_tau5.5" -> {'N': 10, 'tau': 5.5}
		"""
		parts = freq_phase_key.split(', ')
		if len(parts) < 3:
			return {}
		
		override_str = parts[2]  # e.g., "N10_tau5.5"
		override_dict = {}
		
		# Split by underscore to get individual param=value pairs
		for part in override_str.split('_'):
			match = re.match(r'([a-zA-Z_]+)([0-9.]+)', part)
			if match:
				param = match.group(1)
				value = float(match.group(2))
				override_dict[param] = value
		
		return override_dict
	
	def sort_key_for_freq_phase(freq_phase_key):
		"""
		Generate sort key for freq_phase_key.
		Sorts by: (freq, phase, N, tau_ms, lfrac, ...) in that order.
		"""
		parts = freq_phase_key.split(', ')
		freq_info = parts[0] if len(parts) > 0 else ""
		phase_info = parts[1] if len(parts) > 1 else ""
		
		override_params = extract_override_params(freq_phase_key)
		
		# Sort order: freq, phase, then override params in a specific order
		# Common params: N, tau, lfrac, vfrac, PA, DM, RM, width
		param_order = ['N', 'tau', 'lfrac', 'vfrac', 'PA', 'DM', 'RM', 'width']
		
		sort_tuple = [freq_info, phase_info]
		for param in param_order:
			sort_tuple.append(override_params.get(param, 0))  # 0 if param not present
		
		return tuple(sort_tuple)
	
	def extract_freq_phase_key(fname):
		# Extract freq and phase info and any override parameters from filename
		# Pattern: ...freq_{freq}_phase_{phase}[_{overrides}].pkl
		m = re.search(r'_freq_([a-z\-]+)_phase_([a-z\-]+)(?:_(.+?))?\.pkl$', fname)
		if m:
			freq_info = m.group(1)
			phase_info = m.group(2)
			override_info = m.group(3) if m.group(3) else None
			
			if override_info:
				parts = override_info.split('_')
				cleaned_parts = []
				
				for part in parts:
					if not part:
						continue
					match = re.match(r'([a-zA-Z_]+)([0-9.]+)', part)
					if match:
						param = match.group(1)
						value = match.group(2)
						normalized_value = normalise_override_value(value)
						cleaned_parts.append(f"{param}{normalized_value}")
					else:
						cleaned_parts.append(part)
				
				if cleaned_parts:
					override_label = "_".join(cleaned_parts)
					return f"{freq_info}, {phase_info}, {override_label}"
			
			return f"{freq_info}, {phase_info}"
		
		logging.warning(f"Could not parse freq/phase from filename: {fname}")
		return "unknown"  
	
	file_names = [f for f in os.listdir(data) if f.endswith(".pkl")]
	logging.info(f"Found {len(file_names)} .pkl files")
	
	groups = defaultdict(list)
	
	# Group files by freq/phase/override key
	for fname in file_names:
		freq_phase_key = extract_freq_phase_key(fname)
		groups[freq_phase_key].append(fname)
	
	# Sort groups by override parameters
	sorted_keys = sorted(groups.keys(), key=sort_key_for_freq_phase)
	
	logging.info(f"Grouped files into {len(groups)} unique series (sorted by override params):")
	for key in sorted_keys:
		logging.info(f"  '{key}': {len(groups[key])} files")

	all_results = {}
	
	for freq_phase_key in sorted_keys:
		file_list = groups[freq_phase_key]
		
		xname = None
		plot_mode = None
		dspec_params = None
		all_xvals = []
		all_measures = {}
		all_V_params = {}
		all_exp_vars = {}
		all_snrs = {}

		seen_xvals = set()
		logging.info(f"Merging {len(file_list)} files for series '{freq_phase_key}'")
		
		for fname in file_list:
			with open(os.path.join(data, fname), "rb") as f:
				obj = pkl.load(f)
			if not isinstance(obj, dict):
				logging.warning(f"File {fname} does not contain expected dict structure")
				continue
			
			if xname is None:
				xname = obj.get("xname")
				plot_mode = obj.get("plot_mode")
				dspec_params = obj.get("dspec_params")

			xvals = obj.get("xvals", [])
			snrs = obj.get("snrs", {})
			V_params = obj.get("V_params", {})
			exp_vars = obj.get("exp_vars", {})
			measures = obj.get("measures", {})

			# Merge xvals and associated data
			for v in xvals:
				if v not in seen_xvals:
					seen_xvals.add(v)
					all_xvals.append(v)
				# init per-xval
				if v not in all_measures:
					all_measures[v] = []
					all_snrs[v] = []
					all_V_params[v] = {key: [] for key in V_params.get(v, {}).keys()}
					all_exp_vars[v] = {key: [] for key in exp_vars.get(v, {}).keys()}
				# extend
				all_measures[v].extend(measures.get(v, []))
				all_snrs[v].extend(snrs.get(v, []))
				for key, arr in V_params.get(v, {}).items():
					all_V_params[v][key].extend(arr)
				for key, arr in exp_vars.get(v, {}).items():
					all_exp_vars[v][key].extend(arr)
		
		all_xvals = sorted(all_xvals)
		all_results[freq_phase_key] = {
			'xname': xname,
			'xvals': all_xvals,
			'measures': all_measures,     
			'V_params': all_V_params,
			'exp_vars': all_exp_vars,
			'dspec_params': dspec_params,
			'plot_mode': plot_mode,
			'snrs': all_snrs
		}
		
		logging.info(
			f"Merged '{freq_phase_key}': {len(all_xvals)} unique xvals, "
			f"range {min(all_xvals):.1f}-{max(all_xvals):.1f}"
		)

	logging.info(f"Returning {len(all_results)} unique series (sorted)\n")
	return all_results


def _generate_dspec(xname, mode, var, plot_multiple_frb, target_snr=None, **params):
	"""Generate dynamic spectrum based on mode."""
	var = var if plot_multiple_frb else None

	# Choose the correct function
	if mode == 'psn':
		dspec_func = psn_dspec

	# Get the argument names for the selected function
	sig = inspect.signature(dspec_func)
	allowed_args = set(sig.parameters.keys())

	# Always pass tau_ms as v
	params_filtered = {
		k: v for k, v in params.items()
		if k in allowed_args and k not in ("xname")
	}
 
	if mode == 'psn':
		return psn_dspec(**params_filtered, variation_parameter=var, xname=xname, plot_multiple_frb=plot_multiple_frb,
								target_snr=target_snr)


def _process_task(task, xname, mode, plot_mode, **params):
	"""
	Process a single task (combination of timescale and realization).
	Now returns the per-segment measures from psn_dspec, not a single window value.
	"""
	var, realization = task
	base_seed = params.get("seed", None)
	current_seed = (base_seed + realization) if base_seed is not None else None

	local_params = dict(params)
	local_params["seed"] = current_seed
	
	requires_multiple_frb = plot_mode.requires_multiple_frb

	# Generate dynamic spectrum + segments
	# psn_dspec now returns: dspec, snr, V_params, exp_vars, measures
	dspec, snr, V_params, exp_vars, measures = _generate_dspec(
		xname=xname,
		mode=mode,
		var=var,
		plot_multiple_frb=requires_multiple_frb,
		**local_params
	)

	# Return the entire measures dict for this realization
	return var, measures, V_params, snr, exp_vars


def generate_frb(data, frb_id, out_dir, mode, seed, nseed, write, sim_file, gauss_file, scint_file,
				sefd, n_cpus, plot_mode, phase_window, freq_window, buffer_frac, sweep_mode, obs_data, obs_params,
				target_snr=None, param_overrides=None):
	"""
	Generate a simulated FRB with a dispersed and scattered dynamic spectrum.
	"""
	sim_file = load_params("simparams", sim_file, "simulation")

	# Extract frequency and time parameters
	f_start = float(sim_file['f0'])
	f_end   = float(sim_file['f1'])
	t_start = float(sim_file['t0'])
	t_end   = float(sim_file['t1'])
	f_res   = float(sim_file['f_res'])
	t_res   = float(sim_file['t_res'])

	scatter_idx = float(sim_file['scattering_index'])
	ref_freq 	= float(sim_file['reference_freq'])

	# Generate frequency and time arrays
	freq_mhz = np.arange(f_start, f_end + f_res, f_res, dtype=float)
	time_ms  = np.arange(t_start, t_end + t_res, t_res, dtype=float)

	# Load Gaussian parameters
	gauss_params = np.loadtxt(gauss_file)

	# Split structural rows
	stddev_row   = -4   # Gaussian std dev (psn micro variation)
	start_row    = -3   # Sweep start
	stop_row     = -2   # Sweep stop
	step_row     = -1   # Sweep step

	# Means (main components)
	gdict = {
		't0'             : gauss_params[:stddev_row, 0],
		'width_ms'       : gauss_params[:stddev_row, 1],
		'A'              : gauss_params[:stddev_row, 2],
		'spec_idx'       : gauss_params[:stddev_row, 3],
		'tau_ms'         : gauss_params[:stddev_row, 4],
		'DM'             : gauss_params[:stddev_row, 5],
		'RM'             : gauss_params[:stddev_row, 6],
		'PA'             : gauss_params[:stddev_row, 7],
		'lfrac'          : gauss_params[:stddev_row, 8],
		'vfrac'          : gauss_params[:stddev_row, 9],
		'dPA'            : gauss_params[:stddev_row, 10],
		'band_centre_mhz': gauss_params[:stddev_row, 11],
		'band_width_mhz' : gauss_params[:stddev_row, 12],
		'N'              : gauss_params[:stddev_row, 13],
		'mg_width_low'   : gauss_params[:stddev_row, 14],
		'mg_width_high'  : gauss_params[:stddev_row, 15]
	}

	# Apply parameter overrides
	if param_overrides:
		for key, value in param_overrides.items():
			if key in gdict:
				# Override all components with the same value
				original_shape = gdict[key].shape
				gdict[key] = np.full(original_shape, value, dtype=float)
				logging.info(f"Override applied: {key} = {value} (shape: {original_shape})")
			elif key in sd_dict:
				# Allow overriding standard deviations too
				sd_dict[key] = float(value)
				logging.info(f"Override applied: {key} = {value} (std dev)")
			else:
				logging.warning(f"Override key '{key}' not found in gdict or sd_dict. Ignoring.")


	# Micro (psn) σ (std dev) values (scalars per column)
	sd_dict = {
		'sd_t0'             : gauss_params[stddev_row, 0],
		'sd_width_ms'       : gauss_params[stddev_row, 1],
		'sd_A'       : gauss_params[stddev_row, 2],
		'sd_spec_idx'       : gauss_params[stddev_row, 3],
		'sd_tau_ms'         : gauss_params[stddev_row, 4],
		'sd_DM'             : gauss_params[stddev_row, 5],
		'sd_RM'             : gauss_params[stddev_row, 6],
		'sd_PA'             : gauss_params[stddev_row, 7],
		'sd_lfrac'          : gauss_params[stddev_row, 8],
		'sd_vfrac'          : gauss_params[stddev_row, 9],
		'sd_dPA'            : gauss_params[stddev_row,10],
		'sd_band_centre_mhz': gauss_params[stddev_row,11],
		'sd_band_width_mhz' : gauss_params[stddev_row,12]
	}

	# Sweep specification (used only for multi-FRB modes)
	sweep_start = gauss_params[start_row]
	sweep_stop  = gauss_params[stop_row]
	sweep_step  = gauss_params[step_row]

	active_cols = np.where(sweep_step != 0.0)[0]
	if plot_mode.requires_multiple_frb and data is None:
		if active_cols.size == 0:
			logging.error("No sweep defined (all step = 0) but multi-FRB plot requested.")
			sys.exit(1)
		if active_cols.size > 1:
			logging.error("ERROR: More than one sweep column (multiple non-zero steps).")
			sys.exit(1)
	sweep_col = active_cols[0] if active_cols.size == 1 else None

	sweep_spec = {
		'col_index': sweep_col,
		'start'    : sweep_start[sweep_col] if sweep_col is not None else None,
		'stop'     : sweep_stop[sweep_col]  if sweep_col is not None else None,
		'step'     : sweep_step[sweep_col]  if sweep_col is not None else None
	}

	if scint_file is not None:
		scint = load_params("scparams", scint_file, "scintillation")
		if scint.get("derive_from_tau", False):
			# Derive decorrelation bandwidth at the REFERENCE frequency
			tau_ms_ref = float(gdict["tau_ms"][0])          # τ_sc at ref_freq (from gparams)
			tau_s_ref  = 1e-3 * tau_ms_ref                  # s
			nu_s_hz    = 1.0 / (2.0 * np.pi * tau_s_ref)    # Hz
			scint["nu_s"] = float(nu_s_hz)
			logging.info(
				f"Derived nu_s at reference {ref_freq:.1f} MHz: "
				f"tau={tau_ms_ref:.3f} ms -> nu_s={nu_s_hz:.2f} Hz"
			)
	else:
		scint = None

	# Create dynamic spectrum parameters
	dspec_params = dspecParams(
		gdict           = gdict,
		sd_dict         = sd_dict,
		scint_dict      = scint,
		freq_mhz        = freq_mhz,
		freq_res_mhz    = f_res,
		time_ms         = time_ms,
		time_res_ms     = t_res,
		seed            = seed,
		nseed           = nseed,
		sefd            = sefd,
		sc_idx          = scatter_idx,
		ref_freq_mhz    = ref_freq,
		phase_window    = phase_window,
		freq_window     = freq_window,
		buffer_frac     = buffer_frac,
		sweep_mode      = sweep_mode
	)

	tau_ms = gdict['tau_ms']

	if len(np.where(gauss_params[-1,:] != 0.0)[0]) > 1:
		logging.warning("More than one value in the last row of gauss_params is not 0.")
		logging.info("Please ensure that only one value is non-zero in the last row.")
		sys.exit(1)
  
	if np.any(gdict['lfrac'] + gdict['vfrac']) > 1.0:
		logging.warning("Linear and circular polarization fractions sum to more than 1.0.")

	plot_multiple_frb = plot_mode.requires_multiple_frb
	if not plot_multiple_frb:
		# Single FRB generation branch
		if data != None:
			dspec, freq_mhz, time_ms = load_data(obs_data, obs_params)
			snr, (left, right) = snr_onpulse(np.nansum(dspec[0], axis=0), frac=0.95, buffer_frac=buffer_frac)
			logging.info(f"Loaded data S/N: {snr:.2f}, on-pulse window: {left}-{right} ({time_ms[left]:.2f}-{time_ms[right]:.2f} ms)")  
			if tau_ms[0] > 0:
				dspec = _scatter_loaded_dspec(dspec, freq_mhz, time_ms, tau_ms[0], scatter_idx, ref_freq)
			if sefd > 0:
				dspec, sigma_ch, snr = add_noise(dspec=dspec, sefd=sefd, f_res=f_res, t_res=t_res, 
													plot_multiple_frb=plot_multiple_frb)
			
			# Update dspec_params with new time and frequency arrays
			dspec_params = dspec_params._replace(time_ms=time_ms, freq_mhz=freq_mhz)

		else:
			dspec, snr, _, _ = _generate_dspec(
			xname=None,
			mode=mode,
			var=None,
			plot_multiple_frb=False,
			target_snr=target_snr,			
			**dspec_params._asdict()
		)
		_, _, _, noise_spec = process_dspec(dspec, freq_mhz, gdict, buffer_frac)
		frb_data = simulated_frb(
			frb_id, dspec, dspec_params, snr
		)
		if write:
			tau = f"{tau_ms[0]:.2f}"
			if mode == 'psn':
				out_file = (
					f"{out_dir}{frb_id}_mode_{mode}_sc_{tau}_"
					f"seed_{seed}_nseed_{nseed}_PA_{gdict['PA'][-1]:.2f}.pkl"
				)
			with open(out_file, 'wb') as frb_file:
				pkl.dump(frb_data, frb_file)
		return frb_data, noise_spec, gdict

	if plot_multiple_frb:
		if data != None:
			files = [f for f in os.listdir(data) if f.endswith('.pkl')]
			if len(files) > 1:
				frb_dict = _load_multiple_data_grouped(data)
			elif len(files) == 1:
				with open(os.path.join(data, files[0]), 'rb') as f:
					frb_dict = pkl.load(f)
			else:
				logging.error(f"No .pkl files found in {data}.")
				sys.exit(1)
			return frb_dict
		else:
			# Validate sweep definition
			if sweep_mode == "none":
				raise ValueError(
					f"Plot mode '{plot_mode.name}' requires a parameter sweep. "
					"Use --sweep-mode mean or --sweep-mode sd and set exactly one non-zero step in gparams."
				)

			if np.all(gauss_params[step_row, :] == 0.0):
				logging.error("No sweep defined (all step sizes zero) but a multi-FRB plot was requested.")
				logging.info("Edit the last three rows (start/stop/step) of gparams for exactly one column.")
				sys.exit(1)

			col_idx = sweep_spec['col_index']
			if col_idx is None:
				raise ValueError("Could not determine sweep column (no non-zero step).")

			start = sweep_spec['start']; stop = sweep_spec['stop']; step = sweep_spec['step']
			if step is None or step <= 0:
				raise ValueError("Sweep step must be > 0.")

			n_steps = int(np.round((stop - start) / step))
			end = start + n_steps * step
			xvals = np.linspace(start, end, n_steps + 1)

			gdict_keys = list(gdict.keys())
			xname = gdict_keys[col_idx]

			# Slurm / chunking support
			def _slurm_array_size():
				cnt = os.environ.get("SLURM_ARRAY_TASK_COUNT")
				if cnt is not None:
					return int(cnt)
				min_s = os.environ.get("SLURM_ARRAY_TASK_MIN")
				max_s = os.environ.get("SLURM_ARRAY_TASK_MAX")
				if max_s is not None:
					min_i = int(min_s) if min_s is not None else 0
					return int(max_s) - min_i + 1
				return int(os.environ.get("FIRESSWEEP_COUNT", "1"))
			def _slurm_array_id():
				min_s = os.environ.get("SLURM_ARRAY_TASK_MIN")
				task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
				if task_id is not None:
					if min_s is not None:
						return int(task_id) - int(min_s)
					return int(task_id)
				return int(os.environ.get("FIRESSWEEP_ID", "0"))

			_array_count = _slurm_array_size()
			_array_id = _slurm_array_id()
			if _array_count > 1:
				total = len(xvals)
				base = total // _array_count
				rem = total % _array_count
				if _array_id < rem:
					start_idx = _array_id * (base + 1)
					end_idx = start_idx + (base + 1)
				else:
					start_idx = rem * (base + 1) + (_array_id - rem) * base
					end_idx = start_idx + base

				if start_idx >= total or start_idx == end_idx:
					logging.info(f"Array task {_array_id}/{_array_count - 1}: no assigned xvals.")
					sys.exit(0)

				xvals = xvals[start_idx:min(end_idx, total)]
				logging.info(
					f"Array task {_array_id}/{_array_count - 1}: "
					f"processing {len(xvals)} sweep values (idx {start_idx}:{min(end_idx, total)} of {total})."
				)

			tasks = list(product(xvals, range(nseed)))

			with ProcessPoolExecutor(max_workers=n_cpus) as executor:
				partial_func = functools.partial(
					_process_task,
					xname=xname,
					mode=mode,
					plot_mode=plot_mode,
					target_snr=target_snr,
					**dspec_params._asdict()
				)
				results = list(tqdm(
					executor.map(partial_func, tasks),
					total=len(tasks),
					desc=f"Processing sweep of {xname} ({sweep_mode} mode)"
				))

			measures = {v: [] for v in xvals}
			V_params = {
				v: {key: [] for key in [
					't0_i','A_i','width_ms_i','spec_idx_i','tau_ms_i','PA_i',
					'DM_i','RM_i','lfrac_i','vfrac_i','dPA_i','band_centre_mhz_i','band_width_mhz_i'
				]} for v in xvals
			}
			snrs = {v: [] for v in xvals}
			exp_vars = {
				v: {key: [] for key in [
					'exp_var_t0','exp_var_A','exp_var_width_ms','exp_var_spec_idx','exp_var_tau_ms','exp_var_PA',
					'exp_var_DM','exp_var_RM','exp_var_lfrac','exp_var_vfrac','exp_var_dPA','exp_var_band_centre_mhz','exp_var_band_width_mhz'
				]} for v in xvals
			}

			for var, seg_measures, params_dict, snr, exp_var_psi_deg2 in results:
				measures[var].append(seg_measures)
				snrs[var].append(snr)
				for key, value in params_dict.items():
					V_params[var][key].append(value)
				for key, value in exp_var_psi_deg2.items():
					exp_vars[var][key].append(value)

			frb_dict = {
				"xname": xname,
				"xvals": xvals,
				"measures": measures,         # <- replaces 'yvals'
				"V_params": V_params,
				"exp_vars": exp_vars,
				"dspec_params": dspec_params,
				"plot_mode": plot_mode,
				"snrs": snrs,
			}

			print("_".join([f"{k}{v}" for k, v in param_overrides.items()]) if param_overrides else "")
			if write:
				xvals_str = f"{xvals[0]:.2f}-{xvals[-1]:.2f}" if len(xvals) > 1 else f"{xvals[0]:.2f}"
				override_str = ""
				if param_overrides:
					formatted_overrides = []
					for k, v in param_overrides.items():
						if isinstance(v, float) and v.is_integer():
							formatted_overrides.append(f"{k}{int(v)}")
						else:
							formatted_overrides.append(f"{k}{v:.2f}")
					override_str = "_".join(formatted_overrides)
				
				out_file = (
					f"{out_dir}{frb_id}_plot_{plot_mode.name}_xname_{xname}_xvals_{xvals_str}_"
					f"mode_{mode}_freq_{freq_window}_phase_{phase_window}"
					f"{'_' + override_str if override_str else ''}.pkl"
				)
				with open(out_file, 'wb') as frb_file:
					pkl.dump((frb_dict), frb_file)
				logging.info(f"Saved FRB data to {out_file}")

			return frb_dict