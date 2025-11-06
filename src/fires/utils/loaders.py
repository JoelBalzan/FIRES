# -----------------------------------------------------------------------------
# loaders.py
# FIRES I/O utilities for loading observed data and grouped simulation outputs
# -----------------------------------------------------------------------------

import logging
import os
import pickle as pkl
import re
from collections import defaultdict

import numpy as np

from fires.utils.utils import dspecParams
from fires.utils.config import load_params

logging.basicConfig(level=logging.INFO)


def get_parameters(filepath):
	"""
	Parse a parameters.txt file with key = value format.
	
	Extracts relevant parameters for FRB analysis and converts to FIRES format.
	"""
	valid_keys = {
		't0','width','A','spec_idx','tau','DM','RM','PA',
		'lfrac','vfrac','dPA','band_centre','band_width',
		'N','mg_width_low','mg_width_high'
	}
	params = {}
	with open(filepath,'r') as f:
		for line in f:
			line = line.strip()
			if not line or line.startswith('#') or line.startswith('****'):
				continue
			if '=' in line:
				k,v = line.split('=',1)
				k = k.strip()
				v = v.split('#')[0].strip()
				if k in valid_keys:
					params[k] = v

	gdict = {}
	def parse_list(val):
		return np.array([float(x) for x in re.split(r'[,\s]+', val) if x], dtype=float)

	# Parse each provided key exactly; do not fabricate missing ones here.
	for k,v in params.items():
		if k in ('width','tau','lfrac','vfrac','dPA','mg_width_low','mg_width_high'):
			gdict[k] = parse_list(v)
		else:
			try:
				gdict[k] = np.array([float(v)], dtype=float)
			except ValueError:
				# Skip invalid numeric
				continue

	# Simple label fallback
	gdict['label'] = params.get('label','FRB')
	return gdict


def load_data(obs_data_path, obs_params_path, gauss_file=None, sim_file=None, scint_file=None):
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
	
	gdict = {}
	if obs_params_path is None:
		obs_params_path = os.path.join(data_dir, "parameters.txt")

	if os.path.exists(obs_params_path):
		logging.info(f"Loading parameters from {os.path.basename(obs_params_path)}")
		gdict = get_parameters(obs_params_path)
	else:
		logging.warning(f"Parameters file not found: {obs_params_path}")

	# Merge gauss_file only for truly missing keys (not for present zeros)
	sd_dict = None
	if gauss_file is not None:
		gauss_params = np.loadtxt(gauss_file)
		stddev_row = -4  # fourth last row = std dev
		mean_slice = gauss_params[:stddev_row, :]  # all mean rows (allow multi-component)
		col_map = {
			't0'            : 0,
			'width'         : 1,
			'A'             : 2,
			'spec_idx'      : 3,
			'tau'           : 4,
			'DM'            : 5,
			'RM'            : 6,
			'PA'            : 7,
			'lfrac'         : 8,
			'vfrac'         : 9,
			'dPA'           : 10,
			'band_centre'   : 11,
			'band_width'    : 12,
			'N'             : 13,
			'mg_width_low'  : 14,
			'mg_width_high' : 15
		}
		for k,c in col_map.items():
			if k not in gdict:
				gdict[k] = mean_slice[:, c]

		sd_dict = {
			'sd_t0'            : gauss_params[stddev_row, 0],
			'sd_width'         : gauss_params[stddev_row, 1],
			'sd_A'             : gauss_params[stddev_row, 2],
			'sd_spec_idx'      : gauss_params[stddev_row, 3],
			'sd_tau'           : gauss_params[stddev_row, 4],
			'sd_DM'            : gauss_params[stddev_row, 5],
			'sd_RM'            : gauss_params[stddev_row, 6],
			'sd_PA'            : gauss_params[stddev_row, 7],
			'sd_lfrac'         : gauss_params[stddev_row, 8],
			'sd_vfrac'         : gauss_params[stddev_row, 9],
			'sd_dPA'           : gauss_params[stddev_row,10],
			'sd_band_centre'   : gauss_params[stddev_row,11],
			'sd_band_width'    : gauss_params[stddev_row,12]
		}

	# Mandatory keys (exact gparams names)
	mandatory = [
		't0','width_ms','A','spec_idx','tau_ms','DM','RM','PA',
		'lfrac','vfrac','dPA','band_centre_mhz','band_width_mhz',
		'N','mg_width_low','mg_width_high'
	]
	for key in mandatory:
		if key not in gdict:
			gdict[key] = np.array([0.0], dtype=float)

	if sim_file is None:
		sim_file = load_params("simparams", None, "simulation")
	else:
		sim_file = load_params("simparams", sim_file, "simulation")

	f_res       = float(sim_file['f_res'])
	t_res       = float(sim_file['t_res'])
	scatter_idx = float(sim_file['scattering_index'])
	ref_freq    = float(sim_file['reference_freq'])

	if scint_file is not None:
		scint = load_params("scparams", scint_file, "scintillation")
		if scint.get("derive_from_tau", False):
			tau_ms_ref = float(gdict["tau"][0])  # tau already in ms
			tau_s_ref  = 1e-3 * tau_ms_ref
			nu_s_hz    = 1.0 / (2.0 * np.pi * tau_s_ref)
			scint["nu_s"] = float(nu_s_hz)
			logging.info(f"Derived nu_s at reference {ref_freq:.1f} MHz: tau={tau_ms_ref:.3f} ms -> nu_s={nu_s_hz:.2f} Hz")
	else:
		scint = None

	dspec_params = dspecParams(
		gdict        = gdict,
		sd_dict      = sd_dict,
		scint_dict   = scint,
		freq_mhz     = freq_mhz,
		freq_res_mhz = f_res,
		time_ms      = time_ms,
		time_res_ms  = t_res,
		seed         = None,
		nseed        = None,
		sefd         = None,
		sc_idx       = scatter_idx,
		ref_freq_mhz = ref_freq,
		phase_window = None,
		freq_window  = None,
		buffer_frac  = None,
		sweep_mode   = None
	)

	logging.info(
		f"Final data shape: {dspec.shape}, "
		f"freq range: {freq_mhz.min():.1f}-{freq_mhz.max():.1f} MHz, "
		f"time range: {time_ms.min():.3f}-{time_ms.max():.3f} ms"
	)

	return dspec, freq_mhz, time_ms, dspec_params


def load_multiple_data_grouped(data):
	"""
	Group simulation outputs by override parameters (e.g., N, tau, lfrac).
	Loads ALL files per group and merges their xvals/measures together.
	Returns unwrapped dict if single series, dict-of-dicts if multiple series.
	"""
	from collections import defaultdict
	import os, re, numpy as np, pickle, logging
	
	logging.info(f"Loading grouped data from {data}")
  
	def normalise_override_value(value_str):
		"""Normalise numeric string for filenames/labels (keep ints as ints)."""
		try:
			val = float(value_str)
			return str(int(val)) if float(val).is_integer() else f"{val:g}"
		except Exception:
			return value_str
	
	def extract_override_key(fname):
		"""
		Extract override parameters from filename for grouping.

		Outputs a label with no underscores:
		  tokens "param=value" joined by '+', with '_' in param names replaced by '.'
		Compatible with old compact suffixes like 'N100_sd_mg_width_low0.2'.
		"""
		m = re.search(r'_mode_psn_(.+?)\.pkl$', fname)
		if not m:
			return ""
		override_str = m.group(1)

		# Scan tokens: optional sd prefix + param + number (support scientific notation)
		pattern = r'(sd_?)?([A-Za-z][A-Za-z0-9_]*?)(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
		tokens = []
		for mm in re.finditer(pattern, override_str):
			sd_prefix, param, value = mm.groups()
			val_norm = normalise_override_value(value)
			prefix = "sd." if sd_prefix else ""
			tokens.append(f"{prefix}{param}={val_norm}")

		# If nothing matched, fall back to a safe echo of the suffix
		if not tokens:
			return override_str

		return ", ".join(tokens)

	def extract_override_params_for_sorting(override_key):
		"""
		Convert override_key to dict for sorting.
		Supports both new 'p=v+p=v' and old 'paramvalue_paramvalue' formats.
		"""
		if not override_key:
			return {}
		override_dict = {}

		# New safe format
		if ('=' in override_key) or ('+' in override_key):
			for tok in override_key.split('+'):
				if not tok or '=' not in tok:
					continue
				k, v = tok.split('=', 1)
				k = k.strip().replace('.', '_')
				try:
					override_dict[k] = float(v)
				except Exception:
					pass
			return override_dict

		# Fallback: compact tokens like 'N100_sd_mg_width_low0.2'
		for param, value in re.findall(r'([A-Za-z_]+)(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)', override_key):
			try:
				override_dict[param] = float(value)
			except Exception:
				pass
		return override_dict
	
	def sort_key_for_overrides(override_key):
		"""
		Generate sort key for override parameters.
		"""
		override_params = extract_override_params_for_sorting(override_key)
		# Normalise keys for sorting
		override_params = {k.replace('.', '_'): v for k, v in override_params.items()}

		param_order = ['N', 'tau', 'tau_ms', 'lfrac', 'vfrac', 'PA', 'DM', 'RM', 'width', 'width_ms']
		sort_tuple = tuple(override_params.get(p, 0) for p in param_order)
		return sort_tuple
	
	file_names = [f for f in os.listdir(data) if f.endswith(".pkl")]
	logging.info(f"Found {len(file_names)} .pkl files")
	
	groups = defaultdict(list)
	
	# Group files by override parameters
	for fname in file_names:
		override_key = extract_override_key(fname)
		groups[override_key].append(fname)
	
	# Sort groups by override parameters
	sorted_keys = sorted(groups.keys(), key=sort_key_for_overrides)
	
	logging.info(f"Grouped files into {len(groups)} unique series (sorted by override params):")
	for override_key in sorted_keys:
		label = override_key if override_key else "baseline"
		logging.info(f"  '{label}': {len(groups[override_key])} files")

	all_results = {}
	
	for override_key in sorted_keys:
		file_list = groups[override_key]
		
		xname = None
		plot_mode = None
		dspec_params = None
		all_xvals = []
		all_measures = {}
		all_V_params = {}
		all_exp_vars = {}
		all_snrs = {}

		seen_xvals = set()
		label = override_key if override_key else "baseline"
		logging.info(f"Merging {len(file_list)} files for series '{label}'")
		
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
		series_key = label  # Use override params as key
		all_results[series_key] = {
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
			f"Merged '{label}': {len(all_xvals)} unique xvals, "
			f"range {min(all_xvals):.1f}-{max(all_xvals):.1f}"
		)

	logging.info(f"Returning {len(all_results)} unique series (sorted)\n")
	
	# If only one series, return unwrapped dict instead of nested structure
	if len(all_results) == 1:
		single_key = list(all_results.keys())[0]
		logging.info(f"Single series detected ('{single_key}'), returning unwrapped dict for window comparison")
		return all_results[single_key]
	
	return all_results