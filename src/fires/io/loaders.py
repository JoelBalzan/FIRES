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
	
	Parameters:
	-----------
	filepath : str
		Path to parameters.txt file
		
	Returns:
	--------
	dict
		Dictionary with parameter arrays (e.g., 'DM', 'RM', 'width', 'tau_ms')
	"""
	params = {}
	
	with open(filepath, 'r') as f:
		for line in f:
			line = line.strip()
			# Skip empty lines and section headers
			if not line or line.startswith('****') or line.startswith('#'):
				continue
			
			# Parse key = value
			if '=' in line:
				parts = line.split('=', 1)
				key = parts[0].strip()
				value = parts[1].strip()
				
				# Remove trailing comments
				if '#' in value:
					value = value.split('#')[0].strip()
				
				params[key] = value
	
	# Convert to FIRES format
	gdict = {}
	
	# DM
	if 'dm_frb' in params:
		try:
			gdict['DM'] = np.array([float(params['dm_frb'])])
		except ValueError:
			gdict['DM'] = np.array([0.0])
	else:
		gdict['DM'] = np.array([0.0])
	
	# RM (not in this format, default to 0)
	gdict['RM'] = np.array([0.0])
	
	# Width (estimate from data or use default)
	gdict['width'] = np.array([1.0])  # Will be updated from data if available
	
	# Tau (scattering timescale, not in this format)
	gdict['tau_ms'] = np.array([0.0])
	
	# Center frequency
	if 'centre_freq_frb' in params:
		try:
			gdict['band_centre_mhz'] = np.array([float(params['centre_freq_frb'])])
		except ValueError:
			pass
	
	# Bandwidth
	if 'bw' in params:
		try:
			gdict['band_width_mhz'] = np.array([float(params['bw'])])
		except ValueError:
			pass
	
	# RA/Dec for label
	if 'label' in params:
		gdict['label'] = params['label']
	elif 'ra_frb' in params and 'dec_frb' in params:
		gdict['label'] = f"RA={params['ra_frb']}, Dec={params['dec_frb']}"
	else:
		gdict['label'] = "FRB"
	
	logging.info(
		f"Parsed parameters: DM={gdict.get('DM', [0])[0]:.2f} pc/cm³, "
		f"center_freq={gdict.get('band_centre_mhz', ['N/A'])[0]} MHz, "
		f"bandwidth={gdict.get('band_width_mhz', ['N/A'])[0]} MHz"
	)
	
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
		gdict = {
			'DM': np.array([0.0]),
			'RM': np.array([0.0]),
		}

	if gauss_file is not None:	
		gauss_params = np.loadtxt(gauss_file)
		stddev_row   = -4  
		gparams = {
			't0'             : gauss_params[:stddev_row, 0],
			'width_ms'     	 : gauss_params[:stddev_row, 1],
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
		for k, v in gparams.items():
			if k not in gdict or (isinstance(v, np.ndarray) and np.all(gdict.get(k, None) == 0)):
				gdict[k] = v
		sd_dict = {
			'sd_t0'             : gauss_params[stddev_row, 0],
			'sd_width_ms'       : gauss_params[stddev_row, 1],
			'sd_A'       		: gauss_params[stddev_row, 2],
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
	else:
		sd_dict = None

	if sim_file is not None:
		sim_file = load_params("simparams", sim_file, "simulation")
		f_res   = float(sim_file['f_res'])
		t_res   = float(sim_file['t_res'])
		scatter_idx = float(sim_file['scattering_index'])
		ref_freq 	= float(sim_file['reference_freq'])
	else:
		f_res        = None
		t_res        = None
		scatter_idx  = None
		ref_freq     = None

	if scint_file is not None:
		scint = load_params("scparams", scint_file, "scintillation")
		if scint.get("derive_from_tau", False):
			tau_ms_ref = float(gdict["tau_ms"][0])          
			tau_s_ref  = 1e-3 * tau_ms_ref                  
			nu_s_hz    = 1.0 / (2.0 * np.pi * tau_s_ref)    
			scint["nu_s"] = float(nu_s_hz)
			logging.info(
				f"Derived nu_s at reference {ref_freq:.1f} MHz: "
				f"tau={tau_ms_ref:.3f} ms -> nu_s={nu_s_hz:.2f} Hz"
			)
	else:
		scint = None

	dspec_params = dspecParams(
		gdict           = gdict,
		sd_dict         = sd_dict,
		scint_dict      = scint,
		freq_mhz        = freq_mhz,
		freq_res_mhz    = f_res,
		time_ms         = time_ms,
		time_res_ms     = t_res,
		seed            = None,
		nseed           = None,
		sefd            = None,
		sc_idx          = scatter_idx,
		ref_freq_mhz    = ref_freq,
		phase_window    = None,
		freq_window     = None,
		buffer_frac     = None,
		sweep_mode      = None
	)

	logging.info(
		f"Final data shape: {dspec.shape}, "
		f"freq range: {freq_mhz.min():.1f}-{freq_mhz.max():.1f} MHz, "
		f"time range: {time_ms.min():.3f}-{time_ms.max():.3f} ms\n"
	)
	
	return dspec, freq_mhz, time_ms, dspec_params


def load_multiple_data_grouped(data):
	"""
	Group simulation outputs by override parameters (e.g., N, tau, lfrac).
	Loads ALL files per group and merges their xvals/measures together.
	Returns unwrapped dict if single series, dict-of-dicts if multiple series.
	"""
	import re
	from collections import defaultdict
	
	logging.info(f"Loading grouped data from {data}/")
  
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
	
	def extract_override_key(fname):
		"""
		Extract override parameters from filename for grouping.
		Pattern: sweep_{idx}_n{nseed}_plot_{mode}_xname_{xname}_xvals_{range}_mode_psn_{overrides}.pkl
	
		Examples:
		- sweep_0_n100_plot_l_frac_xname_PA_xvals_0.00-4.00_mode_psn_N100.pkl -> "N100"
		- sweep_5_n100_plot_l_frac_xname_PA_xvals_25.00-29.00_mode_psn_N1000.pkl -> "N1000"
	
		Returns: string like "N100" or "N100_tau5.5" or "" if no overrides
		"""
		m = re.search(r'_mode_psn_(.+?)\.pkl$', fname)
		if not m:
			return ""
	
		override_str = m.group(1)
	
		override_parts = override_str.split('_')
	
		normalised_parts = []
		for part in override_parts:
			match = re.match(r'^([a-zA-Z]+)([0-9.]+)$', part)
			if match:
				param = match.group(1)
				value = match.group(2)
				normalised_value = normalise_override_value(value)
				normalised_parts.append(f"{param}{normalised_value}")
			else:
				logging.debug(f"Skipping unexpected override part: {part}")
	
		return "_".join(normalised_parts) if normalised_parts else ""
	
	def extract_override_params_for_sorting(override_key):
		"""
		Convert override_key string to dict for sorting.
		Example: "N100_tau5.5" -> {'N': 100, 'tau': 5.5}
		"""
		if not override_key:
			return {}
		
		override_dict = {}
		param_pattern = r'([a-zA-Z_]+)([0-9.]+)'
		matches = re.findall(param_pattern, override_key)
		
		for param, value in matches:
			try:
				override_dict[param] = float(value)
			except ValueError:
				pass
		
		return override_dict
	
	def sort_key_for_overrides(override_key):
		"""
		Generate sort key for override parameters.
		Sorts by: N, tau, lfrac, vfrac, PA, DM, RM, width in that order.
		"""
		override_params = extract_override_params_for_sorting(override_key)
		
		# Sort order for common parameters
		param_order = ['N', 'tau', 'lfrac', 'vfrac', 'PA', 'DM', 'RM', 'width']
		
		sort_tuple = []
		for param in param_order:
			sort_tuple.append(override_params.get(param, 0))  # 0 if param not present
		
		return tuple(sort_tuple)
	
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