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
        Dictionary with parameter arrays (e.g., 'DM', 'RM', 'width_ms', 'tau_ms')
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
    gdict['width_ms'] = np.array([1.0])  # Will be updated from data if available
    
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


def load_multiple_data_grouped(data):
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
						normalised_value = normalise_override_value(value)
						cleaned_parts.append(f"{param}{normalised_value}")
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