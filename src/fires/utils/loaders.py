# -----------------------------------------------------------------------------
# loaders.py
# FIRES I/O utilities for loading observed data and grouped simulation outputs
# -----------------------------------------------------------------------------

import glob
import logging
import math
import os
import pickle as pkl
import re
from collections import OrderedDict, defaultdict

import numpy as np

from fires.utils.config import load_params
from fires.utils.params import COL_MAP
from fires.utils.stokes_npz import read_stokes_npz
from fires.utils.utils import dspecParams

# Mean-only column map for loading observed data
_MEAN_COL_MAP = {k: v for k, v in COL_MAP.items() if not k.startswith('sd_') and v < 16}

logging.basicConfig(level=logging.INFO)


def get_parameters(filepath):
	"""
	Parse a parameters.txt file with key = value format.
	
	Extracts relevant parameters for FRB analysis and converts to FIRES format.
	"""
	valid_keys = {
		't0','width','A','spec_idx','tau','DM','RM','PA',
		'lfrac','vfrac','dPA','band_centre','band_width',
		'N','mg_width_low','mg_width_high',
		'label'  # allow label from parameters.txt
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
		if k == 'label':
			# keep label as a raw string
			gdict[k] = v.strip()
			continue
		if k in ('width','tau','lfrac','vfrac','dPA','mg_width_low','mg_width_high'):
			gdict[k] = parse_list(v)
		else:
			try:
				gdict[k] = np.array([float(v)], dtype=float)
			except ValueError:
				# Skip invalid numeric
				continue

	# Simple label fallback
	gdict['label'] = gdict.get('label', params.get('label','FRB'))
	return gdict


_STOKES = ('I', 'Q', 'U', 'V')
_HTR_SUFFIX = re.compile(r'_htr_ds[IQUV]\.npy$')


def _first_existing(paths):
	"""Return the first path in ``paths`` that exists, or None."""
	return next((p for p in paths if os.path.exists(p)), None)


def _resolve_obs_location(obs_data_path):
	"""Return (data_dir, base_pattern) for a data directory or a single Stokes file path."""
	if os.path.isdir(obs_data_path):
		htr_files = glob.glob(os.path.join(obs_data_path, "*_htr_dsI.npy"))
		base = os.path.basename(htr_files[0]).replace("_htr_dsI.npy", "") if htr_files else ""
		return obs_data_path, base
	basename = os.path.basename(obs_data_path)
	if basename.endswith("_stokes.npz"):
		return os.path.dirname(obs_data_path), basename[: -len("_stokes.npz")]
	m = _HTR_SUFFIX.search(basename)
	return os.path.dirname(obs_data_path), basename[: m.start()] if m else ""


def _stokes_candidates(data_dir, base, stokes):
	"""File names tried, in order, for one Stokes parameter."""
	names = [f"{base}_htr_ds{stokes}.npy"] if base else []
	names += [f"out_{stokes}.npy", f"out_{stokes}_noflagged.npy"]
	if base:
		names += [f"{base}_{stokes}.npy", f"{base}_ds{stokes}.npy"]
	names += [f"{stokes}.npy", f"stokes_{stokes}.npy"]
	return [os.path.join(data_dir, n) for n in names]


def _axis_candidates(data_dir, base, kind):
	"""File names tried, in order, for the frequency ('freq') or time ('time') axis."""
	zoom, generic = {"freq": ("zoom_x_0.npy", ["freq.npy", "frequency.npy", "freqs.npy"]),
					 "time": ("zoom_y_0.npy", ["time.npy", "times.npy"])}[kind]
	paths = [os.path.join(data_dir, f"{base}_htr_{kind}.npy")] if base else []
	paths += [os.path.join(data_dir, n) for n in [zoom] + generic]
	paths += glob.glob(os.path.join(data_dir, f"*_{kind}.npy"))
	paths += glob.glob(os.path.join(data_dir, f"*_htr_{kind}.npy"))
	return paths


def _load_stokes_npy(data_dir, base):
	"""Stack per-Stokes .npy files into [4, nfreq, ntime]; missing Q/U/V become zeros."""
	arrays = []
	for stokes in _STOKES:
		path = _first_existing(_stokes_candidates(data_dir, base, stokes))
		if path is not None:
			logging.info(f"Found Stokes {stokes}: {os.path.basename(path)}")
			arrays.append(np.load(path))
			continue
		logging.warning(f"Stokes {stokes} file not found in {data_dir}")
		if stokes == 'I':
			raise FileNotFoundError(
				f"Stokes I file required but not found in {data_dir}\n"
				f"Tried patterns: out_I.npy, {base}_htr_dsI.npy, etc."
			)
		arrays.append(np.zeros_like(arrays[0]))
	return np.array(arrays)


def _load_axis(data_dir, base, kind, n):
	"""Load the freq/time axis file, or fall back to bin indices if missing or the wrong length."""
	label = {"freq": ("frequency", "MHz", "channels", "freq"), "time": ("time", "ms", "samples", "time")}[kind]
	path = _first_existing(_axis_candidates(data_dir, base, kind))
	axis = None
	if path is not None:
		axis = np.load(path)
		logging.info(f"Loaded {label[0]} array from {os.path.basename(path)}: {len(axis)} {label[2]}")
	else:
		logging.warning(f"No {label[0]} array found, creating default {label[1]} array")
	if axis is not None and len(axis) != n:
		logging.warning(
			f"{label[0].capitalize()} array length ({len(axis)}) doesn't match dspec {label[3]} axis ({n}). "
			"Creating default array."
		)
		axis = None
	return np.arange(n, dtype=float) if axis is None else axis


def _ensure_ascending_freq(dspec, freq_mhz):
	"""Reverse the frequency axis (and data) if it is descending."""
	try:
		diffs = np.diff(freq_mhz.astype(float))
		if np.nanmedian(diffs) < 0 and np.all(np.isfinite(diffs)):
			logging.info("Frequency axis is descending; reversing frequency axis and data.")
			return dspec[:, ::-1, :], freq_mhz[::-1]
		if np.any(diffs == 0):
			logging.warning("Frequency array contains duplicate channels; leaving order unchanged.")
		elif np.any(diffs < 0) and np.any(diffs > 0):
			logging.warning("Frequency array is non-monotonic; not reversing. Consider sorting externally.")
	except Exception as e:
		logging.warning(f"Could not evaluate frequency ordering: {e}")
	return dspec, freq_mhz


def _find_stokes_npz(obs_data_path, data_dir, base):
	"""Path of a shared-format .npz to load, or None to use per-Stokes .npy files.

	An explicit .npz path is always used. A directory is read from a single ``*_stokes.npz``
	only when it has no per-Stokes I file, so existing .npy layouts load exactly as before.
	"""
	if os.path.isfile(obs_data_path) and obs_data_path.endswith(".npz"):
		return obs_data_path
	if os.path.isdir(obs_data_path) and _first_existing(_stokes_candidates(data_dir, base, 'I')) is None:
		archives = glob.glob(os.path.join(data_dir, "*_stokes.npz"))
		if len(archives) == 1:
			return archives[0]
	return None


def _derive_label(gdict, npz_label, base_pattern, data_dir):
	"""Fill in gdict['label']: from the archive, else from the file/directory name, as 'FRB <date code>'."""
	if npz_label and gdict.get('label', 'FRB') == 'FRB':
		gdict['label'] = str(npz_label)
	try:
		current_label = gdict.get('label', 'FRB')
		if current_label == 'FRB' or not isinstance(current_label, str) or not current_label.strip():
			base_candidate = base_pattern if base_pattern else os.path.basename(os.path.normpath(data_dir))
			base_candidate = re.sub(r'_htr.*$', '', base_candidate)
			# Keep optional trailing letter on the date code (e.g., 240318A, 20240318A)
			m = re.search(r'(\d{6,8}[A-Za-z]+|\d{6,8}|\d{5,}[A-Za-z]+|\d{5,})', base_candidate)
			new_label = f"FRB {m.group(1)}" if m else base_candidate
			gdict['label'] = new_label
			logging.info(f"Derived observational label: {new_label}")
	except Exception as e:
		logging.debug(f"Failed to derive label automatically: {e}")

	try:
		lbl = gdict.get('label', '').strip()
		if lbl and not re.match(r'(?i)^FRB\b', lbl) and re.fullmatch(r'\d{5,8}[A-Za-z]?', lbl):
			gdict['label'] = f"FRB {lbl}"
			logging.info(f"Normalised label: {gdict['label']}")
	except Exception as e:
		logging.debug(f"Failed to normalise label: {e}")


# gdict keys that downstream code expects; missing ones default to 0
_MANDATORY_KEYS = ('t0', 'width', 'A', 'spec_idx', 'tau', 'DM', 'RM', 'PA', 'lfrac', 'vfrac', 'dPA',
				   'band_centre_mhz', 'band_width_mhz', 'N', 'mg_width_low', 'mg_width_high')


# Standard-deviation keys in column order of a gauss parameter table
_SD_KEYS = ('sd_t0', 'sd_width', 'sd_A', 'sd_spec_idx', 'sd_tau', 'sd_DM', 'sd_RM', 'sd_PA',
			'sd_lfrac', 'sd_vfrac', 'sd_dPA', 'sd_band_centre', 'sd_band_width')


def _merge_gauss_file(gdict, gauss_file):
	"""
	Fill keys missing from gdict with the mean rows of a numeric gauss parameter table, and
	return its standard-deviation row as sd_dict (None if there is no usable table).
	"""
	if gauss_file is None:
		return None
	try:
		gauss_params = np.atleast_2d(np.loadtxt(gauss_file))
		if gauss_params.shape[0] < 4 or gauss_params.shape[1] < 16:
			raise ValueError(
				f"Expected gauss parameter table with >=4 rows and >=16 cols, got {gauss_params.shape}"
			)
		stddev_row = -4  # fourth last row = std dev
		mean_slice = gauss_params[:stddev_row, :]  # all mean rows (allow multi-component)
		for k, c in _MEAN_COL_MAP.items():
			if k not in gdict:
				gdict[k] = mean_slice[:, c]
		return {key: gauss_params[stddev_row, col] for col, key in enumerate(_SD_KEYS)}
	except Exception as e:
		logging.warning(
			f"Skipping gauss_file merge for '{gauss_file}': {e}. "
			"Expected a numeric gauss parameter table."
		)
		return None


def _resolve_sim_params(sim_file):
	"""
	(f_res, t_res, scattering_index, reference_freq) from a legacy simparams file or a fires.toml
	(simulation.grid and propagation.scattering.index). The packaged config is used if sim_file is None.
	"""
	raw_sim = None
	raw_fires = None
	if sim_file is None:
		try:
			raw_sim = load_params("simparams", None, "simulation")
		except Exception:
			raw_fires = load_params("fires", None, None)
	else:
		try:
			raw_sim = load_params("simparams", sim_file, "simulation")
		except Exception:
			raw_sim = None
		# Always try the full fires config too: sim_file may be a fires.toml path.
		try:
			raw_fires = load_params("fires", sim_file, None)
		except Exception:
			raw_fires = None

	canonical_sim = dict(raw_sim) if isinstance(raw_sim, dict) else {}

	# New schema: simulation.grid in fires.toml
	if 'f_res' not in canonical_sim or 't_res' not in canonical_sim or 'reference_freq' not in canonical_sim:
		grid = None
		if isinstance(raw_sim, dict) and isinstance(raw_sim.get('grid'), dict):
			grid = raw_sim.get('grid')
		elif isinstance(raw_fires, dict):
			sim_section = raw_fires.get('simulation', {})
			if isinstance(sim_section, dict) and isinstance(sim_section.get('grid'), dict):
				grid = sim_section.get('grid')
		if isinstance(grid, dict):
			canonical_sim.setdefault('f_res', grid.get('df_MHz'))
			canonical_sim.setdefault('t_res', grid.get('dt_ms'))
			canonical_sim.setdefault('reference_freq', grid.get('reference_freq_MHz'))

	# New schema: propagation.scattering.index in fires.toml
	if 'scattering_index' not in canonical_sim and isinstance(raw_fires, dict):
		prop = raw_fires.get('propagation', {})
		if isinstance(prop, dict):
			sc = prop.get('scattering', {})
			if isinstance(sc, dict):
				canonical_sim['scattering_index'] = sc.get('index')

	required = ('f_res', 't_res', 'scattering_index', 'reference_freq')
	missing = [k for k in required if canonical_sim.get(k) is None]
	if missing:
		raise KeyError(
			"Missing required simulation keys "
			f"{missing}. Provide a legacy simparams file or a fires.toml with simulation.grid and propagation.scattering.index."
		)
	return tuple(float(canonical_sim[k]) for k in required)


def load_data(obs_data_path, obs_params_path, gauss_file=None, sim_file=None, scint_file=None):
	"""
	Load real FRB data and its parameter file.

	Accepted inputs (``obs_data_path``):
	- a shared-format archive ``{label}_stokes.npz`` (see fires.utils.stokes_npz), or a directory
	  holding exactly one such archive and no per-Stokes files;
	- a directory, or one Stokes file, with per-Stokes files ``out_I.npy`` ... or
	  ``{label}_htr_dsI.npy`` ..., plus ``{label}_htr_freq.npy`` / ``zoom_x_0.npy`` / ``freq.npy``
	  and ``{label}_htr_time.npy`` / ``zoom_y_0.npy`` / ``time.npy``.
	Parameters come from ``obs_params_path`` or ``parameters.txt`` next to the data (optional).

	Returns
	-------
	tuple
		(dspec, freq_mhz, time_ms, dspec_params) with dspec shaped [4, nfreq, ntime] and
		frequency ascending.
	"""
	data_dir, base_pattern = _resolve_obs_location(obs_data_path)
	logging.info(f"Loading real FRB data from {data_dir}")
	if base_pattern:
		logging.info(f"Using base pattern: {base_pattern}")

	npz_label = None
	npz_path = _find_stokes_npz(obs_data_path, data_dir, base_pattern)
	if npz_path is not None:
		dspec, freq_mhz, time_ms, npz_meta = read_stokes_npz(npz_path)
		npz_label = npz_meta.get("label")
		logging.info(f"Loaded Stokes archive {os.path.basename(npz_path)}")
	else:
		dspec = _load_stokes_npy(data_dir, base_pattern)
	logging.info(f"Loaded dynamic spectrum with shape: {dspec.shape} [Stokes, freq, time]")
	if npz_path is None:
		freq_mhz = _load_axis(data_dir, base_pattern, "freq", dspec.shape[1])
	dspec, freq_mhz = _ensure_ascending_freq(dspec, freq_mhz)
	if npz_path is None:
		time_ms = _load_axis(data_dir, base_pattern, "time", dspec.shape[2])
	logging.info("Time resolution: %.6f ms" % (time_ms[1]-time_ms[0]))
	logging.info("Frequency resolution: %.6f MHz" % np.median(np.diff(freq_mhz.astype(float))))

	if obs_params_path is None:
		obs_params_path = os.path.join(data_dir, "parameters.txt")
	if os.path.exists(obs_params_path):
		logging.info(f"Loading parameters from {os.path.basename(obs_params_path)}")
		gdict = get_parameters(obs_params_path)
	else:
		logging.info(f"Parameters file not found: {obs_params_path} (optional for real data)")
		gdict = {}
	_derive_label(gdict, npz_label, base_pattern, data_dir)

	# Common call-site mix-up: passing fires.toml as gauss_file.
	# If gauss_file looks like a TOML config and sim_file is unset, treat it as sim_file.
	if sim_file is None and isinstance(gauss_file, str) and gauss_file.lower().endswith('.toml'):
		logging.info(
			f"Interpreting gauss_file='{gauss_file}' as sim_file (TOML config detected)."
		)
		sim_file = gauss_file
		gauss_file = None

	# Merge gauss_file only for truly missing keys (not for present zeros).
	sd_dict = _merge_gauss_file(gdict, gauss_file)

	for key in _MANDATORY_KEYS:
		if key not in gdict:
			gdict[key] = np.array([0.0], dtype=float)

	f_res, t_res, scatter_idx, ref_freq = _resolve_sim_params(sim_file)

	scint = None
	if scint_file is not None:
		scint = load_params("scparams", scint_file, "scintillation")
		if scint.get("derive_from_tau", False):
			tau_ref = float(gdict["tau"][0])  # tau already in ms
			nu_s_hz = 1.0 / (2.0 * np.pi * (1e-3 * tau_ref))
			scint["nu_s"] = float(nu_s_hz)
			logging.info(f"Derived nu_s at reference {ref_freq:.1f} MHz: tau={tau_ref:.3f} ms -> nu_s={nu_s_hz:.2f} Hz")

	dspec_params = dspecParams(
		gdict        = gdict,
		sd_dict      = sd_dict,
		scint_dict   = scint,
		prop_dict    = {"scattering_index": scatter_idx, "scattering_screen": None, "RM": None, "order": None},
		freq_mhz     = freq_mhz,
		freq_res_mhz = f_res,
		time_ms      = time_ms,
		time_res_ms  = t_res,
		seed         = None,
		nseed        = None,
		sefd         = None,
		ref_freq_mhz = ref_freq,
		phase_window = None,
		freq_window  = None,
		buffer_frac  = None,
		sweep_mode   = None
	)

	logging.info(
		f"Final data shape: {dspec.shape}, "
		f"freq range: {freq_mhz.min():.1f}-{freq_mhz.max():.1f} MHz, "
		f"time range: {time_ms.min():.3f}-{time_ms.max():.3f} ms \n"
	)

	return dspec, freq_mhz, time_ms, dspec_params


def scrunch_dspec(dspec, freq_mhz, time_ms, dspec_params=None, t_scrunch=1, f_scrunch=1):
	"""
	Average (scrunch) a Stokes dynamic spectrum in time and/or frequency.

	Preprocessing step for input data: averages blocks of adjacent bins,
	degrading resolution by the integer factors given.

	Parameters
	----------
	dspec : np.ndarray
		Dynamic spectrum of shape [4, nfreq, ntime] (Stokes I/Q/U/V).
	freq_mhz : np.ndarray
		Frequency axis (MHz).
	time_ms : np.ndarray
		Time axis (ms).
	dspec_params : dspecParams, optional
		If given, returned with freq_mhz/time_ms and resolutions updated
		(resolution scales up by the scrunch factor).
	t_scrunch : int
		Time scrunch factor (>= 1; 1 disables time scrunching).
	f_scrunch : int
		Frequency scrunch factor (>= 1; 1 disables frequency scrunching).

	Returns
	-------
	tuple
		(dspec, freq_mhz, time_ms, dspec_params) after scrunching.
	"""
	t_scrunch = max(int(t_scrunch if t_scrunch is not None else 1), 1)
	f_scrunch = max(int(f_scrunch if f_scrunch is not None else 1), 1)

	if t_scrunch > 1:
		n_t = (dspec.shape[2] // t_scrunch) * t_scrunch
		if n_t > 0:
			dspec = np.nanmean(
				dspec[:, :, :n_t].reshape(
					dspec.shape[0], dspec.shape[1], -1, t_scrunch
				), axis=3
			)
			time_ms = time_ms[:n_t].reshape(-1, t_scrunch).mean(axis=1)
			logging.info(
				f"Time scrunched by factor {t_scrunch}: n_t={n_t} -> {dspec.shape[2]} bins"
			)

	if f_scrunch > 1:
		n_f = (dspec.shape[1] // f_scrunch) * f_scrunch
		if n_f > 0:
			dspec = np.nanmean(
				dspec[:, :n_f, :].reshape(
					dspec.shape[0], -1, f_scrunch, dspec.shape[2]
				), axis=2
			)
			freq_mhz = freq_mhz[:n_f].reshape(-1, f_scrunch).mean(axis=1)
			logging.info(
				f"Frequency scrunched by factor {f_scrunch}: n_f={n_f} -> {dspec.shape[1]} channels"
			)

	if dspec_params is not None:
		dspec_params = dspec_params._replace(
			freq_mhz=freq_mhz,
			freq_res_mhz=dspec_params.freq_res_mhz * f_scrunch,
			time_ms=time_ms,
			time_res_ms=dspec_params.time_res_ms * t_scrunch,
		)
	return dspec, freq_mhz, time_ms, dspec_params


def merge_sweep_results(results):
	"""
	Merge sweep result dicts (as pickled by a sweep run) into one: xvals are united and sorted,
	and per-xval measures, snrs, V_params and exp_vars lists are concatenated. Non-dict entries
	are skipped. xname, plot_mode and dspec_params come from the first result.
	"""
	xname = None
	plot_mode = None
	dspec_params = None
	all_xvals = []
	all_measures = {}
	all_V_params = {}
	all_exp_vars = {}
	all_snrs = {}
	seen_xvals = set()

	for obj in results:
		if not isinstance(obj, dict):
			continue
		if xname is None:
			xname = obj.get("xname")
			plot_mode = obj.get("plot_mode")
			dspec_params = obj.get("dspec_params")

		V_params = obj.get("V_params", {})
		exp_vars = obj.get("exp_vars", {})
		for v in obj.get("xvals", []):
			if v not in seen_xvals:
				seen_xvals.add(v)
				all_xvals.append(v)
			if v not in all_measures:
				all_measures[v] = []
				all_snrs[v] = []
				all_V_params[v] = {key: [] for key in V_params.get(v, {}).keys()}
				all_exp_vars[v] = {key: [] for key in exp_vars.get(v, {}).keys()}
			all_measures[v].extend(obj.get("measures", {}).get(v, []))
			all_snrs[v].extend(obj.get("snrs", {}).get(v, []))
			for key, arr in V_params.get(v, {}).items():
				all_V_params[v].setdefault(key, []).extend(arr)
			for key, arr in exp_vars.get(v, {}).items():
				all_exp_vars[v].setdefault(key, []).extend(arr)

	return {
		'xname': xname,
		'xvals': sorted(all_xvals),
		'measures': all_measures,
		'V_params': all_V_params,
		'exp_vars': all_exp_vars,
		'dspec_params': dspec_params,
		'plot_mode': plot_mode,
		'snrs': all_snrs,
	}


def parse_override_key(override_key):
	"""Parse a canonical override key "param=val+param=val" into {param: float(val)}, skipping non-numeric values."""
	out = {}
	if not override_key:
		return out
	for tok in override_key.split('+'):
		if '=' not in tok:
			continue
		k, v = tok.split('=', 1)
		k = k.strip().replace('.', '_')
		try:
			out[k] = float(v)
		except Exception:
			continue
	return out


_PARAM_TOKEN_RE = re.compile(r'(sd_?)?([A-Za-z][A-Za-z0-9_]*?)(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)')


def _normalise_override_value(value_str):
	try:
		val = float(value_str)
		return str(int(val)) if float(val).is_integer() else f"{val:g}"
	except Exception:
		return value_str


def _extract_override_key(fname):
	"""
	Canonical override key "param=val+param=val" (sorted by param name) from a sweep pickle's
	file name suffix (after "_mode_psn_"). Empty string for the baseline (no overrides).
	"""
	m = re.search(r'_mode_psn_(.+?)\.pkl$', fname)
	if not m:
		return ""
	toks = []
	for mm in _PARAM_TOKEN_RE.finditer(m.group(1)):
		sd_prefix, p, v = mm.groups()
		toks.append((f"sd.{p}" if sd_prefix else p, _normalise_override_value(v)))
	toks.sort(key=lambda x: x[0])
	return "+".join(f"{p}={v}" for p, v in toks)


def _series_sort_key(key_dict):
	"""Order multi-run series by N, then tau, then width, then the other parameters alphabetically; absent values last."""
	inf = math.inf
	rest = sorted((k, v) for k, v in key_dict.items() if k not in ('N', 'tau', 'width'))
	return (key_dict.get('N', inf), key_dict.get('tau', inf), key_dict.get('width', inf)) + tuple(v for _, v in rest)


def load_multiple_data_grouped(data):
	"""
	Group simulation outputs by override parameters (e.g., N, tau, lfrac).
	Loads ALL files per group and merges their xvals/measures together.
	Returns unwrapped dict if single series, OrderedDict (sorted) if multiple.
	"""
	logging.info(f"Loading grouped data from {data}")

	file_names = [f for f in os.listdir(data) if f.endswith(".pkl")]
	logging.info(f"Found {len(file_names)} .pkl files")

	groups = defaultdict(list)
	for fname in file_names:
		groups[_extract_override_key(fname)].append(fname)

	sorted_keys = sorted(groups.keys(), key=lambda k: _series_sort_key(parse_override_key(k)))

	logging.info(f"Grouped into {len(sorted_keys)} series (loader-side sorted).")
	for k in sorted_keys:
		logging.info(f"  {k if k else 'baseline'}: {len(groups[k])} files")

	def load_group(file_list):
		for fname in file_list:
			with open(os.path.join(data, fname), "rb") as f:
				obj = pkl.load(f)
			if not isinstance(obj, dict):
				logging.warning(f"Skipping {fname}: unexpected structure")
			yield obj

	all_results = OrderedDict()
	for override_key in sorted_keys:
		all_results[override_key if override_key else "baseline"] = merge_sweep_results(load_group(groups[override_key]))

	if len(all_results) == 1:
		k = next(iter(all_results))
		logging.info(f"Single series '{k}' -> returning unwrapped dict.")
		return all_results[k]

	return all_results
