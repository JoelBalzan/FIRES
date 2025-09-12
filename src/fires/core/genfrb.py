# -----------------------------------------------------------------------------
# genfrb.py
# FIRES: The Fast, Intense Radio Emission Simulator
#
# This module contains functions for generating Fast Radio Burst (FRB) dynamic
# spectra, handling baseline subtraction, off-pulse window selection, scattering,
# data loading, and parallelized simulation and aggregation of FRB realizations.
# It is a core part of the FIRES simulation pipeline.
#
# Author: JB
# Date: 2025-05-20
# -----------------------------------------------------------------------------

# -------------------------- Import modules ---------------------------
import os
import sys
import inspect
import functools

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import product
from concurrent.futures import ProcessPoolExecutor

from .basicfns import scatter_stokes_chan, add_noise
from .genfns import *
from ..utils.utils import *



def _scatter_loaded_dynspec(dspec, freq_mhz, time_ms, tau_ms, sc_idx, ref_freq_mhz):
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
	for stokes_idx in range(dspec.shape[0]):  # Loop over I, Q, U, V
		for c in range(len(freq_mhz)):
			dspec_scattered[stokes_idx, c] = scatter_stokes_chan(
				dspec[stokes_idx, c], freq_mhz[c], time_ms, tau_ms, sc_idx, ref_freq_mhz
			)
	return dspec_scattered


def _load_data(frb_id, data, freq_mhz, time_ms):
	"""
	Load data from files with optional downsampling.
	
	Args:
		data: Path to data directory or file
		freq_mhz: Frequency array (will be updated if loading from directory)
		time_ms: Time array (will be updated if loading from directory)
		downsample_factor: Integer factor to downsample by (default: 1, no downsampling)
	
	Returns:
		dspec: Dynamic spectrum array
		freq_mhz: Updated frequency array
		time_ms: Updated time array
	"""
	print(f"Loading data from {data}...")

	dspec = np.load(data) if data.endswith('.npy') else None
	dspec = np.flip(dspec, axis=1)  # Flip frequency axis if needed
	
	#summary_file = [f for f in os.listdir(data) if f.endswith(f'.txt')]
	summary = get_parameters("parameters.txt")
	cfreq_mhz = float(summary['centre_freq_frb'])
	freq_mhz = np.flip(np.load(f"{frb_id}_freq.npy")) #np.linspace(cfreq_mhz - bw_MHz / 2, cfreq_mhz + bw_MHz / 2, dspec.shape[1])

	time_ms = np.load(f"{frb_id}_time.npy") #np.arange(0, dspec.shape[2] * time_res_ms, time_res_ms)
	print(f"Loaded data from {data} with frequency range: {freq_mhz[0]} - {freq_mhz[-1]} MHz")
		
	return dspec, freq_mhz, time_ms


def _load_multiple_data_grouped(data):
	"""
	Group simulation outputs by freq and phase info (everything after freq_ and phase_).
	Returns a dictionary: {freq_phase_key: {'xname': ..., 'xvals': ..., 'yvals': ..., ...}, ...}
	"""
	from collections import defaultdict
	import re
	
	print(f"Loading grouped data from {data}...")

	def extract_xvals_value(fname):
		# Match _xvals_<number> or _xvals_<number>-<number>
		m = re.search(r'_xvals_([0-9.]+)(?:-([0-9.]+))?', fname)
		if m:
			return float(m.group(1))
		return float('inf')  # Put files without _xvals_ at the end
	
	def extract_freq_phase_key(fname):
		# Extract freq and phase info from filename
		# Pattern: freq_{freq}_phase_{phase}.pkl
		m = re.search(r'freq_([^_]+)_phase_([^.]+)\.pkl$', fname)
		if m:
			freq_info = m.group(1)
			phase_info = m.group(2)
			return f"{freq_info}, {phase_info}"
		return "unknown"  # fallback for files that don't match pattern
	
	file_names = [f for f in os.listdir(data) if f.endswith(".pkl")]
	file_names = sorted(file_names, key=extract_xvals_value)
	groups = defaultdict(list)
	for fname in file_names:
		freq_phase_key = extract_freq_phase_key(fname)
		groups[freq_phase_key].append(fname)

	all_results = {}
	for freq_phase_key, files in groups.items():
		xname                 = None
		all_xvals             = []
		all_yvals             = {}
		all_errs              = {}
		all_var_params        = {}
		dspec_params          = None
		plot_mode             = None
		all_snrs 			  = {}

		for file_name in files:
			with open(os.path.join(data, file_name), "rb") as f:
				obj = pkl.load(f)
			xname        = obj["xname"]
			plot_mode    = obj["plot_mode"]
			xvals        = obj["xvals"]
			yvals        = obj["yvals"]
			errs         = obj["errs"]
			var_params   = obj["var_params"]
			dspec_params = obj["dspec_params"]
			snrs         = obj["snrs"]

			for v in xvals:
				if v not in all_yvals:
					all_yvals[v]      = []
					all_errs[v]       = []
					all_var_params[v] = {key: [] for key in var_params[v].keys()}
					all_snrs[v]       = []
				all_yvals[v].extend(yvals[v])
				all_errs[v].extend(errs[v])
				for key, values in var_params[v].items():
					all_var_params[v][key].extend(values)
				all_snrs[v].extend(snrs[v])
			all_xvals.extend(xvals)

		all_results[freq_phase_key] = {
			'xname'            : xname,
			'xvals'            : all_xvals,
			'yvals'            : all_yvals,
			'errs'             : all_errs,
			'var_params'       : all_var_params,
			'dspec_params'     : dspec_params,
			'plot_mode'        : plot_mode,
			'snrs'             : all_snrs	
		}

	return all_results


def _generate_dynspec(xname, mode, var, plot_multiple_frb, **params):
	"""Generate dynamic spectrum based on mode."""
	var = var if plot_multiple_frb else None

	# Choose the correct function
	if mode == 'gauss':
		dynspec_func = gauss_dynspec
	else:  # mode == 'psn'
		dynspec_func = m_gauss_dynspec

	# Get the argument names for the selected function
	sig = inspect.signature(dynspec_func)
	allowed_args = set(sig.parameters.keys())

	# Always pass tau_ms as v
	params_filtered = {
		k: v for k, v in params.items()
		if k in allowed_args and k not in ("xname")
	}
 
	if mode == 'psn':
		return m_gauss_dynspec(**params_filtered, variation_parameter=var, xname=xname, plot_multiple_frb=plot_multiple_frb)
	elif mode == 'gauss':
		return gauss_dynspec(**params_filtered, plot_multiple_frb=plot_multiple_frb)


def _process_task(task, xname, mode, plot_mode, **params):
	"""
	Process a single task (combination of timescale and realization).
	Dynamically uses the provided process_func for mode-specific processing.
	"""
	var, realization = task
	current_seed = params["seed"] + realization if params["seed"] is not None else None
	params["seed"] = current_seed
	
	requires_multiple_frb = plot_mode.requires_multiple_frb

	# Generate dynamic spectrum
	dspec, snr, var_params = _generate_dynspec(
		xname=xname,
		mode=mode,
		var=var,
		plot_multiple_frb=requires_multiple_frb,
		**params
	)

	process_func = plot_mode.process_func
 
	# Dynamically select only the needed arguments for process_func
	sig = inspect.signature(process_func)
	allowed_args = set(sig.parameters.keys())
	# Always provide dspec as the first argument
	process_func_args = {'dspec': dspec}
	# Add other allowed arguments from params if present
	process_func_args.update({
		k: params[k]
		for k in allowed_args if k in params and k not in ('dspec')
	})

	xvals, result_err = process_func(**process_func_args)

	return var, xvals, result_err, var_params, snr


def generate_frb(data, frb_id, out_dir, mode, seed, nseed, write,
				 obs_file, gauss_file, tsys, n_cpus, plot_mode, phase_window, freq_window):
	"""
	Generate a simulated FRB with a dispersed and scattered dynamic spectrum.
	"""
	obs_params = get_parameters(obs_file)

	# Extract frequency and time parameters
	f_start = float(obs_params['f0'])
	f_end   = float(obs_params['f1'])
	t_start = float(obs_params['t0'])
	t_end   = float(obs_params['t1'])
	f_res   = float(obs_params['f_res'])
	t_res   = float(obs_params['t_res'])

	scatter_idx = float(obs_params['scattering_index'])
	ref_freq 	= float(obs_params['reference_freq'])

	# Generate frequency and time arrays
	freq_mhz = np.arange(f_start, f_end + f_res, f_res, dtype=float)
	time_ms  = np.arange(t_start, t_end + t_res, t_res, dtype=float)

	# Load Gaussian parameters
	gauss_params = np.loadtxt(gauss_file)

	gdict = {
		't0'             : gauss_params[:-3, 0],
		'width_ms'       : gauss_params[:-3, 1],
		'peak_amp'       : gauss_params[:-3, 2],
		'spec_idx'       : gauss_params[:-3, 3],
		'tau_ms'         : gauss_params[:-3, 4],
		'DM'             : gauss_params[:-3, 5],
		'RM'             : gauss_params[:-3, 6],
		'PA'             : gauss_params[:-3, 7],
		'lfrac'          : gauss_params[:-3, 8],
		'vfrac'          : gauss_params[:-3, 9],
		'dPA'            : gauss_params[:-3, 10],
		'band_centre_mhz': gauss_params[:-3, 11],
		'band_width_mhz' : gauss_params[:-3, 12],
		'ngauss'         : gauss_params[:-3, 13],
		'mg_width_low'   : gauss_params[:-3, 14],
		'mg_width_high'  : gauss_params[:-3, 15]
	}
	
	var_dict = {
		't0_var'             : gauss_params[-3:, 0],
		'width_ms_var'       : gauss_params[-3:, 1],
		'peak_amp_var'       : gauss_params[-3:, 2],
		'spec_idx_var'       : gauss_params[-3:, 3],
		'tau_ms_var'         : gauss_params[-3:, 4],
		'DM_var'             : gauss_params[-3:, 5],
		'RM_var'             : gauss_params[-3:, 6],
		'PA_var'             : gauss_params[-3:, 7],
		'lfrac_var'          : gauss_params[-3:, 8],
		'vfrac_var'          : gauss_params[-3:, 9],
		'dPA_var'            : gauss_params[-3:, 10],
		'band_centre_mhz_var': gauss_params[-3:, 11],
		'band_width_mhz_var' : gauss_params[-3:, 12]
	}

	# Create dynamic spectrum parameters
	dspec_params = DynspecParams(
		gdict           = gdict,
		var_dict        = var_dict,
		freq_mhz        = freq_mhz,
		freq_res_mhz    = f_res,
		time_ms         = time_ms,
		time_res_ms     = t_res,
		seed            = seed,
		nseed           = nseed,
		tsys            = tsys,
		sc_idx          = scatter_idx,
		ref_freq_mhz    = ref_freq,
		phase_window    = phase_window,
		freq_window     = freq_window
	)

	tau_ms = gdict['tau_ms']

	if len(np.where(gauss_params[-1,:] != 0.0)[0]) > 1:
		print("WARNING: More than one value in the last row of gauss_params is not 0.")
		print("Please ensure that only one value is non-zero in the last row.")
		sys.exit(1)
  
	if np.any(gdict['lfrac'] + gdict['vfrac']) > 1.0:
		print("WARNING: Linear and circular polarization fractions sum to more than 1.0.\n")

	plot_multiple_frb = plot_mode.requires_multiple_frb
	if plot_multiple_frb == False:
		
		if data != None:
			dspec, freq_mhz, time_ms = _load_data(frb_id, data, freq_mhz, time_ms)
			snr = snr_onpulse(np.nansum(dspec[0], axis=0), time_ms, frac=0.95)  
			if tau_ms[0] > 0:
				dspec = _scatter_loaded_dynspec(dspec, freq_mhz, time_ms, tau_ms[0], scatter_idx, ref_freq)
			if tsys > 0:
				dspec, snr = add_noise(dynspec=dspec, t_sys=tsys, f_res=f_res, t_res=t_res, 
													time_ms=time_ms, plot_multiple_frb=plot_multiple_frb)
			
			# Update dspec_params with new time and frequency arrays
			dspec_params = dspec_params._replace(time_ms=time_ms, freq_mhz=freq_mhz)

		else:
			dspec, snr, _ = _generate_dynspec(
			xname=None,
			mode=mode,
			var=None,
			plot_multiple_frb=False,
			**dspec_params._asdict()
		)
		_, _, _, noise_spec = process_dynspec(dspec, freq_mhz, gdict)
		frb_data = simulated_frb(
			frb_id, dspec, dspec_params, snr
		)
		if write:
			tau = f"{tau_ms[0]:.2f}"
			if mode == 'gauss':
				out_file = (
					f"{out_dir}{frb_id}_mode_{mode}_sc_{tau}_"
					f"seed_{seed}_PA_{gdict['PA'][-1]:.2f}.pkl"
				)
			else:  # mode == 'psn'
				out_file = (
					f"{out_dir}{frb_id}_mode_{mode}_sc_{tau}_"
					f"seed_{seed}_nseed_{nseed}_PA_{gdict['PA'][-1]:.2f}.pkl"
				)
			with open(out_file, 'wb') as frb_file:
				pkl.dump(frb_data, frb_file)
		return frb_data, noise_spec, gdict

	else:
		if data != None:
			files = [f for f in os.listdir(data) if f.endswith('.pkl')]
			if len(files) > 1:
				frb_dict = _load_multiple_data_grouped(data)
			elif len(files) == 1:
				with open(os.path.join(data, files[0]), 'rb') as f:
					frb_dict = pkl.load(f)
			else:
				print(f"No .pkl files found in {data}.")
				sys.exit(1)
		else:		
			if np.all(gauss_params[-1,:] == 0.0):
				# For var plots, a sweep must be defined in the last three rows
				print("No sweep defined in gparams (last row all zeros), but a multi-FRB plot mode was requested.")
				print("Define start/stop/step in the last three rows for exactly one column.")
				sys.exit(1)
			else:
				# Find which column in gauss_params the final entry is not zero
				col_idx = np.where(gauss_params[-1, :] != 0.0)[0][0]
				start = gauss_params[-3, col_idx]
				stop = gauss_params[-2, col_idx]
				step = gauss_params[-1, col_idx]
				if step <= 0:
					raise ValueError("Sweep step must be > 0.")
				n_steps = int(np.round((stop - start) / step))
				n_steps = max(n_steps, 0)
				end = start + n_steps * step
				xvals = np.linspace(start, end, n_steps + 1)

				# If running under Slurm array, split xvals across tasks
				# Falls back to env FIRESSWEEP_COUNT/ID for local testing.
				def _slurm_array_size():
					cnt = os.environ.get("SLURM_ARRAY_TASK_COUNT")
					if cnt is not None:
						return int(cnt)
					# Derive count from MIN/MAX if COUNT is not provided
					min_s = os.environ.get("SLURM_ARRAY_TASK_MIN")
					max_s = os.environ.get("SLURM_ARRAY_TASK_MAX")
					if max_s is not None:
						min_i = int(min_s) if min_s is not None else 0
						return int(max_s) - min_i + 1
					# Custom override for non-Slurm runs
					return int(os.environ.get("FIRESSWEEP_COUNT", "1"))
				def _slurm_array_id():
					min_s = os.environ.get("SLURM_ARRAY_TASK_MIN")
					task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
					if task_id is not None:
						if min_s is not None:
							return int(task_id) - int(min_s)
						return int(task_id)
					# Custom override for non-Slurm runs
					return int(os.environ.get("FIRESSWEEP_ID", "0"))
				_array_count = _slurm_array_size()
				_array_id = _slurm_array_id()
				if _array_count > 1:
					total = len(xvals)
					chunk = (total + _array_count - 1) // _array_count  # ceil division
					start_idx = _array_id * chunk
					end_idx = min(start_idx + chunk, total)
					if start_idx >= total:
						print(f"Array task {_array_id}/{_array_count - 1}: no assigned xvals (start_idx={start_idx} >= {total}).")
						sys.exit(0)
					xvals = xvals[start_idx:end_idx]
					print(f"Array task {_array_id}/{_array_count - 1}: processing xvals[{start_idx}:{end_idx}] out of {total}")

				# Find the corresponding key in gdict for col_idx
				gdict_keys = list(gdict.keys())
				xname = gdict_keys[col_idx] + "_var"
	 
				tasks = list(product(xvals, range(nseed)))

				with ProcessPoolExecutor(max_workers=n_cpus) as executor:
					partial_func = functools.partial(
						_process_task,
						xname=xname,
						mode=mode,
						plot_mode=plot_mode,
						**dspec_params._asdict()
					)					
	  
					results = list(tqdm(executor.map(partial_func, tasks),
										total=len(tasks),
										desc=f"Processing {xname} variance and realisations"))

			
			yvals = {v: [] for v in xvals}
			errs = {v: [] for v in xvals}
			var_params = {v: {key: [] for key in ['var_t0', 'var_peak_amp', 'var_width_ms', 'var_spec_idx', 'var_tau_ms', 'var_PA', 'var_DM', 'var_RM', 'var_lfrac', 'var_vfrac', 
													'var_dPA', 'var_band_centre_mhz', 'var_band_width_mhz']} for v in xvals}
			snrs = {v: [] for v in xvals}

			for var, val, err, params_dict, snr in results:
				yvals[var].append(val)
				errs[var].append(err)
				snrs[var].append(snr)
				for key, value in params_dict.items():
					var_params[var][key].append(value)

			frb_dict = {
				"xname": xname,
				"xvals": xvals,
				"yvals": yvals,
				"errs": errs,
				"var_params": var_params, 
				"dspec_params": dspec_params,
				"plot_mode": plot_mode,
				"snrs": snrs,
			}

		if write:
			# Create a descriptive filename
			xvals = f"{xvals[0]:.2f}-{xvals[-1]:.2f}" if len(xvals) > 1 else f"{xvals[0]:.2f}"

			tau = f"{tau_ms[0]:.2f}" if len(tau_ms) == 1 else f"{tau_ms[0]:.2f}-{tau_ms[-1]:.2f}"

			out_file = (
				f"{out_dir}{frb_id}_plot_{plot_mode.name}_xname_{xname}_xvals_{xvals}_mode_{mode}_sc_{tau}_"
				f"freq_{freq_window}_phase_{phase_window}.pkl"
			)
			with open(out_file, 'wb') as frb_file:
				pkl.dump((frb_dict), frb_file)
			print(f"Saved FRB data to {out_file}")

		return frb_dict
