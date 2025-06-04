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

from FIRES.functions.basicfns import scatter_stokes_chan, add_noise_to_dynspec
from FIRES.functions.genfns import *
from FIRES.utils.utils import *

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
obs_params_path = os.path.join(parent_dir, "utils/obsparams.txt")
gauss_params_path = os.path.join(parent_dir, "utils/gparams.txt")


def subtract_baseline_offpulse(dspec, off_start, off_end, axis=-1, method='median'):
	"""
	Subtracts the baseline using a specific off-pulse window, specified as fractions (0-1) of the axis length.
	Args:
		dspec: np.ndarray, shape (nstokes, nchan, ntime)
		off_start: float, start of off-pulse window as fraction (0-1)
		off_end: float, end of off-pulse window as fraction (0-1)
		axis: int, axis along which to estimate the baseline (default: time axis)
		method: 'median' or 'mean'
	Returns:
		dspec_baseline_subtracted: np.ndarray, same shape as dspec
	"""
	axis_len = dspec.shape[axis]
	start_idx = int(np.floor(off_start * axis_len))
	end_idx = int(np.floor(off_end * axis_len))
	slicer = [slice(None)] * dspec.ndim
	slicer[axis] = slice(start_idx, end_idx)
	offpulse_region = dspec[tuple(slicer)]

	if method == 'median':
		baseline = np.nanmedian(offpulse_region, axis=axis, keepdims=True)
	elif method == 'mean':
		baseline = np.nanmean(offpulse_region, axis=axis, keepdims=True)
	else:
		raise ValueError("method must be 'median' or 'mean'")
	return dspec - baseline



def select_offpulse_window(profile):
	"""
	Plot the profile and let the user select the off-pulse window by clicking twice.
	Only mouse clicks are accepted.
	Returns:
		off_start_frac, off_end_frac: Start and end as fractions (0-1)
	"""
	fig, ax = plt.subplots(figsize=(10, 4))
	ax.plot(profile, lw=0.5, color='k')
	ax.set_xlim(0, len(profile))
	ax.set_title("Click twice to select the off-pulse window")

	clicks = []

	def onclick(event):
		# Only accept mouse button presses inside the axes
		if event.inaxes == ax and event.button in [1, 2, 3]:
			ax.axvline(event.xdata, color='r', linestyle='--')
			fig.canvas.draw()
			clicks.append(event.xdata)
			if len(clicks) == 2:
				fig.canvas.mpl_disconnect(cid)
				plt.close(fig)

	cid = fig.canvas.mpl_connect('button_press_event', onclick)
	plt.show()

	if len(clicks) < 2:
		raise RuntimeError("Fewer than two mouse clicks registered.")

	idx = sorted([int(round(x)) for x in clicks])
	nbin = len(profile)
	off_start_frac = max(0, idx[0] / nbin)
	off_end_frac = min(1, idx[1] / nbin)
	return off_start_frac, off_end_frac


def scatter_loaded_dynspec(dspec, freq_mhz, time_ms, tau_ms, sc_idx, ref_freq_mhz):
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


def load_data(data, freq_mhz, time_ms):
	if isinstance(data, str):
		if os.path.isdir(data):
			# Expect exactly one file for each Stokes parameter: I, Q, U, V
			stokes_labels = ['I', 'Q', 'U', 'V']
			stokes_files = []
			for s in stokes_labels:
				# Find the file for this Stokes parameter
				matches = [f for f in os.listdir(data) if f.endswith(f'ds{s}_crop.npy')]
				#matches = [f for f in os.listdir(data) if f.endswith(f'{s}.npy')]
				
				if len(matches) != 1:
					raise ValueError(f"Expected one file for Stokes {s}, found {len(matches)}")
				stokes_files.append(os.path.join(data, matches[0]))
			# Load and stack
			arrs = [np.load(f) for f in stokes_files]  # each (nchan, ntime)
			dspec = np.stack(arrs, axis=0)  # shape: (4, nchan, ntime)
			
			summary_file = "parameters.txt"
			summary = get_parameters(os.path.join(data, summary_file))
			cfreq_mhz = float(summary['centre_freq_frb'])
			bw_mhz = float(summary['bw'])
			freq_mhz = np.linspace(cfreq_mhz - bw_mhz / 2, cfreq_mhz + bw_mhz / 2, dspec.shape[1])
			time_res_ms = 0.001 # Convert to milliseconds
			time_ms = np.arange(0, dspec.shape[2] * time_res_ms, time_res_ms)
			print(f"Loaded data from {data} with frequency range: {freq_mhz[0]} - {freq_mhz[-1]} MHz")
			
		elif data.endswith('.pkl'):
			with open(data, 'rb') as f:
				dspec = pkl.load(f)['dspec']
		elif data.endswith('.npy'):
			arr = np.load(data)
			if arr.ndim == 2:
				dspec = arr[np.newaxis, ...]  # shape: (1, nchan, ntime)
			else:
				dspec = arr
		else:
			raise ValueError("Unsupported file format: only .pkl and .npy are supported")
	else:
		raise ValueError("Unsupported data type for 'data'")

	start, stop = select_offpulse_window(np.nansum(dspec[0], axis=0))
	dspec = subtract_baseline_offpulse(dspec, start, stop, axis=-1, method='mean')

	return dspec, freq_mhz, time_ms


def load_multiple_data_grouped(data):
	"""
	Group simulation outputs by prefix (everything before the first underscore).
	Returns a dictionary: {prefix: {'xname': ..., 'xvals': ..., 'yvals': ..., ...}, ...}
	"""
	from collections import defaultdict
	import re

	def extract_sc_value(fname):
		# Match _sc_<number> or _sc_<number>-<number>
		m = re.search(r'_sc_([0-9.]+)(?:-([0-9.]+))?', fname)
		if m:
			return float(m.group(1))
		return float('inf')  # Put files without _sc_ at the end
	
	file_names = [f for f in os.listdir(data) if f.endswith(".pkl")]
	file_names = sorted(file_names, key=extract_sc_value)
	groups = defaultdict(list)
	for fname in file_names:
		prefix = fname.split('_')[0]
		groups[prefix].append(fname)

	all_results = {}
	for prefix, files in groups.items():
		all_xvals             = []
		all_yvals             = {}
		all_errs              = {}
		all_var_PA_microshots = {}
		all_dspecs            = {}
		xname                 = None
		dspec_params          = None
		plot_mode             = None

		for file_name in files:
			with open(os.path.join(data, file_name), "rb") as f:
				obj = pkl.load(f)
			if xname is None:
				xname = obj.get("xname", "unknown")
			if plot_mode is None:
				plot_mode = obj.get("plot_mode", None)
			xvals             = obj["xvals"]
			yvals             = obj["yvals"]
			errs              = obj["errs"]
			var_PA_microshots = obj["var_PA_microshots"]
			dspec_params      = obj["dspec_params"]
			dspecs            = obj.get("dspecs", None)

			for s_val in xvals:
				if s_val not in all_yvals:
					all_yvals[s_val] = []
					all_errs[s_val] = []
					all_var_PA_microshots[s_val] = []
					all_dspecs[s_val] = []
				all_yvals[s_val].extend(yvals[s_val])
				all_errs[s_val].extend(errs[s_val])
				all_var_PA_microshots[s_val].extend(var_PA_microshots[s_val])
				if dspecs is not None:
					all_dspecs[s_val].extend(dspecs[s_val])

			all_xvals.extend(xvals)

		all_results[prefix] = {
			'xname'            : xname,
			'xvals'            : all_xvals,
			'yvals'            : all_yvals,
			'errs'             : all_errs,
			'var_PA_microshots': all_var_PA_microshots,
			'dspec_params'     : dspec_params,
			'plot_mode'        : plot_mode,
			'dspecs'           : all_dspecs,
		}

	return all_results


def generate_dynspec(xname, mode, var, plot_multiple_frb, **params):
	"""Generate dynamic spectrum based on mode."""
	var = var if plot_multiple_frb else params["tau_ms"]

	# Choose the correct function
	if mode == 'gauss':
		dynspec_func = gauss_dynspec
	else:  # mode == 'mgauss'
		dynspec_func = m_gauss_dynspec

	# Get the argument names for the selected function
	sig = inspect.signature(dynspec_func)
	allowed_args = set(sig.parameters.keys())

	# Always pass tau_ms as s_val
	params_filtered = {
		k: v for k, v in params.items()
		if k in allowed_args and k not in ("xname")
	}
	return dynspec_func(**params_filtered, microvar=var, xname=xname)


def process_task(task, xname, mode, plot_mode, **params):
	"""
	Process a single task (combination of timescale and realization).
	Dynamically uses the provided process_func for mode-specific processing.
	"""
	var, realization = task
	current_seed = params["seed"] + realization if params["seed"] is not None else None
	params["seed"] = current_seed
	
	requires_multiple_frb = plot_mode.requires_multiple_frb

	# Generate dynamic spectrum
	dspec, PA_microshot = generate_dynspec(
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
		k: (var if k == "tau_ms" else params[k])
		for k in allowed_args if k in params and k != 'dspec'
	})

	xvals, result_err = process_func(**process_func_args)

	return dspec, var, xvals, result_err, PA_microshot


def generate_frb(data, tau_ms, frb_id, out_dir, mode, n_gauss, seed, nseed, width_range, save,
				 obs_file, gauss_file, snr, n_cpus, plot_mode, phase_window, freq_window):
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
		'DM'             : gauss_params[:-3, 4],
		'RM'             : gauss_params[:-3, 5],
		'PA'             : gauss_params[:-3, 6],
		'lfrac'          : gauss_params[:-3, 7],
		'vfrac'          : gauss_params[:-3, 8],
		'dPA'            : gauss_params[:-3, 9],
		'band_centre_mhz': gauss_params[:-3, 10],
		'band_width_mhz' : gauss_params[:-3, 11]
	}
 
	var_dict = {
		't0_var'             : gauss_params[-3:, 0],
		'width_ms_var'       : gauss_params[-3:, 1],
		'peak_amp_var'       : gauss_params[-3:, 2],
		'spec_idx_var'       : gauss_params[-3:, 3],
		'DM_var'             : gauss_params[-3:, 4],
		'RM_var'             : gauss_params[-3:, 5],
		'PA_var'             : gauss_params[-3:, 6],
		'lfrac_var'          : gauss_params[-3:, 7],
		'vfrac_var'          : gauss_params[-3:, 8],
		'dPA_var'            : gauss_params[-3:, 9],
		'band_centre_mhz_var': gauss_params[-3:, 10],
		'band_width_mhz_var' : gauss_params[-3:, 11]
	}

	# Create dynamic spectrum parameters
	dspec_params = DynspecParams(
		gdict           = gdict,
		var_dict        = var_dict,
		freq_mhz        = freq_mhz,
		time_ms         = time_ms,
		time_res_ms     = t_res,
		seed            = seed,
		nseed           = nseed,
		snr             = snr,
		tau_ms          = tau_ms,
		sc_idx          = scatter_idx,
		ref_freq_mhz    = ref_freq,
		num_micro_gauss = n_gauss,
		width_range     = width_range,
		phase_window    = phase_window,
		freq_window     = freq_window,
	)
 
	if np.any(gauss_params[-1,:] != 0.0) and len(tau_ms) > 1:
		print("WARNING: The last row of gauss_params is not all zeros, but tau_ms has more than one value.")
		print("Please pick only one.")
		sys.exit(1)
  
	if len(np.where(gauss_params[-1,:] != 0.0)[0]) > 1:
		print("WARNING: More than one value in the last row of gauss_params is not 0.")
		print("Please ensure that only one value is non-zero in the last row.")
		sys.exit(1)
  
	if np.any(gdict['lfrac'] + gdict['vfrac']) > 1.0:
		print("WARNING: Linear and circular polarization fractions sum to more than 1.0.\n")

	if plot_mode.requires_multiple_frb == False:
		
		if data != None:
			dspec, freq_mhz, time_ms = load_data(data, freq_mhz, time_ms)
			if tau_ms > 0:
				dspec = scatter_loaded_dynspec(dspec, freq_mhz, time_ms, tau_ms, scatter_idx, ref_freq)
			if snr > 0:
				width_ds = gdict['width_ms'][1] / t_res
				if band_width_mhz[1] == 0.:
					band_width_mhz = freq_mhz[-1] - freq_mhz[0]
				dynspec = add_noise_to_dynspec(dynspec, snr, seed, band_width_mhz, width_ds)
	
	
		else:
			dspec, _ = generate_dynspec(
			xname=None,
			mode=mode,
			var=None,
			plot_multiple_frb=False,
			**dspec_params._asdict()
		)
		_, _, _, noise_spec = process_dynspec(dspec, freq_mhz, time_ms, gdict, tau_ms)
		frb_data = simulated_frb(
			frb_id, freq_mhz, time_ms, tau_ms, scatter_idx, gauss_params, obs_params, dspec
		)
		if save:
			tau = f"{tau_ms:.2f}"
			if mode == 'gauss':
				out_file = (
					f"{out_dir}{frb_id}_mode_{mode}_sc_{tau}_"
					f"seed_{seed}_PA_{gdict['PA'][-1]:.2f}.pkl"
				)
			else:  # mode == 'mgauss'
				out_file = (
					f"{out_dir}{frb_id}_mode_{mode}_sc_{tau}_"
					f"sgwidth_{width_range[0]:.2f}-{width_range[1]:.2f}_"
					f"seed_{seed}_nseed_{nseed}_PA_{gdict['PA'][-1]:.2f}.pkl"
				)
			with open(out_file, 'wb') as frb_file:
				pkl.dump(frb_data, frb_file)
		return frb_data, noise_spec, gdict

	else:
		if data != None:
			files = [f for f in os.listdir(data) if f.endswith('.pkl')]
			if len(files) > 1:
				frb_dict = load_multiple_data_grouped(data)
			elif len(files) == 1:
				with open(os.path.join(data, files[0]), 'rb') as f:
					frb_dict = pkl.load(f)
		else:
			if np.all(gauss_params[-1,:] == 0.0):
				# Create a list of tasks (timescale, realization)
				tasks = list(product(tau_ms, range(nseed)))
				xname = 'tau_ms'

				with ProcessPoolExecutor(max_workers=n_cpus) as executor:
					partial_func = functools.partial(
						process_task,
						xname=xname,
						mode=mode,
						plot_mode=plot_mode,
						**dspec_params._asdict()
					)

					results = list(tqdm(executor.map(partial_func, tasks),
										total=len(tasks),
										desc="Processing scattering timescales and realisations"))
			else:			
				if np.count_nonzero(gauss_params[-1, :]) > 1:
					print("More than one value in the last row of gauss_params is not 0:")
					print(gauss_params[-1, :])
					sys.exit(1)
				else:
					# Find which column in gauss_params the final entry is not zero
					col_idx = np.where(gauss_params[-1, :] != 0.0)[0][0]
					start = gauss_params[-3, col_idx]
					stop = gauss_params[-2, col_idx]
					step = gauss_params[-1, col_idx]
					# Ensure inclusion of the final point
					xvals = np.arange(start, stop + step/2, step)

					# Find the corresponding key in gdict for col_idx
					gdict_keys = list(gdict.keys())
					xname = gdict_keys[col_idx] + '_var'
	 
					tasks = list(product(xvals, range(nseed)))

					with ProcessPoolExecutor(max_workers=n_cpus) as executor:
						partial_func = functools.partial(
							process_task,
							xname=xname,
							mode=mode,
							plot_mode=plot_mode,
							**dspec_params._asdict()
						)					
	  
						results = list(tqdm(executor.map(partial_func, tasks),
											total=len(tasks),
											desc=f"Processing {xname} variance and realisations"))

			# Aggregate results by timescale
			if 'tau_ms' in xname or np.all(gauss_params[-1,:] == 0.0):
				xvals = tau_ms

			dspecs            = {s_val: [] for s_val in xvals}
			yvals             = {s_val: [] for s_val in xvals}
			errs              = {s_val: [] for s_val in xvals}
			var_PA_microshots = {s_val: [] for s_val in xvals}
			
			for dspec, var, val, err, PA_microshot in results:
				dspecs[var].append(dspec)
				yvals[var].append(val)
				errs[var].append(err)
				var_PA_microshots[var].append(PA_microshot)

	
			frb_dict = {
				"xname"            : xname,
				"xvals"            : xvals,
				"plot_mode"        : plot_mode,
				"yvals"            : yvals,
				"errs"             : errs,
				"var_PA_microshots": var_PA_microshots,
				"dspec_params"     : dspec_params
			}
			
		if save:
			# Create a descriptive filename
			if xname == 'tau_ms':
				xvals = f"{tau_ms[0]:.2f}-{tau_ms[-1]:.2f}" if len(tau_ms) > 1 else f"{tau_ms[0]:.2f}"
			else:
				xvals = f"{xvals[0]:.2f}-{xvals[-1]:.2f}" if len(xvals) > 1 else f"{xvals[0]:.2f}"

			tau = f"{tau_ms[0]:.2f}" if len(tau_ms) == 1 else f"{tau_ms[0]:.2f}-{tau_ms[-1]:.2f}"
				
			out_file = (
				f"{out_dir}{frb_id}_plot_{plot_mode}_xname_{xname}_xvals_{xvals}_mode_{mode}_sc_{tau}_"
				f"freq_{freq_window}_phase_{phase_window}.pkl"
			)
			with open(out_file, 'wb') as frb_file:
				pkl.dump((frb_dict), frb_file)
			print(f"Saved FRB data to {out_file}")

		return frb_dict
