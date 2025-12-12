# -----------------------------------------------------------------------------
# genfrb.py
# FIRES: The Fast, Intense Radio Emission Simulator
#
# This module contains functions for generating Fast Radio Burst (FRB) dynamic
# spectra, handling baseline subtraction, off-pulse window selection, scattering,
# data loading, and parallelised simulation and aggregation of FRB realisations.
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

from fires.core.basicfns import (add_noise, boxcar_snr, compute_segments,
                                 correct_baseline, process_dspec,
                                 scale_dspec_to_target_snr,
                                 scatter_loaded_dspec, snr_onpulse)
from fires.core.genfns import psn_dspec
from fires.utils.config import load_params
from fires.utils.loaders import load_data, load_multiple_data_grouped
from fires.utils.utils import dspecParams, simulated_frb

logging.basicConfig(level=logging.INFO)




def _process_task(task, xname, plot_mode, dspec_params, target_snr=None, baseline_correct=None):
	"""
	Process a single task (combination of timescale and realisation).
	"""
	var, realisation = task
	base_seed = dspec_params.seed
	current_seed = (base_seed + realisation) if base_seed is not None else None

	local_params = dspec_params._replace(seed=current_seed)
	
	requires_multiple_frb = plot_mode.requires_multiple_frb


	_, snr, V_params, exp_vars, measures = psn_dspec(
			dspec_params=local_params,
			variation_parameter=var,
			xname=xname,
			plot_multiple_frb=requires_multiple_frb,
			target_snr=target_snr,
			baseline_correct=baseline_correct
		)

	return var, measures, V_params, snr, exp_vars


def _process_obs_task(task, plot_mode, target_snr=None, baseline_correct=None, obs_data=None, obs_params=None, gauss_file=None, sim_file=None, scint_file=None):
	"""
	Process a single task (combination of timescale and realisation).
	"""
	var, _ = task

	requires_multiple_frb = plot_mode.requires_multiple_frb
	dspec, freq_mhz, time_ms, dspec_params_local = load_data(obs_data, obs_params, gauss_file, sim_file, scint_file)
	scatter_idx = dspec_params_local.sc_idx
	ref_freq = dspec_params_local.ref_freq_mhz
	sefd = dspec_params_local.sefd
	f_res = dspec_params_local.freq_res_mhz
	t_res = dspec_params_local.time_res_ms
	buffer_frac = dspec_params_local.buffer_frac
	
	# Scatter with var (tau)
	dspec = scatter_loaded_dspec(dspec, freq_mhz, time_ms, var, scatter_idx, ref_freq)
	
	gdict = dspec_params_local.gdict
	intrinsic_width_bins = gdict["width"][0] / t_res
	if var > 0:
		dspec = correct_baseline(intrinsic_width_bins, buffer_frac, baseline_correct, requires_multiple_frb, dspec_params_local)

	# Add noise
	sefd = scale_dspec_to_target_snr("analytic", target_snr, dspec_params_local, dspec, freq_mhz, t_res, buffer_frac, sefd, requires_multiple_frb, time_ms)
	dspec, _, snr = add_noise(dspec_params_local,
		dspec, sefd,
		(freq_mhz[1] - freq_mhz[0]) * 1e6,
		t_res / 1000.0,
		requires_multiple_frb, buffer_frac=buffer_frac, n_pol=2
	)

#	dspec = correct_baseline(intrinsic_width_bins, buffer_frac, baseline_correct, requires_multiple_frb, dspec_params_local)

	segments = compute_segments(dspec, freq_mhz, time_ms, dspec_params_local, buffer_frac=buffer_frac, skip_rm=True, remove_pa_trend=True)
	
	measures = segments
	V_params = {}
	exp_vars = {}

	return var, measures, V_params, snr, exp_vars


def _setup_sweep(gauss_params, logstep, sweep_spec, sweep_mode, plot_mode, gdict_keys):
	"""
	Set up the sweep parameters for multiple FRB generation.
	"""
	# Validate sweep definition - allow single point sweeps
	if sweep_mode == "none":
		raise ValueError(
			f"Plot mode '{plot_mode.name}' requires a parameter sweep. "
			"Use --sweep-mode mean or --sweep-mode sd and set exactly one non-zero step in gparams."
		)

	# Check if we have a sweep defined (non-zero step) or single point (zero step but specified start)
	has_sweep = not np.all(gauss_params[-1, :] == 0.0)  # step_row
	has_single_point = np.all(gauss_params[-1, :] == 0.0) and not np.all(gauss_params[-3, :] == 0.0)  # start_row

	if not has_sweep and not has_single_point:
		logging.error("No sweep or single point defined. For sweep: set non-zero step. For single point: set start value with zero step.")
		logging.info("Edit the last three rows (start/stop/step) of gparams for exactly one column.")
		sys.exit(1)

	# Find the active column (either non-zero step or non-zero start with zero step)
	if has_sweep:
		active_cols = np.where(gauss_params[-1, :] != 0.0)[0]  # step_row
	else:
		active_cols = np.where(gauss_params[-3, :] != 0.0)[0]  # start_row

	if active_cols.size == 0:
		logging.error("No parameter column identified for sweep/single point.")
		sys.exit(1)
	if active_cols.size > 1:
		logging.error("ERROR: More than one active column (multiple non-zero steps or starts).")
		sys.exit(1)

	col_idx = active_cols[0]
	start = sweep_spec['start']
	stop = sweep_spec['stop']
	step = sweep_spec['step']
	
	# Handle single point case (zero step)
	if step == 0:
		xvals = np.array([start], dtype=float)
		logging.info(f"Using single point: {start} for realizations")
	else:
		# Use logarithmic spacing if logstep is provided, otherwise linear
		if logstep is not None:
			# Logarithmic spacing
			if start <= 0 or stop <= 0:
				raise ValueError(
					f"Logarithmic sweep (--logstep) requires positive start and stop values. "
					f"Got start={start}, stop={stop}"
				)
		
			xvals = np.logspace(np.log10(start), np.log10(stop), logstep)
			logging.info(f"Using logarithmic sweep: {logstep} points from {start} to {stop}")
		else:
			if step is None or step == 0:
				raise ValueError("Linear sweep requires a non-zero step. Use --logstep for logarithmic sweeps.")

			direction = 1.0 if stop >= start else -1.0
			step = abs(step) * direction

			dist = abs(stop - start)
			if dist == 0:
				xvals = np.array([start], dtype=float)
			else:
				n_steps = int(np.floor(dist / abs(step)))
				end = start + n_steps * step
				xvals = np.linspace(start, end, n_steps + 1)

			logging.info(
				f"Using linear sweep: {len(xvals)} points from {xvals[0]} to {xvals[-1]} (step={step})"
			)
			
	xname = gdict_keys[col_idx]
	return xvals, col_idx, xname


def _collect_results(results, xvals):
	measures = {v: [] for v in xvals}
	V_params = {v: {} for v in xvals}
	snrs 	 = {v: [] for v in xvals}
	exp_vars = {v: {} for v in xvals}

	for var, seg_measures, params_dict, snr, exp_var_psi_deg2 in results:
		measures[var].append(seg_measures)
		snrs[var].append(snr)
		for key, value in params_dict.items():
			if key not in V_params[var]:
				V_params[var][key] = []
			V_params[var][key].append(value)
		for key, value in exp_var_psi_deg2.items():
			if key not in exp_vars[var]:
				exp_vars[var][key] = []
			exp_vars[var][key].append(value)
			
	return measures, V_params, snrs, exp_vars

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

def _slurm_chunk_xvals(xvals):

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
	return xvals, _array_id, _array_count


def generate_frb(data, frb_id, out_dir, mode, seed, nseed, write, sim_file, gauss_file, scint_file,
				sefd, n_cpus, plot_mode, phase_window, freq_window, buffer_frac, sweep_mode, obs_data, obs_params,
				logstep=None, target_snr=None, param_overrides=None, baseline_correct=None):
	"""
	Generate a simulated FRB with a dispersed and scattered dynamic spectrum.
	"""
	sim_params = load_params("simparams", sim_file, "simulation")

	# Extract frequency and time parameters
	f_start = float(sim_params['f0'])
	f_end   = float(sim_params['f1'])
	t_start = float(sim_params['t0'])
	t_end   = float(sim_params['t1'])
	f_res   = float(sim_params['f_res'])
	t_res   = float(sim_params['t_res'])

	scatter_idx = float(sim_params['scattering_index'])
	ref_freq 	= float(sim_params['reference_freq'])

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
		'width'     	 : gauss_params[:stddev_row, 1],
		'A'              : gauss_params[:stddev_row, 2],
		'spec_idx'       : gauss_params[:stddev_row, 3],
		'tau'            : gauss_params[:stddev_row, 4],
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

	sd_dict = {
		'sd_t0'             : gauss_params[stddev_row, 0],
		'sd_width'       	: gauss_params[stddev_row, 1],
		'sd_A'              : gauss_params[stddev_row, 2],
		'sd_spec_idx'       : gauss_params[stddev_row, 3],
		'sd_tau'         	: gauss_params[stddev_row, 4],
		'sd_DM'             : gauss_params[stddev_row, 5],
		'sd_RM'             : gauss_params[stddev_row, 6],
		'sd_PA'             : gauss_params[stddev_row, 7],
		'sd_lfrac'          : gauss_params[stddev_row, 8],
		'sd_vfrac'          : gauss_params[stddev_row, 9],
		'sd_dPA'            : gauss_params[stddev_row,10],
		'sd_band_centre_mhz': gauss_params[stddev_row,11],
		'sd_band_width_mhz' : gauss_params[stddev_row,12]
	}


	mean_override_parts = []
	sd_override_parts = []
	if param_overrides:
		for key, value in param_overrides.items():
			if key in gdict:
				original_shape = gdict[key].shape
				gdict[key] = np.full(original_shape, value, dtype=float)
				logging.info(f"Override applied: {key} = {value} (shape: {original_shape})")
			elif key in sd_dict:
				sd_dict[key] = float(value)
				logging.info(f"Override applied: {key} = {value} (std dev)")
			else:
				raise ValueError(f"Override key '{key}' not found in gdict or sd_dict.")
	
			if key.startswith("sd_"):
				base_key = key[3:].replace("_", "")
				append_to = sd_override_parts
			else:
				base_key = key
				append_to = mean_override_parts
	
			if append_to is sd_override_parts:
				if isinstance(value, (int, np.integer)):
					append_to.append(f"sd{base_key}{value}")
				elif isinstance(value, (float, np.floating)):
					if value.is_integer():
						append_to.append(f"sd{base_key}{int(value)}")
					else:
						append_to.append(f"sd{base_key}{value:.2f}")
				else:
					append_to.append(f"sd{base_key}{value}")
			else:
				if isinstance(value, (int, np.integer)):
					append_to.append(f"{base_key}{value}")
				elif isinstance(value, (float, np.floating)):
					if value.is_integer():
						append_to.append(f"{base_key}{int(value)}")
					else:
						append_to.append(f"{base_key}{value:.2f}")
				else:
					append_to.append(f"{base_key}{value}")


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
			tau_ref = float(gdict["tau"][0])          
			tau_s_ref  = 1e-3 * tau_ref                  
			nu_s_hz    = 1.0 / (2.0 * np.pi * tau_s_ref)    
			scint["nu_s"] = float(nu_s_hz)
			logging.info(
				f"Derived nu_s at reference {ref_freq:.1f} MHz: "
				f"tau={tau_ref:.3f} ms -> nu_s={nu_s_hz:.2f} Hz"
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

	tau = gdict['tau']

	if len(np.where(gauss_params[-1,:] != 0.0)[0]) > 1:
		logging.warning("More than one value in the last row of gauss_params is not 0.")
		logging.info("Please ensure that only one value is non-zero in the last row.")
		sys.exit(1)
  
	if np.any(gdict['lfrac'] + gdict['vfrac']) > 1.0:
		logging.warning("Linear and circular polarisation fractions sum to more than 1.0.")

	plot_multiple_frb = plot_mode.requires_multiple_frb
	if not plot_multiple_frb:
		# Single FRB generation branch
		if obs_data != None:
			dspec, freq_mhz, time_ms, dspec_params = load_data(obs_data, obs_params, gauss_file, sim_file, scint_file)
			I_time = np.nansum(dspec[0], axis=0)
			snr, (left, right) = snr_onpulse(dspec_params, I_time, frac=0.95, buffer_frac=buffer_frac)
			#peak_snr, boxcarw = boxcar_snr(I_time, np.nanstd(I_time))

			logging.info(f"Loaded data S/N: {snr:.2f}, on-pulse window: {left}-{right} ({time_ms[left]:.2f}-{time_ms[right]:.2f} ms)")  
			#logging.info(f"Stokes I peak S/N (max boxcar): {peak_snr:.2f} (width={boxcarw})")
			if tau[0] > 0:
				dspec = scatter_loaded_dspec(dspec, freq_mhz, time_ms, tau[0], scatter_idx, ref_freq)
			if sefd > 0:
				dspec, sigma_ch, snr = add_noise(
					dspec_params, dspec=dspec, sefd=sefd, f_res=f_res, t_res=t_res,
					plot_multiple_frb=plot_multiple_frb, buffer_frac=buffer_frac
				)
			segments = compute_segments(dspec, freq_mhz, time_ms, dspec_params, buffer_frac=buffer_frac, skip_rm=True, remove_pa_trend=True)

		else:
			dspec, snr, _, _, segments = psn_dspec(
			xname=None,
			mode=mode,
			var=None,
			plot_multiple_frb=False,
			target_snr=target_snr,
			dspec_params=dspec_params,
			baseline_correct=baseline_correct
		)

		_, corrdspec, _, noise_spec = process_dspec(dspec, freq_mhz, dspec_params, buffer_frac, skip_rm=True, remove_pa_trend=True)
		frb_data = simulated_frb(
			frb_id, corrdspec, dspec_params, snr
		)
		if write:
			tau = f"{tau[0]:.2f}"
			if mode == 'psn':
				out_file = (
					f"{out_dir}{frb_id}_mode_{mode}_sc_{tau}_"
					f"seed_{seed}_nseed_{nseed}_PA_{gdict['PA'][-1]:.2f}.pkl"
				)
			with open(out_file, 'wb') as frb_file:
				pkl.dump(frb_data, frb_file)
		return frb_data, noise_spec, gdict, segments



	if plot_multiple_frb:
		if data != None:
			files = [f for f in os.listdir(data) if f.endswith('.pkl')]
			if len(files) > 1:
				frb_dict = load_multiple_data_grouped(data)
			elif len(files) == 1:
				with open(os.path.join(data, files[0]), 'rb') as f:
					frb_dict = pkl.load(f)
			else:
				logging.error(f"No .pkl files found in {data}.")
				sys.exit(1)
			return frb_dict
		
		elif obs_data != None:
			# Load the observed data once to record original S/N and segments
			if obs_data.isfile():
				logging.info(f"Loading observed data from {obs_data}")
				with open(obs_data, 'rb') as f:
						frb_dict = pkl.load(f)
				return frb_dict

			dspec, freq_mhz, time_ms, dspec_params = load_data(obs_data, obs_params, gauss_file, sim_file, scint_file)
			I_time = np.nansum(dspec[0], axis=0)
			original_snr, (left, right) = snr_onpulse(dspec_params, I_time, frac=0.95, buffer_frac=buffer_frac)
			logging.info(f"Original data S/N: {original_snr:.2f}, on-pulse window: {left}-{right} ({time_ms[left]:.2f}-{time_ms[right]:.2f} ms)")
			
			# Assume sweep is set for tau; validate
			gdict_keys = list(gdict.keys())
			if active_cols.size != 1 or gdict_keys[col_idx] != 'tau':
				logging.error("For obs_data sweep, exactly one sweep column must be set for 'tau'.")
				sys.exit(1)
			
			xvals, col_idx, xname = _setup_sweep(gauss_params, logstep, sweep_spec, sweep_mode, plot_mode, gdict_keys)
			
			xvals, _array_id, _array_count = _slurm_chunk_xvals(xvals)
			
			tasks = list(product(xvals, range(nseed)))
			
			with ProcessPoolExecutor(max_workers=n_cpus) as executor:
				partial_func = functools.partial(
					_process_obs_task,
					plot_mode=plot_mode,
					target_snr=original_snr,
					baseline_correct=baseline_correct,
					obs_data=obs_data,
					obs_params=obs_params,
					gauss_file=gauss_file,
					sim_file=sim_file,
					scint_file=scint_file
				)
				results = list(tqdm(
					executor.map(partial_func, tasks),
					total=len(tasks),
					desc=f"Processing tau sweep on observed data"
				))
			
			measures, V_params, snrs, exp_vars = _collect_results(results, xvals)
			
			frb_dict = {
				"xname": xname,
				"xvals": xvals,
				"measures": measures,       
				"V_params": V_params,
				"exp_vars": exp_vars,
				"dspec_params": dspec_params,
				"plot_mode": plot_mode,
				"snrs": snrs,
			}
			
			if write:
				sweep_idx = _array_id if _array_count > 1 else 0
			
				parts = [f"sweep_{sweep_idx}", f"n{nseed}", f"plot_{plot_mode.name}", f"xname_{xname}"]
			
				if len(xvals) > 0:
					parts.append(f"xvals_{min(xvals):.2f}-{max(xvals):.2f}")
			
				parts.append(f"mode_{mode}")
			
				# Add mean and stddev overrides
				if mean_override_parts:
					parts.extend(mean_override_parts)
				if sd_override_parts:
					parts.extend(sd_override_parts)
			
				fname = "_".join(parts) + ".pkl"
				fpath = os.path.join(out_dir, fname)
			
				with open(fpath, "wb") as f:
					pkl.dump(frb_dict, f)
			
				logging.info(f"Saved results to {fpath}")
			
			return frb_dict

		else:
			xvals, col_idx, xname = _setup_sweep(gauss_params, logstep, sweep_spec, sweep_mode, plot_mode, gdict_keys)

			xvals, _array_id, _array_count = _slurm_chunk_xvals(xvals)

			tasks = list(product(xvals, range(nseed)))

			with ProcessPoolExecutor(max_workers=n_cpus) as executor:
				partial_func = functools.partial(
					_process_task,
					xname=xname,
					plot_mode=plot_mode,
					target_snr=target_snr,
					dspec_params=dspec_params,
					baseline_correct=baseline_correct
				)
				results = list(tqdm(
					executor.map(partial_func, tasks),
					total=len(tasks),
					desc=f"Processing sweep of {xname} ({sweep_mode} mode)"
				))

			measures, V_params, snrs, exp_vars = _collect_results(results, xvals)


			frb_dict = {
				"xname": xname,
				"xvals": xvals,
				"measures": measures,       
				"V_params": V_params,
				"exp_vars": exp_vars,
				"dspec_params": dspec_params,
				"plot_mode": plot_mode,
				"snrs": snrs,
			}

			if write:
				sweep_idx = _array_id if _array_count > 1 else 0
			
				parts = [f"sweep_{sweep_idx}", f"n{nseed}", f"plot_{plot_mode.name}", f"xname_{xname}"]
			
				if len(xvals) > 0:
					parts.append(f"xvals_{min(xvals):.2f}-{max(xvals):.2f}")
			
				parts.append(f"mode_{mode}")
			
				# Add mean and stddev overrides
				if mean_override_parts:
					parts.extend(mean_override_parts)
				if sd_override_parts:
					parts.extend(sd_override_parts)
			
				fname = "_".join(parts) + ".pkl"
				fpath = os.path.join(out_dir, fname)
			
				with open(fpath, "wb") as f:
					pkl.dump(frb_dict, f)
			
				logging.info(f"Saved results to {fpath}")

			return frb_dict