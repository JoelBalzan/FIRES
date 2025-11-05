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

from fires.core.basicfns import (_freq_quarter_slices, _phase_slices_from_peak,
								 add_noise, process_dspec, scatter_dspec,
								 snr_onpulse)
from fires.core.genfns import psn_dspec
from fires.io.loaders import load_data, load_multiple_data_grouped
from fires.utils.config import load_params
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


def _generate_dspec(xname, mode, var, plot_multiple_frb, dspec_params, target_snr=None,):
	"""Generate dynamic spectrum based on mode."""
	var = var if plot_multiple_frb else None

	# Choose the correct function
	if mode == 'psn':
		dspec_func = psn_dspec

	# Get the argument names for the selected function
	sig = inspect.signature(dspec_func)
	allowed_args = set(sig.parameters.keys())

 
	if mode == 'psn':
		return psn_dspec(
			dspec_params=dspec_params,
			variation_parameter=var,
			xname=xname,
			plot_multiple_frb=plot_multiple_frb,
			target_snr=target_snr
		)


def _process_task(task, xname, mode, plot_mode, dspec_params, target_snr=None):
	"""
	Process a single task (combination of timescale and realisation).
	"""
	var, realisation = task
	base_seed = dspec_params.seed
	current_seed = (base_seed + realisation) if base_seed is not None else None

	local_params = dspec_params._replace(seed=current_seed)
	
	requires_multiple_frb = plot_mode.requires_multiple_frb

	_, snr, V_params, exp_vars, measures = _generate_dspec(
		xname=xname,
		mode=mode,
		var=var,
		plot_multiple_frb=requires_multiple_frb,
		dspec_params=local_params,
		target_snr=target_snr
	)

	return var, measures, V_params, snr, exp_vars


def _normalise_freq_key(key: str | None) -> str:
	if key is None:
		return "all"
	k = str(key).lower()
	alias = {
		"full": "all", "full-band": "all", "fullband": "all",
		"lowest-quarter": "1q", "lower-mid-quarter": "2q",
		"upper-mid-quarter": "3q", "highest-quarter": "4q"
	}
	return alias.get(k, k)

def _normalise_phase_key(key: str | None) -> str:
	if key is None:
		return "total"
	k = str(key).lower()
	alias = {"leading": "first", "trailing": "last", "all": "total"}
	return alias.get(k, k)


def _window_dspec(dspec: np.ndarray,
				  freq_mhz: np.ndarray,
				  time_ms: np.ndarray,
				  freq_window=None,
				  phase_window=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Apply optional frequency and phase windows. Supports:
	  - freq_window: [fmin,fmax] MHz or '1q'|'2q'|'3q'|'4q'|'all' (+ synonyms)
	  - phase_window: [tmin,tmax] ms or 'first'|'last'|'total' (+ synonyms)
	Returns windowed (dspec, freq_mhz, time_ms).
	"""
	dspec_w = dspec
	f_w = freq_mhz
	t_w = time_ms

	if freq_window is not None:
		if isinstance(freq_window, (list, tuple, np.ndarray)) and len(freq_window) == 2:
			fmin, fmax = float(freq_window[0]), float(freq_window[1])
			fmask = (f_w >= min(fmin, fmax)) & (f_w <= max(fmin, fmax))
			if np.any(fmask):
				logging.info(f"Applying freq_window [{min(fmin,fmax):.2f}, {max(fmin,fmax):.2f}] MHz "
							 f"-> {np.count_nonzero(fmask)}/{len(f_w)} channels")
				dspec_w = dspec_w[:, fmask, :]
				f_w = f_w[fmask]
			else:
				logging.warning("freq_window produced empty selection; ignoring frequency window.")
		elif isinstance(freq_window, str):
			key = _normalise_freq_key(freq_window)
			slc_dict = _freq_quarter_slices(dspec_w.shape[1])
			slc = slc_dict.get(key, slc_dict["all"])
			prev = dspec_w.shape[1]
			dspec_w = dspec_w[:, slc, :]
			f_w = f_w[slc]
			logging.info(f"Applying freq_window '{freq_window}' (-> '{key}') -> {dspec_w.shape[1]}/{prev} channels")
		else:
			logging.warning(f"Unrecognised freq_window={freq_window}; ignoring.")

	if phase_window is not None:
		if isinstance(phase_window, (list, tuple, np.ndarray)) and len(phase_window) == 2:
			tmin, tmax = float(phase_window[0]), float(phase_window[1])
			tmask = (t_w >= min(tmin, tmax)) & (t_w <= max(tmin, tmax))
			if np.any(tmask):
				logging.info(f"Applying phase_window [{min(tmin,tmax):.2f}, {max(tmin,tmax):.2f}] ms "
							 f"-> {np.count_nonzero(tmask)}/{len(t_w)} time bins")
				dspec_w = dspec_w[:, :, tmask]
				t_w = t_w[tmask]
			else:
				logging.warning("phase_window produced empty selection; ignoring phase window.")
		elif isinstance(phase_window, str):
			key = _normalise_phase_key(phase_window)
			I_time = np.nansum(dspec_w[0], axis=0) if dspec_w.shape[1] > 0 else np.zeros_like(t_w)
			peak_idx = int(np.nanargmax(I_time)) if I_time.size > 0 else 0
			slc_dict = _phase_slices_from_peak(dspec_w.shape[2], peak_idx)
			slc = slc_dict.get(key, slice(None))
			prev = dspec_w.shape[2]
			dspec_w = dspec_w[:, :, slc]
			t_w = t_w[slc]
			logging.info(f"Applying phase_window '{phase_window}' (-> '{key}', peak idx {peak_idx}) "
						 f"-> {dspec_w.shape[2]}/{prev} time bins")
		else:
			logging.warning(f"Unrecognised phase_window={phase_window}; ignoring.")

	return dspec_w, f_w, t_w


def generate_frb(data, frb_id, out_dir, mode, seed, nseed, write, sim_file, gauss_file, scint_file,
				sefd, n_cpus, plot_mode, phase_window, freq_window, buffer_frac, sweep_mode, obs_data, obs_params,
				logstep=None, target_snr=None, param_overrides=None):
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

	sd_dict = {
		'sd_t0'             : gauss_params[stddev_row, 0],
		'sd_width_ms'       : gauss_params[stddev_row, 1],
		'sd_A'              : gauss_params[stddev_row, 2],
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
		logging.warning("Linear and circular polarisation fractions sum to more than 1.0.")

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

			if freq_window != "full-band" or phase_window != "total":
				dspec, freq_mhz, time_ms = _window_dspec(
					dspec, freq_mhz, time_ms,
					freq_window=freq_window, phase_window=phase_window
				)
				logging.info(f"Windowed dspec shape: {dspec.shape}  "
							 f"freq[{freq_mhz[0]:.2f},{freq_mhz[-1]:.2f}] MHz  "
							 f"time[{time_ms[0]:.2f},{time_ms[-1]:.2f}] ms")
			
			# Update dspec_params with new time and frequency arrays
			dspec_params = dspec_params._replace(time_ms=time_ms, freq_mhz=freq_mhz)

		else:
			dspec, snr, _, _, _ = _generate_dspec(
			xname=None,
			mode=mode,
			var=None,
			plot_multiple_frb=False,
			target_snr=target_snr,
			dspec_params=dspec_params
		)

			if freq_window != "full-band" or phase_window != "total":
				dspec, freq_mhz, time_ms = _window_dspec(
					dspec, freq_mhz, time_ms,
					freq_window=freq_window, phase_window=phase_window
				)
				logging.info(f"Windowed dspec shape: {dspec.shape}  "
							 f"freq[{freq_mhz[0]:.2f},{freq_mhz[-1]:.2f}] MHz  "
							 f"time[{time_ms[0]:.2f},{time_ms[-1]:.2f}] ms")
			dspec_params = dspec_params._replace(time_ms=time_ms, freq_mhz=freq_mhz)
			
		_, corrdspec, _, noise_spec = process_dspec(dspec, freq_mhz, dspec_params, buffer_frac)
		frb_data = simulated_frb(
			frb_id, corrdspec, dspec_params, snr
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
				frb_dict = load_multiple_data_grouped(data)
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

			start = sweep_spec['start']
			stop = sweep_spec['stop']
			step = sweep_spec['step']
			
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
				# Linear spacing (original behavior)
				if step is None or step <= 0:
					raise ValueError("Linear sweep requires step > 0. Use --logstep for logarithmic sweeps.")
			
				n_steps = int(np.round((stop - start) / step))
				end = start + n_steps * step
				xvals = np.linspace(start, end, n_steps + 1)
				logging.info(f"Using linear sweep: {len(xvals)} points from {start} to {stop} (step={step})")

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
					dspec_params=dspec_params
				)
				results = list(tqdm(
					executor.map(partial_func, tasks),
					total=len(tasks),
					desc=f"Processing sweep of {xname} ({sweep_mode} mode)"
				))

			measures = {v: [] for v in xvals}
			V_params = {
				v: {key: [] for key in [
					't0_i','A_i','mg_width_i','spec_idx_i','tau_ms_i','PA_i',
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