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

from fires.config.schema import parse_fires_config
from fires.core.basicfns import (add_noise, boxcar_snr, compute_segments,
                                 correct_baseline, process_dspec,
                                 scale_dspec_to_target_snr,
                                 scatter_loaded_dspec, snr_onpulse)
from fires.core.genfns import psn_dspec
from fires.utils.config import load_params
from fires.utils.loaders import load_data, load_multiple_data_grouped
from fires.utils.utils import dspecParams, simulated_frb

logging.basicConfig(level=logging.INFO)


def _normalise_master_amp_sampling(amp_cfg):
	"""Convert master TOML amplitude_distribution block into internal sampling config."""
	out = {
		"dist": "normal",
		"powerlaw_alpha": 3.0,
		"powerlaw_xmin_scale": 0.1,
		"powerlaw_xmax_scale": 10.0,
		"uniform_low_scale": 0.0,
		"uniform_high_scale": 2.0,
		"lognormal_sigma": None,
	}
	if not isinstance(amp_cfg, dict):
		return out

	dist = str(amp_cfg.get("type", "normal")).strip().lower().replace("-", "_")
	out["dist"] = dist

	if isinstance(amp_cfg.get("powerlaw"), dict):
		pw = amp_cfg["powerlaw"]
		out["powerlaw_alpha"] = float(pw.get("alpha", out["powerlaw_alpha"]))
		out["powerlaw_xmin_scale"] = float(pw.get("xmin_scale", out["powerlaw_xmin_scale"]))
		out["powerlaw_xmax_scale"] = float(pw.get("xmax_scale", out["powerlaw_xmax_scale"]))

	if isinstance(amp_cfg.get("uniform"), dict):
		uu = amp_cfg["uniform"]
		out["uniform_low_scale"] = float(uu.get("low_scale", out["uniform_low_scale"]))
		out["uniform_high_scale"] = float(uu.get("high_scale", out["uniform_high_scale"]))

	if isinstance(amp_cfg.get("lognormal"), dict):
		ln = amp_cfg["lognormal"]
		sigma = ln.get("sigma", None)
		out["lognormal_sigma"] = None if sigma is None else float(sigma)

	return out


def _master_to_internal(master_file):
	"""Map fires.toml master schema into legacy internal structures."""
	raw = load_params("fires", override_path=master_file)
	master = parse_fires_config(raw)

	grid = master.simulation.grid
	sim_params = {
		"f0": float(grid.f_start_MHz),
		"f1": float(grid.f_end_MHz),
		"f_res": float(grid.df_MHz),
		"t0": float(grid.t_start_ms),
		"t1": float(grid.t_end_ms),
		"t_res": float(grid.dt_ms),
		"scattering_index": float(master.propagation.scattering.index),
		"reference_freq": float(grid.reference_freq_MHz),
	}

	components = master.emission.components
	if not isinstance(components, list) or len(components) == 0:
		raise ValueError("Master config must include at least one [[emission.components]] entry")

	n_comp = len(components)
	gauss_params = np.zeros((n_comp + 4, 16), dtype=float)

	for i, comp in enumerate(components):
		gauss_params[i, 0] = float(comp.t0_ms)
		gauss_params[i, 1] = float(comp.width_ms)
		gauss_params[i, 2] = float(comp.amplitude_Jy)
		gauss_params[i, 3] = float(comp.spectral_index)
		gauss_params[i, 4] = float(comp.tau_ms)
		gauss_params[i, 5] = float(comp.dm)
		gauss_params[i, 6] = float(comp.rm)
		gauss_params[i, 7] = float(comp.pa_deg)
		gauss_params[i, 8] = float(comp.lfrac)
		gauss_params[i, 9] = float(comp.vfrac)
		gauss_params[i, 10] = float(comp.dpa_deg_per_ms)
		gauss_params[i, 11] = float(comp.band_centre_MHz)
		gauss_params[i, 12] = float(comp.band_width_MHz)

		gauss_params[i, 13] = float(comp.microshots.N)
		gauss_params[i, 14] = 100.0 * float(comp.microshots.width_frac_low)
		gauss_params[i, 15] = 100.0 * float(comp.microshots.width_frac_high)

	scatter = components[0].microshot_scatter
	stddev_row = -4
	gauss_params[stddev_row, 0] = float(scatter.t0_sigma_ms)
	gauss_params[stddev_row, 1] = float(scatter.width_sigma_ms)
	gauss_params[stddev_row, 2] = float(scatter.amplitude_sigma)
	gauss_params[stddev_row, 3] = float(scatter.spectral_index_sigma)
	gauss_params[stddev_row, 4] = float(scatter.tau_sigma_ms)
	gauss_params[stddev_row, 5] = float(scatter.dm_sigma)
	gauss_params[stddev_row, 6] = float(scatter.rm_sigma)
	gauss_params[stddev_row, 7] = float(scatter.pa_sigma_deg)
	gauss_params[stddev_row, 8] = float(scatter.lfrac_sigma)
	gauss_params[stddev_row, 9] = float(scatter.vfrac_sigma)
	gauss_params[stddev_row, 10] = float(scatter.dpa_sigma)
	gauss_params[stddev_row, 11] = float(scatter.band_centre_sigma)
	gauss_params[stddev_row, 12] = float(scatter.band_width_sigma)

	amp_sampling = _normalise_master_amp_sampling(vars(components[0].amplitude_distribution))

	sweep_mode = "none"
	master_logstep = None
	sweep = master.analysis.sweep
	if bool(sweep.enable):
		mode = str(sweep.mode).strip().lower()
		sweep_mode = mode if mode in ("none", "mean", "sd") else "none"
		param = sweep.parameter
		name = str(param.name).strip().lower()
		start = float(param.start)
		stop = float(param.stop)
		step = float(param.step)
		if param.log_steps is not None:
			master_logstep = int(param.log_steps)

		col_map = {
			"t0": 0, "t0_ms": 0, "width": 1, "width_ms": 1, "a": 2, "amplitude": 2, "amplitude_jy": 2,
			"spec_idx": 3, "spectral_index": 3,
			"tau": 4, "tau_ms": 4, "dm": 5, "rm": 6,
			"pa": 7, "pa_deg": 7, "lfrac": 8, "vfrac": 9, "dpa": 10, "dpa_deg_per_ms": 10,
			"band_centre_mhz": 11, "band_centre": 11, "band_width_mhz": 12, "band_width": 12,
			"sd_t0": 0, "t0_sigma_ms": 0, "sd_width": 1, "width_sigma_ms": 1, "sd_a": 2, "amplitude_sigma": 2,
			"sd_spec_idx": 3, "spectral_index_sigma": 3, "sd_tau": 4, "tau_sigma_ms": 4,
			"sd_dm": 5, "dm_sigma": 5, "sd_rm": 6, "rm_sigma": 6,
			"sd_pa": 7, "pa_sigma_deg": 7, "sd_lfrac": 8, "lfrac_sigma": 8, "sd_vfrac": 9, "vfrac_sigma": 9,
			"sd_dpa": 10, "dpa_sigma": 10,
			"sd_band_centre_mhz": 11, "band_centre_sigma": 11,
			"sd_band_width_mhz": 12, "band_width_sigma": 12,
		}
		if name in col_map:
			col = col_map[name]
			gauss_params[-3, col] = start
			gauss_params[-2, col] = stop
			gauss_params[-1, col] = step

	scint = None
	sc_cfg = master.propagation.scintillation
	if bool(sc_cfg.enable):
		scint = {
			"enable": bool(sc_cfg.enable),
			"t_s": float(sc_cfg.timescale_s),
			"nu_s": float(sc_cfg.bandwidth_Hz),
			"N_im": int(sc_cfg.N_images),
			"th_lim": float(sc_cfg.theta_extent),
			"field": bool(sc_cfg.return_field),
			"derive_from_tau": bool(sc_cfg.derive_from_tau),
		}

	return sim_params, gauss_params, amp_sampling, scint, sweep_mode, master_logstep




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
			baseline_correct=baseline_correct,
			diagnostics=False,
		)

	return var, measures, V_params, snr, exp_vars


def _process_obs_task(task, plot_mode, target_snr=None, baseline_correct=None, obs_data=None, obs_params=None, gauss_file=None, sim_file=None, scint_file=None):
	"""
	Process a single task (combination of timescale and realisation) for real data.
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
		dspec = correct_baseline(dspec, intrinsic_width_bins, buffer_frac, baseline_correct, requires_multiple_frb, dspec_params_local)

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


def _pool_workers_and_chunksize(n_cpus, n_tasks):
	"""Choose pool workers/chunksize to avoid oversubscription and reduce IPC overhead."""
	tasks = max(1, int(n_tasks))
	req_workers = int(n_cpus) if n_cpus is not None else 1
	workers = max(1, min(req_workers, tasks))
	# Keep chunks small to avoid long initial stalls with ordered executor.map.
	chunksize = max(1, min(4, tasks // max(1, workers * 16)))
	return workers, chunksize


def generate_frb(data, frb_id, out_dir, mode, seed, nseed, write, sim_file, gauss_file, scint_file,
				sefd, n_cpus, plot_mode, phase_window, freq_window, buffer_frac, sweep_mode, obs_data, obs_params,
				logstep=None, target_snr=None, param_overrides=None, baseline_correct=None, master_file=None):
	"""
	Generate a simulated FRB with a dispersed and scattered dynamic spectrum.
	"""
	if master_file is None:
		raise ValueError("master_file is required. Legacy split configs are no longer supported.")

	master_scint = None
	sim_params, gauss_params, amp_sampling, master_scint, master_sweep_mode, master_logstep = _master_to_internal(master_file)
	if (sweep_mode is None or sweep_mode == "none") and master_sweep_mode is not None:
		sweep_mode = master_sweep_mode
	if logstep is None and master_logstep is not None:
		logstep = master_logstep

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

	# Gaussian parameters are supplied by master config mapping.

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
		'mg_width_high'  : gauss_params[:stddev_row, 15],
		'amp_sampling'   : amp_sampling,
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

	if master_scint is not None:
		scint = dict(master_scint)
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
			plot_multiple_frb=False,
			target_snr=target_snr,
			dspec_params=dspec_params,
			baseline_correct=baseline_correct,
			diagnostics=True,
		)

		_, corrdspec, _, noise_spec = process_dspec(dspec, freq_mhz, dspec_params, buffer_frac, skip_rm=True, remove_pa_trend=True)
		frb_data = simulated_frb(
			frb_id, corrdspec, dspec_params, snr
		)
		if write:
			tau_str = f"{tau[0]:.2f}"
			if mode == 'psn':
				parts = [
					frb_id,
					f"mode_{mode}",
					f"sc_{tau_str}",
					f"seed_{seed}",
				]
				if nseed is not None:
					parts.append(f"nseed_{nseed}")
				parts.append(f"PA_{gdict['PA'][-1]:.2f}")
				out_file = os.path.join(out_dir, "_".join(parts) + ".pkl")
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
			if os.path.isfile(obs_data):
				logging.info(f"Loading observed data from {obs_data}")
				with open(obs_data, 'rb') as f:
						frb_dict = pkl.load(f)
				return frb_dict

			dspec, freq_mhz, time_ms, dspec_params = load_data(obs_data, obs_params, gauss_file, sim_file, scint_file)
			I_time = np.nansum(dspec[0], axis=0)
			original_snr, (left, right) = snr_onpulse(dspec_params, I_time, frac=0.95, buffer_frac=buffer_frac)
			logging.info(f"Original data S/N: {original_snr:.2f}, on-pulse window: {left}-{right} ({time_ms[left]:.2f}-{time_ms[right]:.2f} ms)")
			
			# Assume sweep is set for tau; validate
			gdict_keys = [k for k in gdict.keys() if k != 'amp_sampling']
			if active_cols.size != 1 or gdict_keys[sweep_col] != 'tau':
				logging.error("For obs_data sweep, exactly one sweep column must be set for 'tau'.")
				sys.exit(1)
			
			xvals, col_idx, xname = _setup_sweep(gauss_params, logstep, sweep_spec, sweep_mode, plot_mode, gdict_keys)
			
			xvals, _array_id, _array_count = _slurm_chunk_xvals(xvals)
			
			tasks = list(product(xvals, range(nseed)))
			workers, chunksize = _pool_workers_and_chunksize(n_cpus, len(tasks))
			logging.info(f"Process pool settings: workers={workers}, chunksize={chunksize}, tasks={len(tasks)}")
			
			with ProcessPoolExecutor(max_workers=workers) as executor:
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
					executor.map(partial_func, tasks, chunksize=chunksize),
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
			gdict_keys = [k for k in gdict.keys() if k != 'amp_sampling']

			xvals, col_idx, xname = _setup_sweep(gauss_params, logstep, sweep_spec, sweep_mode, plot_mode, gdict_keys)

			xvals, _array_id, _array_count = _slurm_chunk_xvals(xvals)

			tasks = list(product(xvals, range(nseed)))
			workers, chunksize = _pool_workers_and_chunksize(n_cpus, len(tasks))
			logging.info(f"Process pool settings: workers={workers}, chunksize={chunksize}, tasks={len(tasks)}")

			with ProcessPoolExecutor(max_workers=workers) as executor:
				partial_func = functools.partial(
					_process_task,
					xname=xname,
					plot_mode=plot_mode,
					target_snr=target_snr,
					dspec_params=dspec_params,
					baseline_correct=baseline_correct
				)
				results = list(tqdm(
					executor.map(partial_func, tasks, chunksize=chunksize),
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