import functools
import logging
import os
import pickle as pkl
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, is_dataclass
from itertools import product

import numpy as np
from tqdm import tqdm

from fires.config.schema import parse_fires_config
from fires.core.dspec import scatter_loaded_dspec
from fires.core.genfns import psn_dspec
from fires.core.basicfns import (add_noise, boxcar_snr, compute_segments,
                                  correct_baseline, process_dspec,
                                  scale_dspec_to_target_snr, snr_onpulse)
from fires.utils.config import load_params
from fires.utils.io import (build_override_parts, build_single_output_path,
                             write_frb_dict, write_single_frb)
from fires.utils.loaders import load_data, load_multiple_data_grouped
from fires.utils.params import COL_MAP
from fires.utils.slurm import (pool_workers_and_chunksize, slurm_chunk_xvals)
from fires.utils.utils import dspecParams, simulated_frb

logging.basicConfig(level=logging.INFO)


def _write_stokes_cube(dspec, freq_mhz, time_ms, out_dir, frb_id):
    from fires.utils.io import write_stokes_cube
    return write_stokes_cube(dspec, freq_mhz, time_ms, out_dir, frb_id)


def _normalise_master_amp_sampling(amp_cfg):
    out = {
        "dist": "normal",
        "powerlaw_alpha": 3.0,
        "powerlaw_xmin_scale": 0.1,
        "powerlaw_xmax_scale": 10.0,
        "uniform_low_scale": 0.0,
        "uniform_high_scale": 2.0,
        "lognormal_sigma": None,
    }
    if amp_cfg is None:
        return out
    if is_dataclass(amp_cfg):
        amp_cfg = asdict(amp_cfg)
    elif not isinstance(amp_cfg, dict):
        return out
    dist = str(amp_cfg.get("type", "normal")).strip().lower().replace("-", "_")
    out["dist"] = dist
    pw = amp_cfg.get("powerlaw")
    if is_dataclass(pw):
        pw = asdict(pw)
    if isinstance(pw, dict):
        out["powerlaw_alpha"] = float(pw.get("alpha", out["powerlaw_alpha"]))
        out["powerlaw_xmin_scale"] = float(pw.get("xmin_scale", out["powerlaw_xmin_scale"]))
        out["powerlaw_xmax_scale"] = float(pw.get("xmax_scale", out["powerlaw_xmax_scale"]))
    uu = amp_cfg.get("uniform")
    if is_dataclass(uu):
        uu = asdict(uu)
    if isinstance(uu, dict):
        out["uniform_low_scale"] = float(uu.get("low_scale", out["uniform_low_scale"]))
        out["uniform_high_scale"] = float(uu.get("high_scale", out["uniform_high_scale"]))
    ln = amp_cfg.get("lognormal")
    if is_dataclass(ln):
        ln = asdict(ln)
    if isinstance(ln, dict):
        sigma = ln.get("sigma", None)
        out["lognormal_sigma"] = None if sigma is None else float(sigma)
    return out


def _master_to_internal(master_file, master_raw=None):
    raw = master_raw if master_raw is not None else load_params("fires", override_path=master_file)
    master = parse_fires_config(raw)
    grid = master.simulation.grid
    sim_params = {
        "f0": float(grid.f_start_MHz),
        "f1": float(grid.f_end_MHz),
        "f_res": float(grid.df_MHz),
        "t0": float(grid.t_start_ms),
        "t1": float(grid.t_end_ms),
        "t_res": float(grid.dt_ms),
        "reference_freq": float(grid.reference_freq_MHz),
    }
    prop_params = {
        "scattering_index": float(master.propagation.scattering.index),
        "RM": float(master.propagation.RM.RM),
        "order": str(master.propagation.RM.order),
    }
    components = master.emission.components
    if not isinstance(components, list) or len(components) == 0:
        raise ValueError("Master config must include at least one [[emission.components]] entry")
    rvm_raw = master.emission.rvm_swing
    rvm_swing = {
        "enable": bool(rvm_raw.enable),
        "alpha_deg": float(rvm_raw.alpha_deg),
        "beta_deg": float(rvm_raw.beta_deg),
        "period_ms": float(rvm_raw.period_ms),
        "phase0_ms": float(rvm_raw.phase0_ms),
        "psi0_deg": float(rvm_raw.psi0_deg),
    }
    n_comp = len(components)
    gauss_params = np.zeros((n_comp + 4, 16), dtype=float)
    mean_keys = ['t0_ms', 'width_ms', 'amplitude_Jy', 'spectral_index', 'tau_ms',
                 'dm', 'rm', 'pa_deg', 'lfrac', 'vfrac', 'dpa_deg_per_ms',
                 'band_centre_MHz', 'band_width_MHz']
    micro_keys = ['N', 'width_frac_low', 'width_frac_high']
    for i, comp in enumerate(components):
        for j, key in enumerate(mean_keys):
            gauss_params[i, j] = float(getattr(comp, key))
        gauss_params[i, 13] = float(comp.microshots.N)
        gauss_params[i, 14] = 100.0 * float(comp.microshots.width_frac_low)
        gauss_params[i, 15] = 100.0 * float(comp.microshots.width_frac_high)
    scatter = components[0].microshot_scatter
    stddev_row = -4
    sd_keys = ['t0_sigma_ms', 'width_sigma_ms', 'amplitude_sigma', 'spectral_index_sigma',
               'tau_sigma_ms', 'dm_sigma', 'rm_sigma', 'pa_sigma_deg',
               'lfrac_sigma', 'vfrac_sigma', 'dpa_sigma',
               'band_centre_sigma', 'band_width_sigma']
    for j, key in enumerate(sd_keys):
        gauss_params[stddev_row, j] = float(getattr(scatter, key))
    amp_sampling = _normalise_master_amp_sampling(components[0].amplitude_distribution)
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
        if name in COL_MAP:
            col = COL_MAP[name]
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
    return sim_params, prop_params, gauss_params, amp_sampling, scint, sweep_mode, master_logstep, rvm_swing


def _process_task(task, xname, plot_mode, dspec_params, target_snr=None, baseline_correct=None):
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
        diagnostics=True,
    )
    return var, measures, V_params, snr, exp_vars


def _process_obs_task(task, plot_mode, target_snr=None, baseline_correct=None,
                      obs_data=None, obs_params=None, gauss_file=None,
                      sim_file=None, scint_file=None):
    var, _ = task
    requires_multiple_frb = plot_mode.requires_multiple_frb
    dspec, freq_mhz, time_ms, dspec_params_local = load_data(
        obs_data, obs_params, gauss_file, sim_file, scint_file
    )
    scatter_idx = dspec_params_local.sc_idx
    ref_freq = dspec_params_local.ref_freq_mhz
    sefd = dspec_params_local.sefd
    f_res = dspec_params_local.freq_res_mhz
    t_res = dspec_params_local.time_res_ms
    buffer_frac = dspec_params_local.buffer_frac
    dspec = scatter_loaded_dspec(dspec, freq_mhz, time_ms, var, scatter_idx, ref_freq)
    gdict = dspec_params_local.gdict
    intrinsic_width_bins = gdict["width"][0] / t_res
    if var > 0:
        dspec = correct_baseline(dspec, intrinsic_width_bins, buffer_frac, baseline_correct,
                                 requires_multiple_frb, dspec_params_local)
    sefd = scale_dspec_to_target_snr("analytic", target_snr, dspec_params_local, dspec,
                                     freq_mhz, t_res, buffer_frac, sefd,
                                     requires_multiple_frb, time_ms)
    dspec, _, snr = add_noise(dspec_params_local, dspec, sefd,
                              (freq_mhz[1] - freq_mhz[0]) * 1e6,
                              t_res / 1000.0,
                              requires_multiple_frb, buffer_frac=buffer_frac, n_pol=2)
    segments = compute_segments(dspec, freq_mhz, time_ms, dspec_params_local,
                                buffer_frac=buffer_frac, skip_rm=True, remove_pa_trend=True)
    measures = segments
    V_params = {}
    exp_vars = {}
    return var, measures, V_params, snr, exp_vars


def _setup_sweep(gauss_params, logstep, sweep_spec, sweep_mode, plot_mode, gdict_keys):
    if sweep_mode == "none":
        raise ValueError(
            f"Plot mode '{plot_mode.name}' requires a parameter sweep. "
            "Use --sweep-mode mean or --sweep-mode sd and set exactly one non-zero step in gparams."
        )
    has_sweep = not np.all(gauss_params[-1, :] == 0.0)
    has_single_point = np.all(gauss_params[-1, :] == 0.0) and not np.all(gauss_params[-3, :] == 0.0)
    if not has_sweep and not has_single_point:
        logging.error("No sweep or single point defined. For sweep: set non-zero step. For single point: set start value with zero step.")
        logging.info("Edit the last three rows (start/stop/step) of gparams for exactly one column.")
        sys.exit(1)
    if has_sweep:
        active_cols = np.where(gauss_params[-1, :] != 0.0)[0]
    else:
        active_cols = np.where(gauss_params[-3, :] != 0.0)[0]
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
    if step == 0:
        xvals = np.array([start], dtype=float)
        logging.info(f"Using single point: {start} for realizations")
    else:
        if logstep is not None:
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
            logging.info(f"Using linear sweep: {len(xvals)} points from {xvals[0]} to {xvals[-1]} (step={step})")
    xname = gdict_keys[col_idx]
    return xvals, col_idx, xname


def _collect_results(results, xvals):
    measures = {v: [] for v in xvals}
    V_params = {v: {} for v in xvals}
    snrs = {v: [] for v in xvals}
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


def _run_sweep_parallel(xvals, nseed, n_cpus, task_func, desc):
    tasks = list(product(xvals, range(nseed)))
    workers, chunksize = pool_workers_and_chunksize(n_cpus, len(tasks))
    logging.info(f"Process pool settings: workers={workers}, chunksize={chunksize}, tasks={len(tasks)}")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(
            executor.map(task_func, tasks, chunksize=chunksize),
            total=len(tasks),
            desc=desc,
        ))
    return _collect_results(results, xvals)


def generate_frb(data, frb_id, out_dir, mode, seed, nseed, write, sim_file, gauss_file, scint_file,
                sefd, n_cpus, plot_mode, phase_window, freq_window, buffer_frac, sweep_mode, obs_data, obs_params,
                logstep=None, target_snr=None, param_overrides=None, baseline_correct=None, master_file=None,
                master_raw_config=None, save_dspec=False):
    if master_file is None:
        raise ValueError("master_file is required. Legacy split configs are no longer supported.")
    master_scint = None
    sim_params, prop_params, gauss_params, amp_sampling, master_scint, master_sweep_mode, master_logstep, rvm_swing = _master_to_internal(
        master_file, master_raw=master_raw_config,
    )
    if (sweep_mode is None or sweep_mode == "none") and master_sweep_mode is not None:
        sweep_mode = master_sweep_mode
    if logstep is None and master_logstep is not None:
        logstep = master_logstep
    f_start  = float(sim_params['f0'])
    f_end    = float(sim_params['f1'])
    t_start  = float(sim_params['t0'])
    t_end    = float(sim_params['t1'])
    f_res    = float(sim_params['f_res'])
    t_res    = float(sim_params['t_res'])
    ref_freq = float(sim_params['reference_freq'])
    scatter_idx = float(prop_params['scattering_index'])
    freq_mhz = np.arange(f_start, f_end + f_res, f_res, dtype=float)
    time_ms  = np.arange(t_start, t_end + t_res, t_res, dtype=float)
    stddev_row = -4
    start_row  = -3
    stop_row   = -2
    step_row   = -1
    gdict = {
        't0'             : gauss_params[:stddev_row, 0],
        'width'      : gauss_params[:stddev_row, 1],
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
    gdict['pa_swing'] = rvm_swing
    sd_dict = {
        'sd_t0'             : gauss_params[stddev_row, 0],
        'sd_width'        : gauss_params[stddev_row, 1],
        'sd_A'              : gauss_params[stddev_row, 2],
        'sd_spec_idx'       : gauss_params[stddev_row, 3],
        'sd_tau'          : gauss_params[stddev_row, 4],
        'sd_DM'             : gauss_params[stddev_row, 5],
        'sd_RM'             : gauss_params[stddev_row, 6],
        'sd_PA'             : gauss_params[stddev_row, 7],
        'sd_lfrac'          : gauss_params[stddev_row, 8],
        'sd_vfrac'          : gauss_params[stddev_row, 9],
        'sd_dPA'            : gauss_params[stddev_row, 10],
        'sd_band_centre_mhz': gauss_params[stddev_row, 11],
        'sd_band_width_mhz' : gauss_params[stddev_row, 12],
    }
    mean_override_parts, sd_override_parts = build_override_parts(param_overrides, gdict, sd_dict)
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
        'step'     : sweep_step[sweep_col]  if sweep_col is not None else None,
    }
    if master_scint is not None:
        scint = dict(master_scint)
        logging.info("Scintillation enabled from master config.")
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
        logging.info("Scintillation disabled in master config.")
    dspec_params = dspecParams(
        gdict=gdict, sd_dict=sd_dict, scint_dict=scint, prop_dict=prop_params,
        freq_mhz=freq_mhz, freq_res_mhz=f_res, time_ms=time_ms, time_res_ms=t_res,
        seed=seed, nseed=nseed, sefd=sefd, ref_freq_mhz=ref_freq,
        phase_window=phase_window, freq_window=freq_window,
        buffer_frac=buffer_frac, sweep_mode=sweep_mode,
    )
    tau = gdict['tau']
    if len(np.where(gauss_params[-1, :] != 0.0)[0]) > 1:
        logging.warning("More than one value in the last row of gauss_params is not 0.")
        logging.info("Please ensure that only one value is non-zero in the last row.")
        sys.exit(1)
    if np.any(gdict['lfrac'] + gdict['vfrac']) > 1.0:
        logging.warning("Linear and circular polarisation fractions sum to more than 1.0.")
    plot_multiple_frb = plot_mode.requires_multiple_frb
    if not plot_multiple_frb:
        if obs_data is not None:
            dspec, freq_mhz, time_ms, dspec_params = load_data(
                obs_data, obs_params, gauss_file, sim_file, scint_file
            )
            I_time = np.nansum(dspec[0], axis=0)
            snr, (left, right) = snr_onpulse(dspec_params, I_time, frac=0.95, buffer_frac=buffer_frac)
            logging.info(f"Loaded data S/N: {snr:.2f}, on-pulse window: {left}-{right} ({time_ms[left]:.2f}-{time_ms[right]:.2f} ms)")
            if tau[0] > 0:
                dspec = scatter_loaded_dspec(dspec, freq_mhz, time_ms, tau[0], scatter_idx, ref_freq)
            if sefd > 0:
                dspec, _, snr = add_noise(
                    dspec_params, dspec=dspec, sefd=sefd, f_res=f_res, t_res=t_res,
                    plot_multiple_frb=plot_multiple_frb, buffer_frac=buffer_frac
                )
            segments = compute_segments(dspec, freq_mhz, time_ms, dspec_params,
                                        buffer_frac=buffer_frac, skip_rm=True, remove_pa_trend=True)
        else:
            dspec, snr, _, _, segments = psn_dspec(
                xname=None, plot_multiple_frb=False,
                target_snr=target_snr, dspec_params=dspec_params,
                baseline_correct=baseline_correct, diagnostics=True,
            )
        if save_dspec:
            _write_stokes_cube(dspec, freq_mhz, time_ms, out_dir, frb_id)
        _, corrdspec, _, noise_spec = process_dspec(dspec, freq_mhz, dspec_params, buffer_frac,
                                                     skip_rm=True, remove_pa_trend=True)
        frb_data = simulated_frb(frb_id, corrdspec, dspec_params, snr)
        if write:
            write_single_frb(frb_data, out_dir, frb_id, mode, tau[0], seed, nseed, float(gdict['PA'][-1]))
        return frb_data, noise_spec, gdict, segments
    if plot_multiple_frb:
        if save_dspec:
            logging.warning("Stokes dspec saving is only supported for single FRB runs. Ignoring --save-dspec.")
        if data is not None:
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
        gdict_keys = [k for k in gdict.keys() if k not in ('amp_sampling', 'pa_swing')]
        if obs_data is not None:
            if os.path.isfile(obs_data):
                logging.info(f"Loading observed data from {obs_data}")
                with open(obs_data, 'rb') as f:
                    frb_dict = pkl.load(f)
                return frb_dict
            dspec, freq_mhz, time_ms, dspec_params = load_data(
                obs_data, obs_params, gauss_file, sim_file, scint_file
            )
            I_time = np.nansum(dspec[0], axis=0)
            original_snr, (left, right) = snr_onpulse(dspec_params, I_time, frac=0.95, buffer_frac=buffer_frac)
            logging.info(f"Original data S/N: {original_snr:.2f}, on-pulse window: {left}-{right} ({time_ms[left]:.2f}-{time_ms[right]:.2f} ms)")
            gdict_keys = [k for k in gdict.keys() if k not in ('amp_sampling', 'pa_swing')]
            if active_cols.size != 1 or gdict_keys[sweep_col] != 'tau':
                logging.error("For obs_data sweep, exactly one sweep column must be set for 'tau'.")
                sys.exit(1)
            xvals, col_idx, xname = _setup_sweep(gauss_params, logstep, sweep_spec, sweep_mode, plot_mode, gdict_keys)
            xvals, _array_id, _array_count = slurm_chunk_xvals(xvals)
            partial_func = functools.partial(
                _process_obs_task, plot_mode=plot_mode, target_snr=original_snr,
                baseline_correct=baseline_correct, obs_data=obs_data, obs_params=obs_params,
                gauss_file=gauss_file, sim_file=sim_file, scint_file=scint_file,
            )
            measures, V_params, snrs, exp_vars = _run_sweep_parallel(
                xvals, nseed, n_cpus, partial_func,
                desc=f"Processing tau sweep on observed data"
            )
        else:
            xvals, col_idx, xname = _setup_sweep(gauss_params, logstep, sweep_spec, sweep_mode, plot_mode, gdict_keys)
            xvals, _array_id, _array_count = slurm_chunk_xvals(xvals)
            partial_func = functools.partial(
                _process_task, xname=xname, plot_mode=plot_mode,
                target_snr=target_snr, dspec_params=dspec_params,
                baseline_correct=baseline_correct,
            )
            measures, V_params, snrs, exp_vars = _run_sweep_parallel(
                xvals, nseed, n_cpus, partial_func,
                desc=f"Processing sweep of {xname} ({sweep_mode} mode)"
            )
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
            from fires.utils.io import build_sweep_output_path
            fpath = build_sweep_output_path(
                out_dir, frb_id, mode, xvals, nseed, plot_mode, xname,
                sweep_idx, mean_override_parts, sd_override_parts
            )
            write_frb_dict(frb_dict, fpath)
        return frb_dict
