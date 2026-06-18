import argparse
import logging
import os
import sys
import traceback
from email import parser
from inspect import signature
from pathlib import Path

import numpy as np

from fires.config.schema import parse_fires_config
from fires.core.genfrb import generate_frb
from fires.plotting.plotmodes import plot_modes
from fires.utils import config as cfg
from fires.utils.params import EMISSION_KEYS, MEAN_ALIASES, SD_ALIASES, canonical_emission_key
from fires.utils.utils import LOG, init_logging, normalise_freq_window, normalise_phase_window


def setup_logging(verbose: bool):
    init_logging(verbose)
    if verbose:
        logging.basicConfig(level=logging.DEBUG, force=True)
    else:
        logging.basicConfig(level=logging.INFO, force=True)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
    return logging.getLogger("FIRES")


def parse_param_overrides(overrides):
    emission_overrides = {}
    config_overrides = {}
    if not overrides:
        return {}, {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: '{override}'. Expected 'key=value'.")
        key, value = override.split("=", 1)
        key = key.strip()
        canonical_key = canonical_emission_key(key)
        base_key = canonical_key
        if canonical_key.startswith("sd_"):
            base_key = canonical_key[3:]
        if canonical_key in EMISSION_KEYS or base_key in EMISSION_KEYS:
            try:
                val = float(value)
                emission_overrides[canonical_key] = val
            except ValueError:
                raise ValueError(f"Invalid value for emission override '{key}': '{value}' (must be numeric).")
        else:
            config_overrides[key] = value
    return emission_overrides, config_overrides


def apply_config_overrides(raw_config: dict, overrides: dict) -> None:
    from fires.utils.config import apply_config_overrides as _apply
    _apply(raw_config, overrides)


def apply_plot_overrides(plot_config, overrides):
    if not overrides:
        return plot_config
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid plot override format: '{override}'. Expected 'param=value'.")
        key, value = override.split("=", 1)
        key = key.strip()
        try:
            if value.startswith('[') and value.endswith(']'):
                import ast
                val = ast.literal_eval(value)
            elif value.lower() in ('true', 'false'):
                val = value.lower() == 'true'
            elif value.lower() in ('null', 'none'):
                val = None
            else:
                try:
                    val = float(value)
                    if val.is_integer():
                        val = int(val)
                except ValueError:
                    val = value
        except Exception:
            raise ValueError(f"Invalid value for plot override '{key}': '{value}'")
        if '.' in key:
            sections = key.split('.')
            current = plot_config
            for section in sections[:-1]:
                if section not in current:
                    current[section] = {}
                current = current[section]
            current[sections[-1]] = val
        else:
            if 'general' not in plot_config:
                plot_config['general'] = {}
            plot_config['general'][key] = val
    return plot_config


def parse_compare_windows(specs):
    if not specs:
        return None
    pairs = []
    for spec in specs:
        if ':' not in spec:
            raise ValueError(f"Invalid --compare-windows format: '{spec}'. Expected 'freq:phase'.")
        freq, phase = spec.split(':', 1)
        pairs.append((freq.strip(), phase.strip()))
    return pairs if pairs else None


def print_sweep_info(args):
    """Print sweep info from a fires.toml config for use by Slurm scripts."""
    config_dir = args.config_dir
    resolved = cfg.find_config_file("fires", config_dir=config_dir)
    if not resolved:
        print("ERROR: no fires.toml found", file=sys.stderr)
        sys.exit(1)
    raw = cfg.load_params("fires", override_path=resolved)
    param = raw.get('analysis', {}).get('sweep', {}).get('parameter', {})
    name = str(param.get('name', 'UNKNOWN'))
    start = param.get('start', 0)
    stop = param.get('stop', 0)
    step = param.get('step', 0)
    nm = name.lower().strip()
    prefixes = ['meas_mean_', 'meas_std_', 'meas_var_', 'mean_', 'sd_']
    for p in prefixes:
        if nm.startswith(p):
            nm = nm[len(p):]
    suffixes = ['_mean', '_sigma']
    for s in suffixes:
        if nm.endswith(s):
            nm = nm[:-len(s)]
    nm = nm.replace('_deg_per_ms', '').replace('_deg', '').replace('deg', '')
    nm = nm.replace('_mhz', '').replace('_hz', '').replace('_ms', '')
    mapping = {
        't0': 't0', 'width': 'width', 'a': 'A', 'amplitude': 'A', 'amplitude_jy': 'A',
        'spec_idx': 'spec_idx', 'spectral_index': 'spec_idx',
        'tau': 'tau', 'dm': 'DM', 'rm': 'RM', 'pa': 'PA', 'pa_deg': 'PA',
        'pa_sigma_deg': 'pa_sigma_deg', 'sd_pa': 'pa_sigma_deg', 'pa_sigma': 'pa_sigma_deg',
        'lfrac': 'lfrac', 'vfrac': 'vfrac', 'dpa': 'dPA', 'dpa_deg_per_ms': 'dPA',
        'band_centre': 'band_centre', 'band_centre_mhz': 'band_centre',
        'band_width': 'band_width', 'band_width_mhz': 'band_width',
        'n': 'N', 'mg_width_low': 'mg_width_low', 'mg_width_high': 'mg_width_high',
    }
    canonical = mapping.get(nm, 'UNKNOWN')
    print(f"{canonical}\t{start}\t{stop}\t{step}")


def main():
    parser = argparse.ArgumentParser(
        description="FIRES: The Fast, Intense Radio Emission Simulator. Simulate Fast Radio Bursts (FRBs) with scattering and polarisation effects",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    parser.add_argument(
        "--config-dir", type=str,
        help="Path to fires.toml or directory containing fires.toml (required)."
    )
    parser.add_argument(
        "--init-config", action="store_true",
        help="Create user config from packaged defaults"
    )
    parser.add_argument(
        "--edit-config", choices=["fires", "plotparams"],
        help="Open default config in $EDITOR"
    )
    parser.add_argument(
        "--print-sweep-info", action="store_true",
        help="Print sweep parameter info from fires.toml (for Slurm scripts)"
    )
    parser.add_argument(
        "-f", "--frb_identifier", type=str, default="FRB", metavar="",
        help="Identifier for the simulated FRB."
    )
    parser.add_argument(
        "-d", "--sim-data", type=str, default=None, metavar="",
        help="Path to simulation data files."
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="simfrbs/", metavar="",
        help="Directory to save the simulated FRB data (default: 'simfrbs/')."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose output."
    )
    parser.add_argument(
        "--no-write", dest="write_output", action="store_false", default=True,
        help="Disable writing outputs to disk (default: write outputs)."
    )
    parser.add_argument(
        "--sd", "--save-dspec", "--wd", "--write-dspec",
        dest="save_dspec", action="store_true", default=False,
        help="Save the Stokes dynamic spectrum cube."
    )
    parser.add_argument(
        "-m", "--mode", type=str, default='psn', choices=['psn'], metavar="",
        help="Mode for generating pulses: 'psn'. Default is 'psn.'"
    )
    parser.add_argument(
        "--override-param", type=str, nargs="+", action="extend", default=None,
        metavar="PARAM=VALUE",
        help=("Override any parameter from fires.toml. Use dot notation for nested keys.\n"
              "Emission parameters (numeric):\n"
              "  --override-param N=5 tau=0.5\n"
              "Config parameters (dotted paths):\n"
              "  --override-param propagation.scintillation.enable=false\n")
    )
    parser.add_argument(
        "--phase-window", type=str, default="total",
        choices=['first', 'last', 'all', 'leading', 'trailing', 'total'], metavar="",
        help="Select the phase window for the simulation."
    )
    parser.add_argument(
        "--freq-window", type=str, default="full-band",
        choices=['1q', '2q', '3q', '4q', 'full',
                 'lowest-quarter', 'lower-mid-quarter', 'upper-mid-quarter', 'highest-quarter', 'full-band'],
        metavar="",
        help="Select the frequency window for the simulation."
    )
    parser.add_argument(
        "-p", "--plot", nargs="+", default=['lvpa'],
        choices=['all', 'None', 'iquv', 'lvpa', 'dpa', 'RM', 'pa_var', 'l_frac', 'pa', 'pali'],
        metavar="",
        help="Generate plots. Pass 'all' to generate all plots, or specify one or more plot names."
    )
    parser.add_argument(
        "--plot-config", type=str, default=None, metavar="",
        help="Path to custom plotting configuration file."
    )
    parser.add_argument(
        "--override-plot", nargs="+", action="extend", default=None,
        metavar="PARAM=VALUE",
        help="Override plotting parameters. Provide space-separated key=value pairs."
    )
    parser.add_argument(
        "--pub-col", type=float, default=2, metavar="N",
        help="Number of equal figures per row (1, 2, 3, ...). Figure width = FULL_PAGE_WIDTH_IN / N."
    )
    parser.add_argument(
        "--compare-windows", type=str, nargs="+", metavar="FREQ:PHASE",
        help="Compare multiple freq/phase windows from a SINGLE run on same plot."
    )
    parser.add_argument(
        "--obs-data", type=str, default=None, metavar="",
        help="Path to observational FRB data to overlay on analytic plots."
    )
    parser.add_argument(
        "--obs-params", type=str, default=None, metavar="",
        help="Path to parameters file for observational data."
    )
    args = parser.parse_args()

    LOG = setup_logging(args.verbose)
    LOG.debug("Verbose logging enabled.")

    if args.init_config:
        cfg.init_user_config(overwrite=True, backup=True)
        print(f"Config files synced to: {cfg.user_config_dir()}\n")
        return 0
    if args.edit_config:
        cfg.edit_params(args.edit_config, config_dir=args.config_dir)
        return
    if args.print_sweep_info:
        print_sweep_info(args)
        return 0

    resolved_master = cfg.find_config_file("fires", config_dir=args.config_dir)
    use_master = resolved_master is not None
    master_cfg = None
    try:
        emission_param_overrides, config_overrides = parse_param_overrides(args.override_param)
        raw_master_config = cfg.load_params("fires", override_path=resolved_master)
        if config_overrides:
            apply_config_overrides(raw_master_config, config_overrides)
        master_cfg = parse_fires_config(raw_master_config)
        logging.info("Using master config: %s", resolved_master)
        if args.output_dir == "simfrbs/":
            args.output_dir = str(master_cfg.output.directory)
    except Exception as e:
        parser.error(f"Failed to load master config {resolved_master}: {e}")

    write_output = bool(getattr(args, 'write_output', True))
    try:
        raw_out = raw_master_config.get('output', {}) if raw_master_config is not None else {}
        if 'write' in raw_out:
            logging.warning("Config key 'output.write' is deprecated; use --no-write to disable writing. Ignoring config value.")
    except Exception:
        pass

    seed = master_cfg.meta.seed
    ncpu = int(master_cfg.numerics.n_cpus)
    _slurm_n = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE") or os.environ.get("SLURM_CPUS_PER_NODE")
    if _slurm_n:
        try:
            _slurm_val = int(_slurm_n)
            if _slurm_val > 0:
                ncpu = _slurm_val
                logging.info(f"Overriding n_cpus from environment: {ncpu}")
        except Exception:
            pass
    nseed = int(master_cfg.numerics.nseed)
    buffer_frac = None
    try:
        buffer_frac = float(master_cfg.analysis.buffer_fraction)
    except Exception:
        buffer_frac = None
    sefd = float(master_cfg.observation.sefd)
    target_snr = float(master_cfg.observation.target_snr) if master_cfg.observation.target_snr is not None else None
    if target_snr is not None and target_snr <= 0:
        target_snr = None
    baseline_correct = master_cfg.observation.baseline_correct
    if isinstance(baseline_correct, bool) and not baseline_correct:
        baseline_correct = None
    if bool(master_cfg.analysis.sweep.enable):
        sweep_mode = str(master_cfg.analysis.sweep.mode).lower()
    else:
        sweep_mode = "none"
    logstep = None
    if master_cfg.analysis.sweep.parameter.log_steps is not None:
        logstep = int(master_cfg.analysis.sweep.parameter.log_steps)

    plot_config = {}
    try:
        if args.plot_config:
            resolved_plot = cfg.find_config_file("plotparams", config_dir=args.plot_config)
            plot_config = cfg.load_params("plotparams", str(resolved_plot))
        else:
            resolved_plot = cfg.find_config_file("plotparams", config_dir=args.config_dir)
            plot_config = cfg.load_params("plotparams", str(resolved_plot))
    except Exception as e:
        logging.warning(f"Could not load plot config: {e}. Using defaults.")
        plot_config = {}

    args.freq_window = normalise_freq_window(args.freq_window, target='dspec')
    args.phase_window = normalise_phase_window(args.phase_window, target='dspec')

    try:
        window_pairs = parse_compare_windows(args.compare_windows)
    except ValueError as e:
        parser.error(str(e))

    if args.plot[0] not in plot_modes and args.plot[0] not in ("all", "None"):
        parser.error(f"Invalid plot mode: {args.plot[0]}")

    selected_plot_mode = plot_modes[args.plot[0]] if args.plot[0] in plot_modes else plot_modes['lvpa']

    global data_directory
    data_directory = args.output_dir

    save_plots = plot_config.get('general', {}).get('save_plots', False)
    show_plots = plot_config.get('general', {}).get('show_plots', True)

    if selected_plot_mode.requires_multiple_frb and (not save_plots) and (not show_plots) and (not write_output):
        write_output = True
        logging.info("Batch run with save_plots/show_plots disabled: forcing writing outputs to disk")

    if write_output or save_plots:
        os.makedirs(args.output_dir, exist_ok=True)
        logging.info(f"Output directory: '{data_directory}' \n")

    try:
        if emission_param_overrides:
            logging.info(f"Emission parameter overrides: {emission_param_overrides}")
        all_param_overrides = emission_param_overrides
    except ValueError as e:
        parser.error(str(e))

    try:
        plot_config = apply_plot_overrides(plot_config, args.override_plot)
    except ValueError as e:
        parser.error(str(e))

    from fires.plotting.plot_helper import set_pub_col
    if args.pub_col is not None:
        set_pub_col(args.pub_col)
    from fires.plotting.plotmodes import configure_matplotlib_from_config
    configure_matplotlib_from_config(plot_config)

    # Initialise all conditional variables to None before branching
    FRB = None
    gdict = None
    frb_dict = None
    segments = None

    try:
        base_kwargs = dict(
            data=args.sim_data, frb_id=args.frb_identifier, sim_file=None,
            gauss_file=None, scint_file=None,
            master_file=str(resolved_master) if use_master and resolved_master is not None else None,
            out_dir=args.output_dir, write=write_output, save_dspec=args.save_dspec,
            mode=args.mode, seed=seed, nseed=None, sefd=sefd, n_cpus=None,
            plot_mode=selected_plot_mode, phase_window=args.phase_window,
            freq_window=args.freq_window, buffer_frac=buffer_frac, sweep_mode=None,
            target_snr=target_snr, obs_data=args.obs_data, obs_params=args.obs_params,
            param_overrides=all_param_overrides, logstep=None,
            baseline_correct=baseline_correct, master_raw_config=raw_master_config,
        )
        if selected_plot_mode.requires_multiple_frb:
            if args.sim_data is None:
                logging.info(f"Processing with {ncpu} threads. \n")
            base_kwargs.update(dict(
                nseed=nseed, n_cpus=ncpu, sweep_mode=sweep_mode,
                obs_data=None, obs_params=None, logstep=logstep,
            ))
            frb_dict = generate_frb(**base_kwargs)
        else:
            base_kwargs.update(dict(
                nseed=None, n_cpus=None, sweep_mode=None,
                obs_data=args.obs_data, obs_params=args.obs_params, logstep=None,
            ))
            FRB, noisespec, gdict, segments = generate_frb(**base_kwargs)

        if args.sim_data is None:
            print(f"Simulation completed. \n")

        if args.plot != 'None' and (save_plots or show_plots):
            for plot_mode in args.plot:
                try:
                    plot_mode_obj = plot_modes.get(plot_mode)
                    if plot_mode_obj is None:
                        print(f"Error: Plot mode '{plot_mode}' is not defined in plotmodes.py. \n")
                        continue
                    plotting_args = {
                        "fname": args.frb_identifier,
                        "frb_data": FRB,
                        "mode": plot_mode,
                        "gdict": gdict,
                        "frb_dict": frb_dict,
                        "out_dir": data_directory,
                        "phase_window": args.phase_window,
                        "freq_window": args.freq_window,
                        "compare_windows": window_pairs,
                        "obs_data": args.obs_data,
                        "obs_params": args.obs_params,
                        "gauss_file": str(resolved_master),
                        "sim_file": str(resolved_master),
                        "plot_config": plot_config,
                        "buffer_frac": buffer_frac,
                        "segments": segments,
                    }
                    plot_function = plot_mode_obj.plot_func
                    plot_func_params = signature(plot_function).parameters
                    filtered_args = {key: value for key, value in plotting_args.items() if key in plot_func_params}
                    plot_function(**filtered_args)
                except Exception as e:
                    logging.error(f"An error occurred while plotting '{plot_mode}': {e} \n")
                    if args.verbose:
                        traceback.print_exc()
        else:
            logging.info("No plots generated. \n")
    except Exception as e:
        logging.error(f"Error: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
