# -----------------------------------------------------------------------------
# main.py
# FIRES: The Fast, Intense Radio Emission Simulator
#
# This script serves as the main entry point for simulating Fast Radio Bursts (FRBs)
# with scattering and polarisation effects. It parses command-line arguments,
# manages simulation and output options, and calls the appropriate functions for
# generating FRBs and plotting results.
#
# Author: JB
# Date: 2025-05-20
# -----------------------------------------------------------------------------

import argparse
import logging
#	--------------------------	Import modules	---------------------------
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
from fires.utils.utils import (LOG, init_logging, normalise_freq_window,
                               normalise_phase_window)


def setup_logging(verbose: bool):
	"""Initialize logging according to verbosity flag and return root logger."""
	init_logging(verbose)
	if verbose:
		logging.basicConfig(level=logging.DEBUG, force=True)
	else:
		logging.basicConfig(level=logging.INFO, force=True)
	logging.getLogger("PIL").setLevel(logging.WARNING)
	logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
	return logging.getLogger("FIRES")


def parse_param_overrides(overrides):
	"""Parse list of 'key=value' strings into mean, std, and config override dicts.

	Returns (emission_param_overrides, config_overrides).
	"""
	# Emission parameter keys that can be overridden via gdict/sd_dict
	emission_keys = {
		't0', 'width', 'A', 'spec_idx', 'tau', 'DM', 'RM', 'PA',
		'lfrac', 'vfrac', 'dPA', 'band_centre_mhz', 'band_width_mhz',
		'N', 'mg_width_low', 'mg_width_high', 'amp_sampling'
	}
	
	emission_overrides = {}
	config_overrides = {}
	
	if not overrides:
		return {}, {}
	
	for override in overrides:
		if "=" not in override:
			raise ValueError(f"Invalid override format: '{override}'. Expected 'key=value'.")
		key, value = override.split("=", 1)
		key = key.strip()
		
		# Check if this is an emission parameter (handle both mean and std variants)
		base_key = key
		if key.startswith("sd_"):
			base_key = key[3:]
		elif key.endswith("_sd"):
			base_key = key[:-3]
		
		# If it looks like an emission parameter, try numeric parse
		if base_key in emission_keys or key.startswith("sd_") or key.endswith("_sd"):
			try:
				val = float(value)
				emission_overrides[key] = val
			except ValueError:
				raise ValueError(f"Invalid value for emission override '{key}': '{value}' (must be numeric).")
		else:
			# Config override - parse type intelligently
			config_overrides[key] = value
	
	return emission_overrides, config_overrides


def apply_config_overrides(raw_config: dict, overrides: dict) -> None:
	"""Apply dotted-key overrides to a nested config dictionary in-place.
	
	Examples:
		apply_config_overrides(cfg, {"propagation.scintillation.enable": "true"})
		apply_config_overrides(cfg, {"observation.sefd": "1.5"})
	"""
	for key_path, value_str in overrides.items():
		keys = key_path.split(".")
		current = raw_config
		
		# Navigate to the parent of the target
		for k in keys[:-1]:
			if k not in current:
				current[k] = {}
			current = current[k]
		
		# Coerce value to appropriate type
		value_str = str(value_str).strip()
		if value_str.lower() in ("true", "false"):
			final_value = value_str.lower() == "true"
		elif value_str.lower() in ("null", "none"):
			final_value = None
		else:
			try:
				if "." in value_str or "e" in value_str.lower():
					final_value = float(value_str)
				else:
					final_value = int(value_str)
			except ValueError:
				# Keep as string if not numeric
				final_value = value_str
		
		current[keys[-1]] = final_value
		logging.info(f"Config override applied: {key_path} = {final_value}")


def apply_plot_overrides(plot_config, overrides):
	"""Apply plot overrides (list of key=value) into plot_config dict.

	Returns modified plot_config (in-place modification as well).
	"""
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
	"""Parse --compare-windows specs into list of (freq, phase) tuples."""
	if not specs:
		return None
	pairs = []
	for spec in specs:
		if ':' not in spec:
			raise ValueError(f"Invalid --compare-windows format: '{spec}'. Expected 'freq:phase'.")
		freq, phase = spec.split(':', 1)
		pairs.append((freq.strip(), phase.strip()))
	return pairs if pairs else None


def main():
	parser = argparse.ArgumentParser(description="FIRES: The Fast, Intense Radio Emission Simulator. Simulate Fast Radio Bursts (FRBs) with scattering and polarisation effects",
								  formatter_class=argparse.RawTextHelpFormatter)

	# Show help when no args are provided (no traceback)
	if len(sys.argv) == 1:
		parser.print_help()
		return 0

	# =====================================================================
	# Configuration Management
	# =====================================================================
	parser.add_argument(
		"--config-dir", 
		type=str, 
		help="Path to fires.toml or directory containing fires.toml (required)."
	)
	parser.add_argument(
		"--init-config", 
		action="store_true", 
		help="Create user config from packaged defaults"
	)
	parser.add_argument(
		"--edit-config", 
		choices=["fires", "plotparams"], 
		help="Open default config in $EDITOR"
	)

	# =====================================================================
	# Input/Output
	# =====================================================================
	parser.add_argument(
		"-f", "--frb_identifier",
		type=str,
		default="FRB",
		metavar="",
		help="Identifier for the simulated FRB."
	)
	parser.add_argument(
		"-d", "--sim-data",
		type=str,
		default=None,
		metavar="",
		help="Path to simulation data files. If provided, the simulation will use this data instead of generating new data."
	)
	parser.add_argument(
		"-o", "--output-dir",
		type=str,
		default="simfrbs/",
		metavar="",
		help="Directory to save the simulated FRB data (default: 'simfrbs/')."
	)
	parser.add_argument(
		"-v", "--verbose",
		action="store_true",
		help="Enable verbose output."
	)

	# =====================================================================
	# Simulation Parameters
	# =====================================================================
	parser.add_argument(
		"-m", "--mode",
		type=str,
		default='psn',
		choices=['psn'],
		metavar="",
		help=("Mode for generating pulses: 'psn'. Default is 'psn.'\n"
			  "'psn' will generate a gaussian distribution of gaussian micro-shots."
		   )
	)
	parser.add_argument(
		"--override-param",
		type=str,
		nargs="+",
		action="extend", 
		default=None,
		metavar="PARAM=VALUE",
		help=(
			"Override any parameter from fires.toml. Use dot notation for nested keys. Provide space-separated key=value pairs.\n"
			"Emission parameters (numeric):\n"
			"  --override-param N=5 tau=0.5\n"
			"  --override-param lfrac=0.8\n"
			"Config parameters (dotted paths):\n"
			"  --override-param propagation.scintillation.enable=false\n"
			"  --override-param propagation.scintillation.timescale_s=100\n"
			"  --override-param observation.sefd=1.5\n"
			"Mixed example:\n"
			"  --override-param tau=0.5 propagation.scintillation.enable=false observation.sefd=1.2\n"
			"Note: Boolean values can be specified as 'true'/'false', numeric values are auto-detected."
		)
	)

	# =====================================================================
	# Window Selection
	# =====================================================================
	parser.add_argument(
		"--phase-window",
		type=str,
		default="total",
		choices=['first', 'last', 'all',
	  			'leading', 'trailing', 'total'
		],
		metavar="",
		help=("Select the phase window for the simulation. Uses the pulse peak and bounds of the on-pulse region.\n"
			   "Default is 'total' or 'all'."
	  )
	)
	parser.add_argument(
		"--freq-window",
		type=str,
		default="full-band",
		choices=[
			'1q', '2q', '3q', '4q', 'full',  
			'lowest-quarter', 'lower-mid-quarter', 'upper-mid-quarter', 'highest-quarter', 'full-band' 
		   ],
		metavar="",
		help=("Select the frequency window for the simulation.\n"
				"Default is 'full-band' or 'full'."
	  )
	)
	# =====================================================================
	# Plotting Options - General
	# =====================================================================
	parser.add_argument(
		"-p", "--plot",
		nargs="+",
		default=['lvpa'],
		choices=['all', 'None', 'iquv', 'lvpa', 'dpa', 'RM', 'pa_var', 'l_frac', 'pa'],
		metavar="",
			help=(
			"Generate plots. Pass 'all' to generate all (non-analytical) plots, or specify one or more plot names separated by spaces:\n"
			"Basic Plots:\n"
			"  'iquv': Plot the Stokes parameters (I, Q, U, V) vs. time or frequency.\n"
			"  'lvpa': Plot linear polarisation (L) and polarisation angle (PA) vs. time.\n"
			"  'dpa': Plot the derivative of the polarisation angle (dPA/dt) vs. time.\n"
			"  'RM': Plot the rotation measure (RM) vs. frequency from RM-Tools.\n"
			"Analytical Plots:\n"
			"  'pa_var': Plot the variance of the polarisation angle (PA) vs. swept parameter.\n"
			"  'l_frac': Plot the fraction of linear polarisation (L/I)/(L/I)_0 vs. swept parameter.\n"
			"Pass 'None' to disable all plots."
		)
	)
	parser.add_argument(
		"--plot-config",
		type=str,
		default=None,
		metavar="",
		help="Path to custom plotting configuration file (overrides default plotparams.toml)"
	)
	parser.add_argument(
		"--override-plot",
		#type=str,
		nargs="+",
		action="extend",
		default=None,
		metavar="PARAM=VALUE",
		help=(
			"Override plotting parameters. Provide space-separated key=value pairs.\n"
			"Examples:\n"
			"  --override-plot figsize=[10,8] use_latex=true\n"
			"  --override-plot save_plots=true extension=png\n"
			"  --override-plot styling.font_size=20 general.show_plots=false\n"
		)
	)
	# =====================================================================
	# Plotting Options - Analytic Plots (pa_var, l_frac)
	# =====================================================================
	parser.add_argument(
		"--compare-windows",
		type=str,
		nargs="+",
		metavar="FREQ:PHASE",
		help=("Compare multiple freq/phase windows from a SINGLE run on same plot.\n"
			  "Format: --compare-windows FREQ:PHASE [FREQ:PHASE ...]\n"
			  "Examples:\n"
			  "  --compare-windows all:leading all:trailing all:total\n"
			  "  --compare-windows 1q:total 4q:total all:total\n"
			  "  --compare-windows all:leading 1q:total 4q:trailing\n"
			  "Valid FREQ: all, 1q, 2q, 3q, 4q, full-band, etc.\n"
			  "Valid PHASE: leading, trailing, total, first, last, all\n"
			  "Only works with single-run data (not multi-run sweeps).")
	)
	# =====================================================================
	# Observational Data Overlay
	# =====================================================================
	parser.add_argument(
		"--obs-data",
		type=str,
		default=None,
		metavar="",
		help="Path to observational FRB data to overlay on analytic plots (e.g., for l_frac or pa_var). Should be a .npy or .pkl file with dynamic spectrum."
	)
	parser.add_argument(
		"--obs-params",
		type=str,
		default=None,
		metavar="",
		help="Path to parameters file for observational data (optional). If not provided, will attempt to extract from --obs-data directory."
	)

	# =====================================================================
	# Main Execution
	# =====================================================================
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


	# Resolve master config: prefer explicit --config-dir, else fall back to
	# user config dir, then packaged defaults. `find_config_file` handles
	# this search order and will ensure user defaults are present.
	resolved_master = cfg.find_config_file("fires", config_dir=args.config_dir)
	use_master = resolved_master is not None
	master_cfg = None
	try:
		# Parse config overrides first
		emission_param_overrides, config_overrides = parse_param_overrides(args.override_param)
		
		# Load raw config
		raw_master_config = cfg.load_params("fires", override_path=resolved_master)
		
		# Apply config-level overrides to raw config before parsing
		if config_overrides:
			apply_config_overrides(raw_master_config, config_overrides)
		
		# Now parse the (potentially modified) config
		master_cfg = parse_fires_config(raw_master_config)
		logging.info("Using master config: %s", resolved_master)
		if args.output_dir == "simfrbs/":
			args.output_dir = str(master_cfg.output.directory)
	except Exception as e:
		parser.error(f"Failed to load master config {resolved_master}: {e}")

	write_output = bool(master_cfg.output.write)
	seed = master_cfg.meta.seed
	ncpu = int(master_cfg.numerics.n_cpus)
	# Allow SLURM environment to override configured CPU count
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
	# Buffer fraction applies to both simulated and observational data.
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
			plot_config = cfg.load_params("plotparams",str(resolved_plot))
		else:
			resolved_plot = cfg.find_config_file("plotparams", config_dir=args.config_dir)
			plot_config = cfg.load_params("plotparams",str(resolved_plot))
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


	global data_directory
	data_directory = args.output_dir

	save_plots = plot_config.get('general', {}).get('save_plots', False)

	if write_output or save_plots:
		os.makedirs(args.output_dir, exist_ok=True)
		logging.info(f"Output directory: '{data_directory}' \n")
  

	try:
		# emission_param_overrides and config_overrides already parsed above
		if emission_param_overrides:
			logging.info(f"Emission parameter overrides: {emission_param_overrides}")
		all_param_overrides = emission_param_overrides
	except ValueError as e:
		parser.error(str(e))


	try:
		plot_config = apply_plot_overrides(plot_config, args.override_plot)
	except ValueError as e:
		parser.error(str(e))

	from fires.plotting.plotmodes import configure_matplotlib_from_config
	configure_matplotlib_from_config(plot_config)


	selected_plot_mode = plot_modes[args.plot[0]] if args.plot[0] in plot_modes else plot_modes['lvpa']
	try:
		# Build base kwargs for generate_frb and adjust per-mode to avoid duplicate call sites
		base_kwargs = dict(
			data            = args.sim_data,
			frb_id          = args.frb_identifier,
			sim_file        = None,
			gauss_file      = None,
			scint_file      = None,
			master_file     = str(resolved_master) if use_master and resolved_master is not None else None,
			out_dir         = args.output_dir,
			write           = write_output,
			mode            = args.mode,
			seed            = seed,
			nseed           = None,
			sefd            = sefd,
			n_cpus          = None,
			plot_mode       = selected_plot_mode,
			phase_window    = args.phase_window,
			freq_window     = args.freq_window,
			buffer_frac     = buffer_frac,
			sweep_mode      = None,
			target_snr      = target_snr,
			obs_data        = args.obs_data,
			obs_params      = args.obs_params,
			param_overrides = all_param_overrides,
			logstep         = None,
			baseline_correct = baseline_correct,
			master_raw_config = raw_master_config,
		)

		if selected_plot_mode.requires_multiple_frb:
			if args.sim_data is None:
				logging.info(f"Processing with {ncpu} threads. \n")
			base_kwargs.update(dict(nseed=nseed, n_cpus=ncpu, sweep_mode=sweep_mode, obs_data=None, obs_params=None, logstep=logstep))
			frb_dict = generate_frb(**base_kwargs)
		else:
			base_kwargs.update(dict(nseed=None, n_cpus=None, sweep_mode=None, obs_data=args.obs_data, obs_params=args.obs_params, logstep=None))
			FRB, noisespec, gdict, segments = generate_frb(**base_kwargs)
		# Print simulation status
		if args.sim_data is None:
			print(f"Simulation completed. \n")

		show_plots = plot_config.get('general', {}).get('show_plots', True)
		if args.plot != 'None' and (save_plots or show_plots):
			for plot_mode in args.plot:
				try:
					plot_mode_obj = plot_modes.get(plot_mode)
					if plot_mode_obj is None:
						print(f"Error: Plot mode '{plot_mode}' is not defined in plotmodes.py. \n")
						continue
					plotting_args = {
						"fname"            : args.frb_identifier,
						"frb_data"         : FRB if 'FRB' in locals() else None,
						"mode"             : plot_mode,
						"gdict"            : gdict if 'gdict' in locals() else None,
						"frb_dict"         : frb_dict if 'frb_dict' in locals() else None,
						"out_dir"          : data_directory,
						"phase_window"     : args.phase_window,
						"freq_window"      : args.freq_window,
						"compare_windows"  : window_pairs,
						"obs_data"         : args.obs_data,
						"obs_params"       : args.obs_params,
						"gauss_file"       : str(resolved_master),
						"sim_file"         : str(resolved_master),
						"plot_config"      : plot_config,
						"buffer_frac"      : buffer_frac,
						"segments"         : segments if 'segments' in locals() else None,
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
		# Clean error by default; full traceback only with --verbose
		logging.error(f"Error: {e}")
		if args.verbose:
			traceback.print_exc()
		return 1



if __name__ == "__main__":
	sys.exit(main())	