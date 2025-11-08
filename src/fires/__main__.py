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

import numpy as np

from fires.core.genfrb import generate_frb
from fires.plotting.plotmodes import plot_modes
from fires.utils import config as cfg
from fires.utils.utils import (LOG, chi2_fit, gaussian_model, init_logging,
							   normalise_freq_window, normalise_phase_window)


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
		help="Override user config dir (default: ~/.config/fires)"
	)
	parser.add_argument(
		"--init-config", 
		action="store_true", 
		help="Create user config from packaged defaults"
	)
	parser.add_argument(
		"--edit-config", 
		choices=["gparams", "simparams", "scparams", "plotparams"], 
		help="Open config in $EDITOR"
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
		"--write",
		action="store_true",
		help="If set, the simulation data will be pickled."
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
		"--seed",
		type=int,
		default=None,
		metavar="",
		help="Set seed for repeatability in psn mode."
	)
	parser.add_argument(
		"--nseed",
		type=int,
		default=1,
		metavar="",
		help="How many realisations to generate for analytical plots."
	)
	parser.add_argument(
		"--ncpu",
		type=int,
		default=1,
		metavar="",
		help="Number of CPUs to use for parallel processing. Default is 1 (single-threaded)."
	)
	parser.add_argument(
		"--sefd",
		type=float,
		default=0.0,
		metavar="",
		help="System equivalent flux density in Jansky for adding noise. Default is 0 Jy (no noise)."
	)
	parser.add_argument(
		"--snr",
		type=float,
		default=None,
		metavar="",
		help="Target S/N for the pulse peak. If set, this will override the --sefd option."
	)
	parser.add_argument(
		"--scint",
		action="store_true",
		help="Enable scintillation effects from scparamts.toml."
	)
	parser.add_argument(
		"--chi2-fit",
		action="store_true",
		help="Enable chi-squared fitting on the final profiles (plot != analytical)."
	)
	parser.add_argument(
		"--override-param",
		type=str,
		nargs="+",
		action="extend", 
		default=None,
		metavar="PARAM=VALUE",
		help=(
			"Override gparams parameters. Provide space-separated key=value pairs.\n"
			"Examples:\n"
			"  --override-param N=5 tau=0.5\n"
			"  --override-param lfrac=0.8\n"
			"  --override-param tau=0.5 tau_std=0.2\n"
			"  --override-param N=5 --override-param mg_width_low=20 mg_width_high=40\n"
			"Note: Use PARAM=VALUE to override the mean, and PARAM_sd=VALUE or sd_PARAM=VALUE to override the std dev."
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
	parser.add_argument(
		"--buffer",
		type=float,
		default=1,
		metavar="",
		help="Buffer time in between on- and off-pulse regions as a fraction of the intrinsic pulse width for noise estimation. Default is 1."
	)
	# =====================================================================
	# Plotting Options - General
	# =====================================================================
	parser.add_argument(
		"-p", "--plot",
		nargs="+",
		default=['lvpa'],
		choices=['all', 'None', 'iquv', 'lvpa', 'dpa', 'RM', 'pa_var', 'l_frac'],
		metavar="",
		help=(
			"Generate plots. Pass 'all' to generate all (non-analytical) plots, or specify one or more plot names separated by spaces:\n"
			"Basic Plots:\n"
			"  'iquv': Plot the Stokes parameters (I, Q, U, V) vs. time or frequency.\n"
			"  'lvpa': Plot linear polarisation (L) and polarisation angle (PA) vs. time.\n"
			"  'dpa': Plot the derivative of the polarisation angle (dPA/dt) vs. time.\n"
			"  'RM': Plot the rotation measure (RM) vs. frequency from RM-Tools.\n"
			"Analytical Plots:\n"
			"  'pa_var': Plot the variance of the polarisation angle (PA) vs. swept parameter in gparams.\n"
			"  'l_frac': Plot the fraction of linear polarisation (L/I)/(L/I)_0 vs. swept parameter in gparams.\n"
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
		type=str,
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
		"--logstep",
		type=int,
		default=None,
		help="Number of steps for logarithmic parameter sweeps --- overrides default linear step in gparams (default: None --- will use default linear step)."
	)
	parser.add_argument(
		"--sweep-mode",
		type=str,
		default="none",
		choices=["none", "mean", "sd"],
		metavar="",
		help=("Parameter sweep mode for analytical plots:\n"
			  "  none      : disable sweeping (use means + micro std dev only)\n"
			  "  mean      : sweep the mean value (std dev forced to 0 for that param)\n"
			  "  sd		   : keep mean fixed, sweep the micro std dev\n")
	)
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

	init_logging(args.verbose)
	if args.verbose:
		LOG.debug("Verbose logging enabled.")


	if args.init_config:
		cfg.init_user_config(overwrite=True, backup=True)
		print(f"Config files synced to: {cfg.user_config_dir()}\n")
		return 0
	if args.edit_config:
		cfg.edit_params(args.edit_config, config_dir=args.config_dir)
		return


	resolved_sim   = str(cfg.find_config_file("simparams", config_dir=args.config_dir))
	resolved_gauss = str(cfg.find_config_file("gparams",   config_dir=args.config_dir))
	if args.scint:
		resolved_scint = str(cfg.find_config_file("scparams", config_dir=args.config_dir))
	else:
		resolved_scint = None

	plot_config = {}
	try:
		if args.plot_config:
			resolved_plot = cfg.find_config_file("plotparams", config_file=args.plot_config)
			plot_config = cfg.load_params("plotparams",str(resolved_plot))
		else:
			resolved_plot = cfg.find_config_file("plotparams", config_dir=args.config_dir)
			plot_config = cfg.load_params("plotparams",str(resolved_plot))
	except Exception as e:
		logging.warning(f"Could not load plot config: {e}. Using defaults.")
		plot_config = {}

	args.freq_window = normalise_freq_window(args.freq_window, target='dspec')
	args.phase_window = normalise_phase_window(args.phase_window, target='dspec')


	window_pairs = None
	if args.compare_windows:
		pairs = []
		for spec in args.compare_windows:
			if ':' not in spec:
				parser.error(f"Invalid --compare-windows format: '{spec}'. Expected 'freq:phase'.")
			freq, phase = spec.split(':', 1)
			pairs.append((freq.strip(), phase.strip()))
		window_pairs = pairs if pairs else None


	if args.plot[0] not in plot_modes and args.plot[0] not in ("all", "None"):
			parser.error(f"Invalid plot mode: {args.plot[0]}")


	global data_directory
	data_directory = args.output_dir

	save_plots = plot_config.get('general', {}).get('save_plots', False)

	if args.write or save_plots:
		os.makedirs(args.output_dir, exist_ok=True)
		logging.info(f"Output directory: '{data_directory}' \n")
  

	param_overrides = {}
	param_std_overrides = {}
	if args.override_param:
		for override in args.override_param:
			if "=" not in override:
				parser.error(f"Invalid override format: '{override}'. Expected 'param=value'.")
			key, value = override.split("=", 1)
			key = key.strip()
			try:
				val = float(value)
			except ValueError:
				parser.error(f"Invalid value for override '{key}': '{value}' (must be numeric).")
			if key.startswith("sd_"):
				param_std_overrides[key] = val
			elif key.endswith("_sd"):
				base_key = key.rsplit("_", 1)[0]
				sd_key = f"sd_{base_key}"
				param_std_overrides[sd_key] = val
			else:
				param_overrides[key] = val
		logging.info(f"Parameter mean overrides: {param_overrides}")
		if param_std_overrides:
			logging.info(f"Parameter std dev overrides: {param_std_overrides}")
	all_param_overrides = {**param_overrides, **param_std_overrides}


	if args.override_plot:
		for override in args.override_plot:
			if "=" not in override:
				parser.error(f"Invalid plot override format: '{override}'. Expected 'param=value'.")
			key, value = override.split("=", 1)
			key = key.strip()
			
			try:
				# Handle lists like [10,8]
				if value.startswith('[') and value.endswith(']'):
					import ast
					val = ast.literal_eval(value)  # Safer than eval
				elif value.lower() in ('true', 'false'):
					val = value.lower() == 'true'
				elif value.lower() in ('null', 'none'):
					val = None
				else:
					try:
						val = float(value)
						# Convert to int if it's a whole number
						if val.is_integer():
							val = int(val)
					except ValueError:
						val = value  # Keep as string
			except Exception:
				parser.error(f"Invalid value for plot override '{key}': '{value}'")
			
			# Set the override in the config
			if '.' in key:
				# Handle nested keys like "styling.font_size"
				sections = key.split('.')
				current = plot_config
				for section in sections[:-1]:
					if section not in current:
						current[section] = {}
					current = current[section]
				current[sections[-1]] = val
			else:
				# Handle top-level or assume 'general' section
				if 'general' not in plot_config:
					plot_config['general'] = {}
				plot_config['general'][key] = val

	from fires.plotting.plotmodes import configure_matplotlib_from_config
	configure_matplotlib_from_config(plot_config)


	selected_plot_mode = plot_modes[args.plot[0]] if args.plot[0] in plot_modes else plot_modes['lvpa']
	try:
		if selected_plot_mode.requires_multiple_frb:
			if args.sim_data is None:
				logging.info(f"Processing with {args.ncpu} threads. \n")
   
			frb_dict = generate_frb(
				data            = args.sim_data,
				frb_id          = args.frb_identifier,
				sim_file        = resolved_sim,
				gauss_file      = resolved_gauss,
				scint_file      = resolved_scint,
				out_dir         = args.output_dir,
				write           = args.write,
				mode            = args.mode,
				seed            = args.seed,
				nseed           = args.nseed,
				sefd            = args.sefd,
				n_cpus          = args.ncpu,
				plot_mode       = selected_plot_mode,
				phase_window    = args.phase_window,
				freq_window     = args.freq_window,
				buffer_frac     = args.buffer,
				sweep_mode      = args.sweep_mode,
				target_snr      = args.snr,
				obs_data        = None,
				obs_params      = None,
				param_overrides = all_param_overrides,
				logstep           = args.logstep
				)
		else:
			FRB, noisespec, gdict = generate_frb(
				data            = args.sim_data,
				frb_id          = args.frb_identifier,
				sim_file        = resolved_sim,
				gauss_file      = resolved_gauss,
				scint_file      = resolved_scint,
				out_dir         = args.output_dir,
				write           = args.write,
				mode            = args.mode,
				seed            = args.seed,
				nseed           = None,
				sefd            = args.sefd,
				n_cpus          = None,
				plot_mode       = selected_plot_mode,
				phase_window    = args.phase_window,
				freq_window     = args.freq_window,
				buffer_frac     = args.buffer,
				sweep_mode      = None,
				target_snr      = args.snr,
				obs_data        = args.obs_data,
				obs_params      = args.obs_params,
				param_overrides = all_param_overrides
			)
			if args.chi2_fit:
				logging.info("Performing chi-squared fitting on the final profiles... \n")
				x_data = FRB.time_ms_array 
				y_data = FRB.dynamic_spectrum[0].mean(axis=0)  
				y_err = noisespec[0].mean(axis=0)  

				initial_guess = [np.max(y_data), np.mean(x_data), np.std(x_data)]  
				popt, chi2 = chi2_fit(x_data, y_data, y_err, gaussian_model, initial_guess)

				if popt is not None:
					print(f"Best-fit parameters: {popt}")
					print(f"Chi-squared value: {chi2} \n")
				else:
					print("Chi-squared fitting failed. \n")

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
						"gauss_file"       : resolved_gauss,
						"sim_file"         : resolved_sim,
						"plot_config"      : plot_config,
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