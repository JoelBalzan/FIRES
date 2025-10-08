# -----------------------------------------------------------------------------
# main.py
# FIRES: The Fast, Intense Radio Emission Simulator
#
# This script serves as the main entry point for simulating Fast Radio Bursts (FRBs)
# with scattering and polarization effects. It parses command-line arguments,
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
from inspect import signature

import numpy as np

from fires.core.genfrb import generate_frb
from fires.plotting.plotmodes import configure_matplotlib, plot_modes
from fires.utils import config as cfg
from fires.utils.utils import (LOG, chi2_fit, gaussian_model, init_logging,
                               window_map)


def main():
	parser = argparse.ArgumentParser(description="FIRES: The Fast, Intense Radio Emission Simulator. Simulate Fast Radio Bursts (FRBs) with scattering and polarisation effects",
								  formatter_class=argparse.RawTextHelpFormatter)

	# Show help when no args are provided (no traceback)
	if len(sys.argv) == 1:
		parser.print_help()
		return 0


	# Input Parameters
	parser.add_argument(
		"-f", "--frb_identifier",
		type=str,
		default="FRB",
		metavar="",
		help="Identifier for the simulated FRB."
	)
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
		choices=["gparams", "obsparams", "scparams"], 
		help="Open config in $EDITOR"
	)


	# Output Options
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
		help="If set, the simulation will be saved to disk. Default is False."
	)
	parser.add_argument(
		"--phase-window",
		type=str,
		default="all",
		choices=['first', 'last', 'all',
	  			'leading', 'trailing', 'total'
		],
		metavar="",
		help=("Window for plotting PA variance and L fraction.\n"
  			 "Choose 'leading', 'trailing', or 'total'. Default is 'total'."
	  )
	)
	parser.add_argument(
		"--freq-window",
		type=str,
		default="full",
		choices=[
			'1q', '2q', '3q', '4q', 'full',  # abbreviated
			'lowest-quarter', 'lower-mid-quarter', 'upper-mid-quarter', 'highest-quarter', 'full-band'  # long
   		],
		metavar="",
		help=("Frequency window for plotting PA variance and L/I.\n"
  			  "Choose 'lowest-quarter', 'lower-mid-quarter', 'upper-mid-quarter', 'highest-quarter', or 'full-band'. Default is 'full-band'."
	  )
	)
	parser.add_argument(
		"--buffer",
		type=float,
		default=0.1,
		metavar="",
		help="Buffer time in between on- and off-pulse regions as a fraction of the pulse width for noise estimation. Default is 0.1."
	)
	parser.add_argument(
		"-v", "--verbose",
		action="store_true",
		help="Enable verbose output."
	)

	# Plotting Options
	parser.add_argument(
		"-p", "--plot",
		nargs="+",
		default=['lvpa'],
		choices=['all', 'None', 'iquv', 'lvpa', 'dpa', 'RM', 'pa_var', 'l_frac'],
		metavar="",
		help=(
			"Generate plots. Pass 'all' to generate all (non-pa_var and l_frac) plots, or specify one or more plot names separated by spaces:\n"
			"  'iquv': Plot the Stokes parameters (I, Q, U, V) vs. time or frequency.\n"
			"  'lvpa': Plot linear polarization (L) and polarization angle (PA) vs. time.\n"
			"  'dpa': Plot the derivative of the polarization angle (dPA/dt) vs. time.\n"
			"  'RM': Plot the rotation measure (RM) vs. frequency from RM-Tools.\n"
			"  'pa_var': Plot the variance of the polarization angle (PA) vs. swept parameter in gparams.\n"
			"  'l_frac': Plot the fraction of linear polarization (L/I)/(L/I)_0 vs. swept parameter in gparams.\n"
			"Pass 'None' to disable all plots."
		)
	)
	parser.add_argument(
		"-s", "--save-plots",
		action="store_true",
		help="Save plots to disk."
	)
	parser.add_argument(
		"--disable-plots",
		action="store_false",
		dest="show_plots",
		help="Disable plot display. "
	)
	parser.add_argument(
		"--figsize",
		type=float,
		nargs=2,
		default=None,
		metavar=("WIDTH", "HEIGHT"),
		help="Figure size for plots. Provide two values: width and height (in inches)."
	)
	parser.add_argument(
		"-e", "--extension",
		type=str,
		default="pdf",
		metavar="",
		help="File extension for saved plots. Default is 'pdf'."
	)
	parser.add_argument(
		"--plot-scale",
		type=str,
		default="linear",
		choices=['linear', 'logx', 'logy', 'loglog'],
		metavar="",
		help="Scale for pa_var and l_frac plots. Choose 'linear', 'logx', 'logy' or 'loglog'. Default is 'linear'."
	)
	parser.add_argument(
		"--fit",
		nargs="+",
		default=None,
		metavar="",
		help=("Fit function for pa_var and l_frac plots.\n"
  			 "Options: 'exp', 'power', 'log', 'linear', 'constant', 'broken-power' or 'power,N', 'poly,N' for power/polynomial of degree N."
	  		)
	)
	parser.add_argument(
		"--no-legend",
		action="store_false",
		help="Disable legends in plots."
	)
	parser.add_argument(
		"--no-info",
		action="store_false",
		help="Disable info text in plots."
	)
	parser.add_argument(
		"--show-onpulse",
		action="store_true",
		help="Show on-pulse region in plots."
	)
	parser.add_argument(
		"--show-offpulse",
		action="store_true",
		help="Show off-pulse region in plots."
	)
	parser.add_argument(
		"--use-latex",
		action="store_true",
		help="Use LaTeX for plot text."
	)
	parser.add_argument(
		"--sweep-mode",
		type=str,
		default="none",
		choices=["none", "mean", "variance"],
		metavar="",
		help=("Parameter sweep mode for pa_var and l_frac plots:\n"
			  "  none      : disable sweeping (use means + micro std dev only)\n"
			  "  mean      : sweep the mean value (std dev forced to 0 for that param)\n"
			  "  variance  : keep mean fixed, sweep the micro std dev\n")
	)

	# Simulation Options
	parser.add_argument(
		"-d", "--data",
		type=str,
		default=None,
		metavar="",
		help="Path to the data file. If provided, the simulation will use this data instead of generating new data."
	)
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
		help="How many realisations to generate for pa_var and l_frac plots."
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
		"--ncpu",
		type=int,
		default=1,
		metavar="",
		help="Number of CPUs to use for parallel processing. Default is 1 (single-threaded)."
	)
	parser.add_argument(
		"--chi2-fit",
		action="store_true",
		help="Enable chi-squared fitting on the final profiles (plot != pa_var and l_frac)."
	)
	parser.add_argument(
		"--scint",
		action="store_true",
		help="Enable scintillation effects."
	)

	args = parser.parse_args()

	# Initialize logging 
	init_logging(args.verbose)
	if args.verbose:
		LOG.debug("Verbose logging enabled.")

	# Handle config management
	if args.init_config:
		# Overwrite user config with packaged defaults, creating timestamped backups
		cfg.init_user_config(overwrite=True, backup=True)
		print(f"Config files synced to: {cfg.user_config_dir()}\n")
		return 0
	if args.edit_config:
		cfg.edit_params(args.edit_config, config_dir=args.config_dir)
		return

	# Resolve parameter file paths 
	resolved_obs   = str(cfg.find_config_file("obsparams", config_dir=args.config_dir))
	resolved_gauss = str(cfg.find_config_file("gparams",   config_dir=args.config_dir))
	if args.scint:
		resolved_scint = str(cfg.find_config_file("scparams", config_dir=args.config_dir))
	else:
		resolved_scint = None


	# Map long freq-window names to abbreviated forms
	if args.freq_window in window_map:
		args.freq_window = window_map[args.freq_window]
	if args.phase_window in window_map:
		args.phase_window = window_map[args.phase_window]

	if args.plot[0] not in plot_modes and args.plot[0] not in ("all", "None"):
			parser.error(f"Invalid plot mode: {args.plot[0]}")

	# Set the global data directory variable
	global data_directory
	data_directory = args.output_dir

	# Check if the output directory exists, if not create it
	if args.write or args.save_plots:
		os.makedirs(args.output_dir, exist_ok=True)
		print(f"Output directory: '{data_directory}' \n")
  

	selected_plot_mode = plot_modes[args.plot[0]] if args.plot[0] in plot_modes else plot_modes['lvpa']

	try:
		if selected_plot_mode.requires_multiple_frb:
			if args.data is None:
				logging.info(f"Processing with {args.ncpu} threads. \n")
   
			frb_dict = generate_frb(
				data         = args.data,
				frb_id       = args.frb_identifier,
				obs_file     = resolved_obs,
				gauss_file   = resolved_gauss,
				scint_file   = resolved_scint,
				out_dir      = args.output_dir,
				write        = args.write,
				mode         = args.mode,
				seed         = args.seed,
				nseed        = args.nseed,
				sefd         = args.sefd,
				n_cpus       = args.ncpu,
				plot_mode    = selected_plot_mode,
				phase_window = args.phase_window,
				freq_window  = args.freq_window,
				buffer_frac  = args.buffer,
				sweep_mode   = args.sweep_mode,
				target_snr   = args.snr
				)
		else:
			FRB, noisespec, gdict = generate_frb(
				data         = args.data,
				frb_id       = args.frb_identifier,
				obs_file     = resolved_obs,
				gauss_file   = resolved_gauss,
				scint_file   = resolved_scint,
				out_dir      = args.output_dir,
				write        = args.write,
				mode         = args.mode,
				seed         = args.seed,
				nseed        = None,
				sefd         = args.sefd,
				n_cpus       = None,
				plot_mode    = selected_plot_mode,
				phase_window = None,
				freq_window  = None,
				buffer_frac  = args.buffer,
				sweep_mode   = None,
				target_snr   = args.snr
			)
			if args.chi2_fit:
				logging.info("Performing chi-squared fitting on the final profiles... \n")
				# Fit a Gaussian to the Stokes I profile
				x_data = FRB.time_ms_array  # Replace with the appropriate x-axis data
				y_data = FRB.dynamic_spectrum[0].mean(axis=0)  # Mean Stokes I profile
				y_err = noisespec[0].mean(axis=0)  

				initial_guess = [np.max(y_data), np.mean(x_data), np.std(x_data)]  # Initial guess for Gaussian parameters
				popt, chi2 = chi2_fit(x_data, y_data, y_err, gaussian_model, initial_guess)

				if popt is not None:
					print(f"Best-fit parameters: {popt}")
					print(f"Chi-squared value: {chi2} \n")
				else:
					print("Chi-squared fitting failed. \n")

		# Print simulation status
		print(f"Simulation completed. \n")

		# Call the plotting function if required
		if args.plot != 'None' and (args.save_plots == True or args.show_plots == True):
			for plot_mode in args.plot:
				configure_matplotlib(use_latex=args.use_latex)
				try:
					plot_mode_obj = plot_modes.get(plot_mode)
					if plot_mode_obj is None:
						print(f"Error: Plot mode '{plot_mode}' is not defined in plotmodes.py. \n")
						continue
					plotting_args = {
						"fname"        : args.frb_identifier,
						"frb_data"     : FRB if 'FRB' in locals() else None,
						"mode"         : plot_mode,
						"gdict"        : gdict if 'gdict' in locals() else None,
						"frb_dict"     : frb_dict if 'frb_dict' in locals() else None,
						"out_dir"      : data_directory,
						"save"         : args.save_plots,
						"figsize"      : args.figsize,
						"show_plots"   : args.show_plots,
						"scale"        : args.plot_scale,
						"phase_window" : args.phase_window,
						"freq_window"  : args.freq_window,
						"fit"          : args.fit,
						"extension"    : args.extension,
						"legend"       : args.no_legend,
						"info"         : args.no_info,
						"buffer_frac"  : args.buffer,
						"show_onpulse" : args.show_onpulse,
						"show_offpulse": args.show_offpulse,
						"use_latex"    : args.use_latex
					}
		
					plot_function = plot_mode_obj.plot_func
					plot_func_params = signature(plot_function).parameters
					filtered_args = {key: value for key, value in plotting_args.items() if key in plot_func_params}
		
					# Call the plotting function with the filtered arguments
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