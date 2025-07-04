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

#	--------------------------	Import modules	---------------------------
import numpy as np
import argparse
import os
import sys
import traceback
from inspect import signature

from FIRES.functions.genfrb import generate_frb, obs_params_path, gauss_params_path
from FIRES.utils.utils import chi2_fit, gaussian_model, window_map
from FIRES.functions.plotmodes import plot_modes


def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
	if len(sys.argv) == 1:
		print("FIRES: The Fast, Intense Radio Emission Simulator \n")
		print("For help, run:\n  FIRES --help\n")
		return

	parser = argparse.ArgumentParser(description="FIRES: The Fast, Intense Radio Emission Simulator. Simulate Fast Radio Bursts (FRBs) with scattering and polarization effects",
								  formatter_class=argparse.RawTextHelpFormatter)

	# Input Parameters
	parser.add_argument(
		"-t", "--tau_ms",
		type=str,
		nargs="+",  # Allow multiple values
		default=[0.0],
		metavar="",
		help=("Scattering time scale(s) in milliseconds.\n"
			  "Provide one or more values. Use '(start,stop,step)' for ranges. Default is 0.0 ms."
		   )
	)
	parser.add_argument(
		"-f", "--frb_identifier",
		type=str,
		default="FRB",
		metavar="",
		help="Identifier for the simulated FRB."
	)
	parser.add_argument(
		"-o", "--obs_params",
		type=str,
		default=obs_params_path, 
		metavar="",
		help="Observation parameters for the simulated FRB."
	)
	parser.add_argument(
		"-g", "--gauss_params",
		type=str,
		default=gauss_params_path,  
		metavar="",
		help="Gaussian parameters for the simulated FRB."
	)

	# Output Options
	parser.add_argument(
		"-d", "--output-dir",
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
	# Plotting Options
	parser.add_argument(
		"-p", "--plot",
		nargs="+",
		default=['lvpa'],
		choices=['all', 'None', 'iquv', 'lvpa', 'dpa', 'RM', 'pa_var', 'lfrac'],
		metavar="",
		help=(
			"Generate plots. Pass 'all' to generate all plots, or specify one or more plot names separated by spaces:\n"
			"  'iquv': Plot the Stokes parameters (I, Q, U, V) vs. time or frequency.\n"
			"  'lvpa': Plot linear polarization (L) and polarization angle (PA) vs. time.\n"
			"  'dpa': Plot the derivative of the polarization angle (dPA/dt) vs. time.\n"
			"  'RM': Plot the rotation measure (RM) vs. frequency from RM-Tools.\n"
			"  'pa_var': Plot the variance of the polarization angle (PA) vs. scattering timescale or microshot variation.\n"
			"  'lfrac': Plot the fraction of linear polarization (L/I) vs. scattering timescale or microshot variation.\n"
			"Pass 'None' to disable all plots."
		)
	)
	parser.add_argument(
		"-s", "--save-plots",
		action="store_true",
		help="Save plots to disk."
	)
	parser.add_argument(
		"--show-plots",
		type=str2bool,
		default=True,
		help="Display plots. Default is True. Set to False to disable plot display."
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
		help="Scale for plots. Choose 'linear', 'logx', 'logy' or 'loglog'. Default is 'linear'."
	)
	parser.add_argument(
		"--fit",
		nargs="+",
		default=None,
		metavar="",
		help=("Fit function for pa_var and lfrac plots.\n"
  			 "Options: 'exp', 'power', 'log', 'linear', 'constant', 'broken-power' or 'power,N', 'poly,N' for power/polynomial of degree N."
	  		)
	)
	# Simulation Options
	parser.add_argument(
		"--data",
		type=str,
		default=None,
		metavar="",
		help="Path to the data file. If provided, the simulation will use this data instead of generating new data."
	)
	parser.add_argument(
		"-m", "--mode",
		type=str,
		default='gauss',
		choices=['gauss', 'mgauss'],
		metavar="",
		help=("Mode for generating pulses: 'gauss' or 'mgauss'. Default is 'gauss.'\n"
			  "'mgauss' will generate a gaussian distribution of gaussian micro-shots."
		   )
	)
	parser.add_argument(
		"--n-gauss",
		nargs="+",  # Expect one or more values
		type=int,
		metavar="",
		help="Number of micro-shots to generate for each main Gaussian. Required if --mode is 'mgauss'."
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=None,
		metavar="",
		help="Set seed for repeatability in mgauss mode."
	)
	parser.add_argument(
		"--nseed",
		type=int,
		default=1,
		metavar="",
		help="How many realisations to generate at each scattering timescale for mgauss mode."
	)
	parser.add_argument(
		"--mg-width",
		nargs=2,
		type=float,
		default=[10, 50],
		metavar=("MIN_WIDTH", "MAX_WIDTH"),
		help=("Minimum and maximum percentage of the main gaussian width to generate micro-gaussians.\n"
  			  "Only if --mode mgauss is set. Default is [10, 50] percent of the main gaussian width."
	 )
	)
	parser.add_argument(
		"--noise",
		action="store_true",
		help="Add radiometric noise to the simulated FRB."
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
		help="Enable chi-squared fitting on the final profiles (plot!=pa_var)."
	)


	args = parser.parse_args()

	# Map long freq-window names to abbreviated forms
	args.freq_window = window_map[args.freq_window]
	args.phase_window = window_map[args.phase_window]

	# Parse scattering timescale(s)
	
	scattering_timescales = np.array([])
	for value in args.tau_ms:
		if value.startswith("(") and value.endswith(")"):  # Check if it's a range
			try:
				start, stop, step = map(float, value.strip("()").split(","))
				range_values = np.arange(start, stop + step, step)  # Include the stop value
				scattering_timescales = np.concatenate((scattering_timescales, range_values))  # Append to array
			except ValueError:
				raise ValueError("Invalid range format for scattering timescales. Use '(start,stop,step)'.")
		else:
			scattering_timescales = np.append(scattering_timescales, float(value))  # Append single value

	args.tau_ms = scattering_timescales

	print(f"Scattering timescales: {args.tau_ms} ms \n")

	# Set the global data directory variable
	global data_directory
	data_directory = args.output_dir

	# Check if the output directory exists, if not create it
	if args.write or args.save_plots:
		os.makedirs(args.output_dir, exist_ok=True)
		print(f"Output directory: '{data_directory}' \n")
  
	if args.plot[0] not in plot_modes:
		raise ValueError(f"Invalid plot mode: {args.plot[0]}")
	selected_plot_mode = plot_modes[args.plot[0]]

	
	try:
		if selected_plot_mode.requires_multiple_frb:
			print(f"Processing with {args.ncpu} threads. \n")
   
			frb_dict = generate_frb(
				data         = args.data,
				tau_ms   	 = args.tau_ms,
				frb_id       = args.frb_identifier,
				obs_file     = obs_params_path,
				gauss_file   = gauss_params_path,
				out_dir      = args.output_dir,
				save         = args.write,
				mode         = args.mode,
				n_gauss      = args.n_gauss,
				seed         = args.seed,
				nseed        = args.nseed,
				width_range  = args.mg_width,
				noise          = args.noise,
				n_cpus       = args.ncpu,
				plot_mode    = selected_plot_mode,
				phase_window = args.phase_window,
				freq_window  = args.freq_window,

				)
		else:
			FRB, noisespec, gdict = generate_frb(
				data         = args.data,
				tau_ms   	 = args.tau_ms,
				frb_id       = args.frb_identifier,
				obs_file     = obs_params_path,
				gauss_file   = gauss_params_path,
				out_dir      = args.output_dir,
				save         = args.write,
				mode         = args.mode,
				n_gauss      = args.n_gauss,
				seed         = args.seed,
				nseed        = None,
				width_range  = args.mg_width,
				noise          = args.noise,
				n_cpus       = None,
				plot_mode    = selected_plot_mode,
				phase_window = None,
				freq_window  = None,
			)
			if args.chi2_fit:
				print("Performing chi-squared fitting on the final profiles... \n")
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
				try:
					plot_mode_obj = plot_modes.get(plot_mode)
					if plot_mode_obj is None:
						print(f"Error: Plot mode '{plot_mode}' is not defined in plotmodes.py. \n")
						continue
					
					plotting_args = {
						"fname"       : args.frb_identifier,
						"frb_data"    : FRB if 'FRB' in locals() else None,
						"mode"        : plot_mode,
						"gdict"       : gdict if 'gdict' in locals() else None,
						"frb_dict"    : frb_dict if 'frb_dict' in locals() else None,
						"out_dir"     : data_directory,
						"save"        : args.save_plots,
						"figsize"     : args.figsize,
						"tau_ms"  	  : args.tau_ms,
						"show_plots"  : args.show_plots,
						"scale"       : args.plot_scale,
						"phase_window": args.phase_window,
						"freq_window" : args.freq_window,
						"fit"         : args.fit,
						"extension"   : args.extension,
					}
		
					plot_function = plot_mode_obj.plot_func
					plot_func_params = signature(plot_function).parameters
					filtered_args = {key: value for key, value in plotting_args.items() if key in plot_func_params}
		
					# Call the plotting function with the filtered arguments
					plot_function(**filtered_args)
						
				except Exception as e:
					print(f"An error occurred while plotting '{plot_mode}': {e} \n")
					traceback.print_exc()
		else:
			print("No plots generated. \n")
	except Exception as e:
		print(f"An error occurred during the simulation: {e} \n")
		traceback.print_exc()



if __name__ == "__main__":
	main()