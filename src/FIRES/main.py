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
import traceback
from inspect import signature

from FIRES.functions.genfrb import generate_frb, obs_params_path, gauss_params_path
from FIRES.utils.utils import chi2_fit, gaussian_model
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

	parser = argparse.ArgumentParser(description="FIRES: The Fast, Intense Radio Emission Simulator. Simulate Fast Radio Bursts (FRBs) with scattering and polarization effects",
								  formatter_class=argparse.RawTextHelpFormatter)

	# Input Parameters
	parser.add_argument(
		"-t", "--scattering_timescale_ms",
		type=str,
		nargs="+",  # Allow multiple values
		default=[0.0],
		metavar="",
		help="Scattering time scale(s) in milliseconds. Provide one or more values. Use '(start,stop,step)' for ranges. Default is 0.0 ms."
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

	# Plotting Options
	parser.add_argument(
		"-p", "--plot",
		nargs="+",
		default=['lvpa'],
		choices=['all', 'None', 'iquv', 'lvpa', 'dpa', 'rm', 'pa_var', 'lfrac'],
		metavar="",
		help=(
			"Generate plots. Pass 'all' to generate all plots, or specify one or more plot names separated by spaces:\n"
			"  'iquv': Plot the Stokes parameters (I, Q, U, V) as a function of time or frequency.\n"
			"  'lvpa': Plot linear polarization (L) and polarization angle (PA) as a function of time.\n"
			"  'dpa': Plot the derivative of the polarization angle (dPA/dt) as a function of time.\n"
			"  'rm': Plot the rotation measure (RM) as a function of frequency from RM-Tools.\n"
			"  'pa_var': Plot the variance of the polarization angle (PA) as a function of scattering timescale.\n"
			"  'lfrac': Plot the fraction of linear polarization (L/I) as a function of time.\n"
			"Pass 'None' to disable all plots."
	)
)
	parser.add_argument(
		"-s", "--save-plots",
		action="store_true",
		help="Save plots to disk. Default is False."
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
		default=[6, 10],
		metavar=("WIDTH", "HEIGHT"),
		help="Figure size for plots. Provide two values: width and height (in inches)."
	)
	parser.add_argument(
		"--plot-scale",
		type=str,
		default="linear",
		choices=['linear', 'log', 'loglog'],
		metavar="",
		help="Scale for plots. Choose 'linear' or 'log' (for y-axis) or loglog. Default is 'linear'."
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
		help="Mode for generating pulses: 'gauss' or 'mgauss'. Default is 'gauss.' 'mgauss' will generate a gaussian distribution of gaussian micro-shots."
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
		help="Minimum and maximum percentage of the main gaussian width to generate micro-gaussians with if --mode is 'mgauss.'"
	)
	parser.add_argument(
		"--SNR",
		type=float,
		default=0,
		metavar="",
		help="Signal-to-noise ratio (SNR) for the simulated FRB. Default is 0 (no noise)."
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
	parser.add_argument(
		"--phase-window",
		type=str,
		default="all",
		choices=['first', 'last', 'all'],
		metavar="",
		help="Window for plotting PA variance and L fraction. Choose 'first', 'last', or 'all'. Default is 'all'."
	)
	parser.add_argument(
		"--freq-window",
		type=str,
		default="all",
		choices=['1q', '2q', '3q', '4q', 'all'],
		metavar="",
		help="Frequency window for plotting PA variance and L fraction. Choose '1q', '2q', '3q', '4q', or 'all'. Default is 'all'."
	)

	args = parser.parse_args()


	# Parse scattering timescale(s)
	scattering_timescales = np.array([])
	for value in args.scattering_timescale_ms:
		if value.startswith("(") and value.endswith(")"):  # Check if it's a range
			try:
				start, stop, step = map(float, value.strip("()").split(","))
				range_values = np.arange(start, stop + step, step)  # Include the stop value
				scattering_timescales = np.concatenate((scattering_timescales, range_values))  # Append to array
			except ValueError:
				raise ValueError("Invalid range format for scattering timescales. Use '(start,stop,step)'.")
		else:
			scattering_timescales = np.append(scattering_timescales, float(value))  # Append single value

	args.scattering_timescale_ms = scattering_timescales

	print(f"Scattering timescales: {args.scattering_timescale_ms} ms \n")

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
		if selected_plot_mode.requires_multiple_tau:
			print(f"Processing with {args.ncpu} threads. \n")
   
			values, errors, width_ms, var_PA_microshots = generate_frb(
				data=args.data,
				scatter_ms=args.scattering_timescale_ms,
				frb_id=args.frb_identifier,
				obs_file=obs_params_path,
				gauss_file=gauss_params_path,
				out_dir=args.output_dir,
				save=args.write,
				mode=args.mode,
				n_gauss=args.n_gauss,
				seed=args.seed,
				nseed=args.nseed,
				width_range=args.mg_width,
				noise=args.SNR,
				n_cpus=args.ncpu,
				plot_mode=selected_plot_mode,
				phase_window=args.phase_window,
				freq_window=args.freq_window,

				)
		else:
			FRB, noisespec, rm = generate_frb(
				data=args.data,
				scatter_ms=args.scattering_timescale_ms,
				frb_id=args.frb_identifier,
				obs_file=obs_params_path,
				gauss_file=gauss_params_path,
				out_dir=args.output_dir,
				save=args.write,
				mode=args.mode,
				n_gauss=args.n_gauss,
				seed=args.seed,
				nseed=None,
				width_range=args.mg_width,
				noise=args.SNR,
				n_cpus=None,
				plot_mode=selected_plot_mode,
				phase_window=None,
				freq_window=None,
			)
			if args.chi2_fit:
				if args.noise == 0:
					print("No noise added to the dynamic spectrum. Skipping chi-squared fitting. \n")
				else:
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
		if args.plot != 'None':
			for plot_mode in args.plot:
				try:
					plot_mode_obj = plot_modes.get(plot_mode)
					if plot_mode_obj is None:
						print(f"Error: Plot mode '{plot_mode}' is not defined in plotmodes.py. \n")
						continue
					
					plotting_args = {
						"fname": args.frb_identifier,
						"frb_data": FRB if 'FRB' in locals() else None,
						"mode": plot_mode,
						"rm": rm if 'rm' in locals() else None,
						"vals": values if 'values' in locals() else None,
						"errs": errors if 'errors' in locals() else None,
						"width_ms": width_ms if 'width_ms' in locals() else None,
						"out_dir": data_directory,
						"save": args.save_plots,
						"figsize": args.figsize,
						"scatter_ms": args.scattering_timescale_ms,
						"show_plots": args.show_plots,
						"var_PA_microshots": var_PA_microshots if 'var_PA_microshots' in locals() else None,
						"scale": args.plot_scale,
					}
		
					plot_function = plot_mode_obj.plot_func
					plot_func_params = signature(plot_function).parameters
					filtered_args = {key: value for key, value in plotting_args.items() if key in plot_func_params}
		
					# Call the plotting function with the filtered arguments
					plot_function(**filtered_args)
						
				except Exception as e:
					print(f"An error occurred while plotting '{plot_mode}': {e} \n")
					traceback.print_exc()

	except Exception as e:
		print(f"An error occurred during the simulation: {e} \n")
		traceback.print_exc()



if __name__ == "__main__":
	main()