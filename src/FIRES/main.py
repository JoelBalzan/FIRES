import numpy as np
import argparse
import os
import traceback
from .functions.genfrb import generate_frb, obs_params_path, gauss_params_path
from .functions.processfrb import plots

def main():

	parser = argparse.ArgumentParser(description="Simulate a Fast Radio Burst (FRB) with scattering.")

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
		choices=['all', 'None', 'iquv', 'lvpa', 'dpa', 'rm', 'pa_rms'],
		metavar="PLOT_NAME",
		help="Generate plots. Pass 'all' to generate all plots, or specify one or more plot names: 'iquv', 'lvpa', 'dpa', 'rm', 'pa_rms'."
	)
	parser.add_argument(
		"-s", "--save-plots",
		action="store_true",
		help="Save plots to disk. Default is False."
	)
	parser.add_argument(
		"--show-plots",
		type=bool,
		default=True,
		help="Display plots. Default is True. Set to False to disable plot display."
	)
	parser.add_argument(
		"--tz",
		nargs=2,
		type=float,
		default=[0, 0],
		metavar=("START_TIME", "END_TIME"),
		help="Time zoom range for plots. Provide two values: start time and end time (in milliseconds)."
	)
	parser.add_argument(
		"--fz",
		nargs=2,
		type=float,
		default=[0, 0],
		metavar=("START_FREQ", "END_FREQ"),
		help="Frequency zoom range for plots. Provide two values: start frequency and end frequency (in MHz)."
	)
	parser.add_argument(
		"--figsize",
		type=float,
		nargs=2,
		default=[6, 10],
		metavar=("WIDTH", "HEIGHT"),
		help="Figure size for plots. Provide two values: width and height (in inches)."
	)

	# Simulation Options
	parser.add_argument(
		"-m", "--mode",
		type=str,
		default='gauss',
		choices=['gauss', 'sgauss'],
		metavar="",
		help="Mode for generating pulses: 'gauss' or 'sgauss'. Default is 'gauss.' 'sgauss' will generate a gaussian distribution of gaussian sub-pulses."
	)
	parser.add_argument(
		"--n-gauss",
		nargs="+",  # Expect one or more values
		type=int,
		metavar="",
		help="Number of sub-Gaussians to generate for each main Gaussian. Required if --mode is 'sgauss'."
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=None,
		metavar="",
		help="Set seed for repeatability in sgauss mode."
	)
	parser.add_argument(
		"--sg-width",
		nargs=2,
		type=float,
		default=[10, 50],
		metavar=("MIN_WIDTH", "MAX_WIDTH"),
		help="Minimum and maximum percentage of the main gaussian width to generate micro-gaussians with if --mode is 'sgauss.'"
	)
	parser.add_argument(
		"--noise",
		type=float,
		default=0,
		metavar="",
		help="For setting noise scale in dynamic spectrum. This value is multiplied by the standard deviation of each Stokes I frequency channel."
	)
	parser.add_argument(
		"--scatter",
		action="store_true",
		default=True,
		help="Enable scattering. Use --no-scatter to disable it."
	)
	parser.add_argument(
		"--no-scatter",
		action="store_false",
		dest="scatter",
		help="Disable scattering. Overrides --scatter if both are provided."
	)

	args = parser.parse_args()


	# Parse scattering timescale(s)
	if args.scatter:
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
	
		# If only one scattering time is passed, convert to a float
		if len(scattering_timescales) == 1:
			scattering_timescales = float(scattering_timescales[0])
	
		args.scattering_timescale_ms = scattering_timescales
	else:
		args.scattering_timescale_ms = False
	print(f"Scattering timescales: {args.scattering_timescale_ms}")

	# Check if multiple scattering timescales are provided
	if isinstance(args.scattering_timescale_ms, np.ndarray) and args.plot != ['pa_rms']:
		print("Multiple scattering timescales detected. Setting plot mode to 'pa_rms'")
		args.plot = ['pa_rms']


	# Set the global data directory variable
	global data_directory
	data_directory = args.output_dir

	# Check if the output directory exists, if not create it
	if args.write or args.save_plots and not os.path.exists(data_directory):
		os.makedirs(args.output_dir)
		print(f"Output directory '{data_directory}' created.")

	
	# Call the generate_frb function 
	try:
		# Generate the FRB or PA RMS data
		if args.plot == ['pa_rms']:
			pa_rms_values, pa_rms_errors = generate_frb(
				scattering_timescale_ms=args.scattering_timescale_ms,
				frb_identifier=args.frb_identifier,
				obs_params=obs_params_path,
				gauss_params=gauss_params_path,
				data_dir=args.output_dir,
				write=args.write,
				mode=args.mode,
				num_micro_gauss=args.n_gauss,
				seed=args.seed,
				width_range=args.sg_width,
				noise=args.noise,
				scatter=args.scatter,
				plot=args.plot,
				startms=args.tz[0],
				stopms=args.tz[1],
				startchan=args.fz[0],
				endchan=args.fz[1]
			)
		else:
			FRB, rm = generate_frb(
				scattering_timescale_ms=args.scattering_timescale_ms,
				frb_identifier=args.frb_identifier,
				obs_params=obs_params_path,
				gauss_params=gauss_params_path,
				data_dir=args.output_dir,
				write=args.write,
				mode=args.mode,
				num_micro_gauss=args.n_gauss,
				seed=args.seed,
				width_range=args.sg_width,
				noise=args.noise,
				scatter=args.scatter,
				plot=args.plot,
				startms=args.tz[0],
				stopms=args.tz[1],
				startchan=args.fz[0],
				endchan=args.fz[1]
			)

		# Print simulation status
		save_status = "Data saved to" if args.write else "Data not saved."
		print(f"Simulation completed for scattering timescale {args.scattering_timescale_ms} ms. {save_status}")

		# Call the plotting function if required
		if args.plot != 'None':
			for plot_mode in args.plot:
				if plot_mode == 'pa_rms':
					# Call the plotting function specifically for 'pa_rms'
					plots(
						fname=args.frb_identifier,
						FRB_data=None,  # No FRB data for 'pa_rms'
						pa_rms=pa_rms_values,
						dpa_rms=pa_rms_errors,
						mode=plot_mode,
						startms=args.tz[0],
						stopms=args.tz[1],
						startchan=args.fz[0],
						endchan=args.fz[1],
						rm=None,  # No RM for 'pa_rms'
						outdir=data_directory,
						save=args.save_plots,
						figsize=args.figsize,
						scattering_timescale=args.scattering_timescale_ms,
						show_plots=args.show_plots
					)
				else:
					# Ensure FRB_data is not None for other plot modes
					if FRB is None:
						print("Error: FRB data is not available for the selected plot mode.")
						continue
					
					# Call the plotting function for other modes
					plots(
						fname=args.frb_identifier,
						FRB_data=FRB,
						pa_rms=None,
						dpa_rms=None,
						mode=plot_mode,
						startms=args.tz[0],
						stopms=args.tz[1],
						startchan=args.fz[0],
						endchan=args.fz[1],
						rm=rm,
						outdir=data_directory,
						save=args.save_plots,
						figsize=args.figsize,
						scattering_timescale=args.scattering_timescale_ms,
						show_plots=args.show_plots
					)

	
	except Exception as e:
		print(f"An error occurred during the simulation: {e}")
		traceback.print_exc()

if __name__ == "__main__":
	main()