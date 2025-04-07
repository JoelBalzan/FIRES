import argparse
import os
import traceback
from .functions.genfrb import generate_frb, obs_params_path, gauss_params_path
from .functions.processfrb import plots

def main():

    parser = argparse.ArgumentParser(description="Simulate a Fast Radio Burst (FRB) with scattering.")
    parser.add_argument(
        "-t", "--scattering_timescale_ms",
        type=float,
        required=True,
        metavar="",
        help="Scattering time scale in milliseconds."
    )
    parser.add_argument(
        "-f", "--frb_identifier",
        type=str,
        default="FRB",
        metavar="",
        help="Identifier for the simulated FRB."
    )
    parser.add_argument(
        "-d", "--output-dir",
        type=str,
        default="simfrbs/",
        metavar="",
        help="Directory to save the simulated FRB data (default: 'simfrbs/')."
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
    parser.add_argument(
        "--write",
        action="store_true",
        help="If set, the simulation will be saved to disk. Default is False."
    )
    parser.add_argument(
        "-p", "--plot",
        nargs="?",
        const=all,
        default='lvpa',
        choices=['all', 'None', 'iquv', 'lvpa', 'dpa', 'rm'],
        metavar="PLOT_NAME",
        help="Generate plots. Pass 'all' to generate all plots, or specify a plot name: 'ds', 'iquv', 'lvpa', 'dpa', 'rm'."
    )
    parser.add_argument(
        "-s", "--save-plots",
        action="store_true",
        help="Save plots to disk. Default is False."
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
    #parser.add_argument(
    #    "--rm",
    #    type=int,
    #    default=-2,
    #    metavar="",
    #    help="Index RM gaussian component to use for RM correction. Default is -2 (RM of last gaussian in gparams.txt)."
    #)
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
        default=[10,50],
        metavar=("MIN_WIDTH", "MAX_WIDTH"),
        help="Minimum and maximum percentage of the main gaussian width to generate micro-gaussians with if --mode is 'sgauss.'"
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=2,
        metavar="",
        help="For setting noise scale in dynamic spectrum. This value is multiplied by the standard deviation of each frequency channel."
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
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[6,10],
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size for plots. Provide two values: width and height (in inches)."
    )

    
    args = parser.parse_args()

    # Set the global data directory variable
    global data_directory
    data_directory = args.output_dir

    # Check if the output directory exists, if not create it
    if args.write and not os.path.exists(data_directory):
        os.makedirs(args.output_dir)
        print(f"Output directory '{data_directory}' created.")



    # Call the generate_frb function
    try:
        FRB, rm = generate_frb(scattering_timescale_ms=args.scattering_timescale_ms, frb_identifier=args.frb_identifier, obs_params=obs_params_path, 
                     gauss_params=gauss_params_path, data_dir=args.output_dir, write=args.write, mode=args.mode, num_micro_gauss=args.n_gauss, 
                     seed=args.seed, width_range=args.sg_width, noise=args.noise, scatter=args.scatter
                     )
        if not args.write:
            print("Simulation completed. Data returned instead of being saved.")
            if args.plot:
                # Call the plotting function with the specified arguments
                plots(fname=args.frb_identifier, FRB_data=FRB, mode=args.plot, startms=args.tz[0], stopms=args.tz[1], 
                      startchan=args.fz[0], endchan=args.fz[1], rm=rm, outdir=data_directory, save=args.save_plots, figsize=args.figsize)
        else:
            print(f"Simulation completed. Data saved to {args.output_dir}")
            if args.plot:
                # Call the plotting function with the specified arguments
                plots(fname=args.frb_identifier, FRB_data=FRB, mode=args.plot, startms=args.tz[0], stopms=args.tz[1], 
                      startchan=args.fz[0], endchan=args.fz[1], rm=rm, outdir=data_directory, save=args.save_plots, figsize=args.figsize)

    except Exception as e:
        print(f"An error occurred during the simulation: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()