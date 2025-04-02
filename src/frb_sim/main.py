import argparse
import os
import traceback
from .functions.genfrb import generate_frb, obs_params_path, gauss_params_path
from .functions.processfrb import plots

def main():
    """
    Main entry point for the FRB simulation package.
    """
    # Parse command-line arguments
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
        "--no-write",
        action="store_true",
        help="If set, the simulation will not be saved to disk and will return the data instead."
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const=all,
        choices=['all', 'iquv', 'lvpa', 'dpa', 'rm'],
        metavar="PLOT_NAME",
        help="Generate plots. Pass 'all' to generate all plots, or specify a plot name: 'iquv', 'lvpa', 'dpa', 'rm'."
    )
    parser.add_argument(
        "--save-plots",
        type=bool,
        default=False,
        metavar="",
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
    parser.add_argument(
        "--rm",
        type=int,
        default=-1,
        metavar="",
        help="Gaussian component to use for RM correction. Default is -1 (last component)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default='gauss',
        choices=['gauss', 'sgauss'],
        metavar="",
        help="Mode for generating pulses: 'gauss' or 'sgauss'. Default is 'gauss.' 'sgauss' will generate a gaussian distribution of gaussian sub-pulses."
    )
    parser.add_argument(
        "--n-gauss",
        type=int,
        metavar="",
        help="Number of Gaussians to generate. Required if --mode is 'sgauss'."
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
        default=[0,50],
        metavar=("MIN_WIDTH", "MAX_WIDTH"),
        help="Minimum and maximum percentage of the main gaussian width to generate micro-gaussians with if --mode is 'sgauss.'"
    )


    args = parser.parse_args()

    # Set the global data directory variable
    global data_directory
    data_directory = args.output_dir

    # Check if the output directory exists, if not create it
    if not args.no_write and not os.path.exists(data_directory):
        os.makedirs(args.output_dir)
        print(f"Output directory '{data_directory}' created.")




    # Call the generate_frb function
    try:
        FRB, rm = generate_frb(scattering_timescale_ms=args.scattering_timescale_ms, frb_identifier=args.frb_identifier, obs_params=obs_params_path, 
                     gauss_params=gauss_params_path, data_dir=args.output_dir, write=not args.no_write, mode=args.mode, num_micro_gauss=args.n_gauss, 
                     seed=args.seed, width_range=args.sg_width
                     )
        if args.no_write:
            print("Simulation completed. Data returned instead of being saved.")
            if args.plot:
                # Call the plotting function with the specified arguments
                plots(fname=args.frb_identifier, FRB_data=FRB, mode=args.plot, scattering_timescale_ms=args.scattering_timescale_ms, 
                      startms=args.tz[0], stopms=args.tz[1], startchan=args.fz[0], endchan=args.fz[1], rm=rm[-1], outdir=data_directory,
                      save=args.save_plots)
        else:
            print(f"Simulation completed. Data saved to {args.output_dir}")
            if args.plot:
                # Call the plotting function with the specified arguments
                plots(fname=args.frb_identifier, FRB_data=FRB, mode=args.plot, scattering_timescale_ms=args.scattering_timescale_ms, 
                      startms=args.tz[0], stopms=args.tz[1], startchan=args.fz[0], endchan=args.fz[1], rm=rm[-1], outdir=data_directory,
                      save=args.save_plots)

    except Exception as e:
        print(f"An error occurred during the simulation: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()