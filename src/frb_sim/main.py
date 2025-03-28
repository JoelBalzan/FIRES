import argparse
import os
from .genfrb import generate_frb

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
        default="../../simfrbs/",
        metavar="",
        help="Directory to save the simulated FRB data (default: '../../simfrbs/')."
    )
    parser.add_argument(
        "-o", "--obs_params",
        type=str,
        default="../utils/obsparams.txt", 
        metavar="",
        help="Observation parameters for the simulated FRB."
    )
    parser.add_argument(
        "-g", "--gauss_params",
        type=str,
        default="../utils/gparams.txt",  
        metavar="",
        help="Gaussian parameters for the simulated FRB."
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="If set, the simulation will not be saved to disk and will return the data instead."
    )

    args = parser.parse_args()

    # Ensure the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set the global data directory variable
    global data_directory
    data_directory = args.output_dir

    # Call the generate_frb function
    try:
        result = generate_frb(scattering_timescale_ms=args.scattering_timescale_ms, frb_identifier=args.frb_identifier, data_dir=args.output_dir, write=not args.no_write)
        if args.no_write:
            print("Simulation completed. Data returned instead of being saved.")
            print(result)
        else:
            print(f"Simulation completed. Data saved to {args.output_dir}")
    except Exception as e:
        print(f"An error occurred during the simulation: {e}")

if __name__ == "__main__":
    main()