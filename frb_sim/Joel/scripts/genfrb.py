#
#	Simulating scattering 
#
#								AB, Sep 2024
#                                                               
#	--------------------------	Import modules	---------------------------

import os
import sys

import matplotlib as mpl
import numpy as np
from genfns import *
from utils import *


def print_instructions():
    """
    Print instructions to terminal
    """
    print("\n            You probably need some assistance here!\n")
    print("Arguments are       --- <scattering time scale> <name>\n")	
    print("\n            Now let's try again!\n")
    
    return 0

#	--------------------------	Read inputs	-------------------------------

if len(sys.argv) < 3:
    print_instructions()
    sys.exit()

scattering_timescale_ms = float(sys.argv[1])  # Scattering time scale (msec)
frb_identifier = sys.argv[2]  # FRB identifier

#	-------------------------	Execute steps	-------------------------------

# Array of frequency channels
frequency_mhz_array = np.arange(
    central_frequency_mhz - (num_channels * channel_width_mhz) / 2.0,
    central_frequency_mhz + (num_channels * channel_width_mhz) / 2.0,
    channel_width_mhz,
    dtype=float
)

# Array of time bins
time_ms_array = np.arange(
    -time_window_ms,
    time_window_ms,
    time_resolution_ms,
    dtype=float
)

# Load Gaussian parameters from gparams.txt
gaussian_params = np.loadtxt('gparams.txt')

# Generate initial dispersed dynamic spectrum with Gaussian components
initial_dynspec,polarisation_angles = gauss_dynspec(
    frequency_mhz_array,
    time_ms_array,
    channel_width_mhz,
    time_resolution_ms,
    gaussian_params[:, 3],
    gaussian_params[:, 2],
    gaussian_params[:, 1],
    gaussian_params[:, 0],
    gaussian_params[:, 4],
    gaussian_params[:, 6],
    gaussian_params[:, 7],
    gaussian_params[:, 8],
    gaussian_params[:, 9],
    gaussian_params[:, 5],
    time_per_bin_ms
)

# Scatter the dynamic spectrum
scattered_dynspec = scatter_dynspec(
    initial_dynspec,
    frequency_mhz_array,
    time_ms_array,
    channel_width_mhz,
    time_resolution_ms,
    scattering_timescale_ms,
    scattering_index,
    polangles=polarisation_angles
)

# Apply scintillation (if needed)
# scintillated_dynspec = apply_scintillation(scattered_dynspec)

# 'Pickle' the simulated FRB and save it to the disk
simulated_frb_data = simulated_frb(
    frb_identifier,
    frequency_mhz_array,
    time_ms_array,
    scattering_timescale_ms,
    reference_frequency_mhz,
    scattering_index,
    gaussian_params,
    scattered_dynspec
)

# Create the data directory, keep all simulated FRBs
output_filename = "{}{}_sc_{:.2f}.pkl".format(data_directory, frb_identifier, scattering_timescale_ms)
with open(output_filename, 'wb') as frbfile:
    pkl.dump(simulated_frb_data, frbfile)


































































