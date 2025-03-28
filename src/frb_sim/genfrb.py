#
#	Simulating scattering 
#
#								AB, Sep 2024
#                                                               
#	--------------------------	Import modules	---------------------------
from importlib.resources import files

import matplotlib as mpl
import numpy as np

from ..functions.genfns import *
from ..utils.utils import *

import os

current_dir = os.path.dirname(__file__)
obs_params_path = os.path.join(current_dir, "../utils/obsparams.txt")
gauss_params_path = os.path.join(current_dir, "../utils/gparams.txt")


#	-------------------------	Execute steps	-------------------------------
def generate_frb(scattering_timescale_ms, frb_identifier, data_dir, write=True, obs_params=obs_params_path, gauss_params=gauss_params_path):
    """
    Generate a simulated FRB with a dispersed and scattered dynamic spectrum
    """

    # Load observation parameters from obsparams.txt
    obsparams = get_parameters(obs_params)

    start_frequency_mhz = float(obsparams['f0'])
    end_frequency_mhz   = float(obsparams['f1'])
    channel_width_mhz   = float(obsparams['f_res'])
    start_time_ms       = float(obsparams['t0'])
    end_time_ms         = float(obsparams['t1'])
    time_resolution_ms  = float(obsparams['t_res'])
    scattering_index    = float(obsparams['scattering_index'])

    central_frequency_mhz = (start_frequency_mhz + end_frequency_mhz) / 2.0  # Central frequency in MHz
    num_channels = int((end_frequency_mhz - start_frequency_mhz) / channel_width_mhz)  # Number of frequency channels
    time_window_ms = (end_time_ms - start_time_ms) / 2.0  # Time window in ms
    num_time_bins = int(2 * time_window_ms / time_resolution_ms)  # Number of time bins

    # Array of frequency channels
    frequency_mhz_array = np.arange(
        start_frequency_mhz,
        end_frequency_mhz+ channel_width_mhz,
        channel_width_mhz,
        dtype=float
    )

    # Array of time bins
    time_ms_array = np.arange(
        -time_window_ms,
        time_window_ms+ time_resolution_ms,
        time_resolution_ms,
        dtype=float
    )

    # Load Gaussian parameters from gparams.txt
    gaussian_params = np.loadtxt(gauss_params)
    t0              = gaussian_params[:, 0]  # Time of the first Gaussian component
    width           = gaussian_params[:, 1]  # Width of the Gaussian component
    peak_amp        = gaussian_params[:, 2]  # Peak amplitude of the Gaussian component
    spec_idx        = gaussian_params[:, 3]  # Spectral index of the Gaussian component
    dm              = gaussian_params[:, 4]  # Dispersion measure of the Gaussian component
    rm              = gaussian_params[:, 5]  # Rotation measure of the Gaussian component
    pol_angle       = gaussian_params[:, 6]  # Polarization angle of the Gaussian component
    lin_pol_frac    = gaussian_params[:, 7]  # Linear polarization fraction of the Gaussian component
    circ_pol_frac   = gaussian_params[:, 8]  # Circular polarization fraction of the Gaussian component
    delta_pol_angle = gaussian_params[:, 9]  # Change in polarization angle with time of the Gaussian component

    # Check if linear and circular polarization fractions sum to more than 1.0
    if (lin_pol_frac + circ_pol_frac).any() > 1.0:
        print("WARNING: Linear and circular polarization fractions sum to more than 1.0")

    if (t0.any()<(-time_window_ms)) or (t0.any()>time_window_ms):
        print("WARNING: Gaussian component(s) outside the time window")




    # Generate initial dispersed dynamic spectrum with Gaussian components
    initial_dynspec = gauss_dynspec(
        frequency_mhz_array,
        time_ms_array,
        channel_width_mhz,
        time_resolution_ms,
        spec_idx,
        peak_amp,
        width,
        t0,
        dm,
        pol_angle,
        lin_pol_frac,
        circ_pol_frac,
        delta_pol_angle,
        rm
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
        rm=np.max(rm)
    )

    # 'Pickle' the simulated FRB and save it to the disk
    simulated_frb_data = simulated_frb(
        frb_identifier,
        frequency_mhz_array,
        time_ms_array,
        scattering_timescale_ms,
        scattering_index,
        gaussian_params,
        scattered_dynspec
    )

    if write:
        # Create the data directory, keep all simulated FRBs
        output_filename = "{}{}_sc_{:.2f}.pkl".format(data_dir, frb_identifier, scattering_timescale_ms)
        with open(output_filename, 'wb') as frbfile:
            pkl.dump(simulated_frb_data, frbfile)
    else:
        return simulated_frb_data


























































