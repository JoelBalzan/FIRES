#
#	Simulating scattering 
#
#								AB, Sep 2024
#                                                               
#	--------------------------	Import modules	---------------------------
from importlib.resources import files

import matplotlib as mpl
import numpy as np

from FIRES.functions.genfns import *
from FIRES.utils.utils import *

import os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
obs_params_path = os.path.join(parent_dir, "utils/obsparams.txt")
gauss_params_path = os.path.join(parent_dir, "utils/gparams.txt")


#	-------------------------	Execute steps	-------------------------------
def generate_frb(scattering_timescale_ms, frb_identifier, data_dir, mode, num_micro_gauss, seed, width_range, write, 
                 obs_params, gauss_params, noise, scatter, plot, startms, stopms, startchan, endchan):
    """
    Generate a simulated FRB with a dispersed and scattered dynamic spectrum
    """
    
    obsparams = get_parameters(obs_params)
    # Extract frequency and time parameters from observation parameters
    start_frequency_mhz = float(obsparams['f0'])
    end_frequency_mhz   = float(obsparams['f1'])
    channel_width_mhz   = float(obsparams['f_res'])
    time_resolution_ms  = float(obsparams['t_res'])
    start_time_ms       = float(obsparams['t0'])
    end_time_ms         = float(obsparams['t1'])

    # Extract scattering and reference frequency parameters
    scattering_index = float(obsparams['scattering_index'])
    reference_frequency_mhz = float(obsparams['reference_freq'])

    # Generate frequency and time arrays
    frequency_mhz_array = np.arange(
        start=start_frequency_mhz,
        stop=end_frequency_mhz + channel_width_mhz,
        step=channel_width_mhz,
        dtype=float
    )
    time_ms_array = np.arange(
        start=start_time_ms,
        stop=end_time_ms + time_resolution_ms,
        step=time_resolution_ms,
        dtype=float
    )

    # Load Gaussian parameters from file
    gaussian_params = np.loadtxt(gauss_params)

    # Extract individual Gaussian parameters
    t0              = gaussian_params[:, 0]  # Time of peak
    width           = gaussian_params[:, 1]  # Width of the Gaussian
    peak_amp        = gaussian_params[:, 2]  # Peak amplitude
    spec_idx        = gaussian_params[:, 3]  # Spectral index
    dm              = gaussian_params[:, 4]  # Dispersion measure
    rm              = gaussian_params[:, 5]  # Rotation measure
    pol_angle       = gaussian_params[:, 6]  # Polarization angle
    lin_pol_frac    = gaussian_params[:, 7]  # Linear polarization fraction
    circ_pol_frac   = gaussian_params[:, 8]  # Circular polarization fraction
    delta_pol_angle = gaussian_params[:, 9]  # Change in polarization angle
    band_centre_mhz = gaussian_params[:, 10]  # Band centre frequency
    band_width_mhz  = gaussian_params[:, 11]  # Band width

    if (lin_pol_frac + circ_pol_frac).any() > 1.0:
        print("WARNING: Linear and circular polarization fractions sum to more than 1.0 \n")

    def process_dynspec_with_pa_rms(dynspec):
        tsdata, corrdspec, noisespec, noistks = process_dynspec(
            dynspec, frequency_mhz_array, time_ms_array, rm
        )
        tsdata.phits[tsdata.iquvt[0] < 10.0 * noistks[0]] = np.nan
        tsdata.dphits[tsdata.iquvt[0] < 10.0 * noistks[0]] = np.nan

        pa_rms = np.sqrt(np.nanmean(tsdata.phits**2))
        pa_rms_error = np.sqrt(np.nansum((2 * tsdata.phits * tsdata.dphits)**2)) / (2 * len(tsdata.phits))
        
        return pa_rms, pa_rms_error

    def generate_dynspec(mode, s=None):
        if mode == 'gauss':
            return gauss_dynspec(
                frequency_mhz_array, time_ms_array, channel_width_mhz, time_resolution_ms, spec_idx, peak_amp, width, t0,
                dm, pol_angle, lin_pol_frac, circ_pol_frac, delta_pol_angle, rm, seed, noise,
                scatter, scattering_timescale_ms, scattering_index, reference_frequency_mhz, band_centre_mhz, band_width_mhz
            )
        else:  # mode == 'sgauss'
            return sub_gauss_dynspec(
                frequency_mhz_array, time_ms_array, channel_width_mhz, time_resolution_ms, spec_idx, peak_amp, width, t0,
                dm, pol_angle, lin_pol_frac, circ_pol_frac, delta_pol_angle, rm, num_micro_gauss, seed, width_range, noise,
                scatter, s if plot == ['pa_rms'] else scattering_timescale_ms, scattering_index, reference_frequency_mhz,
                band_centre_mhz, band_width_mhz
            )

    if plot != ['pa_rms']:
        dynspec, _ = generate_dynspec(mode)
        tsdata, corrdspec, noisespec, noistks = process_dynspec(
            dynspec, frequency_mhz_array, time_ms_array, rm
        )
        simulated_frb_data = simulated_frb(frb_identifier, frequency_mhz_array, time_ms_array, scattering_timescale_ms,
                                           scattering_index, gaussian_params, dynspec)
        if write:
            output_filename = f"{data_dir}{frb_identifier}_sc_{scattering_timescale_ms:.2f}.pkl"
            with open(output_filename, 'wb') as frbfile:
                pkl.dump(simulated_frb_data, frbfile)
        return simulated_frb_data, noisespec, rm

    elif plot == ['pa_rms']:
        pa_rms_values, pa_rms_errors = [], []
        for s in scattering_timescale_ms:
            dynspec, rms_pol_angles = generate_dynspec(mode, s)
            pa_rms, pa_rms_error = process_dynspec_with_pa_rms(dynspec)
            pa_rms_values.append(pa_rms)
            pa_rms_errors.append(pa_rms_error)
        if write:
            output_filename = f"{data_dir}{frb_identifier}_pa_rms.pkl"
            with open(output_filename, 'wb') as frbfile:
                pkl.dump((pa_rms_values, pa_rms_errors), frbfile)
        return np.array(pa_rms_values), np.array(pa_rms_errors), width[1], rms_pol_angles

    else:
        print("Invalid mode specified. Please use 'gauss' or 'sgauss'. \n")



























































