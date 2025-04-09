#
#	Simulating scattering 
#
#								AB, Sep 2024
#                                                               
#	--------------------------	Import modules	---------------------------
from importlib.resources import files

import matplotlib as mpl
import numpy as np

from .genfns import *
from ..utils.utils import *

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
    start_frequency_mhz, end_frequency_mhz = float(obsparams['f0']), float(obsparams['f1'])
    channel_width_mhz, time_resolution_ms = float(obsparams['f_res']), float(obsparams['t_res'])
    start_time_ms, end_time_ms = float(obsparams['t0']), float(obsparams['t1'])
    scattering_index, reference_frequency_mhz = float(obsparams['scattering_index']), float(obsparams['reference_freq'])

    frequency_mhz_array = np.arange(start_frequency_mhz, end_frequency_mhz + channel_width_mhz, channel_width_mhz, dtype=float)
    time_ms_array = np.arange(start_time_ms, end_time_ms + time_resolution_ms, time_resolution_ms, dtype=float)

    gaussian_params = np.loadtxt(gauss_params)
    t0, width, peak_amp, spec_idx = gaussian_params[:, 0], gaussian_params[:, 1], gaussian_params[:, 2], gaussian_params[:, 3]
    dm, rm, pol_angle = gaussian_params[:, 4], gaussian_params[:, 5], gaussian_params[:, 6]
    lin_pol_frac, circ_pol_frac, delta_pol_angle = gaussian_params[:, 7], gaussian_params[:, 8], gaussian_params[:, 9]

    if (lin_pol_frac + circ_pol_frac).any() > 1.0:
        print("WARNING: Linear and circular polarization fractions sum to more than 1.0")

    def process_dynspec_with_pa_rms(dynspec):
        tsdata, corrdspec, noisespec = process_dynspec(
            dynspec, frequency_mhz_array, time_ms_array, startms, stopms, startchan, endchan, rm
        )
        pa_rms = np.sqrt(np.nanmean(tsdata.phits**2))
        pa_rms_error = np.sqrt(np.nansum((2 * tsdata.phits * tsdata.dphits)**2)) / (2 * len(tsdata.phits))
        return pa_rms, pa_rms_error

    def generate_dynspec(mode, s=None):
        if mode == 'gauss':
            return gauss_dynspec(
                frequency_mhz_array, time_ms_array, channel_width_mhz, time_resolution_ms, spec_idx, peak_amp, width, t0,
                dm, pol_angle, lin_pol_frac, circ_pol_frac, delta_pol_angle, rm, seed, noise,
                scatter, scattering_timescale_ms, scattering_index, reference_frequency_mhz
            )
        else:  # mode == 'sgauss'
            return sub_gauss_dynspec(
                frequency_mhz_array, time_ms_array, channel_width_mhz, time_resolution_ms, spec_idx, peak_amp, width, t0,
                dm, pol_angle, lin_pol_frac, circ_pol_frac, delta_pol_angle, rm, num_micro_gauss, seed, width_range, noise,
                scatter, s if plot == ['pa_rms'] else scattering_timescale_ms, scattering_index, reference_frequency_mhz
            )

    if plot != ['pa_rms']:
        dynspec = generate_dynspec(mode)
        simulated_frb_data = simulated_frb(frb_identifier, frequency_mhz_array, time_ms_array, scattering_timescale_ms,
                                           scattering_index, gaussian_params, dynspec)
        if write:
            output_filename = f"{data_dir}{frb_identifier}_sc_{scattering_timescale_ms:.2f}.pkl"
            with open(output_filename, 'wb') as frbfile:
                pkl.dump(simulated_frb_data, frbfile)
        return simulated_frb_data, rm

    elif plot == ['pa_rms']:
        pa_rms_values, pa_rms_errors = [], []
        for s in scattering_timescale_ms:
            dynspec = generate_dynspec(mode, s)
            pa_rms, pa_rms_error = process_dynspec_with_pa_rms(dynspec)
            pa_rms_values.append(pa_rms)
            pa_rms_errors.append(pa_rms_error)
        return np.array(pa_rms_values), np.array(pa_rms_errors)

    else:
        print("Invalid mode specified. Please use 'gauss' or 'sgauss'.")



























































