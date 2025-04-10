# Simulating scattering
# AB, Sep 2024
# -------------------------- Import modules ---------------------------
from importlib.resources import files
import matplotlib as mpl
import numpy as np
from FIRES.functions.genfns import *
from FIRES.utils.utils import *
import os
import pickle as pkl  # Import pickle
import concurrent.futures
import functools

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
obs_params_path = os.path.join(parent_dir, "utils/obsparams.txt")
gauss_params_path = os.path.join(parent_dir, "utils/gparams.txt")


# ------------------------- Helper functions -------------------------------

def process_dynspec_with_pa_rms(dynspec, frequency_mhz_array, time_ms_array, startms, stopms, startchan, endchan, rm):
    """Process dynamic spectrum to calculate PA RMS."""
    tsdata, corrdspec, noisespec, noistks = process_dynspec(
        dynspec, frequency_mhz_array, time_ms_array, startms, stopms, startchan, endchan, rm)
    
    tsdata.phits[tsdata.iquvt[0] < 10.0 * noistks[0]] = np.nan
    tsdata.dphits[tsdata.iquvt[0] < 10.0 * noistks[0]] = np.nan

    pa_rms = np.sqrt(np.nanmean(tsdata.phits**2))
    pa_rms_error = np.sqrt(np.nansum((2 * tsdata.phits * tsdata.dphits)**2)) / (2 * len(tsdata.phits))
    
    return pa_rms, pa_rms_error

def generate_dynspec(mode, frequency_mhz_array, time_ms_array, channel_width_mhz, time_resolution_ms, spec_idx, peak_amp, 
                     width, t0, dm, pol_angle, lin_pol_frac, circ_pol_frac, delta_pol_angle, rm, seed, noise, scatter, 
                     scattering_timescale_ms, scattering_index, reference_frequency_mhz, num_micro_gauss, width_range, s, plot_pa_rms):
    """Generate dynamic spectrum based on mode."""
    if mode == 'gauss':
        return gauss_dynspec(
            frequency_mhz_array, time_ms_array, channel_width_mhz, time_resolution_ms, spec_idx, peak_amp, width, t0,
            dm, pol_angle, lin_pol_frac, circ_pol_frac, delta_pol_angle, rm, seed, noise,
            scatter, scattering_timescale_ms, scattering_index, reference_frequency_mhz
        )
    else:  # mode == 'sgauss'
        s_value = s if plot_pa_rms else scattering_timescale_ms
        return sub_gauss_dynspec(
            frequency_mhz_array, time_ms_array, channel_width_mhz, time_resolution_ms, spec_idx, peak_amp, width, t0,
            dm, pol_angle, lin_pol_frac, circ_pol_frac, delta_pol_angle, rm, num_micro_gauss, seed, width_range, noise,
            scatter, s_value, scattering_index, reference_frequency_mhz
        )

def process_scattering_timescale(s, mode, frequency_mhz_array, time_ms_array, channel_width_mhz, time_resolution_ms, 
                                 spec_idx, peak_amp, width, t0, dm, pol_angle, lin_pol_frac, circ_pol_frac, delta_pol_angle, 
                                 rm, seed, noise, scatter, scattering_timescale_ms, scattering_index, reference_frequency_mhz, 
                                 num_micro_gauss, width_range, startms, stopms, startchan, endchan):
    """Process a single scattering timescale."""
    dynspec = generate_dynspec(mode, frequency_mhz_array, time_ms_array, channel_width_mhz, time_resolution_ms, spec_idx, peak_amp, 
                               width, t0, dm, pol_angle, lin_pol_frac, circ_pol_frac, delta_pol_angle, rm, seed, noise, scatter, 
                               scattering_timescale_ms, scattering_index, reference_frequency_mhz, num_micro_gauss, width_range, s, 
                               plot_pa_rms=True)
    pa_rms, pa_rms_error = process_dynspec_with_pa_rms(dynspec, frequency_mhz_array, time_ms_array, startms, stopms, startchan, endchan, rm)
    return pa_rms, pa_rms_error

# ------------------------- Main function -------------------------------

def generate_frb_parallel(scattering_timescale_ms, frb_identifier, data_dir, mode, num_micro_gauss, seed, width_range, write,
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

    # Check if running on OzSTAR
    if 'SLURM_JOB_ID' in os.environ:  # Example: Check for SLURM environment variable
        num_processes = int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))  # Get from environment, default to 1
        print(f"Running on OzSTAR, setting max_workers to {num_processes}")
    else:
        num_processes = None  # Use default ProcessPoolExecutor behavior
        print("Not running on OzSTAR, using default max_workers")

    if plot != ['pa_rms']:
        dynspec = generate_dynspec(mode, frequency_mhz_array, time_ms_array, channel_width_mhz, time_resolution_ms, spec_idx, peak_amp, 
                                   width, t0, dm, pol_angle, lin_pol_frac, circ_pol_frac, delta_pol_angle, rm, seed, noise, scatter, 
                                   scattering_timescale_ms, scattering_index, reference_frequency_mhz, num_micro_gauss, width_range)
        simulated_frb_data = simulated_frb(frb_identifier, frequency_mhz_array, time_ms_array, scattering_timescale_ms,
                                            scattering_index, gaussian_params, dynspec)
        if write:
            output_filename = f"{data_dir}{frb_identifier}_sc_{scattering_timescale_ms:.2f}.pkl"
            with open(output_filename, 'wb') as frbfile:
                pkl.dump(simulated_frb_data, frbfile)
        return simulated_frb_data, rm
    
    elif plot == ['pa_rms']:
        pa_rms_values, pa_rms_errors = [], []

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            # Create a partial function with all the fixed arguments
            partial_process = functools.partial(
                process_scattering_timescale,
                mode=mode,
                frequency_mhz_array=frequency_mhz_array,
                time_ms_array=time_ms_array,
                channel_width_mhz=channel_width_mhz,
                time_resolution_ms=time_resolution_ms,
                spec_idx=spec_idx,
                peak_amp=peak_amp,
                width=width,
                t0=t0,
                dm=dm,
                pol_angle=pol_angle,
                lin_pol_frac=lin_pol_frac,
                circ_pol_frac=circ_pol_frac,
                delta_pol_angle=delta_pol_angle,
                rm=rm,
                seed=seed,
                noise=noise,
                scatter=scatter,
                scattering_timescale_ms=scattering_timescale_ms,
                scattering_index=scattering_index,
                reference_frequency_mhz=reference_frequency_mhz,
                num_micro_gauss=num_micro_gauss,
                width_range=width_range,
                startms=startms,
                stopms=stopms,
                startchan=startchan,
                endchan=endchan
            )
            # Map the partial function to the scattering timescales
            results = list(executor.map(partial_process, scattering_timescale_ms))


        pa_rms_values, pa_rms_errors = zip(*results)
        pa_rms_values = list(pa_rms_values)
        pa_rms_errors = list(pa_rms_errors)

        if write:
            output_filename = f"{data_dir}{frb_identifier}_pa_rms.pkl"
            with open(output_filename, 'wb') as frbfile:
                pkl.dump((pa_rms_values, pa_rms_errors), frbfile)

        return np.array(pa_rms_values), np.array(pa_rms_errors), width[1]
    else:
        print("Invalid mode specified. Please use 'gauss' or 'sgauss'.")
