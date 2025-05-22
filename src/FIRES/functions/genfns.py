# -----------------------------------------------------------------------------
# genfns.py
# FIRES: The Fast, Intense Radio Emission Simulator
#
# This module provides core functions for generating FRB dynamic spectra,
# applying Faraday rotation, dispersion, noise, and polarization effects.
# It includes both single-Gaussian and multi-Gaussian (micro-shot) models,
# as well as helper routines for Stokes parameter calculation.
#
# Author: JB
# Date: 2025-05-20
# -----------------------------------------------------------------------------

#	--------------------------	Import modules	---------------------------


import numpy as np

from FIRES.utils.utils import *
from FIRES.functions.basicfns import *
from FIRES.functions.plotfns import *


#    --------------------------	Functions ---------------------------

def apply_faraday_rotation(pol_angle_arr, rm, lambda_sq, median_lambda_sq):
	return pol_angle_arr + rm * (lambda_sq - median_lambda_sq)


def calculate_dispersion_delay(dm, freq, ref_freq):
	return 4.15 * dm * ((1.0e3 / freq) ** 2 - (1.0e3 / ref_freq) ** 2)


def add_noise_to_stokes_I(temp_dynspec_chan, peak_amp, noise):
    # Set noise level based on desired SNR (signal-to-noise ratio)
    # SNR = peak signal / noise stddev => noise stddev = peak signal / SNR
    # Use the intended envelope peak as the reference signal
    signal_level = np.nanmax(peak_amp)
    noise_std = signal_level / noise  # 'noise' parameter is now SNR
    noise_I = np.random.normal(loc=0.0, scale=noise_std, size=temp_dynspec_chan.shape)
    
    return temp_dynspec_chan + noise_I


def calculate_stokes(temp_dynspec, lin_pol_frac, circ_pol_frac, faraday_rot_angle):
	stokes_q = temp_dynspec * lin_pol_frac * np.cos(2 * faraday_rot_angle)
	stokes_u = temp_dynspec * lin_pol_frac * np.sin(2 * faraday_rot_angle)
	stokes_v = temp_dynspec * circ_pol_frac
	return stokes_q, stokes_u, stokes_v





# -------------------------- FRB generator functions ---------------------------
def gauss_dynspec(freq_mhz, time_ms, time_res_ms, spec_idx, peak_amp, width_ms, loc_ms, 
                  dm, pol_angle, lin_pol_frac, circ_pol_frac, delta_pol_angle, rm, seed, noise,
                  tau_ms, sc_idx, ref_freq_mhz, band_centre_mhz, band_width_mhz):
    """
    Generate dynamic spectrum for Gaussian pulses.
    Inputs:
        - freq_mhz: Frequency array in MHz
        - time_ms: Time array in ms
        - chan_width_mhz: Frequency resolution in MHz
        - time_res_ms: Time resolution in ms
        - spec_idx: Spectral index array
        - peak_amp: Peak amplitude array
        - width_ms: Width of the Gaussian pulse in ms
        - loc_ms: Location of the Gaussian pulse in ms
        - dm: Dispersion measure in pc/cm^3
        - pol_angle: Polarization angle array
        - lin_pol_frac: Linear polarization fraction array
        - circ_pol_frac: Circular polarization fraction array
        - delta_pol_angle: Change in polarization angle with time
        - rm: Rotation measure array
    """


    if seed is not None:
        np.random.seed(seed)

    # Initialize dynamic spectrum for all Stokes parameters
    dynspec = np.zeros((4, freq_mhz.shape[0], time_ms.shape[0]), dtype=float)  # [I, Q, U, V]
    lambda_sq = (speed_of_light_cgs * 1.0e-8 / freq_mhz) ** 2
    median_lambda_sq = np.nanmedian(lambda_sq)
    num_gauss = len(spec_idx) - 2

    all_pol_angles = []  
    for g in range(num_gauss):
        temp_dynspec = np.zeros_like(dynspec)
        norm_amp = peak_amp[g] * (freq_mhz / ref_freq_mhz) ** spec_idx[g]
        pol_angle_arr = pol_angle[g] + (time_ms - loc_ms[g]) * delta_pol_angle[g]
        all_pol_angles.append(pol_angle)

        # Apply Gaussian spectral profile if band_centre_mhz and band_width_mhz are provided
        if band_width_mhz[g] != 0.:
            if band_centre_mhz[g] == 0.:
                band_centre_mhz[g] = np.median(freq_mhz)
            spectral_profile = np.exp(-((freq_mhz - band_centre_mhz[g]) ** 2) / (2 * (band_width_mhz[g] / 2.355) ** 2)) #2.355 is the FWHM factor
            norm_amp *= spectral_profile

        for c in range(len(freq_mhz)):
            faraday_rot_angle = apply_faraday_rotation(pol_angle_arr, rm[g], lambda_sq[c], median_lambda_sq)
            temp_dynspec[0, c] = gaussian_model(time_ms, norm_amp[c], loc_ms[g], width_ms[g])
            
            if int(dm[g]) != 0:
                disp_delay_ms = calculate_dispersion_delay(dm[g], freq_mhz[c], ref_freq_mhz)
                temp_dynspec[0, c] = np.roll(temp_dynspec[0, c], int(np.round(disp_delay_ms / time_res_ms)))
            
            # Apply scattering if enabled
            if tau_ms > 0:
                temp_dynspec[0, c] = scatter_stokes_chan(temp_dynspec[0, c], freq_mhz[c], time_ms, tau_ms, sc_idx, ref_freq_mhz)

            temp_dynspec[1, c], temp_dynspec[2, c], temp_dynspec[3, c] = calculate_stokes(
                temp_dynspec[0, c], lin_pol_frac[g], circ_pol_frac[g], faraday_rot_angle
            )  # Stokes Q, U, V

        dynspec += temp_dynspec
    var_pol_angles = np.nanvar(np.array(all_pol_angles))
    
    if noise > 0:
        width_ds = width_ms[1] / time_res_ms
        if band_width_mhz[1] == 0.:
            band_width_mhz = freq_mhz[-1] - freq_mhz[0]
        dynspec = add_noise_to_dynspec(dynspec, noise, seed, band_width_mhz, width_ds)

    return dynspec, var_pol_angles




def m_gauss_dynspec(freq_mhz, time_ms, time_res_ms, spec_idx, peak_amp, width_ms, loc_ms, 
                      dm, pol_angle, lin_pol_frac, circ_pol_frac, delta_pol_angle, rm, num_micro_gauss, seed, 
                      width_range, noise, tau_ms, sc_idx, ref_freq_mhz, band_centre_mhz, band_width_mhz):
    """
    Generate dynamic spectrum for multiple main Gaussians, each with a distribution of micro-shots.
    Optionally apply a Gaussian spectral profile to create band-limited pulses.
    """
    # Set the random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    dynspec = np.zeros((4, freq_mhz.shape[0], time_ms.shape[0]), dtype=float)  # Initialize dynamic spectrum array
    lambda_sq = (speed_of_light_cgs * 1.0e-8 / freq_mhz) ** 2  # Lambda squared array
    median_lambda_sq = np.nanmedian(lambda_sq)  # Median lambda squared

    num_main_gauss = len(spec_idx) - 2  # Number of main Gaussian components (-1 for the the variation row and -1 for the plot variation row)

    # Use the last value in each array as the variation factor
    peak_amp_var        = peak_amp[-2]
    pol_angle_var       = pol_angle[-2]
    lin_pol_frac_var    = lin_pol_frac[-2]
    circ_pol_frac_var   = circ_pol_frac[-2]
    delta_pol_angle_var = delta_pol_angle[-2]
    rm_var              = rm[-2]

    if lin_pol_frac_var > 0.0 and circ_pol_frac_var > 0.0:
        input("Linear and circular polarisation variations are both > 0.0. Choose one to vary (l/c).")
        if input("l/c: ") == 'l':
            circ_pol_frac_var = 0.0
        else:
            lin_pol_frac_var = 0.0

    all_pol_angles = []  
    for g in range(num_main_gauss):
        for _ in range(num_micro_gauss[g]):
            # Generate random variations for the micro-Gaussian parameters
            var_peak_amp        = peak_amp[g] + np.random.normal(0, peak_amp_var * peak_amp[g])
            # Sample the micro width as a percentage of the main width
            var_width_ms        = width_ms[g] * np.random.uniform(width_range[0] / 100, width_range[1] / 100)
            var_loc_ms          = np.random.normal(loc=loc_ms[g], scale=width_ms[g])
            var_pol_angle       = pol_angle[g] + np.random.normal(0, pol_angle_var)
            var_lin_pol_frac    = lin_pol_frac[g] + np.random.normal(0, lin_pol_frac_var * lin_pol_frac[g])
            var_circ_pol_frac   = circ_pol_frac[g] + np.random.normal(0, circ_pol_frac_var * circ_pol_frac[g])
            var_delta_pol_angle = delta_pol_angle[g] + np.random.normal(0, delta_pol_angle_var * np.abs(delta_pol_angle[g]))
            var_rm              = rm[g] + np.random.normal(0, rm_var)

            if circ_pol_frac_var > 0.0:
                var_circ_pol_frac = np.clip(var_circ_pol_frac, 0.0, 1.0)
                var_lin_pol_frac = np.clip(1.0 - var_circ_pol_frac, 0.0, 1.0)

            elif lin_pol_frac_var > 0.0:
                var_lin_pol_frac = np.clip(var_lin_pol_frac, 0.0, 1.0)
                var_circ_pol_frac = np.clip(1.0 - var_lin_pol_frac, 0.0, 1.0)

            all_pol_angles.append(var_pol_angle)

            # Initialize a temporary array for the current micro-shot
            temp_dynspec = np.zeros_like(dynspec)

            # Calculate the normalized amplitude for each frequency
            norm_amp = var_peak_amp * (freq_mhz / ref_freq_mhz) ** spec_idx[g]
            
            # Apply Gaussian spectral profile if band_centre_mhz and band_width_mhz are provided
            if band_width_mhz[g] != 0.:
                if band_centre_mhz[g] == 0.:
                    band_centre_mhz[g] = np.median(freq_mhz)
                spectral_profile = np.exp(-((freq_mhz - band_centre_mhz[g]) ** 2) / (2 * (band_width_mhz[g] / 2.355) ** 2)) #2.355 is the FWHM factor
                norm_amp *= spectral_profile

            pol_angle_arr = var_pol_angle + (time_ms - var_loc_ms) * delta_pol_angle[g]

            for c in range(len(freq_mhz)):
                # Apply Faraday rotation
                faraday_rot_angle = apply_faraday_rotation(pol_angle_arr, var_rm, lambda_sq[c], median_lambda_sq)
                # Add the Gaussian pulse to the temporary dynamic spectrum
                temp_dynspec[0, c] = gaussian_model(time_ms, norm_amp[c], var_loc_ms, var_width_ms)

                # Calculate the dispersion delay
                if int(dm[g]) != 0:
                    disp_delay_ms = calculate_dispersion_delay(dm[g], freq_mhz[c], ref_freq_mhz)
                    temp_dynspec[0, c] = np.roll(temp_dynspec[0, c], int(np.round(disp_delay_ms / time_res_ms)))

                # Apply scattering if enabled
                if tau_ms > 0:
                    temp_dynspec[0, c] = scatter_stokes_chan(temp_dynspec[0, c], freq_mhz[c], time_ms, tau_ms, sc_idx, ref_freq_mhz)

                # Calculate Stokes Q, U, V
                temp_dynspec[1, c], temp_dynspec[2, c], temp_dynspec[3, c] = calculate_stokes(
                    temp_dynspec[0, c], var_lin_pol_frac, var_circ_pol_frac, faraday_rot_angle
                )

            # Accumulate the contributions from the current micro-shot
            dynspec += temp_dynspec
    
    # Normalize dynspec so its maximum matches the intended envelope peak
    if np.nanmax(dynspec[0]) > 0 and np.nanmax(peak_amp) > 0:
        dynspec *= np.nanmax(peak_amp) / np.nanmax(dynspec[0])

    var_pol_angles = np.nanvar(np.array(all_pol_angles))
    
    if noise > 0:
        width_ds = width_ms[1] / time_res_ms
        if band_width_mhz[1] == 0.:
            band_width_mhz = freq_mhz[-1] - freq_mhz[0]
        dynspec = add_noise_to_dynspec(dynspec, noise, seed, band_width_mhz, width_ds)
    
    return dynspec, var_pol_angles