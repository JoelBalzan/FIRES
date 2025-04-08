#
#	Functions for simulating scattering 
#
#								AB, May 2024
#								TC, Sep 2024
#
#	Function list
#
#	gauss_dynspec(fmhzarr, tmsarr, df_mhz, dtms, specind, peak, wms, locms, dmpccc):
#		Generate dynamic spectrum for a Gaussian pulse
#
#	scatter_dynspec(dspec, fmhzarr, tmsarr, df_mhz, dtms, taums, scindex):
#		Scatter a given dynamic spectrum
#
#	--------------------------	Import modules	---------------------------

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scintools.scint_sim import Simulation
from scipy.interpolate import griddata
from ..utils.utils import *
from .basicfns import *
from .plotfns import *


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['savefig.dpi'] = 600
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 8
mpl.rcParams["xtick.major.size"] = 3
mpl.rcParams["ytick.major.size"] = 3

#	--------------------------	Analysis functions	-------------------------------

def gauss_dynspec(freq_mhz, time_ms, chan_width_mhz, time_res_ms, spec_idx, peak_amp, width_ms, loc_ms, 
                  dm, pol_angle, lin_pol_frac, circ_pol_frac, delta_pol_angle, rm, seed, noise, scatter,
                  tau_ms, sc_idx, ref_freq_mhz):
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

    for g in range(num_gauss):
        temp_dynspec = np.zeros_like(dynspec)
        norm_amp = peak_amp[g + 1] * (freq_mhz / ref_freq_mhz) ** spec_idx[g + 1]
        pulse = np.exp(-(time_ms - loc_ms[g + 1]) ** 2 / (2 * (width_ms[g + 1] ** 2)))
        pol_angle_arr = pol_angle[g + 1] + (time_ms - loc_ms[g + 1]) * delta_pol_angle[g + 1]

        for c in range(len(freq_mhz)):
            faraday_rot_angle = apply_faraday_rotation(pol_angle_arr, rm[g + 1], lambda_sq[c], median_lambda_sq)
            temp_dynspec[0, c] = norm_amp[c] * pulse  # Stokes I
            if int(dm[g + 1]) != 0:
                disp_delay_ms = calculate_dispersion_delay(dm[g + 1], freq_mhz[c], ref_freq_mhz)
                temp_dynspec[0, c] = np.roll(temp_dynspec[0, c], int(np.round(disp_delay_ms / time_res_ms)))
            
            # Apply scattering if enabled
            if scatter:
                temp_dynspec[0, c] = scatter_stokes_chan(temp_dynspec[0, c], freq_mhz[c], time_ms, tau_ms, sc_idx, ref_freq_mhz)

            # Add Gaussian noise to Stokes I before calculating Q, U, V
            noise_I = np.random.normal(loc=0.0, scale=np.nanstd(temp_dynspec[0, c]) * noise, size=temp_dynspec[0, c].shape)
            temp_dynspec[0, c] += noise_I

            temp_dynspec[1, c], temp_dynspec[2, c], temp_dynspec[3, c] = calculate_stokes(
                temp_dynspec[0, c], lin_pol_frac, circ_pol_frac, faraday_rot_angle, g + 1
            )  # Stokes Q, U, V

        dynspec += temp_dynspec

    return dynspec





#	--------------------------------------------------------------------------------

def sub_gauss_dynspec(freq_mhz, time_ms, chan_width_mhz, time_res_ms, spec_idx, peak_amp, width_ms, loc_ms, 
                      dm, pol_angle, lin_pol_frac, circ_pol_frac, delta_pol_angle, rm, 
                      num_sub_gauss, seed, width_range, noise, scatter, tau_ms, sc_idx, ref_freq_mhz):
    """
    Generate dynamic spectrum for multiple main Gaussians, each with a distribution of sub-Gaussians.
    """
    # Set the random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    dynspec = np.zeros((4, freq_mhz.shape[0], time_ms.shape[0]), dtype=float)  # Initialize dynamic spectrum array
    lambda_sq = (speed_of_light_cgs * 1.0e-8 / freq_mhz) ** 2  # Lambda squared array
    median_lambda_sq = np.nanmedian(lambda_sq)  # Median lambda squared

    num_main_gauss = len(spec_idx) - 2  # Number of main Gaussian components (-1 for the dummy component and -1 for the variation row)
  
    for g in range(num_main_gauss):
        # Use the last value in each array as the variation factor
        peak_amp_var        = peak_amp[-1]
        pol_angle_var       = pol_angle[-1]
        lin_pol_frac_var    = lin_pol_frac[-1]
        circ_pol_frac_var   = circ_pol_frac[-1]
        delta_pol_angle_var = delta_pol_angle[-1]
        rm_var              = rm[-1]

        for _ in range(num_sub_gauss[g]):
            # Generate random variations for the micro-Gaussian parameters
            var_peak_amp        = peak_amp[g + 1] + np.random.normal(0, peak_amp_var * peak_amp[g + 1])
            # Sample the micro width as a percentage of the main width
            var_width_ms        = width_ms[g + 1] * np.random.uniform(width_range[0] / 100, width_range[1] / 100)
            var_loc_ms          = np.random.normal(loc=loc_ms[g + 1], scale=width_ms[g + 1])
            var_pol_angle       = pol_angle[g + 1] + np.random.normal(0, pol_angle_var * np.abs(pol_angle[g + 1]))
            var_lin_pol_frac    = lin_pol_frac[g + 1] + np.random.normal(0, lin_pol_frac_var * lin_pol_frac[g + 1])
            var_circ_pol_frac   = circ_pol_frac[g + 1] + np.random.normal(0, circ_pol_frac_var * circ_pol_frac[g + 1])
            #var_delta_pol_angle = delta_pol_angle[g + 1] + np.random.normal(0, delta_pol_angle_var * np.abs(delta_pol_angle[g + 1]))
            var_rm              = rm[g + 1] + np.random.normal(0, rm_var * rm[g + 1])

            # Initialize a temporary array for the current sub-Gaussian
            temp_dynspec = np.zeros_like(dynspec)

            # Calculate the normalized amplitude for each frequency
            norm_amp = var_peak_amp * (freq_mhz / ref_freq_mhz) ** spec_idx[g + 1]
            pulse = np.exp(-(time_ms - var_loc_ms) ** 2 / (2 * (var_width_ms ** 2)))
            pol_angle_arr = var_pol_angle + (time_ms - var_loc_ms) * delta_pol_angle[g + 1]

            for c in range(len(freq_mhz)):
                # Apply Faraday rotation
                faraday_rot_angle = pol_angle_arr + var_rm * (lambda_sq[c] - median_lambda_sq)

                # Add the Gaussian pulse to the temporary dynamic spectrum
                temp_dynspec[0, c] = norm_amp[c] * pulse

                # Calculate the dispersion delay
                if int(dm[g + 1]) != 0:
                    disp_delay_ms = 4.15 * dm[g + 1] * ((1.0e3 / freq_mhz[c]) ** 2 - (1.0e3 / ref_freq_mhz) ** 2)
                    temp_dynspec[0, c] = np.roll(temp_dynspec[0, c], int(np.round(disp_delay_ms / time_res_ms)))

                # Apply scattering if enabled
                if scatter:
                    temp_dynspec[0, c] = scatter_stokes_chan(temp_dynspec[0, c], freq_mhz[c], time_ms, tau_ms, sc_idx, ref_freq_mhz)

                # Add Gaussian noise to Stokes I
                if noise > 0:
                    noise_I = np.random.normal(loc=0.0, scale=np.nanstd(temp_dynspec[0, c]) * noise, size=temp_dynspec[0, c].shape)
                    temp_dynspec[0, c] += noise_I

                # Calculate Stokes Q, U, V
                temp_dynspec[1, c], temp_dynspec[2, c], temp_dynspec[3, c] = calculate_stokes(
                    temp_dynspec[0, c], var_lin_pol_frac, var_circ_pol_frac, faraday_rot_angle
                )

            # Accumulate the contributions from the current sub-Gaussian
            dynspec += temp_dynspec

    return dynspec