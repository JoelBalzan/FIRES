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
from utils import *
from basicfns import *
from plotfns import *


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['savefig.dpi'] = 600
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 8
mpl.rcParams["xtick.major.size"] = 3
mpl.rcParams["ytick.major.size"] = 3

#	--------------------------	Analysis functions	-------------------------------

def gauss_dynspec(freq_mhz, time_ms, chan_width_mhz, time_res_ms, spec_idx, peak_amp, width_ms, 
                  loc_ms, dm, pol_angle, lin_pol_frac, circ_pol_frac, 
                  delta_pol_angle, rm, time_per_bin_ms):
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
    dynspec = np.zeros((4, freq_mhz.shape[0], time_ms.shape[0]), dtype=float)  # Initialize dynamic spectrum array
    num_gauss = len(spec_idx) - 1  # Number of Gaussian components
    ref_freq_mhz = np.nanmedian(freq_mhz)  # Reference frequency
    lambda_sq = (speed_of_light_cgs * 1.0e-8 / freq_mhz) ** 2  # Lambda squared array
    median_lambda_sq = np.nanmedian(lambda_sq)  # Median lambda squared

    for g in range(0, num_gauss):
        # Calculate the normalized amplitude for each frequency
        norm_amp = peak_amp[g + 1] * (freq_mhz / np.nanmedian(freq_mhz)) ** spec_idx[g + 1]
        # Calculate the Gaussian pulse shape
        pulse = np.exp(-(time_ms - loc_ms[g + 1]) ** 2 / (2 * (width_ms[g + 1] ** 2)))
        # Calculate the polarization angle array
        pol_angle_arr = pol_angle[g + 1] + (time_ms - loc_ms[g + 1]) * delta_pol_angle[g + 1]

        for c in range(0, len(freq_mhz)):
            # Apply Faraday rotation
            faraday_rot_angle = pol_angle_arr + rm[g + 1] * (lambda_sq[c] - median_lambda_sq)
            # Add the Gaussian pulse to the dynamic spectrum
            dynspec[0, c] += norm_amp[c] * pulse
            # Calculate the dispersion delay
            disp_delay_ms = 4.15 * dm[g + 1] * ((1.0e3 / freq_mhz[c]) ** 2 - (1.0e3 / ref_freq_mhz) ** 2)
            # Apply the dispersion delay
            dynspec[0, c] = np.roll(dynspec[0, c], int(np.round(disp_delay_ms / time_res_ms)))
            # Calculate the Stokes parameters
            dynspec[1, c] += dynspec[0, c] * lin_pol_frac[g + 1] * np.cos(2 * faraday_rot_angle)  # Q
            dynspec[2, c] += dynspec[0, c] * lin_pol_frac[g + 1] * np.sin(2 * faraday_rot_angle)  # U
            dynspec[3, c] += dynspec[0, c] * circ_pol_frac[g + 1]  # V

    print("\nGenerating dynamic spectrum with %d Gaussian component(s)\n" % (num_gauss))
    plt.imshow(dynspec[0, :], aspect='auto', interpolation='none', origin='lower', cmap='seismic', 
               vmin=-np.nanmax(np.abs(dynspec)), vmax=np.nanmax(np.abs(dynspec)))
    plt.show()

    return dynspec

#	--------------------------------------------------------------------------------

def scatter_dynspec(dspec, freq_mhz, time_ms, chan_width_mhz, time_res_ms, tau_ms, sc_idx, rm):
    """	
    Scatter a given dynamic spectrum.
    Inputs:
        - dspec: Dynamic spectrum array
        - freq_mhz: Frequency array in MHz
        - time_ms: Time array in ms
        - chan_width_mhz: Frequency resolution in MHz
        - time_res_ms: Time resolution in ms
        - tau_ms: Scattering time scale in ms
        - sc_idx: Scattering index
    """
    sc_dspec = np.zeros(dspec.shape, dtype=float)  # Initialize scattered dynamic spectrum array
    tau_cms = tau_ms * ((freq_mhz / np.nanmedian(freq_mhz)) ** sc_idx)  # Calculate the scattering time scale for each frequency
    
    for c in range(len(freq_mhz)):
        # Calculate the impulse response function
        irf = np.heaviside(time_ms, 1.0) * np.exp(-time_ms / tau_cms[c]) / tau_cms[c]
        for stk in range(4): 
            # Convolve the dynamic spectrum with the impulse response function
            sc_dspec[stk, c] = np.convolve(dspec[stk, c], irf, mode='same')
    
    # Add noise to the scattered dynamic spectrum
    for stk in range(4): 
        sc_dspec[stk] = sc_dspec[stk] + np.random.normal(loc=0.0, scale=1.0, size=(freq_mhz.shape[0], time_ms.shape[0]))

    # Linear polarisation
    L = np.sqrt(np.nanmean(sc_dspec[1,:], axis=0)**2 + np.nanmean(sc_dspec[2,:], axis=0)**2)

    ############################################
    # Polarisation angle
    ## Estimate Noise spectra
    noisespec	=	estimate_noise(sc_dspec, time_ms, np.min(time_ms), np.max(time_ms)) # add the arguments here 
    noistks		=	np.sqrt(np.nansum(noisespec[:,:]**2,axis=1))/len(freq_mhz)

    corrdspec	=	rm_correct_dynspec(sc_dspec, freq_mhz, rm)
    tsdata		=	est_profiles(corrdspec, freq_mhz, time_ms, noisespec, np.argmin(freq_mhz), np.argmax(freq_mhz))
    
    phits = tsdata.phits
    dphits = tsdata.dphits

    ntp=5
    dpadt = np.zeros(phits.shape, dtype=float)
    edpadt = np.zeros(phits.shape, dtype=float)	
    dpadt[:ntp] = np.nan
    edpadt[:ntp] = np.nan
    dpadt[-ntp:] = np.nan
    edpadt[-ntp:] = np.nan
    
    phits[tsdata.iquvt[0] < 10.0 * noistks[0]] = np.nan
    dphits[tsdata.iquvt[0] < 10.0 * noistks[0]] = np.nan
    ############################################


    print(f"--- Scattering time scale = {tau_ms:.2f} ms, {np.nanmin(tau_cms):.2f} ms to {np.nanmax(tau_cms):.2f} ms")
    
    fig, axs = plt.subplots(6, figsize=(10, 6), constrained_layout=True)
    # Plot polarisation angle
    axs[0].errorbar(time_ms, phits, dphits, fmt='b*', markersize=5, lw=0.5, capsize=2)
    axs[0].set_xlim(time_ms[0], time_ms[-1])
    axs[0].set_ylabel("PA [deg]")
    
    # Plot the mean across all frequency channels (axis 0)
    axs[1].plot(np.nanmean(sc_dspec[0,:], axis=0), markersize=2 ,label='I', color='Black')
    #axs[1].plot(np.nanmean(dspec[0,:], axis=0), markersize=2, label='I_unsc', color='Gray')
    axs[1].plot(np.nanmean(sc_dspec[1,:], axis=0), markersize=2, label='Q', color='Green')
    axs[1].plot(np.nanmean(sc_dspec[2,:], axis=0), markersize=2, label='U', color='Orange')
    axs[1].plot(np.nanmean(sc_dspec[3,:], axis=0), markersize=2, label='V', color='Blue')
    axs[1].plot(L, markersize=2, label='L', color='Red')
    axs[1].set_xlim(0, len(time_ms))
    axs[1].legend(loc='upper right')
    axs[1].set_ylabel("Flux Density (arb.)")

    # Plot the 2D scattered dynamic spectrum
    ## Calculate the mean and standard deviation of the dynamic spectrum
    mn = np.mean(sc_dspec[0,:], axis=(0, 1))
    std = np.std(sc_dspec[0,:], axis=(0, 1))
    ## Set appropriate minimum and maximum values for imshow (Thanks to Dr. M. Lower)
    vmin = mn - 3*std
    vmax = mn + 7*std

    axs[2].imshow(sc_dspec[0], aspect='auto', interpolation='none', origin='lower', cmap='plasma',
        vmin=vmin, vmax=vmax)
    axs[2].set_title("Dynamic Spectra: I, Q, U, V")

    axs[3].imshow(sc_dspec[1], aspect='auto', interpolation='none', origin='lower', cmap='plasma',
        vmin=vmin, vmax=vmax)

    axs[4].imshow(sc_dspec[2], aspect='auto', interpolation='none', origin='lower', cmap='plasma',
        vmin=vmin, vmax=vmax)

    axs[5].imshow(sc_dspec[3], aspect='auto', interpolation='none', origin='lower', cmap='plasma',
           vmin=vmin, vmax=vmax)
    axs[5].set_xlabel("Time (samples)")
    axs[5].set_ylabel("Frequency (MHz)")

    plt.show()
    
    return sc_dspec
