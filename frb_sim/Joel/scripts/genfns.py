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

import os, sys
import matplotlib as mpl
import numpy as np
import matplotlib.ticker as ticker
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scintools.scint_sim import Simulation
from utils import *

mpl.rcParams['pdf.fonttype']	= 42
mpl.rcParams['ps.fonttype'] 	= 42
mpl.rcParams['savefig.dpi'] 	= 600
mpl.rcParams['font.family'] 	= 'sans-serif'
mpl.rcParams['font.size']		= 8
mpl.rcParams["xtick.major.size"]= 3
mpl.rcParams["ytick.major.size"]= 3

#	--------------------------	Analysis functions	-------------------------------
 
def gauss_dynspec(frequency_mhz_array, time_ms_array, channel_width_mhz, time_resolution_ms, spectral_index, peak_amplitude, width_ms, location_ms, dispersion_measure, polarization_angle, linear_polarization_fraction, circular_polarization_fraction, change_in_polarization_angle, rotation_measure):
    """
    Generate dynamic spectrum for Gaussian pulses.
    Inputs:
        - frequency_mhz_array: Frequency array in MHz
        - time_ms_array: Time array in ms
        - channel_width_mhz: Frequency resolution in MHz
        - time_resolution_ms: Time resolution in ms
        - spectral_index: Spectral index array
        - peak_amplitude: Peak amplitude array
        - width_ms: Width of the Gaussian pulse in ms
        - location_ms: Location of the Gaussian pulse in ms
        - dispersion_measure: Dispersion measure in pc/cm^3
        - polarization_angle: Polarization angle array
        - linear_polarization_fraction: Linear polarization fraction array
        - circular_polarization_fraction: Circular polarization fraction array
        - change_in_polarization_angle: Change in polarization angle with time
        - rotation_measure: Rotation measure array
    """
    dynamic_spectrum = np.zeros((4, frequency_mhz_array.shape[0], time_ms_array.shape[0]), dtype=float)  # Initialize dynamic spectrum array
    num_gaussians = len(spectral_index) - 1  # Number of Gaussian components
    pulse_array = np.zeros(len(time_ms_array), dtype=float)  # Initialize pulse array
    reference_frequency_mhz = np.nanmedian(frequency_mhz_array)  # Reference frequency
    lambda_squared_array = (speed_of_light_cgs * 1.0e-8 / frequency_mhz_array) ** 2  # Lambda squared array
    median_lambda_squared = np.nanmedian(lambda_squared_array)  # Median lambda squared

    for g in range(0, num_gaussians):
        # Calculate the normalized amplitude for each frequency
        normalized_amplitude = peak_amplitude[g + 1] * (frequency_mhz_array / np.nanmedian(frequency_mhz_array)) ** spectral_index[g + 1]
        # Calculate the Gaussian pulse shape
        pulse_array = np.exp(-(time_ms_array - location_ms[g + 1]) ** 2 / (2 * (width_ms[g + 1] ** 2)))
        print(np.min(pulse_array), np.max(pulse_array))
        # Calculate the polarization angle array
        polarization_angle_array = polarization_angle[g + 1] + (time_ms_array - location_ms[g + 1]) * change_in_polarization_angle[g + 1]

        for c in range(0, len(frequency_mhz_array)):
            # Apply Faraday rotation
            faraday_rotation_angle = polarization_angle_array + rotation_measure[g + 1] * (lambda_squared_array[c] - median_lambda_squared)
            # Add the Gaussian pulse to the dynamic spectrum
            dynamic_spectrum[0, c] += normalized_amplitude[c] * pulse_array
            # Calculate the dispersion delay
            dispersion_delay_ms = 4.15 * dispersion_measure[g + 1] * ((1.0e3 / frequency_mhz_array[c]) ** 2 - (1.0e3 / reference_frequency_mhz) ** 2)
            # Apply the dispersion delay
            dynamic_spectrum[0, c] = np.roll(dynamic_spectrum[0, c], int(np.round(dispersion_delay_ms / time_resolution_ms)))
            # Calculate the Stokes parameters
            dynamic_spectrum[1, c] += dynamic_spectrum[0, c] * linear_polarization_fraction[g + 1] * np.cos(2 * faraday_rotation_angle)  # Q
            dynamic_spectrum[2, c] += dynamic_spectrum[0, c] * linear_polarization_fraction[g + 1] * np.sin(2 * faraday_rotation_angle)  # U
            dynamic_spectrum[3, c] += dynamic_spectrum[0, c] * circular_polarization_fraction[g + 1]  # V

    print("\nGenerating dynamic spectrum with %d Gaussian component(s)\n" % (num_gaussians))
    plt.imshow(dynamic_spectrum[0, :], aspect='auto', interpolation='none', origin='lower', cmap='seismic', vmin=-np.nanmax(np.abs(dynamic_spectrum)), vmax=np.nanmax(np.abs(dynamic_spectrum)))
    plt.show()

    return dynamic_spectrum, polarization_angle_array

#	--------------------------------------------------------------------------------

def scatter_dynspec(dspec, fmhzarr, tmsarr, df_mhz, dtms, taums, scindex, polangles):
    """	
    Scatter a given dynamic spectrum.
    Inputs:
        - dspec: Dynamic spectrum array
        - fmhzarr: Frequency array in MHz
        - tmsarr: Time array in ms
        - df_mhz: Frequency resolution in MHz
        - dtms: Time resolution in ms
        - taums: Scattering time scale in ms
        - scindex: Scattering index
    """
    scdspec = np.zeros(dspec.shape, dtype=float)  # Initialize scattered dynamic spectrum array
    taucms 	= taums * ((fmhzarr / np.nanmedian(fmhzarr)) ** scindex)  # Calculate the scattering time scale for each frequency
    
    for c in range(len(fmhzarr)):
        # Calculate the impulse response function
        irfarr = np.heaviside(tmsarr, 1.0) * np.exp(-tmsarr / taucms[c]) / taucms[c]
        for stk in range(4): 
            # Convolve the dynamic spectrum with the impulse response function
            scdspec[stk, c] = np.convolve(dspec[stk, c], irfarr, mode='same')
    
    # Add noise to the scattered dynamic spectrum
    for stk in range(4): 
        scdspec[stk] = scdspec[stk] + np.random.normal(loc=0.0, scale=1.0, size=(fmhzarr.shape[0], tmsarr.shape[0]))

    # Linear polarisation
    L = np.sqrt(np.nanmean(scdspec[1,:], axis=0)**2 + np.nanmean(scdspec[2,:], axis=0)**2)

    print(f"--- Scattering time scale = {taums:.2f} ms, {np.nanmin(taucms):.2f} ms to {np.nanmax(taucms):.2f} ms")
    
    fig, axs = plt.subplots(6, figsize=(10, 6), constrained_layout=True)
    fig.suptitle('Scattered Dynamic Spectrum')
    # Plot polarisation angle
    axs[0].scatter(tmsarr, polangles, label='Polarisation Angle', color='Black')
    axs[0].set_title("Polarisation Angle over Time")
    
    # Plot the mean across all frequency channels (axis 0)
    axs[1].plot(np.nanmean(scdspec[0,:], axis=0), markersize=2 ,label='I', color='Black')
    axs[1].plot(np.nanmean(scdspec[1,:], axis=0), markersize=2, label='Q', color='Green')
    axs[1].plot(np.nanmean(scdspec[2,:], axis=0), markersize=2, label='U', color='Orange')
    axs[1].plot(np.nanmean(scdspec[3,:], axis=0), markersize=2, label='V', color='Blue')
    axs[1].plot(L, markersize=2, label='L', color='Red')
    axs[1].set_title("Mean Scattered Signal over Time")
    axs[1].legend(loc='upper right')

    # Plot the 2D scattered dynamic spectrum
    axs[2].imshow(scdspec[0], aspect='auto', interpolation='none', origin='lower', cmap='plasma',
        vmin=-np.nanmax(np.abs(dspec[0])), vmax=np.nanmax(np.abs(dspec[0]))*5)
    axs[2].set_title("Mean Scattered Dynamic Spectrum Across Frequency Channels")

    axs[3].imshow(scdspec[1], aspect='auto', interpolation='none', origin='lower', cmap='plasma',
        vmin=-np.nanmax(np.abs(dspec[0])), vmax=np.nanmax(np.abs(dspec[0]))*2.5)

    axs[4].imshow(scdspec[2], aspect='auto', interpolation='none', origin='lower', cmap='plasma',
        vmin=-np.nanmax(np.abs(dspec[0])), vmax=np.nanmax(np.abs(dspec[0]))*2.5)

    axs[5].imshow(scdspec[3], aspect='auto', interpolation='none', origin='lower', cmap='plasma',
           vmin=-np.nanmax(np.abs(dspec[0])), vmax=np.nanmax(np.abs(dspec[0]))*5)
    axs[5].set_xlabel("Time (samples)")
    axs[5].set_ylabel("Frequency (MHz)")

    #plt.tight_layout()
    plt.show()
    
    return scdspec

#	--------------------------------------------------------------------------------
 ### THIS IS WRONG ###
def apply_scintillation(dynspec):
    """
    Apply scintillation effects to a given dynamic spectrum. 
    Uses Daniel Reardon's Scintools package for simulating scintillation:
    https://github.com/danielreardon/scintools

    Parameters:
    -----------
    dynspec : numpy.ndarray
        Input dynamic spectrum with shape (4, time bins, frequency channels).
        The first dimension represents the four Stokes parameters.
    mb2 : float, optional
        Max Born parameter for the strength of scattering. Default is 2.
    rf : float, optional
        Fresnel scale. Default is 1.
    ds : float, optional
        Spatial step size with respect to rf. Default is 0.01.
    alpha : float, optional
        Structure function exponent (Kolmogorov = 5/3). Default is 5/3.
    ar : float, optional
        Anisotropy axial ratio. Default is 1.
    psi : float, optional
        Anisotropy orientation in degrees. Default is 0.
    inner : float, optional
        Inner scale w.r.t rf - should generally be smaller than ds. Default is 0.001.
    seed : int, optional
        Seed number for the random number generator, or use "-1" to shuffle. Default is None.
    verbose : bool, optional
        Flag to enable verbose output. Default is False.

    Returns:
    --------
    scintillated_dynspec : numpy.ndarray
        The scintillated dynamic spectrum with the same shape as the input.
    """

    npol, nsub, nchan = dynspec.shape  # Extract dimensions
    # Set simulation parameters based on dynamic spectrum dimensions
    scint_sim = Simulation(mb2=mb2, rf=rf, ds=time_resolution_ms, alpha=alpha, ar=ar, psi=psi, inner=inner, 
                              ns=nsub, nf=nchan, dlam=dlam, 
                              lamsteps=lamsteps, seed=seed, nx=nsub, ny=nchan, dx=ds, dy=ds)
    
    # Generate the intensity pattern
    xyi = scint_sim.xyi  # Calculate the intensity
    intensity_pattern = xyi / np.max(xyi)  # Normalize intensity pattern

    scintillated_dynspec = np.zeros_like(dynspec)  # Initialize output

    for pol in range(npol):
        # Apply the intensity pattern to each Stokes parameter
        scintillated_dynspec[pol, :, :] = dynspec[pol, :, :] * intensity_pattern

    
    # Plot the scintillated dynamic spectrum
    fig, axs = plt.subplots(5, figsize=(10, 6))
    fig.suptitle('Scintillated Dynamic Spectrum')
    
    # Plot the mean across all frequency channels (axis 0)
    axs[0].plot(np.nanmean(scintillated_dynspec[0,:], axis=0), markersize=2 ,label='I')
    axs[0].plot(np.nanmean(scintillated_dynspec[1,:], axis=0), markersize=2, label='Q')
    axs[0].plot(np.nanmean(scintillated_dynspec[2,:], axis=0), markersize=2, label='U')
    axs[0].plot(np.nanmean(scintillated_dynspec[3,:], axis=0), markersize=2, label='V')
    axs[0].set_title("Mean Scintillated Signal over Time")
    axs[0].legend(loc='upper right')

    # Plot the 2D scattered dynamic spectrum
    axs[1].imshow(scintillated_dynspec[0], aspect='auto', interpolation='none', origin='lower', cmap='plasma',
        vmin=-np.nanmax(np.abs(scintillated_dynspec[0])), vmax=np.nanmax(np.abs(scintillated_dynspec[0])))
    axs[1].set_title("Mean Scintillated Dynamic Spectrum Across Frequency Channels")

    axs[2].imshow(scintillated_dynspec[1], aspect='auto', interpolation='none', origin='lower', cmap='plasma',
        vmin=-np.nanmax(np.abs(scintillated_dynspec[0])), vmax=np.nanmax(np.abs(scintillated_dynspec[0])))

    axs[3].imshow(scintillated_dynspec[2], aspect='auto', interpolation='none', origin='lower', cmap='plasma',
        vmin=-np.nanmax(np.abs(scintillated_dynspec[0])), vmax=np.nanmax(np.abs(scintillated_dynspec[0])))

    axs[4].imshow(scintillated_dynspec[3], aspect='auto', interpolation='none', origin='lower', cmap='plasma',
           vmin=-np.nanmax(np.abs(scintillated_dynspec[0])), vmax=np.nanmax(np.abs(scintillated_dynspec[0])))
    axs[4].set_xlabel("Time (samples)")
    axs[4].set_ylabel("Frequency (MHz)")

    plt.tight_layout()
    plt.show()


    plt.imshow(scintillated_dynspec[0], aspect='auto', interpolation='none', origin='lower', cmap='plasma', vmin=-np.nanmax(np.abs(scintillated_dynspec)), vmax=np.nanmax(np.abs(scintillated_dynspec)))
    plt.show()
    

    return scintillated_dynspec
