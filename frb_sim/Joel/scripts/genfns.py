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
 
def gauss_dynspec(fmhzarr, tmsarr, df_mhz, dtms, specind, peak, wms, locms, dmpccc, pa, l, v, dpadt, rm):
    """
    Generate dynamic spectrum for Gaussian pulses.
    Inputs:
        - fmhzarr: Frequency array in MHz
        - tmsarr: Time array in ms
        - df_mhz: Frequency resolution in MHz
        - dtms: Time resolution in ms
        - specind: Spectral index array
        - peak: Peak amplitude array
        - wms: Width of the Gaussian pulse in ms
        - locms: Location of the Gaussian pulse in ms
        - dmpccc: Dispersion measure in pc/cm^3
        - pa: Polarization angle array
        - l: Linear polarization fraction array
        - v: Circular polarization fraction array
        - dpadt: Change in polarization angle with time
        - rm: Rotation measure array
    """
    gpdspec	=	np.zeros((4, fmhzarr.shape[0], tmsarr.shape[0]), dtype=float)  # Initialize dynamic spectrum array
    ngp		=	len(specind) - 1  # Number of Gaussian components
    plsarr	=	np.zeros(len(tmsarr), dtype=float)  # Initialize pulse array
    fmhzref	=	np.nanmedian(fmhzarr)  # Reference frequency
    lm2arr	=	(speed_of_light_cgs*1.0e-8 / fmhzarr)**2  # Lambda squared array
    lm20 	= 	np.nanmedian(lm2arr)  # Median lambda squared

    for g in range(0, ngp):
        # Calculate the normalized amplitude for each frequency
        nrmarr	=	peak[g+1] * (fmhzarr / np.nanmedian(fmhzarr)) ** specind[g+1]
        # Calculate the Gaussian pulse shape
        plsarr	=	np.exp(-(tmsarr - locms[g+1]) ** 2 / (2 * (wms[g+1] ** 2)))
        # Calculate the polarization angle array
        pa_arr 	= 	pa[g+1] + (tmsarr - locms[g+1]) * dpadt[g+1]
        
        for c in range(0, len(fmhzarr)):
            # Apply Faraday rotation
            pa_farr = pa_arr + rm[g+1] * (lm2arr[c] - lm20)
            # Add the Gaussian pulse to the dynamic spectrum
            gpdspec[0, c] = gpdspec[0, c] + nrmarr[c] * plsarr
            # Calculate the dispersion delay
            disdelms = 4.15 * dmpccc[g+1] * ((1.0e3 / fmhzarr[c]) ** 2 - (1.0e3 / fmhzref) ** 2)
            # Apply the dispersion delay
            gpdspec[0, c] = np.roll(gpdspec[0, c], int(np.round(disdelms / dtms)))
            # Calculate the Stokes parameters
            gpdspec[1, c] = gpdspec[0, c] * l[g+1] * np.cos(2 * pa_farr)  # Q
            gpdspec[2, c] = gpdspec[0, c] * l[g+1] * np.sin(2 * pa_farr)  # U
            gpdspec[3, c] = gpdspec[0, c] * v[g+1]  # V
                        
    print("\nGenerating dynamic spectrum with %d Gaussian component(s)\n" % (ngp))
    plt.imshow(gpdspec[0, :], aspect='auto', interpolation='none', origin='lower', cmap='seismic', vmin=-np.nanmax(np.abs(gpdspec)), vmax=np.nanmax(np.abs(gpdspec)))
    plt.show()
    
    return gpdspec

#	--------------------------------------------------------------------------------

def scatter_dynspec(dspec, fmhzarr, tmsarr, df_mhz, dtms, taums, scindex):
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
    
    fig, axs = plt.subplots(5, figsize=(10, 6))
    fig.suptitle('Scattered Dynamic Spectrum')
    
    # Plot the mean across all frequency channels (axis 0)
    axs[0].plot(np.nanmean(scdspec[0,:], axis=0), markersize=2 ,label='I', color='Black')
    axs[0].plot(np.nanmean(scdspec[1,:], axis=0), markersize=2, label='Q')
    axs[0].plot(np.nanmean(scdspec[2,:], axis=0), markersize=2, label='U')
    axs[0].plot(np.nanmean(scdspec[3,:], axis=0), markersize=2, label='V', color='Red')
    axs[0].plot(L, markersize=2, label='L', color='Blue')
    axs[0].set_title("Mean Scattered Signal over Time")
    axs[0].legend(loc='upper right')

    # Plot the 2D scattered dynamic spectrum
    axs[1].imshow(scdspec[0], aspect='auto', interpolation='none', origin='lower', cmap='plasma',
        vmin=-np.nanmax(np.abs(dspec[0])), vmax=np.nanmax(np.abs(dspec[0]))*5)
    axs[1].set_title("Mean Scattered Dynamic Spectrum Across Frequency Channels")

    axs[2].imshow(scdspec[1], aspect='auto', interpolation='none', origin='lower', cmap='plasma',
        vmin=-np.nanmax(np.abs(dspec[0])), vmax=np.nanmax(np.abs(dspec[0]))*2.5)

    axs[3].imshow(scdspec[2], aspect='auto', interpolation='none', origin='lower', cmap='plasma',
        vmin=-np.nanmax(np.abs(dspec[0])), vmax=np.nanmax(np.abs(dspec[0]))*2.5)

    axs[4].imshow(scdspec[3], aspect='auto', interpolation='none', origin='lower', cmap='plasma',
           vmin=-np.nanmax(np.abs(dspec[0])), vmax=np.nanmax(np.abs(dspec[0]))*5)
    axs[4].set_xlabel("Time (samples)")
    axs[4].set_ylabel("Frequency (MHz)")

    plt.tight_layout()
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
