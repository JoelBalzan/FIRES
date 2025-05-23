# -----------------------------------------------------------------------------
# basicfns.py
# FIRES: The Fast, Intense Radio Emission Simulator
#
# This module provides core processing functions for the FIRES simulation
# pipeline, including RM synthesis, RM correction, profile and spectra
# estimation, dynamic spectrum processing, window estimation, and utility
# routines for handling floating point dictionary keys and noise addition.
#
# Author: JB
# Date: 2025-05-20
# -----------------------------------------------------------------------------

#	--------------------------	Import modules	---------------------------

import os
import sys

import numpy as np
from scipy.signal import fftconvolve
from RMtools_1D.do_RMclean_1D import run_rmclean
from RMtools_1D.do_RMsynth_1D import run_rmsynth
from FIRES.utils.utils import *

#	---------------------------------------------------------------------------------

def rm_synth(freq_ghz, iquv, diquv, outdir, save, show_plots):
	"""
	Determine RM using RM synthesis with RMtool.
	Inputs:
		- freq_ghz: Frequencies in GHz
		- iquv: I Q U V spectrum
		- diquv: I Q U V noise spectrum
	Returns:
		- res: List containing RM, RM error, polarization angle, and polarization angle error
	"""
	
	# Prepare the data for RM synthesis
	rm_data = np.array([freq_ghz * 1.0e9, iquv[0], iquv[1], iquv[2], diquv[0], diquv[1], diquv[2]])
	
	# Run RM synthesis
	rm_synth_data, rm_synth_ad = run_rmsynth(rm_data, polyOrd=3, phiMax_radm2=1.0e3, dPhi_radm2=1.0, nSamples=100.0, weightType='variance', fitRMSF=False, noStokesI=False, phiNoise_radm2=1000000.0, \
						nBits=32, showPlots=show_plots, debug=False, verbose=False, log=print, units='Jy/beam', prefixOut=os.path.join(outdir,"rm"), saveFigures=save, fit_function='log')
	
	# Run RM clean
	rm_clean_data = run_rmclean(rm_synth_data, rm_synth_ad, 0.1, maxIter=1000, gain=0.1, nBits=32, showPlots=show_plots, verbose=False, log=print)
	
	#print(rm_clean_data[0])
	
	# Extract results
	res = [rm_clean_data[0]['phiPeakPIfit_rm2'], rm_clean_data[0]['dPhiPeakPIfit_rm2'], rm_clean_data[0]['polAngle0Fit_deg'], rm_clean_data[0]['dPolAngle0Fit_deg']]
	
	return res


def estimate_rm(dynspec, freq_mhz, time_ms, noisespec, phi_range, dphi, outdir, save, show_plots):
	"""
	Estimate rotation measure.
	Inputs:
		- dynspec: Dynamic spectrum array
		- freq_mhz: Frequency array in MHz
		- time_ms: Time array in ms
		- noisespec: Noise spectrum
		- left_window_ms: Left window in ms for RM estimation
		- right_window_ms: Right window in ms for RM estimation
		- phi_range: Range of RM values to search
		- dphi: Step size for RM search
		- start_chan: Starting channel index
		- end_chan: Ending channel index
	Returns:
		- res_rmtool: List containing RM, RM error, polarization angle, and polarization angle error
	"""
	
	left_window, right_window = estimate_windows(np.nansum(dynspec[0], axis=0), time_ms, threshold=0.1)
		
   
	# Calculate the mean spectra for each Stokes parameter
	ispec  = np.nansum(dynspec[0, :, left_window:right_window], axis=1)
	vspec  = np.nansum(dynspec[3, :, left_window:right_window], axis=1)
	qspec0 = np.nansum(dynspec[1, :, left_window:right_window], axis=1)
	uspec0 = np.nansum(dynspec[2, :, left_window:right_window], axis=1)
	noispec = noisespec / np.sqrt(float(right_window + 1 - left_window))	
		

	iquv = (ispec, qspec0, uspec0, vspec)
	eiquv = (noispec[0], noispec[1], noispec[2], noispec[3])
		
	# Run RM synthesis
	res_rmtool = rm_synth(freq_mhz / 1.0e3, iquv, eiquv, outdir, save, show_plots)
		
	print("\nResults from RMtool (RM synthesis) \n")
	print("RM = %.2f +/- %.2f rad/m2   PolAng0 = %.2f +/- %.2f deg\n" % (res_rmtool[0], res_rmtool[1], res_rmtool[2], res_rmtool[3]))
	
	return res_rmtool


def rm_correct_dynspec(dynspec, freq_mhz, rm0):
	"""
	Generate RM corrected dynamic spectrum.
	Inputs:
		- dynspec: Dynamic spectrum array
		- freq_mhz: Frequency array in MHz
		- rm0: Rotation measure to correct for
	Returns:
		- new_dynspec: RM corrected dynamic spectrum
	"""
	
	# Initialize the new dynamic spectrum
	new_dynspec = np.zeros(dynspec.shape, dtype=float)
	new_dynspec[0] = dynspec[0]
	new_dynspec[3] = dynspec[3]
	
	# Calculate the lambda squared array
	lambda_sq = (speed_of_light_cgs * 1.0e-8 / freq_mhz) ** 2
	lambda_sq_median = np.nanmedian(lambda_sq)
		
	# Apply RM correction to Q and U spectra
	for ci in range(len(lambda_sq)):
		rot_angle = -2 * rm0 * (lambda_sq[ci] - lambda_sq_median)
		new_dynspec[1, ci] = dynspec[1, ci] * np.cos(rot_angle) - dynspec[2, ci] * np.sin(rot_angle)
		new_dynspec[2, ci] = dynspec[2, ci] * np.cos(rot_angle) + dynspec[1, ci] * np.sin(rot_angle)

	return new_dynspec


def est_profiles(dynspec, freq_mhz, time_ms, noise_stokes):
	"""
	Estimate time profiles.
	Inputs:
		- dynspec: Dynamic spectrum array
		- freq_mhz: Frequency array in MHz
		- time_ms: Time array in ms
		- noisespec: Noise spectrum
		- start_chan: Starting channel index
		- end_chan: Ending channel index
	Returns:
		- frb_time_series: Object containing time profiles
	"""
	with np.errstate(invalid='ignore', divide='ignore', over='ignore'):

		iquvt = np.nansum(dynspec, axis=1)
  	
		I = iquvt[0]
  
		threshold = 0.05 * np.nanmax(I)
		mask = I <= threshold
  
		itsub = np.where(mask, np.nan, iquvt[0])
		qtsub = np.where(mask, np.nan, iquvt[1])
		utsub = np.where(mask, np.nan, iquvt[2])
		vtsub = np.where(mask, np.nan, iquvt[3])
		
		# Calculate the linear polarization intensity
		lts = np.sqrt(utsub ** 2 + qtsub ** 2)			
		elts = np.sqrt((qtsub * noise_stokes[1]) ** 2 + (utsub * noise_stokes[2]) ** 2) / lts
		# Calculate the total polarization intensity
		pts = np.sqrt(lts ** 2 + vtsub ** 2)
		epts = np.sqrt((qtsub * noise_stokes[1]) ** 2 + (utsub * noise_stokes[2]) ** 2 + (vtsub * noise_stokes[3]) ** 2) / pts
  
		# Calculate the polarization angles
		phits = np.rad2deg(0.5 * np.arctan2(utsub, qtsub))		
		dphits = np.rad2deg(0.5 * np.sqrt((utsub * noise_stokes[1]) ** 2 + (qtsub * noise_stokes[2]) ** 2) / (utsub ** 2 + qtsub ** 2))						
		psits = np.rad2deg(0.5 * np.arctan2(vtsub, lts))		
		dpsits = np.rad2deg(0.5 * np.sqrt((vtsub * elts) ** 2 + (lts * noise_stokes[3]) ** 2) / (vtsub ** 2 + lts ** 2))
  
		
		
		# Calculate the fractional polarizations
		qfrac = qtsub / itsub
		ufrac = utsub / itsub
		vfrac = vtsub / itsub
  
		lfrac = lts / itsub
		pfrac = pts / itsub		
  
		# Set large errors to NaN
		mask = iquvt[0] < noise_stokes[0]
		phits[mask] = np.nan
		dphits[mask] = np.nan
		psits[mask] = np.nan
		dpsits[mask] = np.nan
	
		evfrac = np.abs(vfrac) * np.sqrt((noise_stokes[3] / vtsub) ** 2 + (noise_stokes[0] / itsub) ** 2)
		eqfrac = np.abs(qfrac) * np.sqrt((noise_stokes[1] / qtsub) ** 2 + (noise_stokes[0] / itsub) ** 2)
		eufrac = np.abs(ufrac) * np.sqrt((noise_stokes[2] / utsub) ** 2 + (noise_stokes[0] / itsub) ** 2)
		elfrac = np.abs(lfrac) * np.sqrt((elts / lts) ** 2 + (noise_stokes[0] / itsub) ** 2)
		epfrac = np.abs(pfrac) * np.sqrt((epts / pts) ** 2 + (noise_stokes[0] / itsub) ** 2)
			
		# Return the time profiles as a frb_time_series object
	return frb_time_series(iquvt, lts, elts, pts, epts, phits, dphits, psits, dpsits, qfrac, eqfrac, ufrac, eufrac, vfrac, evfrac, lfrac, elfrac, pfrac, epfrac)


def est_spectra(dynspec, freq_mhz, time_ms, noisespec, left_window_ms, right_window_ms):
	"""
	Estimate spectra.
	Inputs:
		- dynspec: Dynamic spectrum array
		- freq_mhz: Frequency array in MHz
		- time_ms: Time array in ms
		- noisespec: Noise spectrum
		- left_window_ms: Left window in ms for spectra estimation
		- right_window_ms: Right window in ms for spectra estimation
	Returns:
		- frb_spectrum: Object containing spectra
	"""
	 
	# Average the dynamic spectrum over the specified time range
	iquvspec = np.nansum(dynspec[:, :, left_window_ms:right_window_ms + 1], axis=2)
	
	# Extract the Stokes parameters
	ispec = iquvspec[0]
	vspec = iquvspec[3]
	qspec = iquvspec[1]
	uspec = iquvspec[2]		
	
	# Calculate the noise for each Stokes parameter
	noispec0 = noisespec / np.sqrt(float(right_window_ms + 1 - left_window_ms))
	# Calculate the linear polarization intensity
	lspec = np.sqrt(uspec ** 2 + qspec ** 2)
	# Calculate the error in linear polarization intensity
	dlspec = np.sqrt((uspec * noispec0[2]) ** 2 + (qspec * noispec0[1]) ** 2) / lspec
	# Calculate the total polarization intensity
	pspec = np.sqrt(lspec ** 2 + vspec ** 2)
	# Calculate the error in total polarization intensity
	dpspec = np.sqrt((vspec * dlspec) ** 2 + (lspec * noispec0[3]) ** 2) / pspec

	# Calculate the fractional polarizations
	qfracspec = qspec / ispec
	ufracspec = uspec / ispec
	vfracspec = vspec / ispec
	# Calculate the errors in fractional polarizations
	dqfrac = np.sqrt((qspec * noispec0[0]) ** 2 + (ispec * noispec0[1]) ** 2) / (ispec ** 2)
	dufrac = np.sqrt((uspec * noispec0[0]) ** 2 + (ispec * noispec0[2]) ** 2) / (ispec ** 2)
	dvfrac = np.sqrt((vspec * noispec0[0]) ** 2 + (ispec * noispec0[3]) ** 2) / (ispec ** 2)

	# Calculate the fractional linear and total polarizations
	lfracspec = lspec / ispec
	dlfrac = np.sqrt((lspec * noispec0[0]) ** 2 + (ispec * dlspec) ** 2) / (ispec ** 2)
	pfracspec = pspec / ispec
	dpfrac = np.sqrt((pspec * noispec0[0]) ** 2 + (ispec * dpspec) ** 2) / (ispec ** 2)

	# Calculate the polarization angles
	phispec = np.rad2deg(0.5 * np.arctan2(uspec, qspec))		
	dphispec = np.rad2deg(0.5 * np.sqrt((uspec * noispec0[1]) ** 2 + (qspec * noispec0[2]) ** 2) / (uspec ** 2 + qspec ** 2))

	psispec = np.rad2deg(0.5 * np.arctan2(vspec, lspec))		
	dpsispec = np.rad2deg(0.5 * np.sqrt((vspec * dlspec) ** 2 + (lspec * noispec0[2]) ** 2) / (vspec ** 2 + lspec ** 2))

	# Return the spectra as a frb_spectrum object
	return frb_spectrum(iquvspec, noispec0, lspec, dlspec, pspec, dpspec, qfracspec, dqfrac, ufracspec, dufrac, vfracspec, dvfrac, lfracspec, dlfrac, pfracspec, dpfrac, phispec, dphispec, psispec, dpsispec)




def process_dynspec(dynspec, frequency_mhz_array, time_ms_array, rm):
	"""
	Process the dynamic spectrum: RM correction, noise estimation, and profile extraction.
	"""

	max_rm = rm[np.argmax(np.abs(rm))]
	
	corrdspec = rm_correct_dynspec(dynspec, frequency_mhz_array, max_rm)

	I = np.nansum(corrdspec[0], axis=0)

	threshold = 0.05 * np.nanmax(I)
	mask = I >= threshold
	noise_region = np.where(mask, np.nan, corrdspec)  # Mask signal regions with NaN

	noisespec = np.nanstd(noise_region, axis=2)
	noistks = np.sqrt(np.nanmean(noisespec**2, axis=1))

	tsdata = est_profiles(corrdspec, frequency_mhz_array, time_ms_array, noistks)
 
	return tsdata, corrdspec, noisespec, noistks


def estimate_windows(itsub, time_ms, threshold=0.1):
	"""
	Estimate left_window and right_window based on the total intensity profile.

	Args:
		itsub (array): Total intensity profile (1D array).
		time_ms (array): Time array in milliseconds (1D array).
		threshold (float): Fraction of the peak intensity to define the window.

	Returns:
		tuple: (left_window, right_window) indices.
	"""
	# Normalize the intensity profile
	normalized_intensity = itsub / np.nanmax(itsub)

	# Find indices where intensity exceeds the threshold
	significant_indices = np.where(normalized_intensity > threshold)[0]

	if len(significant_indices) == 0:
		raise ValueError("No significant intensity found above the threshold.")

	# Determine the left and right window indices
	left_window = significant_indices[0]
	right_window = significant_indices[-1]

	# Convert indices to time values if needed
	left_time = time_ms[left_window]
	right_time = time_ms[right_window]

	print(f"RM: Estimated left_window: {left_time} ms, right_window: {right_time} ms")
	return left_window, right_window


def median_percentiles(vals, scatter_ms, ndigits=3):
	med_vals = []
	percentile_errs = []
	# Round all keys in vals for consistent lookup
	vals_rounded = {round(float(k), ndigits): v for k, v in vals.items()}
	for s_val in scatter_ms:
		key = round(float(s_val), ndigits)
		v = vals_rounded.get(key, None)
		if v is not None and isinstance(v, (list, np.ndarray)) and len(v) > 0:
			median_val = np.median(v)
			lower_percentile = np.percentile(v, 16)
			upper_percentile = np.percentile(v, 84)
			med_vals.append(median_val)
			percentile_errs.append((lower_percentile, upper_percentile))
		else:
			med_vals.append(np.nan)
			percentile_errs.append((np.nan, np.nan))
	return med_vals, percentile_errs


def weight_dict(scatter_ms, vals, weights_dict, ndigits=3):
	# Round all keys in vals and weights_dict
	vals_rounded = {round(float(k), ndigits): v for k, v in vals.items()}
	weights_rounded = {round(float(k), ndigits): v for k, v in weights_dict.items()}
	normalised_vals = {}
	for s_val in scatter_ms:
		key = round(float(s_val), ndigits)
		if key in vals_rounded and key in weights_rounded:
			normalised_vals[key] = [val / pa if pa != 0 else 0 for val, pa in zip(vals_rounded[key], weights_rounded[key])]
		else:
			normalised_vals[key] = None  # or handle missing keys as needed
	return normalised_vals
	
 
def scatter_stokes_chan(chan, freq_mhz, time_ms, tau_ms, sc_idx, ref_freq_mhz):
	"""
	Apply scattering to chan using a causal exponential IRF,
	with padding to prevent boundary artifacts.

	Inputs:
		- stokes_I: 1D array of chan (len(time_ms))
		- freq_mhz: Channel frequency in MHz
		- time_ms: 1D array of time values in ms (uniformly spaced)
		- tau_ms: Reference scattering timescale (ms) at ref_freq_mhz
		- sc_idx: Scattering index (e.g. -4)
		- ref_freq_mhz: Reference frequency in MHz

	Returns:
		- sc_stokes_I: Scattered chan (same shape as input)
		- tau_cms: Scattering timescale at freq_mhz
	"""
	# Calculate frequency-dependent scattering timescale
	tau_cms = tau_ms * (freq_mhz / ref_freq_mhz) ** sc_idx

	# Time resolution
	dt = time_ms[1] - time_ms[0]

	# Pad to cover tail (~5 tau)
	n_pad = int(np.ceil(5 * tau_cms / dt))
	padded_I = np.pad(chan, (0, n_pad), mode='constant')  # Pad only at end

	# Create IRF time axis
	irf_t = np.arange(0, (n_pad + 1)) * dt
	irf = np.exp(-irf_t / tau_cms)
	irf /= np.sum(irf)  # Normalize

	# Convolve and trim back to original size
	convolved = fftconvolve(padded_I, irf, mode='full')
	sc_chan = convolved[:len(chan)]

	return sc_chan



def add_noise_to_dynspec(data, desired_snr, bandwidth_mhz, width_ds,
                                     tsys_k=75.0, gain_k_jy=0.3, npol=2, 
                                     use_radiometer_floor=True):
    """
    Add noise with optional radiometer equation floor constraint.
    
    Parameters:
    -----------
    data : ndarray, shape (4, nchan, ntime)
        Input IQUV dynamic spectrum data
    desired_snr : float
        Desired SNR in the final pulse profile
    bandwidth_mhz : float
        Total bandwidth in MHz
    width_ds : float
        Time width of first gaussian envelope in bins
    tsys_k : float, optional
        System temperature in Kelvin
    gain_k_jy : float, optional
        System gain in K/Jy
    npol : int, optional
        Number of polarizations
    use_radiometer_floor : bool, optional
        If True, ensures noise is at least as high as radiometer equation predicts
    
    Returns:
    --------
    noisy_data : ndarray, shape (4, nchan, ntime)
        IQUV data with added noise
    profile_snr_achieved : float
        Actual SNR achieved in the summed profile
    noise_std_used : float
        Actual noise standard deviation used per channel
    radiometer_noise : float
        Theoretical noise from radiometer equation
    """
    
    nstokes, nchan, ntime = data.shape
    
    # Calculate radiometer noise per channel

    bandwidth_mhz = bandwidth_mhz * 1e6  # Convert MHz to Hz
    #radiometer_noise = tsys_k / (gain_k_jy * np.sqrt(npol * bandwidth_mhz * width_ds))
    radiometer_noise = tsys_k / (np.sqrt(bandwidth_mhz * width_ds))
    
    # Calculate required noise for target SNR (as before)
    stokes_i_profile = np.sum(data[0], axis=0)
    signal_peak_profile = np.max(stokes_i_profile)
    signal_peak_per_channel = signal_peak_profile / nchan
    
    snr_per_channel_needed = desired_snr / np.sqrt(nchan)
    noise_from_snr = signal_peak_per_channel / snr_per_channel_needed
    
    # Use the higher of the two noise levels
    if use_radiometer_floor:
        noise_std_used = max(noise_from_snr, radiometer_noise)
    else:
        noise_std_used = noise_from_snr
    
    # Generate and add noise
    noise = np.random.normal(0, noise_std_used, data.shape)
    noisy_data = data + noise

    
    return noisy_data