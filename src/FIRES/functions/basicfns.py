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
from scipy.ndimage import gaussian_filter1d
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
						nBits=32, showPlots=show_plots, debug=False, verbose=False, log=print, units='Jy/beam', prefixOut=os.path.join(outdir,"RM"), saveFigures=save, fit_function='log')
	
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
	
	windows = find_onpulse_matched_filter(np.nansum(dynspec[0], axis=0), width_bins=10, snr_thresh=3)
	if not windows:
		raise ValueError("No on-pulse window found for RM estimation.")
	left_window, right_window = windows[0]
		
   
	# Calculate the mean spectra for each Stokes parameter
	ispec   = np.nansum(dynspec[0, :, left_window:right_window], axis=1)
	vspec   = np.nansum(dynspec[3, :, left_window:right_window], axis=1)
	qspec0  = np.nansum(dynspec[1, :, left_window:right_window], axis=1)
	uspec0  = np.nansum(dynspec[2, :, left_window:right_window], axis=1)
	noispec = noisespec / np.sqrt(float(right_window + 1 - left_window))
		

	iquv  = (ispec, qspec0, uspec0, vspec)
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
	new_dynspec    = np.zeros(dynspec.shape, dtype=float)
	new_dynspec[0] = dynspec[0]
	new_dynspec[3] = dynspec[3]
	
	# Calculate the lambda squared array
	lambda_sq 		 = (speed_of_light_cgs * 1.0e-8 / freq_mhz) ** 2
	lambda_sq_median = np.nanmedian(lambda_sq)
		
	# Apply RM correction to Q and U spectra
	for ci in range(len(lambda_sq)):
		rot_angle = -2 * rm0 * (lambda_sq[ci] - lambda_sq_median)
		new_dynspec[1, ci] = dynspec[1, ci] * np.cos(rot_angle) - dynspec[2, ci] * np.sin(rot_angle)
		new_dynspec[2, ci] = dynspec[2, ci] * np.cos(rot_angle) + dynspec[1, ci] * np.sin(rot_angle)

	return new_dynspec


def est_profiles(dynspec, time_ms, noise_stokes, width_ms):
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
		lts  = np.sqrt(utsub ** 2 + qtsub ** 2)			
		elts = np.sqrt((qtsub * noise_stokes[1]) ** 2 + (utsub * noise_stokes[2]) ** 2) / lts
		# Calculate the total polarization intensity
		pts  = np.sqrt(lts ** 2 + vtsub ** 2)
		epts = np.sqrt((qtsub * noise_stokes[1]) ** 2 + (utsub * noise_stokes[2]) ** 2 + (vtsub * noise_stokes[3]) ** 2) / pts
  
		# Calculate the polarization angles
		phits  = np.rad2deg(0.5 * np.arctan2(utsub, qtsub))		
		dphits = np.rad2deg(0.5 * np.sqrt((utsub * noise_stokes[1]) ** 2 + (qtsub * noise_stokes[2]) ** 2) / (utsub ** 2 + qtsub ** 2))						
		psits  = np.rad2deg(0.5 * np.arctan2(vtsub, lts))		
		dpsits = np.rad2deg(0.5 * np.sqrt((vtsub * elts) ** 2 + (lts * noise_stokes[3]) ** 2) / (vtsub ** 2 + lts ** 2))
  
		
		
		# Calculate the fractional polarizations
		qfrac = qtsub / itsub
		ufrac = utsub / itsub
		vfrac = vtsub / itsub
  
		lfrac = lts / itsub
		pfrac = pts / itsub		
  
		# Set large errors to NaN
		mask = iquvt[0] < noise_stokes[0]
		phits[mask]  = np.nan
		dphits[mask] = np.nan
		psits[mask]  = np.nan
		dpsits[mask] = np.nan

		invalid_pa = ~np.isfinite(phits) | ~np.isfinite(dphits)
		phits[invalid_pa] = np.nan
		dphits[invalid_pa] = np.nan
  
		# Mask PA outside all signal windows using robust on-pulse finder
		try:
			t_res = time_ms[1] - time_ms[0]
			# Support width_ms as scalar or list/array
			if isinstance(width_ms, (list, np.ndarray)):
				windows = []
				for w in width_ms:
					width_bins = int(np.round(float(w) / t_res))
					windows.extend(find_onpulse_matched_filter(I, width_bins=width_bins, snr_thresh=1))
			else:
				width_bins = int(np.round(float(width_ms) / t_res))
				windows = find_onpulse_matched_filter(I, width_bins=width_bins, snr_thresh=1)
			pa_mask = np.zeros_like(phits, dtype=bool)
			for left, right in windows:
				pa_mask[left:right+1] = True
			phits[~pa_mask] = np.nan
			dphits[~pa_mask] = np.nan
		except Exception as e:
			print(f"Warning: find_onpulse_matched_filter failed: {e}")
			phits[:] = np.nan
			dphits[:] = np.nan
	
		evfrac = np.abs(vfrac) * np.sqrt((noise_stokes[3] / vtsub) ** 2 + (noise_stokes[0] / itsub) ** 2)
		eqfrac = np.abs(qfrac) * np.sqrt((noise_stokes[1] / qtsub) ** 2 + (noise_stokes[0] / itsub) ** 2)
		eufrac = np.abs(ufrac) * np.sqrt((noise_stokes[2] / utsub) ** 2 + (noise_stokes[0] / itsub) ** 2)
		elfrac = np.abs(lfrac) * np.sqrt((elts / lts) ** 2 + (noise_stokes[0] / itsub) ** 2)
		epfrac = np.abs(pfrac) * np.sqrt((epts / pts) ** 2 + (noise_stokes[0] / itsub) ** 2)
			
		# Return the time profiles as a frb_time_series object
	return frb_time_series(iquvt, lts, elts, pts, epts, phits, dphits, psits, dpsits, qfrac, eqfrac, ufrac, eufrac, vfrac, evfrac, lfrac, elfrac, pfrac, epfrac)


def est_spectra(dynspec, noisespec, left_window_ms, right_window_ms):
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
	lspec  = np.sqrt(uspec ** 2 + qspec ** 2)
	dlspec = np.sqrt((uspec * noispec0[2]) ** 2 + (qspec * noispec0[1]) ** 2) / lspec
	pspec  = np.sqrt(lspec ** 2 + vspec ** 2)
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
	dlfrac	  = np.sqrt((lspec * noispec0[0]) ** 2 + (ispec * dlspec) ** 2) / (ispec ** 2)
	pfracspec = pspec / ispec
	dpfrac 	  = np.sqrt((pspec * noispec0[0]) ** 2 + (ispec * dpspec) ** 2) / (ispec ** 2)

	# Calculate the polarization angles
	phispec  = np.rad2deg(0.5 * np.arctan2(uspec, qspec))		
	dphispec = np.rad2deg(0.5 * np.sqrt((uspec * noispec0[1]) ** 2 + (qspec * noispec0[2]) ** 2) / (uspec ** 2 + qspec ** 2))

	psispec  = np.rad2deg(0.5 * np.arctan2(vspec, lspec))		
	dpsispec = np.rad2deg(0.5 * np.sqrt((vspec * dlspec) ** 2 + (lspec * noispec0[2]) ** 2) / (vspec ** 2 + lspec ** 2))

	# Return the spectra as a frb_spectrum object
	return frb_spectrum(iquvspec, noispec0, lspec, dlspec, pspec, dpspec, qfracspec, dqfrac, ufracspec, dufrac, vfracspec, dvfrac, lfracspec, dlfrac, pfracspec, dpfrac, phispec, dphispec, psispec, dpsispec)




def process_dynspec(dynspec, frequency_mhz_array, time_ms_array, gdict):
    """
    Process the dynamic spectrum: RM correction, noise estimation, and profile extraction.
    """
    RM = gdict["RM"]
    width_ms = gdict["width_ms"]

    max_rm = RM[np.argmax(np.abs(RM))]
    corrdspec = rm_correct_dynspec(dynspec, frequency_mhz_array, max_rm)

    # Use Stokes I to find the off-pulse window
    I = np.nansum(corrdspec[0], axis=0)
    off_start_frac, off_end_frac = find_offpulse_window(I, window_frac=0.2)
    nbin = len(I)
    off_start = int(off_start_frac * nbin)
    off_end = int(off_end_frac * nbin)

    # Estimate noise in each Stokes parameter using the off-pulse window
    noisespec = np.zeros(corrdspec.shape[:1] + (corrdspec.shape[1],))
    for s in range(4):
        # shape: (nchan, ntime)
        noisespec[s] = np.nanstd(corrdspec[s, :, off_start:off_end], axis=1)
    noistks = np.sqrt(np.nanmean(noisespec**2, axis=1))

    tsdata = est_profiles(corrdspec, time_ms_array, noistks, width_ms=width_ms)

    return tsdata, corrdspec, noisespec, noistks


def find_onpulse_matched_filter(I, width_bins, snr_thresh=3):
	"""
	Find on-pulse windows using matched filtering and SNR threshold.
	Returns a list of (left, right) index tuples for all detected bursts.
	"""

	I = np.nan_to_num(I)
	# Smooth to suppress noise
	smoothed = gaussian_filter1d(I, sigma=width_bins/2)
	# Estimate noise from off-pulse
	noise = np.std(smoothed)
	snr = (smoothed - np.median(smoothed)) / (noise if noise > 0 else 1)
	above = snr > snr_thresh

	# Find contiguous regions above threshold
	windows = []
	in_burst = False
	for i, val in enumerate(above):
		if val and not in_burst:
			start = i
			in_burst = True
		elif not val and in_burst:
			end = i - 1
			windows.append((start, end))
			in_burst = False
	if in_burst:
		windows.append((start, len(above) - 1))
	return windows


def median_percentiles(yvals, x, ndigits=3):
	med_vals = []
	percentile_errs = []
	# Round all keys in yvals for consistent lookup
	vals_rounded = {round(float(k), ndigits): v for k, v in yvals.items()}
	for var in x:
		key = round(float(var), ndigits)
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


def weight_dict(x, yvals, weights_dict, ndigits=3):
	# Round all keys in yvals and weights_dict
	vals_rounded = {round(float(k), ndigits): v for k, v in yvals.items()}
	weights_rounded = {round(float(k), ndigits): v for k, v in weights_dict.items()}
	normalised_vals = {}
	for var in x:
		key = round(float(var), ndigits)
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



def find_offpulse_window(profile, window_frac=0.2):
	"""
	Find the off-pulse window in a 1D profile by sliding a window and finding the minimum mean.
	Args:
		profile: 1D numpy array (summed over frequency and Stokes if needed)
		window_frac: Fractional width of the window (e.g., 0.2 for 20%)
	Returns:
		off_start_frac, off_end_frac: Start and end as fractions (0-1)
	"""
	nbin = len(profile)
	win_size = int(window_frac * nbin)
	min_mean = np.inf
	min_start = 0
	for i in range(nbin - win_size + 1):
		win_mean = np.mean(profile[i:i+win_size])
		if win_mean < min_mean:
			min_mean = win_mean
			min_start = i
	off_start_frac = min_start / nbin
	off_end_frac = (min_start + win_size) / nbin
	return off_start_frac, off_end_frac

