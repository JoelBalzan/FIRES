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
	Perform rotation measure (RM) synthesis using RMtools to determine the RM and polarization properties.

	Parameters:
	-----------
	freq_ghz : array_like
		Observation frequencies in GHz
	iquv : tuple of arrays
		Stokes parameters (I, Q, U, V) spectra in flux density units
	diquv : tuple of arrays  
		Uncertainties/noise in Stokes parameters (dI, dQ, dU, dV) spectra
	outdir : str
		Output directory path for saving results and plots
	save : bool
		Whether to save output figures and data files
	show_plots : bool
		Whether to display plots during processing
		
	Returns:
	--------
	list
		Four-element list containing:
		[0] RM value in rad/m²
		[1] RM uncertainty in rad/m²
		[2] Intrinsic polarization angle at λ²=0 in degrees
		[3] Uncertainty in polarization angle in degrees
		
	Notes:
	------
	- Uses polynomial order 3 for fitting
	- Searches RM space up to ±1000 rad/m² with 1 rad/m² resolution
	- Applies RM CLEAN with 0.1 threshold to remove sidelobes
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
	Estimate the rotation measure (RM) of an FRB from its dynamic spectrum using RM synthesis.

	Parameters:
	-----------
	dynspec : ndarray, shape (4, n_freq, n_time)
		4D dynamic spectrum array with Stokes I, Q, U, V
	freq_mhz : array_like
		Frequency channels in MHz
	time_ms : array_like  
		Time samples in milliseconds
	noisespec : array_like, shape (4, n_freq)
		Noise levels for each Stokes parameter and frequency channel
	phi_range : float
		Maximum RM range to search (±phi_range rad/m²) - currently unused
	dphi : float
		RM step size for search grid - currently unused  
	outdir : str
		Output directory for saving RM synthesis results
	save : bool
		Whether to save diagnostic plots and data
	show_plots : bool
		Whether to display plots during processing
		
	Returns:
	--------
	list
		RM synthesis results: [RM, dRM, pol_angle_0, dpol_angle_0]
		- RM: Rotation measure in rad/m²
		- dRM: Uncertainty in RM in rad/m²  
		- pol_angle_0: Intrinsic polarization angle in degrees
		- dpol_angle_0: Uncertainty in polarization angle in degrees
	"""


	w95_ms, left, right = boxcar_width(np.nansum(dynspec[0], axis=0), time_ms, frac=0.95)

	# Calculate the mean spectra for each Stokes parameter
	ispec   = np.nansum(dynspec[0, :, left:right], axis=1)
	vspec   = np.nansum(dynspec[3, :, left:right], axis=1)
	qspec0  = np.nansum(dynspec[1, :, left:right], axis=1)
	uspec0  = np.nansum(dynspec[2, :, left:right], axis=1)
	noispec = noisespec / np.sqrt(float(right + 1 - left))


	iquv  = (ispec, qspec0, uspec0, vspec)
	eiquv = (noispec[0], noispec[1], noispec[2], noispec[3])
		
	# Run RM synthesis
	res_rmtool = rm_synth(freq_mhz / 1.0e3, iquv, eiquv, outdir, save, show_plots)
		
	print("\nResults from RMtool (RM synthesis) \n")
	print("RM = %.2f +/- %.2f rad/m2   PolAng0 = %.2f +/- %.2f deg\n" % (res_rmtool[0], res_rmtool[1], res_rmtool[2], res_rmtool[3]))
	
	return res_rmtool


def rm_correct_dynspec(dynspec, freq_mhz, rm0):
	"""
	Apply rotation measure correction to remove Faraday rotation from a dynamic spectrum.
 
	Parameters:
	-----------
	dynspec : ndarray, shape (4, n_freq, n_time)
		Input dynamic spectrum with Stokes I, Q, U, V
	freq_mhz : array_like
		Frequency channels in MHz
	rm0 : float
		Rotation measure to correct for in rad/m²
		
	Returns:
	--------
	ndarray, shape (4, n_freq, n_time)
		RM-corrected dynamic spectrum where:
		- Stokes I and V are unchanged (not affected by Faraday rotation)
		- Stokes Q and U are rotated to remove Faraday rotation effects
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


def est_profiles(dynspec, time_ms, noise_stokes, left, right):
	"""
	Extract and analyze time-resolved polarization profiles from a dynamic spectrum.
	
	Parameters:
	-----------
	dynspec : ndarray, shape (4, n_freq, n_time)  
		Dynamic spectrum with Stokes I, Q, U, V
	time_ms : array_like
		Time axis in milliseconds
	noise_stokes : array_like, shape (4,)
		RMS noise levels for each Stokes parameter
	left : int
		Starting time bin index for integration window
	right : int
		Ending time bin index for integration window
		
	Returns:
	--------
	frb_time_series
		Object containing time-resolved polarization measurements:
		- iquvt: Time series of I, Q, U, V
		- lts, elts: Linear polarization intensity and error
		- pts, epts: Total polarization intensity and error  
		- phits, dphits: Linear polarization angle and error (degrees)
		- psits, dpsits: Circular polarization angle and error (degrees)
		- Fractional polarizations: qfrac, ufrac, vfrac, lfrac, pfrac with errors
	"""

	with np.errstate(invalid='ignore', divide='ignore', over='ignore'):

		iquvt = np.nansum(dynspec, axis=1)

		itsub = iquvt[0]
		qtsub = iquvt[1]
		utsub = iquvt[2]
		vtsub = iquvt[3]
		
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
  
		# Calculate the errors in fractional polarizations
		evfrac = np.abs(vfrac) * np.sqrt((noise_stokes[3] / vtsub) ** 2 + (noise_stokes[0] / itsub) ** 2)
		eqfrac = np.abs(qfrac) * np.sqrt((noise_stokes[1] / qtsub) ** 2 + (noise_stokes[0] / itsub) ** 2)
		eufrac = np.abs(ufrac) * np.sqrt((noise_stokes[2] / utsub) ** 2 + (noise_stokes[0] / itsub) ** 2)
		elfrac = np.abs(lfrac) * np.sqrt((elts / lts) ** 2 + (noise_stokes[0] / itsub) ** 2)
		epfrac = np.abs(pfrac) * np.sqrt((epts / pts) ** 2 + (noise_stokes[0] / itsub) ** 2)  
  
  
		# Set large errors to NaN
		mask = itsub < noise_stokes[0]
		phits[mask]  = np.nan
		dphits[mask] = np.nan
		psits[mask]  = np.nan
		dpsits[mask] = np.nan

		# Mask PA where linear polarization is not significant
		#lts_noise = np.sqrt(noise_stokes[1]**2 + noise_stokes[2]**2)
		#snr_L = lts / lts_noise
		#snr_threshold = 50.0  # or 5.0 for stricter masking
#
		#low_snr_mask = snr_L < snr_threshold
		#phits[low_snr_mask] = np.nan
		#dphits[low_snr_mask] = np.nan
  
		# Mask PA outside all signal windows using on-pulse finder
		pa_mask = np.zeros_like(phits, dtype=bool)
		pa_mask[left:right+1] = True
		phits[~pa_mask] = np.nan
		dphits[~pa_mask] = np.nan

		# Return the time profiles as a frb_time_series object
	return frb_time_series(iquvt, lts, elts, pts, epts, phits, dphits, psits, dpsits, qfrac, eqfrac, ufrac, eufrac, vfrac, evfrac, lfrac, elfrac, pfrac, epfrac)


def est_spectra(dynspec, noisespec, left_window_ms, right_window_ms):
	"""
	Extract frequency-resolved polarization spectra by integrating over a specified time window.

	Parameters:
	-----------
	dynspec : ndarray, shape (4, n_freq, n_time)
		Dynamic spectrum containing Stokes I, Q, U, V  
	noisespec : array_like, shape (4, n_freq)
		Noise spectrum for each Stokes parameter and frequency channel
	left_window_ms : int
		Starting time bin index for integration window
	right_window_ms : int  
		Ending time bin index for integration window
		
	Returns:
	--------
	frb_spectrum
		Object containing frequency-resolved polarization measurements:
		- iquvspec: Integrated Stokes I, Q, U, V spectra
		- noispec0: Noise spectra scaled for integration time
		- lspec, dlspec: Linear polarization intensity and error vs frequency
		- pspec, dpspec: Total polarization intensity and error vs frequency
		- Fractional polarization spectra with errors: qfracspec, ufracspec, etc.
		- phispec, dphispec: Linear polarization angle and error vs frequency  
		- psispec, dpsispec: Circular polarization angle and error vs frequency
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




def process_dynspec(dynspec, freq_mhz, time_ms, gdict, tau_ms):
	"""
	Complete pipeline for processing FRB dynamic spectra: RM correction, noise estimation, and profile extraction.
	
	Parameters:
	-----------
	dynspec : ndarray, shape (4, n_freq, n_time)
		Input dynamic spectrum with Stokes I, Q, U, V
	freq_mhz : array_like
		Frequency channels in MHz
	time_ms : array_like
		Time samples in milliseconds  
	gdict : dict
		Dictionary containing analysis parameters, must include:
		- "RM": array of rotation measure values (rad/m²)
	tau_ms : float
		Pulse width parameter (currently unused in this function)
		
	Returns:
	--------
	tuple
		Four-element tuple containing:
		- tsdata: frb_time_series object with time-resolved polarization profiles
		- corrdspec: RM-corrected dynamic spectrum  
		- noisespec: Noise spectrum for each Stokes parameter and frequency
		- noise_stokes: Average noise levels for each Stokes parameter
	"""
	RM = gdict["RM"]

	max_rm = RM[np.argmax(np.abs(RM))]
	if max_rm > 0:
		corrdspec = rm_correct_dynspec(dynspec, freq_mhz, max_rm)
	else:
		corrdspec = dynspec.copy()

	# Use Stokes I to find the on-pulse window
	I = np.nansum(corrdspec[0], axis=0)
	_, left, right = boxcar_width(I, time_ms, frac=0.95)

	# Estimate noise in each Stokes parameter using off-pulse region
	offpulse_mask = np.ones(I.shape, dtype=bool)
	offpulse_mask[left:right+1] = False  # Exclude on-pulse region

	npol = corrdspec.shape[0]
	noise_stokes = np.zeros(npol)
	for s in range(npol):
		# Flatten all off-pulse samples across all channels for this Stokes parameter
		offpulse_samples = corrdspec[s, :, offpulse_mask].ravel()
		noise_stokes[s] = np.nanstd(offpulse_samples)

	noisespec = np.nanstd(corrdspec[:, :, offpulse_mask], axis=2)

	tsdata = est_profiles(corrdspec, time_ms, noise_stokes, left, right)

	return tsdata, corrdspec, noisespec, noise_stokes



def boxcar_width(profile, time_ms, frac=0.95):
	"""
	Find the minimum contiguous time window that contains a specified fraction of the total burst energy.
	
	Parameters:
	-----------
	profile : array_like
		1D intensity profile (typically integrated over frequency)
	time_ms : array_like  
		Time axis corresponding to profile bins, in milliseconds
	frac : float, optional
		Fraction of total flux to enclose (default: 0.95 for 95%)
		
	Returns:
	--------
	tuple
		Three-element tuple containing:
		- width_ms: Width of optimal window in milliseconds
		- best_start: Starting index of optimal window  
		- best_end: Ending index of optimal window
	"""
	prof = np.nan_to_num(profile)
	total = np.sum(prof)
	n = len(prof)
	min_width = n
	best_start, best_end = 0, n-1

	for start in range(n):
		cumsum = 0.0
		for end in range(start, n):
			cumsum += prof[end]
			if cumsum >= frac * total:
				width = end - start + 1
				if width < min_width:
					min_width = width
					best_start, best_end = start, end
				break  # No need to check longer windows from this start

	width_ms = time_ms[best_end] - time_ms[best_start] if min_width < n else 0.0
	return width_ms, best_start, best_end


def median_percentiles(yvals, x, ndigits=3):
	"""
	Calculate median values and percentile-based error bars from grouped data.

	Parameters:
	-----------
	yvals : dict
		Dictionary where keys are parameter values and values are lists/arrays of measurements
	x : array_like
		Array of parameter values for which to compute statistics  
	ndigits : int, optional
		Number of decimal places for rounding keys during lookup (default: 3)
		
	Returns:
	--------
	tuple
		Two-element tuple containing:
		- med_vals: List of median values for each x value
		- percentile_errs: List of (lower_percentile, upper_percentile) tuples
	"""
 
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
	
 
def scatter_stokes_chan(chan, time_res_ms, tau_cms):
	"""
	Normalize grouped measurement data by corresponding weight factors.
	
	Parameters:
	-----------
	x : array_like
		Parameter values for which to perform normalization
	yvals : dict  
		Dictionary with parameter values as keys and measurement lists as values
	weights_dict : dict
		Dictionary with parameter values as keys and weight factor lists as values
	ndigits : int, optional
		Number of decimal places for rounding keys during lookup (default: 3)
		
	Returns:
	--------
	dict
		Dictionary with same keys as input, containing normalized values:
		normalized_value = original_value / weight_factor
	"""
	# Pad to cover tail (~5 tau)
	n_pad = int(np.ceil(5 * tau_cms / time_res_ms))
	padded_I = np.pad(chan, (0, n_pad), mode='constant')  # Pad only at end

	# Create IRF time axis
	irf_t = np.arange(0, (n_pad + 1)) * time_res_ms
	irf = np.exp(-irf_t / tau_cms)
	irf /= np.sum(irf)  # Normalize

	# Convolve and trim back to original size
	convolved = fftconvolve(padded_I, irf, mode='full')
	sc_chan = convolved[:len(chan)]

	return sc_chan



def add_noise(dynspec, t_sys, f_res, t_res, time_ms, n_pol=1):
	"""
	Add Gaussian noise to a clean Stokes IQUV dynamic spectrum based on the radiometer equation.

	Parameters
	----------
	dynspec : np.ndarray
		3D array with shape (4, nchan, ntime), clean dynamic spectrum [I, Q, U, V]
	t_sys : float
		System temperature (or equivalent noise level) in arbitrary units
	f_res : float
		Frequency resolution in Hz
	t_res : float
		Time resolution in seconds
	time_ms : np.ndarray
		Time array in milliseconds
	n_pol : int
		Number of polarizations (default 2 for Stokes I)
	
	Returns:
	-------
	noisy_dynspec : np.ndarray
		Noisy Stokes IQUV dynamic spectrum
	"""

	# Calculate RMS noise using the radiometer equation
	sigma = t_sys / np.sqrt(n_pol * f_res * t_res)

	noise = np.random.normal(0.0, sigma, dynspec.shape)
	
	noisy_dynspec = dynspec + noise

	
	#snr, _ = boxcar_snr(np.nansum(noisy_dynspec[0], axis=0), sigma)
	#print(f"Stokes I SNR (boxcar method): {snr:.2f}")
 
	snr = snr_onpulse(np.nansum(noisy_dynspec[0], axis=0), time_ms, frac=0.95)
	print(f"Stokes I SNR (on-pulse method): {snr:.2f}")

	return noisy_dynspec, snr



def boxcar_snr(ys, rms):
	"""
	Calculates "max boxcar S/N" using boxcar's method.
	
	Parameters:
	-----------
	ys : array_like
		Input signal profile
	rms : float
		RMS noise level
	plot : bool, optional
		Whether to show the diagnostic plot
		
	Returns:
	--------
	global_maxSNR_normalized : float
		Maximum S/N ratio (normalized by RMS)
	boxcarw : int
		Optimal boxcar width
	"""
	
	ys = np.asarray(ys)
	maxSNR = np.zeros(ys.size)
	wmax = np.zeros(ys.size, dtype=int)
	
	for i1 in range(ys.size):
		n2 = int(ys.size - i1)
		
		for i2 in range(n2):
			w = i2 + 1
			SNR = np.sum(ys[i1:i1+w]) / w**0.5
			
			if SNR > maxSNR[i1]:
				maxSNR[i1] = SNR
				wmax[i1] = w
				
	global_maxSNR = np.max(maxSNR)
	boxcarw = wmax[np.argmax(maxSNR)]

	return (global_maxSNR/rms, boxcarw)


def snr_onpulse(profile, time_ms, frac=0.95):
	"""
	Calculate S/N using the on-pulse window and off-pulse RMS.
	"""
	# Find on-pulse window
	_, left, right = boxcar_width(profile, time_ms, frac=frac)
	onpulse = profile[left:right+1]
	# Off-pulse mask
	mask = np.ones_like(profile, dtype=bool)
	mask[left:right+1] = False
	offpulse = profile[mask]
	# Estimate RMS from off-pulse
	rms = np.nanstd(offpulse)
	# S/N calculation
	snr = np.nansum(onpulse) / (rms * np.sqrt(len(onpulse)))
	return snr