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

import logging
import os

import numpy as np
from scipy.stats import circvar
from RMtools_1D.do_RMclean_1D import run_rmclean
from RMtools_1D.do_RMsynth_1D import run_rmsynth
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve

from fires.utils.utils import frb_spectrum, frb_time_series, speed_of_light_cgs

logging.basicConfig(level=logging.INFO)

#	---------------------------------------------------------------------------------

def rm_synth(freq_ghz, iquv, diquv, outdir, save, show_plots):
	"""
	Perform rotation measure (RM) synthesis using RMtools to determine the RM and polarisation properties.

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

	--------
	list
		Four-element list containing:
		[0] RM value in rad/m²
		[1] RM uncertainty in rad/m²
		[2] Intrinsic polarisation angle at λ²=0 in degrees
		[3] Uncertainty in polarisation angle in degrees
		
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


def estimate_rm(dspec, freq_mhz, time_ms, noisespec, phi_range, dphi, outdir, save, show_plots):
	"""
	Estimate the rotation measure (RM) of an FRB from its dynamic spectrum using RM synthesis.

	Parameters:
	-----------
	dspec : ndarray, shape (4, n_freq, n_time)
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
		- pol_angle_0: Intrinsic polarisation angle in degrees
		- dpol_angle_0: Uncertainty in polarisation angle in degrees
	"""


	left, right = boxcar_width(np.nansum(dspec[0], axis=0), frac=0.95)

	# Calculate the mean spectra for each Stokes parameter
	ispec   = np.nansum(dspec[0, :, left:right], axis=1)
	vspec   = np.nansum(dspec[3, :, left:right], axis=1)
	qspec0  = np.nansum(dspec[1, :, left:right], axis=1)
	uspec0  = np.nansum(dspec[2, :, left:right], axis=1)
	noispec = noisespec / np.sqrt(float(right + 1 - left))


	iquv  = (ispec, qspec0, uspec0, vspec)
	eiquv = (noispec[0], noispec[1], noispec[2], noispec[3])
		
	# Run RM synthesis
	res_rmtool = rm_synth(freq_mhz / 1.0e3, iquv, eiquv, outdir, save, show_plots)

	logging.info("\nResults from RMtool (RM synthesis) \n")
	logging.info("RM = %.2f +/- %.2f rad/m2   PolAng0 = %.2f +/- %.2f deg\n" % (res_rmtool[0], res_rmtool[1], res_rmtool[2], res_rmtool[3]))

	return res_rmtool


def rm_correct_dspec(dspec, freq_mhz, rm0):
	"""
	Apply rotation measure correction to remove Faraday rotation from a dynamic spectrum.
 
	Parameters:
	-----------
	dspec : ndarray, shape (4, n_freq, n_time)
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

	
	# Initialise the new dynamic spectrum
	new_dspec    = np.zeros(dspec.shape, dtype=float)
	new_dspec[0] = dspec[0]
	new_dspec[3] = dspec[3]
	
	# Calculate the lambda squared array
	lambda_sq 		 = (speed_of_light_cgs * 1.0e-8 / freq_mhz) ** 2
	lambda_sq_median = np.nanmedian(lambda_sq)
		
	# Apply RM correction to Q and U spectra
	for ci in range(len(lambda_sq)):
		rot_angle = -2 * rm0 * (lambda_sq[ci] - lambda_sq_median)
		new_dspec[1, ci] = dspec[1, ci] * np.cos(rot_angle) - dspec[2, ci] * np.sin(rot_angle)
		new_dspec[2, ci] = dspec[2, ci] * np.cos(rot_angle) + dspec[1, ci] * np.sin(rot_angle)

	return new_dspec


def est_profiles(dspec, noise_stokes, left, right):
	"""
	Extract and analyze time-resolved polarisation profiles from a dynamic spectrum.
	
	Parameters:
	Parameters:
	-----------
	dspec : ndarray, shape (4, n_freq, n_time)  
		Dynamic spectrum with Stokes I, Q, U, V
	noise_stokes : array_like, shape (4,)
		RMS noise levels for each Stokes parameter
	left : int
		Starting time bin index for integration window
	right : int
		Ending time bin index for integration window
		
	Returns:
	--------
	frb_time_series
		Object containing time-resolved polarisation measurements:
		- iquvt: Time series of I, Q, U, V
		- lts, elts: Linear polarisation intensity and error
		- pts, epts: Total polarisation intensity and error  
		- phits, dphits: Linear polarisation angle and error (degrees)
		- psits, dpsits: Circular polarisation angle and error (degrees)
		- Fractional polarisations: qfrac, ufrac, vfrac, lfrac, pfrac with errors
	"""
	pa_mask_sigma = 2.0  # PA detection threshold in sigma_L

	with np.errstate(invalid='ignore', divide='ignore', over='ignore'):

		iquvt = np.nansum(dspec, axis=1)

		Its = iquvt[0]
		Qts = iquvt[1]
		Uts = iquvt[2]
		Vts = iquvt[3]

		Its_rms = noise_stokes[0]
		Qts_rms = noise_stokes[1]
		Uts_rms = noise_stokes[2]
		Vts_rms = noise_stokes[3]

		L_meas = np.sqrt(Qts**2 + Uts**2)
		eps = 1e-12

		sigma_L = np.sqrt(Qts**2 * Qts_rms**2 + Uts**2 * Uts_rms**2) / np.maximum(L_meas, eps)
		r = L_meas / np.maximum(sigma_L, eps)
		cutoff = 1.57

		# Debias (Everett & Weisberg 2001 / Wardle-Kronberg threshold 1.57)
		Lts = np.zeros_like(L_meas)
		det = r >= cutoff
		Lts[det] = np.sqrt(np.maximum(L_meas[det]**2 - sigma_L[det]**2, 0.0))
		eLts = np.full_like(Lts, np.nan)
		eLts[det] = sigma_L[det]
	
		# Total polarisation
		Pts = np.sqrt(Lts**2 + Vts**2)
		ePts = np.sqrt((Lts**2 * eLts**2) + (Vts**2 * Vts_rms**2)) / np.maximum(Pts, eps)

		# Position angle (keep in radians internally)
		phits = np.full_like(Lts, np.nan)
		ephits = np.full_like(Lts, np.nan)

		# PA detection mask
		pa_det = (Lts >= pa_mask_sigma * sigma_L)

		phits[pa_det] = 0.5 * np.arctan2(Uts[pa_det], Qts[pa_det])
		# Stable σ_PA approximation for detected bins
		ephits[pa_det] = 0.5 * np.sqrt(
			(Qts[pa_det]**2 * Uts_rms**2 + Uts[pa_det]**2 * Qts_rms**2)
			/ np.maximum((Qts[pa_det]**2 + Uts[pa_det]**2)**2, eps)
		)

		# Restrict PA to on-pulse window
		win_mask = np.zeros_like(pa_det, dtype=bool)
		win_mask[left:right+1] = True
		keep = pa_det & win_mask
		phits[~keep] = np.nan
		ephits[~keep] = np.nan

		# Fractional polarisations
		qfrac = Qts / Its
		ufrac = Uts / Its
		vfrac = Vts / Its
		lfrac = Lts / Its
		pfrac = Pts / Its

		# Fractional errors (guard zeros)
		def _frac_err(val, err_val, base, err_base):
			return np.abs(val / np.maximum(base, eps)) * np.sqrt(
				(err_val / np.maximum(val, eps))**2 + (err_base / np.maximum(base, eps))**2
			)

		eqfrac = _frac_err(Qts, Qts_rms, Its, Its_rms)
		eufrac = _frac_err(Uts, Uts_rms, Its, Its_rms)
		evfrac = _frac_err(Vts, Vts_rms, Its, Its_rms)
		elfrac = _frac_err(Lts, np.nan_to_num(eLts, nan=0.0), Its, Its_rms)
		epfrac = _frac_err(Pts, ePts, Its, Its_rms)

	return frb_time_series(
		iquvt, Lts, eLts, Pts, ePts,
		phits, ephits,  # linear PA + error
		np.full_like(phits, np.nan),  # placeholder circular PA
		np.full_like(phits, np.nan),  # placeholder circular PA err
		qfrac, eqfrac, ufrac, eufrac, vfrac, evfrac, lfrac, elfrac, pfrac, epfrac
	)

def est_spectra(dspec, noisespec, left_window_ms, right_window_ms):
	"""
	Extract frequency-resolved polarisation spectra by integrating over a specified time window.

	Parameters:
	-----------
	dspec : ndarray, shape (4, n_freq, n_time)
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
		Object containing frequency-resolved polarisation measurements:
		- iquvspec: Integrated Stokes I, Q, U, V spectra
		- noispec0: Noise spectra scaled for integration time
		- lspec, dlspec: Linear polarisation intensity and error vs frequency
		- pspec, dpspec: Total polarisation intensity and error vs frequency
		- Fractional polarisation spectra with errors: qfracspec, ufracspec, etc.
		- phispec, dphispec: Linear polarisation angle and error vs frequency  
		- psispec, dpsispec: Circular polarisation angle and error vs frequency
	"""
	 
	# Average the dynamic spectrum over the specified time range
	iquvspec = np.nansum(dspec[:, :, left_window_ms:right_window_ms + 1], axis=2)
	
	# Extract the Stokes parameters
	ispec = iquvspec[0]
	vspec = iquvspec[3]
	qspec = iquvspec[1]
	uspec = iquvspec[2]		
	
	# Calculate the noise for each Stokes parameter
	noispec0 = noisespec / np.sqrt(float(right_window_ms + 1 - left_window_ms))
	lspec  = np.sqrt(uspec ** 2 + qspec ** 2)
	dlspec = np.sqrt((uspec * noispec0[2]) ** 2 + (qspec * noispec0[1]) ** 2) / np.maximum(lspec, 1e-12)
	pspec  = np.sqrt(lspec ** 2 + vspec ** 2)
	dpspec = np.sqrt((lspec * dlspec) ** 2 + (vspec * noispec0[3]) ** 2) / np.maximum(pspec, 1e-12)

	# Calculate the fractional polarisations
	qfracspec = qspec / ispec
	ufracspec = uspec / ispec
	vfracspec = vspec / ispec
	# Calculate the errors in fractional polarisations
	dqfrac = np.sqrt((qspec * noispec0[0]) ** 2 + (ispec * noispec0[1]) ** 2) / (ispec ** 2)
	dufrac = np.sqrt((uspec * noispec0[0]) ** 2 + (ispec * noispec0[2]) ** 2) / (ispec ** 2)
	dvfrac = np.sqrt((vspec * noispec0[0]) ** 2 + (ispec * noispec0[3]) ** 2) / (ispec ** 2)

	# Calculate the fractional linear and total polarisations
	lfracspec = lspec / ispec
	dlfrac	  = np.sqrt((lspec * noispec0[0]) ** 2 + (ispec * dlspec) ** 2) / (ispec ** 2)
	pfracspec = pspec / ispec
	dpfrac 	  = np.sqrt((pspec * noispec0[0]) ** 2 + (ispec * dpspec) ** 2) / (ispec ** 2)

	# Calculate the polarisation angles
	phispec  = np.rad2deg(0.5 * np.arctan2(uspec, qspec))		
	dphispec = np.rad2deg(0.5 * np.sqrt(uspec**2 * noispec0[1]**2 + qspec**2 * noispec0[2]**2) / np.maximum(uspec ** 2 + qspec ** 2, 1e-12))

	psispec  = np.rad2deg(0.5 * np.arctan2(vspec, lspec))		
	dpsispec = np.rad2deg(0.5 * np.sqrt(vspec**2 * noispec0[3]**2 + lspec**2 * dlspec**2) / np.maximum(vspec ** 2 + lspec ** 2, 1e-12))

	# Return the spectra as a frb_spectrum object
	return frb_spectrum(iquvspec, noispec0, lspec, dlspec, pspec, dpspec, qfracspec, dqfrac, ufracspec, dufrac, vfracspec, dvfrac, lfracspec, dlfrac, pfracspec, dpfrac, phispec, dphispec, psispec, dpsispec)


def make_onpulse_mask(n_time, left, right):
	"""
	Return a boolean mask marking the on-pulse window [left, right] inclusive.
	"""
	on_mask = np.zeros(int(n_time), dtype=bool)
	l = max(0, int(left))
	r = min(int(n_time) - 1, int(right))
	if r >= l:
		on_mask[l:r+1] = True
	return on_mask


def make_offpulse_mask(n_time, left, right, buffer_bins=0):
	"""
	Return a boolean mask for off-pulse samples, excluding the on-pulse window
	and an extra buffer on each side of the on-pulse window.
	"""
	n = int(n_time)
	l_on = max(0, int(left))
	r_on = min(n - 1, int(right))
	buf = max(0, int(buffer_bins))
	l_excl = max(0, l_on - buf)
	r_excl = min(n - 1, r_on + buf)

	off_mask = np.ones(n, dtype=bool)
	if r_excl >= l_excl:
		off_mask[l_excl:r_excl+1] = False
	return off_mask


def on_off_pulse_masks_from_profile(profile,
									frac=0.95,
									buffer_frac=None,
									one_sided_offpulse=False,
									tail_frac=None,
									max_tail_mult=5):
	"""
	Construct on- and off-pulse masks from a 1-D profile.

	Enhancements:
	  - Optional tail inclusion (tail_frac): extend right edge to include scattered tail
		where profile > tail_frac * peak (capped by max_tail_mult * initial_width).
	  - Optional one-sided off-pulse region (pre-burst only) to avoid tail leakage.
	  - Buffer computed from the INITIAL (pre-tail) width for stable behaviour.

	Parameters
	----------
	profile : 1D array
		Intensity profile (e.g. summed Stokes I over frequency).
	frac : float
		Fraction of total energy to enclose for initial compact window (boxcar_width).
	buffer_frac : float or None
		Fraction of initial on-pulse width excluded on EACH side from off-pulse.
	one_sided_offpulse : bool
		If True, use only samples strictly before the (buffered) on-pulse start.
	tail_frac : float or None
		If set (e.g. 0.003), extend right edge while profile > tail_frac * peak.
	max_tail_mult : int
		Cap on tail extension in units of the initial width.
	return_details : bool
		If True, also return a dict with window metadata.

	Returns
	-------
	on_mask : bool array
	off_mask: bool array
	(left, right) : tuple(int, int)
		Final on-pulse indices inclusive.
	details (optional) : dict
		width_init, width_final, peak, tail_used, buffer_bins
	"""
	prof = np.asarray(profile, dtype=float)
	n = prof.size
	left, right = boxcar_width(prof, frac=frac)
	peak_val = np.nanmax(prof) if n > 0 else 0.0
	init_width = max(1, right - left + 1)

	# Tail expansion
	tail_used = False
	if tail_frac is not None and peak_val > 0:
		thr = tail_frac * peak_val
		max_right = min(n - 1, right + int(max_tail_mult * init_width))
		r = right
		while r + 1 <= max_right and prof[r + 1] > thr:
			r += 1
		if r != right:
			tail_used = True
			right = r

	buffer_bins = int(float(buffer_frac) * init_width) if buffer_frac is not None else 0

	on_mask = make_onpulse_mask(n, left, right)

	if one_sided_offpulse:
		off_mask = np.zeros(n, dtype=bool)
		end_off = max(0, left - buffer_bins - 1)
		if end_off >= 0:
			off_mask[0:end_off + 1] = True
	else:
		off_mask = make_offpulse_mask(n, left, right, buffer_bins=buffer_bins)

	return on_mask, off_mask, (left, right)


def estimate_noise_with_offpulse_mask(corrdspec, offpulse_mask, robust=False, ddof=1):
	"""
	Estimate noise using an off-pulse mask (True = off-pulse).

	Returns
	-------
	noise_stokes : (n_stokes,)
		RMS of the frequency-summed time series for each Stokes parameter
		(quad-sum of per-channel RMS).
	noisespec : (n_stokes, n_chan)
		Per-Stokes, per-channel RMS over off-pulse times.

	Parameters
	----------
	corrdspec : array, shape (n_stokes, n_chan, n_time)
	offpulse_mask : bool array, shape (n_time,)
	robust : bool, default False
		If True, use MAD (scaled) instead of standard deviation.
	ddof : int, default 1
		Delta degrees of freedom for std (ignored if robust=True).
	"""
	corrdspec = np.asarray(corrdspec, dtype=float)
	if corrdspec.ndim != 3:
		raise ValueError("corrdspec must have shape (n_stokes, n_chan, n_time)")
	if offpulse_mask.dtype != bool or offpulse_mask.ndim != 1:
		raise ValueError("offpulse_mask must be 1-D boolean (time axis)")

	n_stokes, n_chan, n_time = corrdspec.shape

	if offpulse_mask.size != n_time:
		raise ValueError("offpulse_mask length does not match time dimension")

	if not np.any(offpulse_mask):
		noise_stokes = np.full(n_stokes, np.nan)
		noisespec = np.full((n_stokes, n_chan), np.nan)
		return noise_stokes, noisespec

	# Slice off-pulse data: (n_stokes, n_chan, n_off)
	offcube = corrdspec[:, :, offpulse_mask]

	if robust:
		# Median Absolute Deviation per (stokes, chan)
		med = np.nanmedian(offcube, axis=2, keepdims=True)
		mad = np.nanmedian(np.abs(offcube - med), axis=2)
		noisespec = 1.4826 * mad
	else:
		# Standard deviation over time axis (ddof)
		if offcube.shape[2] - ddof <= 0:
			# Fall back to population std
			noisespec = np.nanstd(offcube, axis=2)
		else:
			noisespec = np.nanstd(offcube, axis=2, ddof=ddof)

	# Frequency-summed time-series RMS for each Stokes:
	# Sum over channels -> variances add (assume independence)
	noise_stokes = np.sqrt(np.nansum(noisespec**2, axis=1))

	return noise_stokes, noisespec


def process_dspec(dspec, freq_mhz, gdict, buffer_frac):
	"""
	Complete pipeline for processing FRB dynamic spectra: RM correction, noise estimation, and profile extraction.
	"""
	RM = gdict["RM"]

	max_rm = RM[np.argmax(np.abs(RM))]
	if np.abs(max_rm) > 0:
		corrdspec = rm_correct_dspec(dspec, freq_mhz, max_rm)
	else:
		corrdspec = dspec.copy()

	# Use Stokes I to find the on-pulse window
	I = np.nansum(corrdspec[0], axis=0)
	left, right = boxcar_width(I, frac=0.95)

	# New: buffer around on-pulse window for off-pulse noise estimation
	_, offpulse_mask, _ = on_off_pulse_masks_from_profile(I, frac=0.95, buffer_frac=buffer_frac)

	# Estimate noise using off-pulse region with buffer
	noise_stokes, noisespec = estimate_noise_with_offpulse_mask(corrdspec, offpulse_mask)

	tsdata = est_profiles(corrdspec, noise_stokes, left, right)

	return tsdata, corrdspec, noisespec, noise_stokes


def boxcar_width(profile, frac=0.95):
	"""
	Find the minimum contiguous time window that contains a specified fraction of the total burst energy.
	
	Parameters:
	-----------
	profile : array_like
		1D intensity profile (typically integrated over frequency)
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
	prof = np.nan_to_num(np.squeeze(profile))
	n = len(prof)
	
	# Target flux to enclose
	target_flux = frac * np.sum(prof)
	
	# Compute cumulative sum once
	cumsum = np.cumsum(prof)
	
	min_width = n
	best_start, best_end = 0, n-1
	
	# For each starting position
	for start in range(n):
		# Find the first position where we exceed target flux
		start_flux = cumsum[start-1] if start > 0 else 0
		target_end_flux = start_flux + target_flux
		
		# Binary search or simple search for end position
		end_indices = np.where(cumsum >= target_end_flux)[0]
		if len(end_indices) > 0:
			end = end_indices[0]
			width = end - start + 1
			if width < min_width:
				min_width = width
				best_start, best_end = start, end

	return best_start, best_end

 
def scatter_dspec(dspec, time_res_ms, tau_cms, pad_factor=5):
	"""
	Apply one-sided exponential scattering independently to each frequency channel
	of a Stokes I dynamic spectrum.

	Replaces (renamed from) scatter_stokes_chan which operated on a single 1-D channel.

	Parameters
	----------
	I_dspec : array_like, shape (n_chan, n_time)
		Stokes I dynamic spectrum (no polarisation axis).
	time_res_ms : float
		Time resolution in milliseconds.
	tau_cms : float or array_like, shape (n_chan,)
		Scattering timescale(s) in milliseconds. Scalar applies to all channels.
	pad_factor : float, default 5
		Multiple of tau used to determine zero-padding length (covers tail).

	Returns
	-------
	scattered_dspec : ndarray, shape (n_chan, n_time)
		Dynamic spectrum after convolution with normalised exponential tail.

	Notes
	-----
	- Channels with non-positive or NaN tau are returned unchanged.
	- Uses fftconvolve per channel (OK for typical FRB channel counts).
	"""
	dspec = np.asarray(dspec, dtype=float)
	if dspec.ndim != 2:
		raise ValueError("I_dspec must be 2-D (n_chan, n_time)")

	n_chan, n_time = dspec.shape

	# Prepare per-channel tau array
	if np.isscalar(tau_cms):
		tau_arr = np.full(n_chan, float(tau_cms), dtype=float)
	else:
		tau_arr = np.asarray(tau_cms, dtype=float)
		if tau_arr.shape[0] != n_chan:
			raise ValueError("tau_cms length must match number of channels")

	dspec_scattered = np.empty_like(dspec)

	for ci in range(n_chan):
		tau = tau_arr[ci]
		chan = dspec[ci]

		n_pad = int(np.ceil(pad_factor * tau / time_res_ms))

		# Causal IRF support: t >= 0 (implements Heaviside step)
		t_irf = np.arange(0, n_pad + 1) * time_res_ms

		# Discrete causal exponential kernel: samples of H(t) * exp(-t/τ).
		# We normalise by sum so that the discrete convolution preserves total flux
		# (this absorbs the continuous 1/τ and Δt factors on the sampled grid).
		irf = np.exp(-t_irf / tau)
		irf /= irf.sum()

		# Right-pad only to keep the tail causal within the window
		padded = np.pad(chan, (0, n_pad), mode='constant')
		conv = fftconvolve(padded, irf, mode='full')
		dspec_scattered[ci] = conv[:n_time]

	return dspec_scattered


def compute_required_sefd(dspec, f_res_hz, t_res_s, target_snr, n_pol=2, frac=0.95, buffer_frac=None, 
					one_sided_offpulse=False, tail_frac=None, max_tail_mult=5):
	"""
	Compute SEFD needed for a desired S/N, using adaptive on/off selection
	consistent with snr_onpulse (with tail inclusion).

	Parameters
	----------
	dspec : (4, n_chan, n_time)
	f_res_hz : float
	t_res_s : float
	target_snr : float
	n_pol : int
	frac : float
	buffer_frac : float or None
	subtract_baseline : bool
	one_sided_offpulse : bool
		Use only pre-pulse region for noise (avoids scattered tail).
	tail_frac : float or None
		Include right-side tail bins with flux > tail_frac * peak in on-pulse.
	max_tail_mult : int
		Cap tail extension vs initial width.

	Returns
	-------
	sefd_required : float
	details : dict
	"""
	if target_snr is None or target_snr <= 0:
		raise ValueError("target_snr must be > 0")

	prof = np.nansum(dspec[0], axis=0)

	# Build masks similarly to snr_onpulse but WITHOUT noise (so baseline from left only if one_sided_offpulse)
	left, right = boxcar_width(prof, frac=frac)
	peak_val = np.nanmax(prof)
	init_width = max(1, right - left + 1)

	# Tail expansion
	if tail_frac is not None and peak_val > 0:
		thr = tail_frac * peak_val
		max_right = min(prof.size - 1, right + int(max_tail_mult * init_width))
		r = right
		while r + 1 <= max_right and prof[r + 1] > thr:
			r += 1
		right = r

	buffer_bins = int(float(buffer_frac) * init_width) if buffer_frac is not None else 0

	on_mask = make_onpulse_mask(prof.size, left, right)

	if one_sided_offpulse:
		off_mask = np.zeros(prof.size, dtype=bool)
		end_off = max(0, left - buffer_bins - 1)
		if end_off >= 0:
			off_mask[0:end_off + 1] = True
	else:
		off_mask = make_offpulse_mask(prof.size, left, right, buffer_bins=buffer_bins)

	pulse = prof[on_mask]
	E_on = np.nansum(pulse)
	N_on = int(on_mask.sum())
	N_chan = dspec.shape[1]

	if E_on <= 0 or N_on == 0 or N_chan == 0:
		raise ValueError("Cannot compute SEFD (invalid on-pulse energy).")

	# Invert radiometer SNR relation (SNR ∝ 1/SEFD)
	sefd_req = (E_on * np.sqrt(n_pol * f_res_hz * t_res_s)) / (target_snr * np.sqrt(N_chan * N_on))

	return sefd_req


def add_noise(dspec, sefd, f_res, t_res, plot_multiple_frb, buffer_frac, n_pol=2,
			   stokes_scale=(1.0, 1.0, 1.0, 1.0), add_slow_baseline=False,
			   baseline_frac=0.05, baseline_kernel_ms=5.0, time_res_ms=None):
	"""
	Add thermal (and optional slow baseline) noise using SEFD.

	Parameters
	----------
	dspec : (4, n_chan, n_time) array
		Clean Stokes I,Q,U,V cube.
	sefd : float or (n_chan,) array
		System Equivalent Flux Density in Jy.
	f_res : float or (n_chan,) array
		Channel bandwidth (Hz).
	t_res : float
		Time resolution (s).
	buffer_frac : float
		Passed to snr_onpulse for S/N estimate.
	plot_multiple_frb : bool
		If False, print S/N summary.
	n_pol : int
		Number of summed polarisations (usually 2).
	stokes_scale : tuple of 4 floats
		Multiplicative RMS scale factors for (I,Q,U,V).
	add_slow_baseline : bool
		Add smoothed low-frequency baseline drift.
	baseline_frac : float
		Std of baseline component relative to white RMS.
	baseline_kernel_ms : float
		FWHM of Gaussian smoothing (ms) for baseline.
	time_res_ms : float or None
		Needed if add_slow_baseline=True.

	Returns
	-------
	noisy_dspec : array
		dspec + injected noise.
	sigma_ch : (4, n_chan) array
		Per-Stokes per-channel white-noise RMS used.
	snr : float
		On-pulse S/N of Stokes I.
	"""
	dspec = np.asarray(dspec, dtype=float)

	_, n_chan, n_time = dspec.shape

	# Normalise inputs
	sefd_arr = np.full(n_chan, sefd, dtype=float) if np.isscalar(sefd) else np.asarray(sefd, dtype=float)
	f_res_arr = np.full(n_chan, f_res, dtype=float) if np.isscalar(f_res) else np.asarray(f_res, dtype=float)

	# Radiometer equation: sigma = SEFD / sqrt(n_pol * Δν * Δt)
	sigma_I_ch = sefd_arr / np.sqrt(n_pol * f_res_arr * t_res)  # (n_chan,)

	stokes_scale = np.asarray(stokes_scale, dtype=float)

	sigma_ch = np.vstack([sigma_I_ch * stokes_scale[s] for s in range(4)])  # (4, n_chan)

	noise_white = np.random.normal(0.0, sigma_ch[:, :, None], size=(4, n_chan, n_time))

	if add_slow_baseline:
		if time_res_ms is None:
			raise ValueError("time_res_ms required when add_slow_baseline=True")
		sigma_bins = (baseline_kernel_ms / time_res_ms) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
		baseline = np.random.normal(0.0, sigma_ch[:, :, None] * baseline_frac, size=noise_white.shape)
		baseline = gaussian_filter1d(baseline, sigma=sigma_bins, axis=2, mode='nearest')
		noise = noise_white + baseline
	else:
		noise = noise_white

	noisy_dspec = dspec + noise

	I_time = np.nansum(noisy_dspec[0], axis=0)
	snr, (left, right) = snr_onpulse(I_time, frac=0.95, subtract_baseline=True, robust_rms=True, buffer_frac=buffer_frac)

	if not plot_multiple_frb:
		logging.info(f"Stokes I S/N (on-pulse method): {snr:.2f}")

	return noisy_dspec, sigma_ch, snr


def boxcar_snr(ys, rms):
	"""
	Calculates "max boxcar S/N".
	
	Parameters:
	-----------
	ys : array_like
		Input signal profile
	rms : float
		RMS noise level
	Returns:
	--------
	global_maxSNR_normalised : float
		Maximum S/N ratio (normalised by RMS)
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


def snr_onpulse(profile, frac=0.95, subtract_baseline=True, robust_rms=True, buffer_frac=None, one_sided_offpulse=False,
				tail_frac=None, max_tail_mult=5):
	"""
	Estimate S/N using an on-pulse window and an (adaptive) off-pulse RMS.

	Improvements:
	  - Optional tail expansion: include scattered tail in on-pulse so it is NOT treated as noise.
	  - Optional one-sided off-pulse: use only pre-burst region to avoid tail leakage.

	Parameters
	----------
	profile : 1D array
	frac : float
		Fraction of total flux for initial window (boxcar_width).
	subtract_baseline : bool
	robust_rms : bool
	buffer_frac : float or None
		Fraction of (initial) on-pulse width excluded on each side from off-pulse.
	one_sided_offpulse : bool
		If True, estimate noise ONLY from bins strictly before the on-pulse window (after buffer).
	tail_frac : float or None
		If set (e.g. 0.003), extend the right edge to include all bins with flux > tail_frac * peak
		(up to max_tail_mult * initial_width). Helps include scattering tail in on-pulse.
	max_tail_mult : int
		Safety cap on tail extension (multiple of initial width).

	Returns
	-------
	snr : float
	(left, right) : tuple
		Final on-pulse edges used.
	"""
	prof = np.asarray(profile, dtype=float)
	n = prof.size
	left, right = boxcar_width(prof, frac=frac)
	peak_val = np.nanmax(prof)
	init_width = max(1, right - left + 1)

	# Tail expansion (right side) if requested
	if tail_frac is not None and peak_val > 0:
		thr = tail_frac * peak_val
		max_right = min(n - 1, right + int(max_tail_mult * init_width))
		r = right
		# Continue while profile stays above threshold (monotonicity not required)
		while r + 1 <= max_right and prof[r + 1] > thr:
			r += 1
		right = r

	width = max(1, right - left + 1)
	buffer_bins = int(float(buffer_frac) * init_width) if buffer_frac is not None else 0  # use initial width for buffer sizing

	# On-pulse mask (after possible tail expansion)
	mask_on = make_onpulse_mask(n, left, right)
	onpulse = prof[mask_on]

	# Off-pulse mask
	if one_sided_offpulse:
		# Only LEFT side before (left - buffer_bins)
		off_mask = np.zeros(n, dtype=bool)
		start_off = 0
		end_off = max(0, left - buffer_bins - 1)
		if end_off >= start_off:
			off_mask[start_off:end_off + 1] = True
	else:
		# Two-sided with buffer around expanded on-pulse
		off_mask = make_offpulse_mask(n, left, right, buffer_bins=buffer_bins)

	offpulse = prof[off_mask]

	baseline = 0.0
	if subtract_baseline:
		if offpulse.size > 0:
			baseline = np.nanmedian(offpulse)
			onpulse = onpulse - baseline
			offpulse = offpulse - baseline

	if robust_rms:
		if offpulse.size > 0:
			mad = np.nanmedian(np.abs(offpulse - np.nanmedian(offpulse)))
			sigma = 1.4826 * mad if mad > 0 else np.nanstd(offpulse)
		else:
			sigma = np.nan
	else:
		sigma = np.nanstd(offpulse) if offpulse.size > 0 else np.nan

	N_on = int(mask_on.sum())
	snr = np.nansum(onpulse) / (sigma * np.sqrt(max(N_on, 1))) if (sigma is not None and sigma > 0) else 0.0
	return snr, (left, right)


def _integrated_fractions_from_timeseries(I, Q, U, V, L, on_mask) -> tuple[float, float]:
    """
    Compute integrated L/I and V/I using on-pulse mask.
    Inputs are 1-D time series arrays; L is debiased linear from est_profiles.
    """
    I_masked = np.where(on_mask, I, np.nan)
    Q_masked = np.where(on_mask, Q, np.nan)
    U_masked = np.where(on_mask, U, np.nan)
    V_masked = np.where(on_mask, V, np.nan)
    L_masked = np.where(on_mask, L, np.nan)

    integrated_I = np.nansum(I_masked)
    integrated_V = np.nansum(V_masked)
    integrated_L = np.nansum(L_masked)

    if not np.isfinite(integrated_I) or integrated_I == 0:
        return np.nan, np.nan

    return float(integrated_L / integrated_I), float(integrated_V / integrated_I)


def _pa_variance_deg2(phits: np.ndarray) -> float:
    """
    Circular variance of PA in deg^2. phits in radians, uses circvar on 2*PA, divides by 4.
    Returns deg^2.
    """
    valid = np.isfinite(phits)
    if not np.any(valid):
        return np.nan
    pa_var_rad2 = circvar(2.0 * phits[valid]) / 4.0
    # convert to deg^2 in the same convention as used elsewhere: Var_deg2 = (deg(std))^2
    return float(np.rad2deg(np.sqrt(pa_var_rad2))**2)


def _freq_quarter_slices(n_chan: int) -> dict[str, slice]:
    """
    Build frequency quarter slices over channel index.
    Returns dict for '1q','2q','3q','4q','all'.
    """
    q = max(1, n_chan // 4)
    return {
        "1q": slice(0, q),
        "2q": slice(q, 2*q),
        "3q": slice(2*q, 3*q),
        "4q": slice(3*q, None),
        "all": slice(None)
    }


def _phase_slices_from_peak(n_time: int, peak_index: int) -> dict[str, slice]:
    """
    Build standard phase slices relative to peak index.
    Returns dict with 'first' (leading), 'last' (trailing), and 'total'.
    """
    first = slice(0, max(0, int(peak_index)))
    last = slice(max(0, int(peak_index)), int(n_time))
    total = slice(None)
    return {"first": first, "last": last, "total": total}


def compute_segments(dspec, freq_mhz, time_ms, gdict, buffer_frac=0.1) -> dict:
    """
    Compute per-segment measurements from a single dynamic spectrum:
      - phase segments: first (leading), last (trailing), total
      - freq segments: 1q, 2q, 3q, 4q, all

    Each segment records:
      - Vpsi: Var(psi) in deg^2, measured from PA time series (not micro params)
      - Lfrac: integrated L/I over on-pulse (95% boxcar with buffer)
      - Vfrac: integrated V/I over on-pulse

    Returns:
      {
        'phase': {
          'first' : {'Vpsi':..., 'Lfrac':..., 'Vfrac':...},
          'last'  : {...},
          'total' : {...}
        },
        'freq' : {
          '1q' : {...}, '2q': {...}, '3q': {...}, '4q': {...}, 'all': {...}
        }
      }
    """
    # Process full-band once (RM correction + masks)
    tsdata_full, corr_dspec, _, _ = process_dspec(dspec, freq_mhz, gdict, buffer_frac)

    Its = tsdata_full.iquvt[0]
    Qts = tsdata_full.iquvt[1]
    Uts = tsdata_full.iquvt[2]
    Vts = tsdata_full.iquvt[3]
    Lts = tsdata_full.Lts
    phits = tsdata_full.phits  # radians
    n_time = Its.size

    # On-pulse mask from full-band I (consistent with snr calculations)
    on_mask, _, (left, right) = on_off_pulse_masks_from_profile(Its, frac=0.95, buffer_frac=buffer_frac)
    peak_index = int(np.nanargmax(Its)) if n_time > 0 else 0
    phase_slices = _phase_slices_from_peak(n_time, peak_index)

    def _measure_phase_slice(slc: slice) -> dict:
        # Build slice mask
        slc_mask = np.zeros(n_time, dtype=bool)
        start = 0 if slc.start is None else slc.start
        stop = n_time if slc.stop is None else slc.stop
        if stop > start:
            slc_mask[start:stop] = True
        # Restrict to on-pulse within this slice
        on_mask_slice = on_mask & slc_mask

        Vpsi = _pa_variance_deg2(phits[slc_mask])
        Lfrac, Vfrac = _integrated_fractions_from_timeseries(Its, Qts, Uts, Vts, Lts, on_mask_slice)
        return {"Vpsi": Vpsi, "Lfrac": Lfrac, "Vfrac": Vfrac}

    phase_measures = {name: _measure_phase_slice(slc) for name, slc in phase_slices.items()}

    # Frequency quarters (compute tsdata per slice to keep RM correction + masks consistent)
    n_chan = corr_dspec.shape[1]
    fq = _freq_quarter_slices(n_chan)

    def _measure_freq_slice(slc: slice) -> dict:
        dspec_f = corr_dspec[:, slc, :]
        freq_f = freq_mhz[slc] if isinstance(slc, slice) else freq_mhz
        # Re-run process_dspec on the subset to get consistent on/off windows in this sub-band
        tsdata_f, _, _, _ = process_dspec(dspec_f, freq_f, gdict, buffer_frac)
        I = tsdata_f.iquvt[0]
        Q = tsdata_f.iquvt[1]
        U = tsdata_f.iquvt[2]
        V = tsdata_f.iquvt[3]
        L = tsdata_f.Lts
        ph = tsdata_f.phits
        # New on-pulse for sub-band
        on_m, _, _ = on_off_pulse_masks_from_profile(I, frac=0.95, buffer_frac=buffer_frac)
        Vpsi = _pa_variance_deg2(ph)
        Lfrac, Vfrac = _integrated_fractions_from_timeseries(I, Q, U, V, L, on_m)
        return {"Vpsi": Vpsi, "Lfrac": Lfrac, "Vfrac": Vfrac}

    freq_measures = {name: _measure_freq_slice(slc) for name, slc in fq.items()}

    return {"phase": phase_measures, "freq": freq_measures}
