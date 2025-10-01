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
from ..utils.utils import *

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


	left, right = boxcar_width(np.nansum(dynspec[0], axis=0), frac=0.95)

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

	
	# Initialise the new dynamic spectrum
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


def est_profiles(dynspec, noise_stokes, left, right):
	"""
	Extract and analyze time-resolved polarization profiles from a dynamic spectrum.
	
	Parameters:
	Parameters:
	-----------
	dynspec : ndarray, shape (4, n_freq, n_time)  
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

		Its = iquvt[0]
		Qts = iquvt[1]
		Uts = iquvt[2]
		Vts = iquvt[3]

		Its_rms = noise_stokes[0]
		Qts_rms = noise_stokes[1]
		Uts_rms = noise_stokes[2]
		Vts_rms = noise_stokes[3]
		
		# Calculate the linear polarization intensity
		Lts  = np.sqrt(Uts ** 2 + Qts ** 2)
		eps = 1e-12
		
		# Debias L using Everett & Weisberg+2001 method
		#Lts = np.sqrt((Lts/Its_rms)**2 - 1*Its_rms)
		#Lts[np.isnan(Lts)] = 0.0
		#Lts[Lts/Its_rms < 1.57] = 0.0

		eLts = np.sqrt((Qts**2 * Qts_rms**2) + (Uts**2 * Uts_rms**2)) / np.maximum(Lts, eps)
		Lmask = Lts != 0.0
		eLts[~Lmask] = np.nan
		#Lts[~Lmask] = np.nan
		
		# Calculate the total polarization intensity
		Pts  = np.sqrt(Lts ** 2 + Vts ** 2)
		# Correct error propagation for P = sqrt(L^2 + V^2)
		ePts = np.sqrt((Lts**2 * eLts**2) + (Vts**2 * Vts_rms**2)) / np.maximum(Pts, eps)

		# Calculate the polarization angles
		phits  = 0.5 * np.arctan2(Uts, Qts)
		ephits = 0.5 * np.sqrt(Uts**2 * Qts_rms**2 + Qts**2 * Uts_rms**2) / (Uts**2 + Qts**2)
		psits  = 0.5 * np.arctan2(Vts, Lts)
		epsits = 0.5 * np.sqrt(Vts**2 * eLts**2 + Lts**2 * Vts_rms**2) / (Vts**2 + Lts**2)

		# Calculate the fractional polarizations
		qfrac = Qts / Its
		ufrac = Uts / Its
		vfrac = Vts / Its

		lfrac = Lts / Its
		pfrac = Pts / Its

		# Calculate the errors in fractional polarizations
		evfrac = np.abs(vfrac) * np.sqrt((Vts_rms / Vts) ** 2 + (Its_rms / Its) ** 2)
		eqfrac = np.abs(qfrac) * np.sqrt((Qts_rms / Qts) ** 2 + (Its_rms / Its) ** 2)
		eufrac = np.abs(ufrac) * np.sqrt((Uts_rms / Uts) ** 2 + (Its_rms / Its) ** 2)
		elfrac = np.abs(lfrac) * np.sqrt((eLts / Lts) ** 2 + (Its_rms / Its) ** 2)
		epfrac = np.abs(pfrac) * np.sqrt((ePts / Pts) ** 2 + (Its_rms / Its) ** 2)


		# Set large errors to NaN
		mask = Lts < (2.0 * Its_rms)  # Mask where L is less than 2-sigma
		phits[mask]  = np.nan
		ephits[mask] = np.nan
		psits[mask]  = np.nan
		epsits[mask] = np.nan


		# Mask PA outside all signal windows using on-pulse finder
		pa_mask = np.zeros_like(phits, dtype=bool)
		pa_mask[left:right+1] = True
		phits[~pa_mask] = np.nan
		ephits[~pa_mask] = np.nan

		# Return the time profiles as a frb_time_series object
	return frb_time_series(iquvt, Lts, eLts, Pts, ePts, phits, ephits, psits, epsits, qfrac, eqfrac, ufrac, eufrac, vfrac, evfrac, lfrac, elfrac, pfrac, epfrac)


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
	dlspec = np.sqrt((uspec * noispec0[2]) ** 2 + (qspec * noispec0[1]) ** 2) / np.maximum(lspec, 1e-12)
	pspec  = np.sqrt(lspec ** 2 + vspec ** 2)
	dpspec = np.sqrt((lspec * dlspec) ** 2 + (vspec * noispec0[3]) ** 2) / np.maximum(pspec, 1e-12)

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


def estimate_noise_with_offpulse_mask(corrdspec, offpulse_mask):
	"""
	Estimate per-Stokes scalar noise and per-(Stokes,chan) noise using an
	off-pulse mask (True where off-pulse).
	"""
	npol = corrdspec.shape[0]
	noise_stokes = np.zeros(npol)
	for s in range(npol):
		offpulse_samples = corrdspec[s, :, offpulse_mask].ravel()
		noise_stokes[s] = np.nanstd(offpulse_samples) if offpulse_samples.size > 0 else np.nan

	noisespec = (np.nanstd(corrdspec[:, :, offpulse_mask], axis=2)
				 if np.any(offpulse_mask) else np.full(corrdspec.shape[:2], np.nan))
	return noise_stokes, noisespec


def process_dynspec(dynspec, freq_mhz, gdict, buffer_frac):
	"""
	Complete pipeline for processing FRB dynamic spectra: RM correction, noise estimation, and profile extraction.
	"""
	RM = gdict["RM"]

	max_rm = RM[np.argmax(np.abs(RM))]
	if np.abs(max_rm) > 0:
		corrdspec = rm_correct_dynspec(dynspec, freq_mhz, max_rm)
	else:
		corrdspec = dynspec.copy()

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

 
try:
    from numba import njit, prange
    _NUMBA_OK = True
except Exception:
    _NUMBA_OK = False
# ...existing code...

def scatter_stokes_chan(chan, time_res_ms, tau_cms):
    """
    Fast exponential scattering (causal one-sided) via IIR recursion.

    Kernel: h[k] = (1 - alpha) * alpha^k, alpha = exp(-dt / tau)
    Ensures flux conservation (sum h = 1). O(N).

    Parameters
    ----------
    chan : 1D array
    time_res_ms : float
    tau_cms : float (>0 enables scattering)

    Returns
    -------
    1D array (same shape)
    """
    if tau_cms <= 0:
        return chan

    dt = float(time_res_ms)
    alpha = np.exp(-dt / float(tau_cms))
    one_minus_alpha = 1.0 - alpha

    out = np.empty_like(chan)
    out[0] = one_minus_alpha * chan[0]
    prev = out[0]
    for i in range(1, chan.size):
        prev = one_minus_alpha * chan[i] + alpha * prev
        out[i] = prev
    return out

# Optional vectorised version for an (n_freq, n_time) block
def scatter_block(I_ft, time_res_ms, tau_ms_freq):
    """
    Vectorised scattering across frequency channels.

    I_ft : (nf, nt)
    tau_ms_freq : (nf,)
    Returns new array (nf, nt)
    """
    I_ft = np.asarray(I_ft)
    tau_ms_freq = np.asarray(tau_ms_freq)
    out = np.empty_like(I_ft)
    dt = float(time_res_ms)
    for f in range(I_ft.shape[0]):
        tau = tau_ms_freq[f]
        if tau <= 0:
            out[f] = I_ft[f]
            continue
        alpha = np.exp(-dt / tau)
        oma = 1.0 - alpha
        row = I_ft[f]
        o = out[f]
        o[0] = oma * row[0]
        prev = o[0]
        for t in range(1, row.size):
            prev = oma * row[t] + alpha * prev
            o[t] = prev
    return out

if _NUMBA_OK:
    @njit(parallel=True, fastmath=True)
    def _scatter_block_numba(I_ft, dt, tau_arr):
        nf, nt = I_ft.shape
        out = np.empty_like(I_ft)
        for f in prange(nf):
            tau = tau_arr[f]
            if tau <= 0:
                out[f, :] = I_ft[f, :]
                continue
            alpha = np.exp(-dt / tau)
            oma = 1.0 - alpha
            out[f, 0] = oma * I_ft[f, 0]
            for t in range(1, nt):
                out[f, t] = oma * I_ft[f, t] + alpha * out[f, t-1]
        return out

    def scatter_block(I_ft, time_res_ms, tau_ms_freq):
        return _scatter_block_numba(I_ft, float(time_res_ms), tau_ms_freq.astype(np.float64))


def compute_required_sefd(dynspec, f_res_hz, t_res_s, target_snr, n_pol=2, frac=0.95, buffer_frac=None, 
					one_sided_offpulse=False, tail_frac=None, max_tail_mult=5):
    """
    Compute SEFD needed for a desired S/N, using adaptive on/off selection
    consistent with snr_onpulse (with tail inclusion).

    Parameters
    ----------
    dynspec : (4, n_chan, n_time)
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

    prof = np.nansum(dynspec[0], axis=0)

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
    N_chan = dynspec.shape[1]

    if E_on <= 0 or N_on == 0 or N_chan == 0:
        raise ValueError("Cannot compute SEFD (invalid on-pulse energy).")

    # Invert radiometer SNR relation (SNR ∝ 1/SEFD)
    sefd_req = (E_on * np.sqrt(n_pol * f_res_hz * t_res_s)) / (target_snr * np.sqrt(N_chan * N_on))

    return sefd_req


def add_noise(dynspec, sefd, f_res, t_res, plot_multiple_frb, buffer_frac, n_pol=2,
               stokes_scale=(1.0, 1.0, 1.0, 1.0), add_slow_baseline=False,
               baseline_frac=0.05, baseline_kernel_ms=5.0, time_res_ms=None):
    """
    Add thermal (and optional slow baseline) noise using SEFD.

    Parameters
    ----------
    dynspec : (4, n_chan, n_time) array
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
    noisy_dynspec : array
        dynspec + injected noise.
    sigma_ch : (4, n_chan) array
        Per-Stokes per-channel white-noise RMS used.
    snr : float
        On-pulse S/N of Stokes I.
    """
    dynspec = np.asarray(dynspec, dtype=float)

    _, n_chan, n_time = dynspec.shape

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

    noisy_dynspec = dynspec + noise

    I_time = np.nansum(noisy_dynspec[0], axis=0)
    snr, (left, right) = snr_onpulse(I_time, frac=0.95, subtract_baseline=True, robust_rms=True, buffer_frac=buffer_frac)

    if not plot_multiple_frb:
        print(f"Stokes I S/N (on-pulse method): {snr:.2f}")

    return noisy_dynspec, sigma_ch, snr


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