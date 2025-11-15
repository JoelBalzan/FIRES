# -----------------------------------------------------------------------------
# genfns.py
# FIRES: The Fast, Intense Radio Emission Simulator
#
# This module provides core functions for generating FRB dynamic spectra,
# applying Faraday rotation, dispersion, noise, and polarisation effects.
# It includes both single-Gaussian and multi-Gaussian (micro-shot) models,
# as well as helper routines for Stokes parameter calculation.
#
# Author: JB
# Date: 2025-05-20
# -----------------------------------------------------------------------------

#	--------------------------	Import modules	---------------------------


import logging

import numpy as np

from fires.core.basicfns import (
	add_noise,
	compute_required_sefd,
	compute_segments,
	scatter_dspec,
	rm_correct_dspec,
	estimate_rm,
	on_off_pulse_masks_from_profile,
	estimate_noise_with_offpulse_mask,
)
from fires.scint.lib_ScintillationMaker import simulate_scintillation
from fires.utils.utils import gaussian_model, speed_of_light_cgs

logging.basicConfig(level=logging.INFO)

GAUSSIAN_FWHM_FACTOR = 2 * np.sqrt(2 * np.log(2)) 

#    --------------------------	Functions ---------------------------

def _apply_faraday_rotation(pol_angle_arr, RM, freq_mhz, ref_freq_mhz):
	"""
	Return polarisation angle (radians) after Faraday rotation, using ref_freq_mhz
	as the zero-rotation reference: chi = chi0 + RM * (lambda^2 - lambda_ref^2).
	"""
	chi0 = np.deg2rad(pol_angle_arr)
	lambda_sq = (speed_of_light_cgs * 1.0e-8 / np.asarray(freq_mhz, dtype=float)) ** 2
	lambda_ref_sq = (speed_of_light_cgs * 1.0e-8 / float(ref_freq_mhz)) ** 2
	return chi0 + RM * (lambda_sq - lambda_ref_sq)


def _calculate_dispersion_delay(DM, freq, ref_freq):
	return 4.15 * DM * ((1.0e3 / freq) ** 2 - (1.0e3 / ref_freq) ** 2)


def _roll_rows(arr: np.ndarray, shifts: np.ndarray) -> np.ndarray:
	"""Vectorised np.roll for a (n_rows, n_cols) array with per-row integer shifts."""
	arr = np.asarray(arr)
	nr, nc = arr.shape
	sh = np.asarray(shifts, dtype=int).reshape(nr, 1)
	idx = (np.arange(nc)[None, :] - sh) % nc  # positive shift -> right roll
	return np.take_along_axis(arr, idx, axis=1)


def apply_scintillation(dspec, freq_mhz, time_ms, scint_dict, ref_freq_mhz):
	"""
	Apply diffractive scintillation as a multiplicative gain field to all Stokes.

	scint_dict keys (with defaults):
		t_s        : scintillation timescale (seconds)
		nu_s       : scintillation bandwidth   (Hz)
		N_im       : number of phasor screen components (default 1000)
		th_lim     : angular truncation parameter (default 3.0)
		seed       : optional RNG seed (for reproducibility)
		field      : if True return complex field (else only gain)
		mod_scale  : optional factor to scale modulation index (gain -> 1 + (gain-1)*mod_scale)

	Returns:
		gain (nf, nt) applied in-place to dspec.
	"""
	t_s          = float(scint_dict.get("t_s"))
	nu_s         = float(scint_dict.get("nu_s"))
	N_im         = int(scint_dict.get("N_im", 1000))
	th_lim       = float(scint_dict.get("th_lim", 3.0))
	return_field = bool(scint_dict.get("field", False))

	# Convert grids
	t_sec = time_ms * 1e-3
	nu_hz = freq_mhz * 1e6
	ref_freq_hz = ref_freq_mhz * 1e6

	logging.info(f"Applying scintillation: t_s={t_s}s, nu_s={np.round(nu_s,2)}Hz, N_im={N_im}, th_lim={th_lim}")

	# Simulate complex field: shape (N_t, N_nu)
	E = simulate_scintillation(t_sec, nu_hz, t_s=t_s, nu_s=nu_s, N_im=N_im, th_lim=th_lim, ref_freq_hz=ref_freq_hz)

	# Intensity gain
	gain_tn = np.abs(E) ** 2
	gain_tn /= np.mean(gain_tn)

	# Reorder to (nf, nt)
	gain_fn = gain_tn.T  # (N_nu, N_t)

	# Apply in-place to all Stokes
	dspec *= gain_fn[None, :, :]

	if return_field:
		return gain_fn, E.T  # (nf, nt) each
	return gain_fn, None


def _init_seed(seed: int | None, plot_multiple_frb: bool) -> int:
	"""
	Ensure a concrete seed. If None, draw one from OS entropy, set NumPy RNG, and print it.
	Returns the seed used.
	"""
	if seed is None:
		seed = int(np.random.SeedSequence().generate_state(1)[0])
		np.random.seed(seed)
		if not plot_multiple_frb:
			logging.info(f"Using random seed: {seed}")
	else:
		np.random.seed(seed)
	return seed
	

def _disable_micro_variance_for_swept_base(sd_dict, xname):
	"""
	If sweeping a base parameter (e.g., 'tau'), disable its random micro-variance
	by zeroing the corresponding '*_sd' entry (e.g., 'tau_sd') so the sweep
	reflects only the base change.
	Returns a modified copy of sd_dict (shallow copy; values may be arrays).
	"""
	if xname is None or xname.endswith("_sd"):
		return sd_dict

	var_key = xname + "_sd"
	if var_key is None or var_key not in sd_dict:
		return sd_dict

	# Shallow copy dict; copy value to avoid in-place side effects
	new_sd_dict = dict(sd_dict)
	val = new_sd_dict[var_key]
	arr = np.array(val, dtype=float, copy=True)
	if arr.ndim == 0:
		arr = np.array([arr], dtype=float)
	arr[0] = 0.0
	new_sd_dict[var_key] = arr
	return new_sd_dict


def _make_scattering_kernel(t_ms: np.ndarray, tau: float) -> np.ndarray:
	"""
	Exponential scattering kernel h(t) = (1/tau) exp(-t/tau) for t>=0; delta if tau<=0.
	"""
	if tau is None or tau <= 0:
		h = np.zeros_like(t_ms, dtype=float)
		h[np.argmin(np.abs(t_ms))] = 1.0
		return h
	h = np.zeros_like(t_ms, dtype=float)
	mask = t_ms >= 0.0
	# Avoid division by zero in very small tau
	den = max(float(tau), 1e-12)
	h[mask] = np.exp(-t_ms[mask] / den) / den
	return h


def _gaussian_on_grid(t_ms: np.ndarray, sigma_ms: float, normalise: str) -> np.ndarray:
	"""
	normalise: 'area' -> unit area (sum*dt=1); 'peak' -> unit peak (max=1).
	If sigma<=0, returns a discrete delta at 0 with value 1.0.
	"""
	t = np.asarray(t_ms, dtype=float)
	if sigma_ms is None or sigma_ms <= 0:
		g = np.zeros_like(t, dtype=float)
		g[np.argmin(np.abs(t))] = 1.0
		return g

	# Raw Gaussian shape (amp=1 at mean)
	g = gaussian_model(t, amp=1.0, mean=0.0, stddev=float(sigma_ms))

	if normalise == "area":
		dt = float(t[1] - t[0])
		norm = np.sum(g) * dt
		return g / norm if norm > 0 else g
	elif normalise == "peak":
		mx = float(np.max(g))
		return g / mx if mx > 0 and np.isfinite(mx) else g
	else:
		return g 


def _unit_fluence_response(t_ms: np.ndarray, sigma_w_ms: float, tau: float) -> np.ndarray:
	"""
	Build unit-fluence (unit-area) temporal response h_tau(t; w):
	  1) s(t; w) = unit-area Gaussian (fluence-normalised)
	  2) k_tau(t) = unit-area exponential scattering kernel (or delta if tau<=0)
	  3) h = (s * k_tau) · dt  (automatically unit area since both are)
	"""
	t = np.asarray(t_ms, dtype=float)
	dt = float(t[1] - t[0])

	# s: unit-area Gaussian
	s = _gaussian_on_grid(t, float(sigma_w_ms), normalise="area")

	# k: unit-area scattering kernel (delta if tau<=0)
	tau_eff = float(tau) if (tau is not None and float(tau) > 0) else 0.0
	k = _make_scattering_kernel(t, tau_eff)

	# Convolution (integral)
	h = np.convolve(s, k, mode="same") * dt

	# Normalise area again (numerical drift)
	area = np.sum(h) * dt
	if area > 0 and np.isfinite(area):
		h /= area
	else:
		h = np.zeros_like(t)
		h[np.argmin(np.abs(t))] = 1.0
	return h



def _triple_convolution_with_width_pdf(
	t_ms: np.ndarray,
	tau: float,
	f_arrival: np.ndarray,
	width_mean_fwhm_ms: float,
	mg_width_low: float | None,
	mg_width_high: float | None,
	n_width_samples: int = 31,
) -> tuple[np.ndarray, np.ndarray]:
	"""
	Compute (h*f*g)(t) and (h^2*f*g)(t) where:
	  - h is unit-peak temporal response after scattering,
	  - f is the arrival-time Gaussian PDF (unit area),
	  - g is the width PDF (unit area), here approximated by a uniform PDF over [w_lo, w_hi].

	Returns:
		hf (nt,), h2f (nt,)
	"""
	t = np.asarray(t_ms, dtype=float)
	dt = float(t[1] - t[0])

	# Width sampling (uniform g(w) over [w_lo, w_hi])
	w0 = float(width_mean_fwhm_ms)
	if mg_width_low is None or mg_width_high is None:
		w_samples = np.array([w0], dtype=float)
	else:
		w_lo = max(1e-12, w0 * float(mg_width_low) / 100.0)
		w_hi = max(w_lo,   w0 * float(mg_width_high) / 100.0)
		if np.isclose(w_lo, w_hi):
			w_samples = np.array([w0], dtype=float)
		else:
			nw = max(1, int(n_width_samples))
			w_samples = np.linspace(w_lo, w_hi, nw, dtype=float)

	hf_list, h2f_list, area_list = [], [], []
	for w in w_samples:
		sigma_w = float(w) / GAUSSIAN_FWHM_FACTOR
		h_w = _unit_fluence_response(t, sigma_w_ms=sigma_w, tau=tau)
		area = np.sum(h_w) * dt
		hf_w  = np.convolve(h_w,    f_arrival, mode="same") * dt
		h2f_w = np.convolve(h_w**2, f_arrival, mode="same") * dt

		hf_list.append(hf_w)
		h2f_list.append(h2f_w)
		area_list.append(area)

	area_arr = np.array(area_list)
	hf_arr = np.stack(hf_list, axis=0)
	h2f_arr = np.stack(h2f_list, axis=0)

	# Fluence-weighted averages (see derivation)
	hf  = np.average(area_arr[:, None] * hf_arr, axis=0)
	h2f = np.average((area_arr[:, None]**2) * h2f_arr, axis=0)
	return hf, h2f


def _expected_pa_variance(
	tau, sigma_deg, N, width, A, A_sd,
	time_ms=None, mg_width_low=None, mg_width_high=None, n_width_samples: int = 31
):
	"""
	Compute scalar expected PA variance using time-averaged N_eff over the on-pulse region.
	Returns var_PA_deg2 (float).
	"""

	assert time_ms is not None and len(time_ms) >= 3

	# amplitude moments
	a_mean = float(np.nanmean(A)) if np.ndim(A) > 0 else float(A)
	a2_mean = a_mean**2 + float(A_sd)**2
	sigma_rad = np.deg2rad(float(sigma_deg))

	t = np.asarray(time_ms, dtype=float)
	t_rel = t - np.median(t)

	# arrival PDF
	sigma_arrival_ms = float(width) / GAUSSIAN_FWHM_FACTOR
	f = _gaussian_on_grid(t_rel, sigma_arrival_ms, normalise="area")

	hf, h2f = _triple_convolution_with_width_pdf(
		t_rel, float(tau) if tau is not None else 0.0, f,
		width_mean_fwhm_ms=float(width),
		mg_width_low=mg_width_low,
		mg_width_high=mg_width_high,
		n_width_samples=n_width_samples
	)

	N_eff_t = float(N) * (hf**2) / (h2f + 1e-300)  

	threshold = 0.5 * np.nanmax(hf)
	mask_on = hf >= threshold

	weights = hf[mask_on]
	N_eff = N_eff_t[mask_on]
	
	if np.nansum(weights) > 0:
		N_eff = float(np.nansum(N_eff * weights) / np.nansum(weights))
	else:
		N_eff = float(np.nanmedian(N_eff))

	var_PA_rad2 = (a2_mean / (4.0 * a_mean**2)) * (1.0 / N_eff) * np.sinh(4.0 * sigma_rad**2)
	var_PA_deg2 = var_PA_rad2 * (180.0 / np.pi)**2

	return float(var_PA_deg2), hf, h2f, N_eff_t, N_eff


def _expected_pa_variance_basic(
	width,
	mg_width_low,
	mg_width_high,
	tau,
	sigma_deg,
	N,
	time_res_ms
) -> tuple[float | None, float | None, dict]:
	"""
	Basic PA-variance estimator using broadened widths and an effective shots-per-time argument.

	Assumptions:
	  - Macro width W and micro width w are FWHM-like measures.
	  - Scattering broadens both: W_tot = sqrt(W^2 + tau^2), w_tot = sqrt(w^2 + tau^2).
	  - Effective number per time: N_eff_t = N * (w_tot / W_tot).
	  - PA_rms ~ sigma_deg / sqrt(N_eff_t), Var[PA] ~ PA_rms^2.

	Returns:
	  var_PA_deg2 (float|None), PA_rms_deg (float|None), aux (dict with components).
	"""
	W_fwhm = float(np.nanmean(np.asarray(width, dtype=float)))
	frac_micro = np.clip(0.5 * (float(mg_width_low) + float(mg_width_high)) / 100.0, 1e-6, None)
	w_fwhm = W_fwhm * frac_micro

	tau = float(np.nanmean(np.asarray(tau, dtype=float)))
	# Convert FWHM → σ
	sigma_W = W_fwhm / GAUSSIAN_FWHM_FACTOR
	sigma_w = w_fwhm / GAUSSIAN_FWHM_FACTOR

	# Variance addition (ex-Gaussian variance = σ_g^2 + τ^2)
	sigma_W_tot = np.sqrt(sigma_W**2 + tau**2)
	sigma_w_tot = np.sqrt(sigma_w**2 + tau**2)

	# Effective number scaling ~ fraction of independent shots per macro duration
	N_eff_t = float(N) * (sigma_w_tot / max(sigma_W_tot, 1e-12))

	# Effective number scaling ~ fraction of independent shots per macro duration
	ratio = sigma_w_tot / max(sigma_W_tot, 1e-12)
	N_eff_t = float(N) * ratio

	# Resolution correction in σ space
	sigma_res = time_res_ms / GAUSSIAN_FWHM_FACTOR
	if sigma_w_tot < sigma_res:
		N_eff_t *= sigma_w_tot / sigma_res

	# DOF correction for mean-subtracted variance: (M-1)/M = 1 - w_tot/W_tot
	dof_factor = max(0.0, 1.0 - ratio)

	sigma_pa = float(sigma_deg)
	var_PA_deg2 = (sigma_pa**2 / max(N_eff_t, 1e-12)) * dof_factor

	return var_PA_deg2



def plot_Neff_vs_time(time_ms, Neff_t):
	"""
	Plot N_eff(t) vs time.
	Args:
		time_ms: array of time bins (ms)
		Neff_t: array of N_eff(t) values (same length as time_ms)
		onpulse_mask: optional boolean mask for on-pulse region
		title: optional plot title
	"""
	import matplotlib.pyplot as plt
	
	plt.figure(figsize=(8, 4))
	plt.plot(time_ms, Neff_t, label=r'$N_\mathrm{eff}(t)$', color='C0')

	plt.xlabel('Time (ms)')
	plt.ylabel(r'$N_\mathrm{eff}(t)$')

	plt.legend()
	plt.tight_layout()
	plt.show()
	


def psn_dspec(
	dspec_params,
	plot_multiple_frb,
	variation_parameter=None,
	xname=None,
	target_snr=None
):
	"""
	Generate dynamic spectrum from microshots (polarised shot noise) with optional parameter sweeping.

	Sweep modes:
	  none      : Do not modify means or variances based on sweep values.
	  mean      : (Case 1) Replace the mean of the swept parameter with each sweep value
				  and force its micro variance to zero.
	  variance  : (Case 2) Keep mean fixed; set the micro variance (std dev) to the sweep value.
	"""

	gdict        = dspec_params.gdict
	sd_dict      = dspec_params.sd_dict
	scint_dict   = dspec_params.scint_dict
	freq_mhz     = dspec_params.freq_mhz
	time_ms      = dspec_params.time_ms
	time_res_ms  = dspec_params.time_res_ms
	seed         = dspec_params.seed
	sefd         = dspec_params.sefd
	sc_idx       = dspec_params.sc_idx
	ref_freq_mhz = dspec_params.ref_freq_mhz
	buffer_frac  = dspec_params.buffer_frac
	sweep_mode   = dspec_params.sweep_mode

	seed = _init_seed(seed, plot_multiple_frb)

	gdict = {k: np.array(v, copy=True) for k, v in gdict.items()}
	sd_dict = {k: np.array(v, copy=True) for k, v in sd_dict.items()}

	is_mean_sweep = (sweep_mode == "mean")
	is_variance_sweep = (sweep_mode == "sd")

	if variation_parameter is not None and xname is not None and sweep_mode != "none":
		if is_variance_sweep:
			var_key = f"sd_{xname}"
			if var_key not in sd_dict:
				raise ValueError(f"Variance key '{var_key}' not found for standard deviation sweep.\n",
									f"Ensure {xname} exists in gparams.toml and can be varied.")
			arr = np.array(sd_dict[var_key], copy=True)
			if arr.ndim == 0:
				arr = np.array([arr], dtype=float)
			arr[0] = float(variation_parameter)
			sd_dict[var_key] = arr
		elif is_mean_sweep:
			if xname not in gdict:
				raise ValueError(f"Base parameter '{xname}' not found in gdict for mean sweep.\n",
									f"Ensure {xname} exists in gparams.toml.")
			base = np.array(gdict[xname], copy=True)
			if base.ndim == 0:
				gdict[xname] = float(variation_parameter)
			else:
				gdict[xname] = np.full_like(base, float(variation_parameter), dtype=float)

	# Disable micro variance if mean sweep so only deterministic change remains
	if is_mean_sweep:
		sd_dict = _disable_micro_variance_for_swept_base(sd_dict, xname)
	
	t0              = gdict['t0']
	width       	= gdict['width']
	A               = gdict['A']
	spec_idx        = gdict['spec_idx']
	tau 	   		= gdict['tau']
	PA              = gdict['PA']
	DM              = gdict['DM']
	RM              = gdict['RM']
	lfrac           = gdict['lfrac']
	vfrac           = gdict['vfrac']
	dPA             = gdict['dPA']
	band_centre_mhz = gdict['band_centre_mhz']
	band_width_mhz  = gdict['band_width_mhz']
	N		    	= gdict['N']
	mg_width_low    = gdict['mg_width_low']
	mg_width_high   = gdict['mg_width_high']

	width_range = [[mg_width_low[i], mg_width_high[i]] for i in range(len(mg_width_low))]

	sd_A               = sd_dict['sd_A']
	sd_spec_idx        = sd_dict['sd_spec_idx']
	sd_tau          = sd_dict['sd_tau']
	sd_PA              = sd_dict['sd_PA']
	sd_dm              = sd_dict['sd_DM']
	sd_rm              = sd_dict['sd_RM']
	sd_lfrac           = sd_dict['sd_lfrac']
	sd_vfrac           = sd_dict['sd_vfrac']
	sd_dPA             = sd_dict['sd_dPA']
	sd_band_centre_mhz = sd_dict['sd_band_centre_mhz']
	sd_band_width_mhz  = sd_dict['sd_band_width_mhz']
	
			
	dspec = np.zeros((4, freq_mhz.shape[0], time_ms.shape[0]), dtype=float)  # Initialise dynamic spectrum array

	all_params = {
		't0_i'             : [],
		'A_i'              : [],
		'mg_width_i'       : [],
		'spec_idx_i'       : [],
		'tau_i'         : [],
		'PA_i'             : [],
		'DM_i'             : [],
		'RM_i'             : [],
		'lfrac_i'          : [],
		'vfrac_i'          : [],
		'dPA_i'            : [],
		'band_centre_mhz_i': [],
		'band_width_mhz_i' : []
	}

	num_main_gauss = len(t0) 
	for g in range(num_main_gauss):
		for _ in range(int(N[g])):
			t0_i              = np.random.normal(t0[g], width[g] / GAUSSIAN_FWHM_FACTOR)
			A_i        		  = np.random.normal(A[g], sd_A)
			mg_width_i        = width[g] * np.random.uniform(width_range[g][0] / 100, width_range[g][1] / 100)
			spec_idx_i        = np.random.normal(spec_idx[g], sd_spec_idx)
			tau_i          = np.random.normal(tau[g], sd_tau)
			tau_eff = tau_i if tau_i > 0 else float(tau[g])
			if tau_eff > 0:
				tau_cms = tau_eff * (freq_mhz / ref_freq_mhz) ** sc_idx
			else:
				tau_cms = None

			PA_i              = np.random.normal(PA[g], sd_PA)
			DM_i              = np.random.normal(DM[g], sd_dm)
			RM_i              = np.random.normal(RM[g], sd_rm)
			lfrac_i           = np.random.normal(lfrac[g], sd_lfrac)
			vfrac_i           = np.random.normal(vfrac[g], sd_vfrac)
			dPA_i             = np.random.normal(dPA[g], sd_dPA)
			band_centre_mhz_i = np.random.normal(band_centre_mhz[g], sd_band_centre_mhz)
			band_width_mhz_i  = np.random.normal(band_width_mhz[g], sd_band_width_mhz)

			if sd_vfrac > 0.0:
				vfrac_i = np.clip(vfrac_i, 0.0, 1.0)
				lfrac_i = np.clip(1.0 - vfrac_i, 0.0, 1.0)
			elif sd_lfrac > 0.0:
				lfrac_i = np.clip(lfrac_i, 0.0, 1.0)
				vfrac_i = np.clip(1.0 - lfrac_i, 0.0, 1.0)

			# Record parameters
			all_params['t0_i'].append(t0_i)
			all_params['A_i'].append(A_i)
			all_params['mg_width_i'].append(mg_width_i)
			all_params['spec_idx_i'].append(spec_idx_i)
			all_params['tau_i'].append(tau_i)
			all_params['PA_i'].append(PA_i)
			all_params['DM_i'].append(DM_i)
			all_params['RM_i'].append(RM_i)
			all_params['lfrac_i'].append(lfrac_i)
			all_params['vfrac_i'].append(vfrac_i)
			all_params['dPA_i'].append(dPA_i)
			all_params['band_centre_mhz_i'].append(band_centre_mhz_i)
			all_params['band_width_mhz_i'].append(band_width_mhz_i)

			# Vectorised micro-shot synthesis
			norm_amp = A_i * (freq_mhz / ref_freq_mhz) ** spec_idx_i
			if band_width_mhz[g] != 0.:
				centre_freq = band_centre_mhz_i if band_centre_mhz_i != 0. else np.median(freq_mhz)
				bw_sigma = band_width_mhz_i / GAUSSIAN_FWHM_FACTOR
				if bw_sigma > 0:
					spectral_profile = gaussian_model(freq_mhz, 1.0, centre_freq, bw_sigma)
					norm_amp *= spectral_profile

			base_gauss = gaussian_model(time_ms, 1.0, t0_i, mg_width_i / GAUSSIAN_FWHM_FACTOR)
			I_ft = norm_amp[:, None] * base_gauss[None, :]

			if DM_i != 0:
				shifts = np.round(_calculate_dispersion_delay(DM_i, freq_mhz, ref_freq_mhz) / time_res_ms).astype(int)
				I_ft = _roll_rows(I_ft, shifts)

			pol_angle_arr = PA_i + (time_ms - t0_i) * dPA_i
			faraday_angles = _apply_faraday_rotation(pol_angle_arr[None, :], RM_i, freq_mhz[:, None], ref_freq_mhz)

			Q_ft = I_ft * lfrac_i * np.cos(2 * faraday_angles)
			U_ft = I_ft * lfrac_i * np.sin(2 * faraday_angles)
			V_ft = I_ft * vfrac_i

			if tau_eff > 0:
				I_ft = scatter_dspec(I_ft, time_res_ms, tau_cms)
				Q_ft = scatter_dspec(Q_ft, time_res_ms, tau_cms)
				U_ft = scatter_dspec(U_ft, time_res_ms, tau_cms)
				V_ft = scatter_dspec(V_ft, time_res_ms, tau_cms)

			dspec[0] += I_ft
			dspec[1] += Q_ft
			dspec[2] += U_ft
			dspec[3] += V_ft

	if scint_dict is not None:
		apply_scintillation(dspec, freq_mhz, time_ms, scint_dict, ref_freq_mhz)

	try:
		# On/off mask from frequency-summed I
		I_ts = np.nansum(dspec[0], axis=0)
		intrinsic_width_bins = gdict["width"][0] / time_res_ms
		_, offpulse_mask, _ = on_off_pulse_masks_from_profile(
			I_ts, intrinsic_width_bins=intrinsic_width_bins, frac=0.95, buffer_frac=buffer_frac
		)
		_, noisespec = estimate_noise_with_offpulse_mask(dspec, offpulse_mask, robust=True)

		res_rmtool = estimate_rm(
			dspec, freq_mhz, time_ms, noisespec,
			phi_range=1.0e3, dphi=1.0, outdir='.', save=False, show_plots=False
		)
		measured_rm = float(res_rmtool[0])

		def _int_Lfrac(cube):
			I = np.nansum(cube[0], axis=0)
			Q = np.nansum(cube[1], axis=0)
			U = np.nansum(cube[2], axis=0)
			on_mask, _, _ = on_off_pulse_masks_from_profile(
				I, intrinsic_width_bins=intrinsic_width_bins, frac=0.95, buffer_frac=buffer_frac
			)
			I_int = float(np.nansum(I[on_mask]))
			L_int = float(np.nansum(np.sqrt(Q[on_mask]**2 + U[on_mask]**2)))
			return (L_int / I_int) if I_int > 0 else 0.0

		if np.isfinite(measured_rm) and np.abs(measured_rm) > 0.0:
			cand_pos = rm_correct_dspec(dspec, freq_mhz, +measured_rm, ref_freq_mhz=ref_freq_mhz)
			cand_neg = rm_correct_dspec(dspec, freq_mhz, -measured_rm, ref_freq_mhz=ref_freq_mhz)
			Lpos = _int_Lfrac(cand_pos)
			Lneg = _int_Lfrac(cand_neg)
			dspec = cand_pos if Lpos >= Lneg else cand_neg
			chosen_sign = '+' if Lpos >= Lneg else '-'
			Lbest = max(Lpos, Lneg)
			logging.info("Measured RM = %.2f rad/m2; applied derotation (ref=%.1f MHz, sign=%s); L/I=%.3f",
			             measured_rm, ref_freq_mhz, chosen_sign, Lbest)
		else:
			logging.info("Measured RM not significant; skipping RM correction")
	except Exception as e:
		logging.warning("RM measurement/derotation in psn_dspec failed (%s). Proceeding without derotation.", str(e))

	V_params = {}
	for key, values in all_params.items():
		arr = np.asarray(values, dtype=float)
		var = float(np.nanvar(arr)) if arr.size else np.nan
		V_params[f"meas_var_{key}"] = var

	f_res_hz = (freq_mhz[1] - freq_mhz[0]) * 1e6 
	t_res_s = time_res_ms / 1000.0

	if target_snr is not None:
		sefd_est = compute_required_sefd(
			dspec,
			f_res_hz=f_res_hz,
			t_res_s=t_res_s,
			target_snr=target_snr,
			n_pol=2,
			frac=0.95,
			buffer_frac=buffer_frac,
		)
		sefd_work = sefd_est
		max_iter, tol = 5, 0.02
		for _ in range(max_iter):
			_, _, snr_meas = add_noise(dspec_params,
				dspec, sefd_work, f_res_hz, t_res_s,
				plot_multiple_frb=True, buffer_frac=buffer_frac,
				n_pol=2
			)
			if snr_meas <= 0: break
			ratio = snr_meas / target_snr
			if abs(ratio - 1) < tol: break
			sefd_work *= ratio
		sefd = sefd_work

		if not plot_multiple_frb:
			print(f"SEFD set to {sefd:.3g} Jy for target S/N {target_snr}")

	if sefd > 0:
		dspec, sigma_ch, snr = add_noise(dspec_params,
			dspec, sefd, f_res_hz, t_res_s,
			plot_multiple_frb, buffer_frac=buffer_frac, n_pol=2
		)
	else:
		snr = None

	N_tot = int(np.nansum(N))
	exp_V_PA_deg2 = None
	if sd_PA > 0 and N_tot > 1:
		actual_A_mean = np.mean(A)
		actual_A_std = np.std(A)
		actual_width_mean = np.mean(width)
		actual_tau_mean = np.mean(tau)
		exp_V_PA_deg2, _, _, _, N_eff_diag = _expected_pa_variance(
			tau=actual_tau_mean,  
			sigma_deg=float(sd_PA),
			N=N_tot,
			width=actual_width_mean,
			A=actual_A_mean,
			A_sd=actual_A_std,
			time_ms=time_ms,
			mg_width_low=float(mg_width_low) if np.ndim(mg_width_low) == 0 else float(mg_width_low[0]),
			mg_width_high=float(mg_width_high) if np.ndim(mg_width_high) == 0 else float(mg_width_high[0]),
			n_width_samples=100,
		)

		# Compute on-pulse mask as in your code
		#plot_Neff_vs_time(time_ms, N_eff_t_diag)

		exp_V_PA_deg2_basic = _expected_pa_variance_basic(
			width=float(np.nanmean(width)),
			mg_width_low=float(np.nanmean(mg_width_low)),
			mg_width_high=float(np.nanmean(mg_width_high)),
			tau=float(tau) if np.ndim(tau) == 0 else float(tau[0]),
			sigma_deg=float(sd_PA),
			N=N_tot,
			time_res_ms=time_res_ms
		)

		if not plot_multiple_frb:
			print(f"tau={tau[0]:.2f}:"
				  f"Expected(detailed)={exp_V_PA_deg2:.2f}, "
				  f"Expected(basic)={exp_V_PA_deg2_basic:.2f}")

	else:
		exp_V_PA_deg2 = None
		exp_V_PA_deg2_basic = None

	exp_vars = {
		'exp_var_t0'             : None,
		'exp_var_A'       		 : None,
		'exp_var_width'       : None,
		'exp_var_spec_idx'       : None,
		'exp_var_tau'         : None,
		'exp_var_PA'             : [exp_V_PA_deg2, exp_V_PA_deg2_basic],
		'exp_var_DM'             : None,
		'exp_var_RM'             : None,
		'exp_var_lfrac'          : None,
		'exp_var_vfrac'          : None,
		'exp_var_dPA'            : None,
		'exp_var_band_centre_mhz': None,
		'exp_var_band_width_mhz' : None
	}

	return dspec, snr, V_params, exp_vars, compute_segments(dspec, freq_mhz, time_ms, dspec_params, buffer_frac, skip_rm=True)

