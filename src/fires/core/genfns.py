# -----------------------------------------------------------------------------
# genfns.py
# FIRES: The Fast, Intense Radio Emission Simulator
#
# This module provides core functions for generating FRB dynamic spectra,
# applying Faraday rotation, dispersion, noise, and polarization effects.
# It includes both single-Gaussian and multi-Gaussian (micro-shot) models,
# as well as helper routines for Stokes parameter calculation.
#
# Author: JB
# Date: 2025-05-20
# -----------------------------------------------------------------------------

#	--------------------------	Import modules	---------------------------


import logging

import numpy as np

from fires.core.basicfns import add_noise, compute_required_sefd, scatter_dspec
from fires.scint.lib_ScintillationMaker import simulate_scintillation
from fires.utils.utils import gaussian_model, speed_of_light_cgs

logging.basicConfig(level=logging.INFO)

GAUSSIAN_FWHM_FACTOR = 2 * np.sqrt(2 * np.log(2)) 

#    --------------------------	Functions ---------------------------

def _apply_faraday_rotation(pol_angle_arr, RM, lambda_sq, median_lambda_sq):
	return np.deg2rad(pol_angle_arr) + RM * (lambda_sq - median_lambda_sq)


def _calculate_dispersion_delay(DM, freq, ref_freq):
	return 4.15 * DM * ((1.0e3 / freq) ** 2 - (1.0e3 / ref_freq) ** 2)

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
	If sweeping a base parameter (e.g., 'tau_ms'), disable its random micro-variance
	by zeroing the corresponding '*_sd' entry (e.g., 'tau_ms_sd') so the sweep
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


def _make_scattering_kernel(t_ms: np.ndarray, tau_ms: float) -> np.ndarray:
	"""
	Exponential scattering kernel h(t) = (1/tau) exp(-t/tau) for t>=0; delta if tau<=0.
	"""
	if tau_ms is None or tau_ms <= 0:
		h = np.zeros_like(t_ms, dtype=float)
		h[np.argmin(np.abs(t_ms))] = 1.0
		return h
	h = np.zeros_like(t_ms, dtype=float)
	mask = t_ms >= 0.0
	# Avoid division by zero in very small tau
	den = max(float(tau_ms), 1e-12)
	h[mask] = np.exp(-t_ms[mask] / den) / den
	return h


def _gaussian_on_grid(t_ms: np.ndarray, sigma_ms: float, normalize: str) -> np.ndarray:
    """
    Common Gaussian builder using utils.gaussian_model, with grid-safe normalisation.
    normalize: 'area' -> unit area (sum*dt=1); 'peak' -> unit peak (max=1).
    If sigma<=0, returns a discrete delta at 0 with value 1.0.
    """
    t = np.asarray(t_ms, dtype=float)
    if sigma_ms is None or sigma_ms <= 0:
        g = np.zeros_like(t, dtype=float)
        g[np.argmin(np.abs(t))] = 1.0
        return g

    # Raw Gaussian shape (amp=1 at mean)
    g = gaussian_model(t, amp=1.0, mean=0.0, stddev=float(sigma_ms))

    if normalize == "area":
        dt = float(t[1] - t[0])
        norm = np.sum(g) * dt
        return g / norm if norm > 0 else g
    elif normalize == "peak":
        mx = float(np.max(g))
        return g / mx if mx > 0 and np.isfinite(mx) else g
    else:
        return g 


def _unit_peak_response(t_ms: np.ndarray, sigma_w_ms: float, tau_ms: float) -> np.ndarray:
    """
    Build unit-peak temporal response h_tau(t; w):
      1) s(t; w) = unit-peak Gaussian (shape), std = sigma_w_ms
      2) k_tau(t) = unit-area exponential kernel (or delta if tau<=0)
      3) h_raw = (s * k_tau) · dt
      4) h = h_raw / max(h_raw)  (unit peak)
    """
    t = np.asarray(t_ms, dtype=float)
    dt = float(t[1] - t[0])

    # s: unit-peak Gaussian
    s = _gaussian_on_grid(t, float(sigma_w_ms), normalize="peak")

    # k: unit-area scattering kernel (delta if tau<=0)
    tau_eff = float(tau_ms) if (tau_ms is not None and float(tau_ms) > 0) else 0.0
    k = _make_scattering_kernel(t, tau_eff)

    # Convolution (integral) and unit-peak normalisation
    h_raw = np.convolve(s, k, mode="same") * dt
    mx = float(np.nanmax(h_raw)) if np.any(np.isfinite(h_raw)) else 0.0
    if mx <= 0 or not np.isfinite(mx):
        h = np.zeros_like(t, dtype=float)
        h[np.argmin(np.abs(t))] = 1.0
    else:
        h = h_raw / mx
    return h


def _triple_convolution_with_width_pdf(
    t_ms: np.ndarray,
    tau_ms: float,
    f_arrival: np.ndarray,
    width_mean_fwhm_ms: float,
    width_pct_low: float | None,
    width_pct_high: float | None,
    n_width_samples: int = 31,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute (h*f*g)(t) and (h^2*f*g)(t) where:
      - h is unit-peak temporal response after scattering,
      - f is the arrival-time Gaussian PDF (unit area),
      - g is the width PDF (unit area), here approximated by a uniform PDF over [w_lo, w_hi].

    Returns:
        hfg (nt,), h2fg (nt,)
    """
    t = np.asarray(t_ms, dtype=float)
    dt = float(t[1] - t[0])

    # Width sampling (uniform g(w) over [w_lo, w_hi])
    w0 = float(width_mean_fwhm_ms)
    if width_pct_low is None or width_pct_high is None:
        w_samples = np.array([w0], dtype=float)
    else:
        w_lo = max(1e-12, w0 * float(width_pct_low) / 100.0)
        w_hi = max(w_lo,   w0 * float(width_pct_high) / 100.0)
        if np.isclose(w_lo, w_hi):
            w_samples = np.array([w0], dtype=float)
        else:
            nw = max(1, int(n_width_samples))
            w_samples = np.linspace(w_lo, w_hi, nw, dtype=float)

    hf_list, h2f_list = [], []
    for w in w_samples:
        sigma_w = float(w) / GAUSSIAN_FWHM_FACTOR
        # Build h(t; w) as unit-peak after scattering
        h_w = _unit_peak_response(t, sigma_w_ms=sigma_w, tau_ms=tau_ms)
        # Convolve with arrival PDF (unit area), include dt for integral
        hf_w  = np.convolve(h_w,           f_arrival, mode="same") * dt
        h2f_w = np.convolve(h_w * h_w,     f_arrival, mode="same") * dt
        hf_list.append(hf_w)
        h2f_list.append(h2f_w)

    # Average over uniform g(w)
	# final averaging over widths is effectively the integration over g(w)
    hf  = np.mean(np.stack(hf_list,  axis=0), axis=0)
    h2f = np.mean(np.stack(h2f_list, axis=0), axis=0)
    return hf, h2f



def _expected_pa_variance(
	tau_ms, sigma_deg, ngauss, width_ms, peak_amp, peak_amp_sd,
	time_ms=None, width_pct_low=None, width_pct_high=None, n_width_samples: int = 31,
	onpulse_fraction: float = 0.1
):
	"""
	Compute scalar expected PA variance using time-averaged N_eff over the on-pulse region.
	Returns var_psi_deg2 (float).
	"""
	# Basic checks
	assert time_ms is not None and len(time_ms) >= 3

	# amplitude moments (peak-flux moments)
	a_mean = float(np.nanmean(peak_amp)) if np.ndim(peak_amp) > 0 else float(peak_amp)
	a2_mean = a_mean**2 + float(peak_amp_sd)**2
	sigma_rad = np.deg2rad(float(sigma_deg))

	t = np.asarray(time_ms, dtype=float)
	t_rel = t - np.median(t)

	# arrival PDF
	sigma_arrival_ms = float(width_ms) / GAUSSIAN_FWHM_FACTOR
	f = _gaussian_on_grid(t_rel, sigma_arrival_ms, normalize="area")

	# compute (h*f*g) and (h^2*f*g)
	hfg, h2fg = _triple_convolution_with_width_pdf(
		t_rel, float(tau_ms) if tau_ms is not None else 0.0, f,
		width_mean_fwhm_ms=float(width_ms),
		width_pct_low=width_pct_low,
		width_pct_high=width_pct_high,
		n_width_samples=n_width_samples
	)

	# per-time N_eff
	N_eff_t = float(ngauss) * (hfg**2) / (h2fg + 1e-300)  # avoid zeros

	# on-pulse mask: relative to peak of hfg
	threshold = onpulse_fraction * np.nanmax(hfg)
	mask_on = (hfg >= threshold) & np.isfinite(hfg)

	if not np.any(mask_on):
		# fallback: top 50% by hfg
		thr = 0.5 * np.nanmax(hfg)
		mask_on = (hfg >= thr)

	N_eff_avg = np.nanmean(N_eff_t[mask_on])
	# guard against very small N_eff_avg
	if not np.isfinite(N_eff_avg) or N_eff_avg <= 0:
		N_eff_avg = max(1.0, float(ngauss))

	# variance prefactor (peak amplitude convention)
	pref = (a2_mean / (4.0 * a_mean**2)) * (1.0 / N_eff_avg) * np.sinh(4.0 * sigma_rad**2)

	var_psi_rad2 = pref
	var_psi_deg2 = var_psi_rad2 * (180.0 / np.pi)**2
	return float(var_psi_deg2), hfg, h2fg, N_eff_t, N_eff_avg


def psn_dspec(freq_mhz, time_ms, time_res_ms, seed, gdict, sd_dict, scint_dict,
					sefd, sc_idx, ref_freq_mhz, plot_multiple_frb, buffer_frac, sweep_mode,
					variation_parameter=None, xname=None, target_snr=None):
	"""
	Generate dynamic spectrum from microshots (polarised shot noise) with optional parameter sweeping.

	Sweep modes:
	  none      : Do not modify means or variances based on sweep values.
	  mean      : (Case 1) Replace the mean of the swept parameter with each sweep value
				  and force its micro variance to zero.
	  variance  : (Case 2) Keep mean fixed; set the micro variance (std dev) to the sweep value.
	"""
	seed = _init_seed(seed, plot_multiple_frb)

	gdict = {k: np.array(v, copy=True) for k, v in gdict.items()}
	sd_dict = {k: np.array(v, copy=True) for k, v in sd_dict.items()}

	is_mean_sweep = (sweep_mode == "mean")
	is_variance_sweep = (sweep_mode == "variance")

	if variation_parameter is not None and xname is not None and sweep_mode != "none":
		if is_variance_sweep:
			var_key = f"{xname}_sd"
			if var_key not in sd_dict:
				raise ValueError(f"Variance key '{var_key}' not found for variance sweep.")
			arr = np.array(sd_dict[var_key], copy=True)
			if arr.ndim == 0:
				arr = np.array([arr], dtype=float)
			arr[0] = float(variation_parameter)
			sd_dict[var_key] = arr
		elif is_mean_sweep:
			if xname not in gdict:
				raise ValueError(f"Base parameter '{xname}' not found in gdict for mean sweep.")
			base = np.array(gdict[xname], copy=True)
			if base.ndim == 0:
				gdict[xname] = float(variation_parameter)
			else:
				gdict[xname] = np.full_like(base, float(variation_parameter), dtype=float)

	# Disable micro variance if mean sweep so only deterministic change remains
	if is_mean_sweep:
		sd_dict = _disable_micro_variance_for_swept_base(sd_dict, xname)
	
	t0              = gdict['t0']
	width_ms        = gdict['width_ms']
	peak_amp        = gdict['peak_amp']
	spec_idx        = gdict['spec_idx']
	tau_ms 	   		= gdict['tau_ms']
	PA              = gdict['PA']
	DM              = gdict['DM']
	RM              = gdict['RM']
	lfrac           = gdict['lfrac']
	vfrac           = gdict['vfrac']
	dPA             = gdict['dPA']
	band_centre_mhz = gdict['band_centre_mhz']
	band_width_mhz  = gdict['band_width_mhz']
	ngauss		    = gdict['ngauss']
	mg_width_low    = gdict['mg_width_low']
	mg_width_high   = gdict['mg_width_high']

	# Create width_range list with pairs of [mg_width_low, mg_width_high]
	width_range = [[mg_width_low[i], mg_width_high[i]] for i in range(len(mg_width_low))]

	peak_amp_sd        = sd_dict['peak_amp_sd']
	spec_idx_sd        = sd_dict['spec_idx_sd']
	tau_ms_sd          = sd_dict['tau_ms_sd']
	PA_sd              = sd_dict['PA_sd']
	dm_sd              = sd_dict['DM_sd']
	rm_sd              = sd_dict['RM_sd']
	lfrac_sd           = sd_dict['lfrac_sd']
	vfrac_sd           = sd_dict['vfrac_sd']
	dPA_sd             = sd_dict['dPA_sd']
	band_centre_mhz_sd = sd_dict['band_centre_mhz_sd']
	band_width_mhz_sd  = sd_dict['band_width_mhz_sd']
	
			
	dspec = np.zeros((4, freq_mhz.shape[0], time_ms.shape[0]), dtype=float)  # Initialise dynamic spectrum array
	lambda_sq = (speed_of_light_cgs * 1.0e-8 / freq_mhz) ** 2  # Lambda squared array
	median_lambda_sq = np.nanmedian(lambda_sq)  # Median lambda squared

	all_params = {
		'var_t0'             : [],
		'var_peak_amp'       : [],
		'var_width_ms'       : [],
		'var_spec_idx'       : [],
		'var_tau_ms'         : [],
		'var_PA'             : [],
		'var_DM'             : [],
		'var_RM'             : [],
		'var_lfrac'          : [],
		'var_vfrac'          : [],
		'var_dPA'            : [],
		'var_band_centre_mhz': [],
		'var_band_width_mhz' : []
	}

	num_main_gauss = len(t0) 
	for g in range(num_main_gauss):
		for _ in range(int(ngauss[g])):
			var_t0              = np.random.normal(t0[g], width_ms[g] / GAUSSIAN_FWHM_FACTOR)
			var_peak_amp        = np.random.normal(peak_amp[g], peak_amp_sd)
			var_width_ms        = width_ms[g] * np.random.uniform(width_range[g][0] / 100, width_range[g][1] / 100)
			var_spec_idx        = np.random.normal(spec_idx[g], spec_idx_sd)
			var_tau_ms          = np.random.normal(tau_ms[g], tau_ms_sd)
			tau_eff = var_tau_ms if var_tau_ms > 0 else float(tau_ms[g])
			if tau_eff > 0:
				tau_cms = tau_eff * (freq_mhz / ref_freq_mhz) ** sc_idx
			else:
				tau_cms = None

			var_PA              = np.random.normal(PA[g], PA_sd)
			var_DM              = np.random.normal(DM[g], dm_sd)
			var_RM              = np.random.normal(RM[g], rm_sd)
			var_lfrac           = np.random.normal(lfrac[g], lfrac_sd)
			var_vfrac           = np.random.normal(vfrac[g], vfrac_sd)
			var_dPA             = np.random.normal(dPA[g], dPA_sd)
			var_band_centre_mhz = np.random.normal(band_centre_mhz[g], band_centre_mhz_sd)
			var_band_width_mhz  = np.random.normal(band_width_mhz[g], band_width_mhz_sd)

			if vfrac_sd > 0.0:
				var_vfrac = np.clip(var_vfrac, 0.0, 1.0)
				var_lfrac = np.clip(1.0 - var_vfrac, 0.0, 1.0)
			elif lfrac_sd > 0.0:
				var_lfrac = np.clip(var_lfrac, 0.0, 1.0)
				var_vfrac = np.clip(1.0 - var_lfrac, 0.0, 1.0)

			# Record parameters
			all_params['var_t0'].append(var_t0)
			all_params['var_peak_amp'].append(var_peak_amp)
			all_params['var_width_ms'].append(var_width_ms)
			all_params['var_spec_idx'].append(var_spec_idx)
			all_params['var_tau_ms'].append(var_tau_ms)
			all_params['var_PA'].append(var_PA)
			all_params['var_DM'].append(var_DM)
			all_params['var_RM'].append(var_RM)
			all_params['var_lfrac'].append(var_lfrac)
			all_params['var_vfrac'].append(var_vfrac)
			all_params['var_dPA'].append(var_dPA)
			all_params['var_band_centre_mhz'].append(var_band_centre_mhz)
			all_params['var_band_width_mhz'].append(var_band_width_mhz)

			# Vectorised micro-shot synthesis
			norm_amp = var_peak_amp * (freq_mhz / ref_freq_mhz) ** var_spec_idx
			if band_width_mhz[g] != 0.:
				centre_freq = var_band_centre_mhz if var_band_centre_mhz != 0. else np.median(freq_mhz)
				bw_sigma = var_band_width_mhz / GAUSSIAN_FWHM_FACTOR
				if bw_sigma > 0:
					spectral_profile = gaussian_model(freq_mhz, 1.0, centre_freq, bw_sigma)
					norm_amp *= spectral_profile

			base_gauss = gaussian_model(time_ms, 1.0, var_t0, var_width_ms / GAUSSIAN_FWHM_FACTOR)
			I_ft = norm_amp[:, None] * base_gauss[None, :]

			if var_DM != 0:
				shifts = np.round(_calculate_dispersion_delay(var_DM, freq_mhz, ref_freq_mhz) / time_res_ms).astype(int)
				for c, s in enumerate(shifts):
					if s != 0:
						I_ft[c] = np.roll(I_ft[c], s)

			if tau_eff > 0:
				I_ft = scatter_dspec(I_ft, time_res_ms, tau_cms)

			pol_angle_arr = var_PA + (time_ms - var_t0) * var_dPA
			faraday_angles = _apply_faraday_rotation(pol_angle_arr[None, :], var_RM, lambda_sq[:, None], median_lambda_sq)

			Q_ft = I_ft * var_lfrac * np.cos(2 * faraday_angles)
			U_ft = I_ft * var_lfrac * np.sin(2 * faraday_angles)
			V_ft = I_ft * var_vfrac

			dspec[0] += I_ft
			dspec[1] += Q_ft
			dspec[2] += U_ft
			dspec[3] += V_ft

	# --- Apply scintillation if requested ---
	if scint_dict is not None:
		apply_scintillation(dspec, freq_mhz, time_ms, scint_dict, ref_freq_mhz)

	# Calculate variance for each parameter in var_params
	var_params = {key: np.nanvar(values) for key, values in all_params.items()}

	f_res_hz = (freq_mhz[1] - freq_mhz[0]) * 1e6 
	t_res_s = time_res_ms / 1000.0

	# --- Derive SEFD from target S/N if requested ---
	if target_snr is not None:
		sefd_est = compute_required_sefd(
			dspec,
			f_res_hz=f_res_hz,
			t_res_s=t_res_s,
			target_snr=target_snr,
			n_pol=2,
			frac=0.95,
			buffer_frac=buffer_frac,
			one_sided_offpulse=True,   # use only pre-pulse for noise
		)
		sefd_work = sefd_est
		max_iter, tol = 5, 0.02
		for _ in range(max_iter):
			_, _, snr_meas = add_noise(
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
		dspec, sigma_ch, snr = add_noise(
			dspec, sefd, f_res_hz, t_res_s,
			plot_multiple_frb, buffer_frac=buffer_frac, n_pol=2
		)
	else:
		snr = None

	# Expected PA variance (time-averaged N_eff)
	N_tot = int(np.nansum(ngauss))
	exp_V_psi_deg2 = None
	if PA_sd > 0 and N_tot > 1:
		exp_V_psi_deg2, hfg_diag, h2fg_diag, N_eff_t_diag, N_eff_avg_diag = _expected_pa_variance(
			tau_ms=float(np.nanmean(tau_ms)) if np.ndim(tau_ms) == 0 else float(np.nanmean(tau_ms)),
			sigma_deg=float(PA_sd),
			ngauss=N_tot,
			width_ms=float(np.nanmean(width_ms)),
			peak_amp=float(np.nanmean(peak_amp)),
			peak_amp_sd=float(peak_amp_sd),
			time_ms=time_ms,
			width_pct_low=float(np.nanmean(mg_width_low)),
			width_pct_high=float(np.nanmean(mg_width_high)),
			n_width_samples=31,
			onpulse_fraction=0.10
		)
		logging.info(f"Expected V(PA) (time-avg N_eff) ~ {exp_V_psi_deg2:.3f} deg^2 (N_eff_avg={N_eff_avg_diag:.1f})")
	else:
		exp_V_psi_deg2 = None

	exp_vars = {
		'exp_var_t0'             : None,
		'exp_var_peak_amp'       : None,
		'exp_var_width_ms'       : None,
		'exp_var_spec_idx'       : None,
		'exp_var_tau_ms'         : None,
		'exp_var_PA'             : exp_V_psi_deg2,
		'exp_var_DM'             : None,
		'exp_var_RM'             : None,
		'exp_var_lfrac'          : None,
		'exp_var_vfrac'          : None,
		'exp_var_dPA'            : None,
		'exp_var_band_centre_mhz': None,
		'exp_var_band_width_mhz' : None
	}

	return dspec, snr, var_params, exp_vars

