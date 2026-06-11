import logging

import numpy as np
from scipy.ndimage import gaussian_filter1d

from fires.utils.profiles import (boxcar_width, make_onpulse_mask,
                                    on_off_pulse_masks_from_profile)


def add_noise(dspec_params, dspec, sefd, f_res, t_res, plot_multiple_frb, buffer_frac, n_pol=2,
              stokes_scale=(1.0, 1.0, 1.0, 1.0), add_slow_baseline=False,
              baseline_frac=0.05, baseline_kernel_ms=5.0, time_res_ms=None):
    dspec = np.asarray(dspec, dtype=float)
    n_stokes, n_chan, n_time = dspec.shape
    sefd_arr = np.full(n_chan, sefd, dtype=float) if np.isscalar(sefd) else np.asarray(sefd, dtype=float)
    f_res_arr = np.full(n_chan, f_res, dtype=float) if np.isscalar(f_res) else np.asarray(f_res, dtype=float)
    sigma_I_ch = sefd_arr / np.sqrt(n_pol * f_res_arr * t_res)
    stokes_scale = np.asarray(stokes_scale, dtype=float)
    sigma_ch = np.vstack([sigma_I_ch * stokes_scale[s] for s in range(n_stokes)])
    noise_white = np.random.normal(0.0, sigma_ch[:, :, None], size=(n_stokes, n_chan, n_time))
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
    snr, (left, right) = snr_onpulse(
        dspec_params, I_time, frac=0.95, robust_rms=True, buffer_frac=buffer_frac
    )
    if not plot_multiple_frb:
        logging.info("Added noise with SEFD=%.1f Jy", float(np.mean(sefd_arr)))
        logging.info(f"Stokes I S/N (on-pulse method): {snr:.2f}")
    return noisy_dspec, sigma_ch, snr


def compute_required_sefd(dspec_params, dspec_clean, freq_mhz, target_snr,
                         n_pol=2, buffer_frac=None, robust_rms=True):
    if target_snr is None or target_snr <= 0:
        raise ValueError("target_snr must be > 0")
    profile = np.nansum(dspec_clean[0], axis=0)
    left, right = boxcar_width(profile, frac=0.95)
    init_width_bins = dspec_params.gdict["width"][0] / dspec_params.time_res_ms
    buffer_bins = int(float(buffer_frac) * init_width_bins) if buffer_frac is not None else 0
    on_mask = make_onpulse_mask(profile.size, left, right)
    off_mask = np.zeros(profile.size, dtype=bool)
    end_off = max(0, left - buffer_bins - 1)
    if end_off >= 0:
        off_mask[:end_off + 1] = True
    F_on = float(np.nansum(profile[on_mask]))
    N_on = int(on_mask.sum())
    N_chan = dspec_clean.shape[1]
    if F_on <= 0 or N_on == 0 or N_chan == 0:
        raise ValueError("Insufficient on-pulse energy for SEFD estimation.")
    if hasattr(dspec_params, "channel_bw_hz") and dspec_params.channel_bw_hz is not None:
        ch_bw = float(dspec_params.channel_bw_hz)
    else:
        diffs = np.diff(freq_mhz)
        if np.any(np.abs(diffs - diffs[0]) > 1e-6):
            logging.warning("Irregular channel spacing; using first diff for bandwidth fallback.")
        ch_bw = float(diffs[0] * 1e6)
    t_res_s = dspec_params.time_res_ms / 1000.0
    sefd = F_on * np.sqrt(n_pol * ch_bw * t_res_s) / (target_snr * np.sqrt(N_chan * N_on))
    details = {
        "F_on": F_on, "N_on": N_on, "N_chan": N_chan,
        "left": left, "right": right, "channel_bw_hz": ch_bw,
    }
    return sefd, details


def boxcar_snr(ys, rms):
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
    return (global_maxSNR / rms, boxcarw)


def baseline_stats_from_offpulse(profile, offpulse_mask, subtract_baseline=True, robust_rms=True):
    prof = np.asarray(profile, dtype=float)
    mask = np.asarray(offpulse_mask, dtype=bool)
    offpulse = prof[mask]
    if offpulse.size == 0:
        return 0.0 if subtract_baseline else 0.0, np.nan
    if subtract_baseline:
        baseline = float(np.nanmedian(offpulse))
    else:
        baseline = 0.0
    off_centered = offpulse - baseline
    if robust_rms:
        mad = float(np.nanmedian(np.abs(off_centered - np.nanmedian(off_centered))))
        sigma = 1.4826 * mad if mad > 0 else float(np.nanstd(off_centered))
    else:
        sigma = float(np.nanstd(off_centered)) if off_centered.size > 0 else np.nan
    return baseline, sigma


def apply_baseline_correction(dspec, offpulse_mask, mode="median"):
    if dspec is None or dspec.ndim != 3 or dspec.shape[0] != 4:
        raise ValueError("dspec must have shape (4, nf, nt)")
    nf, nt = dspec.shape[1], dspec.shape[2]
    off = np.asarray(offpulse_mask, dtype=bool)
    if off.size != nt:
        raise ValueError(f"offpulse_mask has wrong length {off.size}; expected {nt}")
    idx_t = np.flatnonzero(off)
    info = {"mode": str(mode), "used_bins": int(idx_t.size)}
    if idx_t.size == 0:
        return dspec, info
    m = mode.lower()
    if m in ("median", "mean"):
        sub = np.take(dspec, idx_t, axis=2)
        if m == "mean":
            offsets = np.nanmean(sub, axis=2)
        else:
            offsets = np.nanmedian(sub, axis=2)
        dspec_corr = dspec - offsets[:, :, None]
        info["offsets"] = offsets
        return dspec_corr, info
    if m in ("zscore", "zscore_per_stokes", "z"):
        sub = np.take(dspec, idx_t, axis=2)
        mu = np.nanmean(sub, axis=2)
        sd = np.nanstd(sub, axis=2, ddof=1)
        sd = np.where(sd > 0, sd, 1.0)
        dspec_corr = (dspec - mu[:, :, None]) / sd[:, :, None]
        info["mu"] = mu
        info["sd"] = sd
        return dspec_corr, info
    if m in ("zscore_i_shared", "z_i"):
        I = dspec[0]
        I_sub = np.take(I, idx_t, axis=1)
        mu_I = np.nanmean(I_sub, axis=1)
        sd_I = np.nanstd(I_sub, axis=1, ddof=1)
        sd_I = np.where(sd_I > 0, sd_I, 1.0)
        dspec_corr = (dspec - mu_I[None, :, None]) / sd_I[None, :, None]
        info["mu_I"] = mu_I
        info["sd_I"] = sd_I
        return dspec_corr, info
    return dspec, info


def estimate_noise_with_offpulse_mask(corrdspec, offpulse_mask, robust=False, ddof=1):
    corrdspec = np.asarray(corrdspec, dtype=float)
    if corrdspec.ndim != 3:
        raise ValueError("corrdspec must have shape (n_stokes, n_chan, n_time)")
    if offpulse_mask.dtype != bool or offpulse_mask.ndim != 1:
        raise ValueError("offpulse_mask must be 1-D boolean (time axis)")
    n_stokes, n_chan, n_time = corrdspec.shape
    if offpulse_mask.size != n_time:
        raise ValueError("offpulse_mask length does not match time dimension")
    if not np.any(offpulse_mask):
        return np.full(n_stokes, np.nan), np.full((n_stokes, n_chan), np.nan)
    offcube = corrdspec[:, :, offpulse_mask]
    if robust:
        med = np.nanmedian(offcube, axis=2, keepdims=True)
        mad = np.nanmedian(np.abs(offcube - med), axis=2)
        noisespec = 1.4826 * mad
    else:
        x = offcube
        with np.errstate(invalid='ignore', divide='ignore'):
            n_valid = np.sum(np.isfinite(x), axis=2).astype(float)
            sum_x   = np.nansum(x, axis=2)
            sum_x2  = np.nansum(x * x, axis=2)
            var_num = sum_x2 - (sum_x * sum_x) / np.maximum(n_valid, 1.0)
            denom = n_valid - (1.0 if ddof == 1 else 0.0)
            var = np.where(denom > 0.0, var_num / denom, np.nan)
            var = np.maximum(var, 0.0)
            noisespec = np.sqrt(var)
    noise_stokes = np.sqrt(np.nansum(noisespec**2, axis=1))
    return noise_stokes, noisespec


def snr_onpulse(dspec_params, profile, frac=0.95, robust_rms=True, buffer_frac=None):
    prof = np.asarray(profile, dtype=float)
    intrinsic_width_bins = dspec_params.gdict["width"][0] / dspec_params.time_res_ms
    on_mask, off_mask, (left, right) = on_off_pulse_masks_from_profile(
        prof, intrinsic_width_bins=intrinsic_width_bins, frac=frac, buffer_frac=buffer_frac
    )
    off = prof[off_mask]
    if off.size == 0 or not np.any(np.isfinite(off)):
        return 0.0, (left, right)
    if robust_rms:
        med = float(np.nanmedian(off))
        mad = float(np.nanmedian(np.abs(off - med)))
        sigma = 1.4826 * mad if mad > 0 else float(np.nanstd(off))
    else:
        sigma = float(np.nanstd(off)) if off.size > 0 else np.nan
    if not np.isfinite(sigma) or sigma <= 0:
        return 0.0, (left, right)
    N_on = int(np.count_nonzero(on_mask))
    if N_on <= 0:
        return 0.0, (left, right)
    S_on = float(np.nansum(prof[on_mask]))
    snr = S_on / (sigma * np.sqrt(N_on))
    return snr, (left, right)


def scale_dspec_to_target_snr(target_snr_mode, target_snr, dspec_params, dspec, freq_mhz,
                               time_res_ms, buffer_frac, sefd, plot_multiple_frb, time_ms):
    if target_snr_mode == "scale_intensity":
        prof = np.nansum(dspec[0], axis=0)
        from fires.core.dspec import compute_segments
        left, right = compute_segments(dspec, freq_mhz, time_ms, dspec_params,
                                       buffer_frac, skip_rm=True)["global"]["window"].values()
        f_res_hz = (freq_mhz[1] - freq_mhz[0]) * 1e6
        t_res_s = time_res_ms / 1000.0
        N_on = right - left + 1
        N_chan = dspec.shape[1]
        F_on = np.nansum(prof[left:right+1])
        sefd_eff = sefd if sefd > 0 else 1.0
        snr_est = F_on * np.sqrt(2 * f_res_hz * t_res_s) / (sefd_eff * np.sqrt(N_chan * N_on))
        if snr_est > 0:
            scale = target_snr / snr_est
            dspec *= scale
            if not plot_multiple_frb:
                logging.info("Applied amplitude scaling factor %.3f for target S/N %g", scale, target_snr)
    else:
        sefd_est, sefd_details = compute_required_sefd(
            dspec_params, dspec, freq_mhz, target_snr=target_snr,
            n_pol=2, buffer_frac=buffer_frac, robust_rms=True
        )
        if target_snr_mode == "iter":
            _, _, snr_meas = add_noise(
                dspec_params, dspec, sefd_est,
                (freq_mhz[1] - freq_mhz[0]) * 1e6,
                time_res_ms / 1000.0,
                plot_multiple_frb=True,
                buffer_frac=buffer_frac,
                n_pol=2
            )
            if snr_meas > 0:
                sefd_est *= (snr_meas / target_snr)
        sefd = sefd_est
        if not plot_multiple_frb:
            logging.info("SEFD set to %.3f Jy (mode=%s) for target S/N %g",
                         sefd, target_snr_mode, target_snr)
    return sefd


def correct_baseline(dspec, intrinsic_width_bins, buffer_frac, baseline_correct, plot_multiple_frb, dspec_params):
    I_ts = np.nansum(dspec[0], axis=0)
    on_mask, off_mask, (left, right) = on_off_pulse_masks_from_profile(
        I_ts, intrinsic_width_bins=intrinsic_width_bins, frac=0.95, buffer_frac=buffer_frac
    )
    dspec, bl_info = apply_baseline_correction(dspec, off_mask, mode=baseline_correct)
    if not plot_multiple_frb:
        logging.info(f"Applied baseline correction mode='{baseline_correct}' using {bl_info.get('used_bins',0)} off-pulse bins.")
    from fires.core.dspec import stokes_consistency_diagnostics
    stokes_consistency_diagnostics(
        dspec, buffer_frac, intrinsic_width_bins,
        label="post-bline", plot_multiple_frb=plot_multiple_frb, snr_min=5.0
    )
    I_ts_bline = np.nansum(dspec[0], axis=0)
    snr_post_bline, _ = snr_onpulse(
        dspec_params, I_ts_bline, frac=0.95, robust_rms=True, buffer_frac=buffer_frac
    )
    if not plot_multiple_frb:
        logging.info(f"Stokes I S/N (post-baseline): {snr_post_bline:.2f}")
    return dspec
