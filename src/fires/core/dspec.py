import logging

import numpy as np
from scipy.stats import circvar

from fires.utils.profiles import (boxcar_width, make_onpulse_mask,
                                    on_off_pulse_masks_from_profile)
from fires.utils.utils import frb_spectrum, frb_time_series


def est_profiles(dspec, noise_stokes, left, right, remove_pa_trend=False):
    pa_mask_sigma = 2.0
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
        Lts = np.zeros_like(L_meas)
        det = r >= cutoff
        Lts[det] = np.sqrt(np.maximum(L_meas[det]**2 - sigma_L[det]**2, 0.0))
        eLts = np.full_like(Lts, np.nan)
        eLts[det] = sigma_L[det]
        Pts = np.sqrt(Lts**2 + Vts**2)
        ePts = np.sqrt((Lts**2 * eLts**2) + (Vts**2 * Vts_rms**2)) / np.maximum(Pts, eps)
        phits = np.full_like(Lts, np.nan)
        ephits = np.full_like(Lts, np.nan)
        pa_det = (Lts >= pa_mask_sigma * sigma_L)
        phits[pa_det] = 0.5 * np.arctan2(Uts[pa_det], Qts[pa_det])
        ephits[pa_det] = 0.5 * np.sqrt(
            (Qts[pa_det]**2 * Uts_rms**2 + Uts[pa_det]**2 * Qts_rms**2)
            / np.maximum((Qts[pa_det]**2 + Uts[pa_det]**2)**2, eps)
        )
        win_mask = np.zeros_like(pa_det, dtype=bool)
        win_mask[left:right+1] = True
        keep = pa_det & win_mask
        phits[~keep] = np.nan
        ephits[~keep] = np.nan
        min_run = 5
        valid = np.isfinite(phits)
        if np.any(valid):
            v = valid.astype(int)
            dv = np.diff(np.concatenate(([0], v, [0])))
            starts = np.where(dv == 1)[0]
            ends = np.where(dv == -1)[0]
            keep_run = np.zeros_like(valid, dtype=bool)
            for s, e in zip(starts, ends):
                if (e - s) >= min_run:
                    keep_run[s:e] = True
            drop = valid & ~keep_run
            phits[drop] = np.nan
            ephits[drop] = np.nan
        qfrac = Qts / Its
        ufrac = Uts / Its
        vfrac = Vts / Its
        lfrac = Lts / Its
        pfrac = Pts / Its
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
        phits, ephits,
        np.full_like(phits, np.nan),
        np.full_like(phits, np.nan),
        qfrac, eqfrac, ufrac, eufrac, vfrac, evfrac, lfrac, elfrac, pfrac, epfrac
    )


def est_spectra(dspec, noisespec, left_window_ms, right_window_ms):
    iquvspec = np.nansum(dspec[:, :, left_window_ms:right_window_ms + 1], axis=2)
    ispec = iquvspec[0]
    vspec = iquvspec[3]
    qspec = iquvspec[1]
    uspec = iquvspec[2]
    noispec0 = noisespec / np.sqrt(float(right_window_ms + 1 - left_window_ms))
    lspec  = np.sqrt(uspec ** 2 + qspec ** 2)
    dlspec = np.sqrt((uspec * noispec0[2]) ** 2 + (qspec * noispec0[1]) ** 2) / np.maximum(lspec, 1e-12)
    pspec  = np.sqrt(lspec ** 2 + vspec ** 2)
    dpspec = np.sqrt((lspec * dlspec) ** 2 + (vspec * noispec0[3]) ** 2) / np.maximum(pspec, 1e-12)
    qfracspec = qspec / ispec
    ufracspec = uspec / ispec
    vfracspec = vspec / ispec
    dqfrac = np.sqrt((qspec * noispec0[0]) ** 2 + (ispec * noispec0[1]) ** 2) / (ispec ** 2)
    dufrac = np.sqrt((uspec * noispec0[0]) ** 2 + (ispec * noispec0[2]) ** 2) / (ispec ** 2)
    dvfrac = np.sqrt((vspec * noispec0[0]) ** 2 + (ispec * noispec0[3]) ** 2) / (ispec ** 2)
    lfracspec = lspec / ispec
    dlfrac = np.sqrt((lspec * noispec0[0]) ** 2 + (ispec * dlspec) ** 2) / (ispec ** 2)
    pfracspec = pspec / ispec
    dpfrac = np.sqrt((pspec * noispec0[0]) ** 2 + (ispec * dpspec) ** 2) / (ispec ** 2)
    phispec  = np.rad2deg(0.5 * np.arctan2(uspec, qspec))
    dphispec = np.rad2deg(0.5 * np.sqrt(uspec**2 * noispec0[1]**2 + qspec**2 * noispec0[2]**2) / np.maximum(uspec ** 2 + qspec ** 2, 1e-12))
    psispec  = np.rad2deg(0.5 * np.arctan2(vspec, lspec))
    dpsispec = np.rad2deg(0.5 * np.sqrt(vspec**2 * noispec0[3]**2 + lspec**2 * dlspec**2) / np.maximum(vspec ** 2 + lspec ** 2, 1e-12))
    return frb_spectrum(
        iquvspec, noispec0, lspec, dlspec, pspec, dpspec,
        qfracspec, dqfrac, ufracspec, dufrac, vfracspec, dvfrac,
        lfracspec, dlfrac, pfracspec, dpfrac, phispec, dphispec, psispec, dpsispec
    )


def _integrated_fractions_from_timeseries(I, Q, U, V, L, on_mask):
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


def _freq_quarter_slices(n_chan):
    n = int(n_chan)
    base, rem = divmod(n, 4)
    sizes = [base + (1 if i < rem else 0) for i in range(4)]
    start = 0
    slices = {}
    for i, sz in enumerate(sizes):
        end = start + sz
        name = f"{i+1}q"
        slices[name] = slice(start, end)
        start = end
    slices["all"] = slice(None)
    return slices


def _phase_slices_from_peak(n_time, peak_index, include_peak="last"):
    pi = int(max(0, min(n_time - 1, peak_index)))
    if include_peak == "first":
        first = slice(0, pi + 1)
        last = slice(pi + 1, int(n_time))
    else:
        first = slice(0, pi)
        last = slice(pi, int(n_time))
    total = slice(None)
    return {"first": first, "last": last, "total": total}


def _timeseries_from_corr(corrdspec, dspec_params, buffer_frac, remove_pa_trend=False):
    I = np.nansum(corrdspec[0], axis=0)
    gdict = dspec_params.gdict
    left, right = boxcar_width(I, frac=0.95)
    _, offpulse_mask, _ = on_off_pulse_masks_from_profile(
        I, gdict["width"][0]/dspec_params.time_res_ms, frac=0.95, buffer_frac=buffer_frac
    )
    from fires.core.noise import estimate_noise_with_offpulse_mask
    noise_stokes, _ = estimate_noise_with_offpulse_mask(corrdspec, offpulse_mask)
    return est_profiles(corrdspec, noise_stokes, left, right, remove_pa_trend=remove_pa_trend)


def compute_segments(dspec, freq_mhz, time_ms, dspec_params, buffer_frac=0.1, skip_rm=False, remove_pa_trend=False):
    from fires.core.rm import rm_correct_dspec
    from fires.core.noise import estimate_noise_with_offpulse_mask
    gdict = dspec_params.gdict
    tsdata_full, corr_dspec, _, _ = process_dspec(
        dspec, freq_mhz, dspec_params, buffer_frac, skip_rm=skip_rm,
        remove_pa_trend=remove_pa_trend
    )
    Its = tsdata_full.iquvt[0]
    Qts = tsdata_full.iquvt[1]
    Uts = tsdata_full.iquvt[2]
    Vts = tsdata_full.iquvt[3]
    Lts = tsdata_full.Lts
    phits = tsdata_full.phits
    n_time = Its.size
    on_mask, off_mask, (left, right) = on_off_pulse_masks_from_profile(
        Its, gdict["width"][0]/dspec_params.time_res_ms, frac=0.95, buffer_frac=buffer_frac
    )
    peak_index = int(np.nanargmax(Its)) if n_time > 0 else 0
    phase_slices = _phase_slices_from_peak(n_time, peak_index, include_peak="first")
    def _masked_stats(arr, mask):
        if mask is None or arr.size == 0:
            return {"mean": np.nan, "median": np.nan, "std": np.nan, "var": np.nan}
        data = arr[mask]
        if data.size == 0 or not np.any(np.isfinite(data)):
            return {"mean": np.nan, "median": np.nan, "std": np.nan, "var": np.nan}
        mean = float(np.nanmean(data))
        med  = float(np.nanmedian(data))
        std  = float(np.nanstd(data, ddof=1))
        var  = float(np.nanvar(data, ddof=1))
        return {"mean": mean, "median": med, "std": std, "var": var}
    def _collect_onoff_stats():
        return {
            "onpulse": {
                "I": _masked_stats(Its, on_mask),
                "Q": _masked_stats(Qts, on_mask),
                "U": _masked_stats(Uts, on_mask),
                "V": _masked_stats(Vts, on_mask),
            },
            "offpulse": {
                "I": _masked_stats(Its, off_mask),
                "Q": _masked_stats(Qts, off_mask),
                "U": _masked_stats(Uts, off_mask),
                "V": _masked_stats(Vts, off_mask),
            },
            "window": {"left": int(left), "right": int(right)}
        }
    def _measure_phase_slice(slc):
        slc_mask = np.zeros(n_time, dtype=bool)
        start = 0 if slc.start is None else slc.start
        stop = n_time if slc.stop is None else slc.stop
        if stop > start:
            slc_mask[start:stop] = True
        on_mask_slice = on_mask & slc_mask
        Vpsi = pa_variance_deg2(phits[on_mask_slice])
        Lfrac, Vfrac = _integrated_fractions_from_timeseries(Its, Qts, Uts, Vts, Lts, on_mask_slice)
        return {"Vpsi": Vpsi, "Lfrac": Lfrac, "Vfrac": Vfrac}
    phase_measures = {name: _measure_phase_slice(slc) for name, slc in phase_slices.items()}
    n_chan = corr_dspec.shape[1]
    fq = _freq_quarter_slices(n_chan)
    def _measure_freq_slice(slc):
        dspec_f = corr_dspec[:, slc, :]
        tsdata_f = _timeseries_from_corr(dspec_f, dspec_params, buffer_frac, remove_pa_trend=remove_pa_trend)
        I = tsdata_f.iquvt[0]; Q = tsdata_f.iquvt[1]; U = tsdata_f.iquvt[2]; V = tsdata_f.iquvt[3]; L = tsdata_f.Lts; ph = tsdata_f.phits
        on_m, _, _ = on_off_pulse_masks_from_profile(
            I, gdict["width"][0]/dspec_params.time_res_ms, frac=0.95, buffer_frac=buffer_frac
        )
        Vpsi = pa_variance_deg2(ph[on_m])
        Lfrac, Vfrac = _integrated_fractions_from_timeseries(I, Q, U, V, L, on_m)
        return {"Vpsi": Vpsi, "Lfrac": Lfrac, "Vfrac": Vfrac}
    freq_measures = {name: _measure_freq_slice(slc) for name, slc in fq.items()}
    global_stats = _collect_onoff_stats()
    return {"phase": phase_measures, "freq": freq_measures, "global": global_stats}


def pa_variance_deg2(phits):
    valid = np.isfinite(phits)
    if not np.any(valid):
        return np.nan
    pa_var = circvar(2.0 * phits[valid], low=-np.pi, high=np.pi) / 4.0
    return (180/np.pi)**2 * pa_var


def format_global_stats(global_stats):
    win = global_stats.get("window", {})
    lines = []
    lines.append(f"On-pulse window: [{win.get('left','?')} , {win.get('right','?')}]")
    for region in ("onpulse", "offpulse"):
        lines.append(f"\n{region.upper()}:")
        stokes_stats = global_stats.get(region, {})
        lines.append("  Stokes  mean        median      std         var")
        for st in ("I", "Q", "U", "V"):
            s = stokes_stats.get(st, {})
            lines.append(
                f"  {st:>5}  {s.get('mean',np.nan):>11.5g}  {s.get('median',np.nan):>11.5g}  "
                f"{s.get('std',np.nan):>11.5g}  {s.get('var',np.nan):>11.5g}"
            )
    return "\n".join(lines)


def print_global_stats(global_stats, logger=True):
    msg = format_global_stats(global_stats)
    if logger:
        logging.debug("\n" + msg)
    else:
        print(msg)


def wrap_pa_deg(pa):
    w = (pa + 90.0) % 180.0 - 90.0
    w[np.isclose(w, -90.0, atol=1e-6)] = 90.0
    return w


def scatter_loaded_dspec(dspec, freq_mhz, time_ms, tau, sc_idx, ref_freq_mhz):
    dspec_scattered = dspec.copy()
    time_res_ms = np.median(np.diff(time_ms))
    tau_cms = tau * (freq_mhz / ref_freq_mhz) ** (-sc_idx)
    dspec_scattered = scatter_dspec(dspec_scattered, time_res_ms, tau_cms)
    return dspec_scattered


def scatter_dspec(dspec, time_res_ms, tau_cms, pad_factor=5, screen="thin"):
    X = np.asarray(dspec, dtype=float)
    if X.ndim == 2:
        return _scatter_2d(X, time_res_ms, tau_cms, pad_factor, screen)
    elif X.ndim == 3:
        return _scatter_4stokes(X, time_res_ms, tau_cms, pad_factor, screen)
    else:
        raise ValueError(f"scatter_dspec: expected 2-D or 3-D input, got {X.ndim}-D")


def _scatter_2d(X, time_res_ms, tau_cms, pad_factor=5, screen="thin"):
    nc, nt = X.shape
    if np.isscalar(tau_cms):
        tau_arr = np.full(nc, float(tau_cms), dtype=float)
    else:
        tau_arr = np.asarray(tau_cms, dtype=float)
        if tau_arr.shape[0] != nc:
            raise ValueError("tau_cms length must match number of channels")
    out = X.copy()
    pos = np.isfinite(tau_arr) & (tau_arr > 0)
    if not np.any(pos):
        return out
    FH, max_pad, L = _build_irf_fft(tau_arr[pos], time_res_ms, nt, pad_factor, screen)
    Xp = np.pad(X[pos], ((0, 0), (0, max_pad)), mode='constant')
    FX = np.fft.rfft(Xp, n=L, axis=1)
    out[pos] = np.fft.irfft(FX * FH, n=L, axis=1)[:, :nt]
    return out


def _scatter_4stokes(X4, time_res_ms, tau_cms, pad_factor=5, screen="thin"):
    _, nc, nt = X4.shape
    if np.isscalar(tau_cms):
        tau_arr = np.full(nc, float(tau_cms), dtype=float)
    else:
        tau_arr = np.asarray(tau_cms, dtype=float)
        if tau_arr.shape[0] != nc:
            raise ValueError("tau_cms length must match number of channels")
    out = X4.copy()
    pos = np.isfinite(tau_arr) & (tau_arr > 0)
    if not np.any(pos):
        return out
    FH, max_pad, L = _build_irf_fft(tau_arr[pos], time_res_ms, nt, pad_factor, screen)
    for s in range(X4.shape[0]):
        Xp = np.pad(X4[s][pos], ((0, 0), (0, max_pad)), mode='constant')
        FX = np.fft.rfft(Xp, n=L, axis=1)
        out[s][pos] = np.fft.irfft(FX * FH, n=L, axis=1)[:, :nt]
    return out


def _build_irf_fft(tau_pos, time_res_ms, nt, pad_factor, screen="thin"):
    n_pad_arr = np.ceil(pad_factor * tau_pos / float(time_res_ms)).astype(int)
    max_pad = int(np.max(n_pad_arr))
    L = nt + max_pad

    # t starts at time_res_ms (index 1) to avoid 1/t singularities in thick/uniform.
    # Thin screen is unaffected since exp(0)=1 and the zero bin would just be
    # absorbed into normalisation anyway.
    t_ms = np.arange(1, max_pad + 2, dtype=float) * float(time_res_ms)  # shape (max_pad+1,)
    tau = tau_pos[:, None]  # shape (nch, 1)
    t   = t_ms[None, :]     # shape (1, max_pad+1)

    if screen == "thin":
        irf = np.exp(-t / tau)

    elif screen == "thick":
        # sqrt(pi * tau_s / (4 * t^3)) * exp(-pi^2 * tau_s / (16 * t))
        irf = np.sqrt(np.pi * tau / (4.0 * t**3)) * np.exp(-(np.pi**2 * tau) / (16.0 * t))

    elif screen == "uniform":
        # sqrt(pi^5 * tau_s^3 / (8 * t^5)) * exp(-pi^2 * tau_s / (4 * t))
        irf = np.sqrt(np.pi**5 * tau**3 / (8.0 * t**5)) * np.exp(-(np.pi**2 * tau) / (4.0 * t))

    else:
        raise ValueError(f"Unknown screen type '{screen}'. Choose 'thin', 'thick', or 'uniform'.")

    # Apply the same per-channel truncation mask as the original
    irf *= (np.arange(1, max_pad + 2)[None, :] <= n_pad_arr[:, None])
    irf /= np.maximum(irf.sum(axis=1, keepdims=True), 1e-300)

    # Pad to length L before FFT (irf has max_pad+1 samples, L = nt + max_pad)
    irf_padded = np.zeros((len(tau_pos), L), dtype=float)
    irf_padded[:, :max_pad + 1] = irf

    return np.fft.rfft(irf_padded, n=L, axis=1), max_pad, L


def stokes_consistency_diagnostics(dspec, buffer_frac, intrinsic_width_bins, label, plot_multiple_frb, snr_min=5.0):
    try:
        I_ts = np.nansum(dspec[0], axis=0)
        Q_ts = np.nansum(dspec[1], axis=0)
        U_ts = np.nansum(dspec[2], axis=0)
        V_ts = np.nansum(dspec[3], axis=0)
        nt = I_ts.size
        on_mask, _, (left, right) = on_off_pulse_masks_from_profile(
            I_ts, intrinsic_width_bins=intrinsic_width_bins, frac=0.95, buffer_frac=buffer_frac
        )
        guard_bins = int(np.ceil(float(buffer_frac) * float(intrinsic_width_bins))) if buffer_frac is not None else 0
        L_end = max(0, int(left) - guard_bins)
        bl_mask = np.zeros(nt, dtype=bool)
        if L_end > 0:
            bl_mask[:L_end] = True
        if not np.any(bl_mask):
            bl_mask = ~on_mask
        sigma_off = float(np.nanstd(I_ts[bl_mask], ddof=1))
        sigma_off = sigma_off if sigma_off > 0 else 1.0
        on_snr = on_mask & (I_ts >= snr_min * sigma_off)
        if not np.any(on_snr):
            if not plot_multiple_frb:
                logging.info(f"[stokes_ts:{label}] no bins above {snr_min:.1f} in on-pulse; skipping.")
            return
        I2 = I_ts**2
        P2 = Q_ts**2 + U_ts**2 + V_ts**2
        R = (I2 - P2)[on_snr]
        p16 = float(np.nanpercentile(R, 16))
        med = float(np.nanmedian(R))
        p84 = float(np.nanpercentile(R, 84))
        mean = float(np.nanmean(R))
        frac_neg = float(np.mean(R < 0.0))
        p = np.sqrt(P2[on_snr]) / np.maximum(I_ts[on_snr], 1e-12)
        p_med = float(np.nanmedian(p))
        p_mean = float(np.nanmean(p))
        p95 = float(np.nanpercentile(p, 95))
        if not plot_multiple_frb:
            logging.info(
                f"[stokes_ts:{label}] R=I^2-P^2: med={med:.3g}, p16={p16:.3g}, p84={p84:.3g}, "
                f"mean={mean:.3g}, frac(R<0)={frac_neg:.3%}"
            )
            logging.info(
                f"[stokes_ts:{label}] p=sqrt(Q^2+U^2+V^2)/I: median={p_med:.3g}, mean={p_mean:.3g}, p95={p95:.3g}"
            )
    except Exception as e:
        if not plot_multiple_frb:
            logging.warning(f"[stokes_ts:{label}] diagnostics failed: {e}")


def process_dspec(dspec, freq_mhz, dspec_params, buffer_frac, skip_rm=False, remove_pa_trend=False):
    from fires.core.rm import estimate_rm, rm_correct_dspec
    from fires.core.noise import estimate_noise_with_offpulse_mask
    gdict = dspec_params.gdict
    RM = gdict["RM"]
    ref_freq_mhz = dspec_params.ref_freq_mhz
    if skip_rm or np.all(RM == 0.0):
        corrdspec = dspec.copy()
    else:
        try:
            n_time = dspec.shape[2]
            time_ms = np.arange(n_time) * float(dspec_params.time_res_ms)
            I = np.nansum(dspec[0], axis=0)
            _, offpulse_mask, _ = on_off_pulse_masks_from_profile(
                I, gdict["width"][0] / dspec_params.time_res_ms, frac=0.95, buffer_frac=buffer_frac
            )
            _, noisespec = estimate_noise_with_offpulse_mask(dspec, offpulse_mask, robust=True)
            res_rmtool = estimate_rm(dspec, freq_mhz, time_ms, noisespec,
                                     phi_range=1.0e3, dphi=1.0, outdir='.', save=False, show_plots=True)
            measured_rm = float(res_rmtool[0])
            def _int_Lfrac(candidate):
                I_ts = np.nansum(candidate[0], axis=0)
                Q_ts = np.nansum(candidate[1], axis=0)
                U_ts = np.nansum(candidate[2], axis=0)
                left, right = boxcar_width(I_ts, frac=0.95)
                on_mask = make_onpulse_mask(I_ts.size, left, right)
                I_int = np.nansum(I_ts[on_mask])
                L_int = np.nansum(np.sqrt(Q_ts[on_mask]**2 + U_ts[on_mask]**2))
                return (L_int / I_int if I_int > 0 else 0.0), (left, right)
            if np.isfinite(measured_rm) and np.abs(measured_rm) > 0.0:
                cand_pos = rm_correct_dspec(dspec, freq_mhz, +measured_rm, ref_freq_mhz=ref_freq_mhz)
                cand_neg = rm_correct_dspec(dspec, freq_mhz, -measured_rm, ref_freq_mhz=ref_freq_mhz)
                Lfrac_pos, _ = _int_Lfrac(cand_pos)
                Lfrac_neg, _ = _int_Lfrac(cand_neg)
                if Lfrac_pos >= Lfrac_neg:
                    corrdspec = cand_pos
                    chosen_sign = '+'
                    Lfrac_best = Lfrac_pos
                else:
                    corrdspec = cand_neg
                    chosen_sign = '-'
                    Lfrac_best = Lfrac_neg
                logging.info("Measured RM = %.2f rad/m2; applied derotation to 2=0 (sign=%s); L/I=%.3f",
                             measured_rm, chosen_sign, Lfrac_best)
                I_ts = np.nansum(corrdspec[0], axis=0)
                left_chk, right_chk = boxcar_width(I_ts, frac=0.95)
                Qint = np.nansum(corrdspec[1,:,left_chk:right_chk+1], axis=1)
                Uint = np.nansum(corrdspec[2,:,left_chk:right_chk+1], axis=1)
                phi_ch = 0.5 * np.arctan2(Uint, Qint)
                lam2 = (299.792458 / np.asarray(freq_mhz, float))**2
                ok = np.isfinite(phi_ch) & np.isfinite(lam2)
                if np.count_nonzero(ok) > 2:
                    y = np.unwrap(2.0 * phi_ch[ok]) / 2.0
                    x = lam2[ok]
                    p = np.polyfit(x, y, deg=1)
                    slope = p[0]
                    logging.info("Residual dPA/d(2) after derotation: %.3e rad/m^2 (expect ~0)", slope)
            else:
                corrdspec = dspec.copy()
                logging.info("Measured RM not significant; skipping RM correction")
        except Exception as e:
            logging.warning("RM measurement/derotation failed (%s). Leaving dspec unchanged.", str(e))
            corrdspec = dspec.copy()
    I = np.nansum(corrdspec[0], axis=0)
    left, right = boxcar_width(I, frac=0.95)
    _, offpulse_mask, _ = on_off_pulse_masks_from_profile(
        I, gdict["width"][0]/dspec_params.time_res_ms, frac=0.95, buffer_frac=buffer_frac
    )
    noise_stokes, noisespec = estimate_noise_with_offpulse_mask(corrdspec, offpulse_mask)
    tsdata = est_profiles(corrdspec, noise_stokes, left, right, remove_pa_trend=remove_pa_trend)
    return tsdata, corrdspec, noisespec, noise_stokes
