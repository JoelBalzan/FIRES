import logging
import os

import numpy as np
from RMtools_1D.do_RMclean_1D import run_rmclean
from RMtools_1D.do_RMsynth_1D import run_rmsynth

from fires.utils.profiles import boxcar_width


def rm_synth(freq_ghz, iquv, diquv, outdir, save, show_plots):
    rm_data = np.array([freq_ghz * 1.0e9, iquv[0], iquv[1], iquv[2], diquv[0], diquv[1], diquv[2]])
    rm_synth_data, rm_synth_ad = run_rmsynth(
        rm_data, polyOrd=3, phiMax_radm2=1.0e3, dPhi_radm2=1.0, nSamples=100.0,
        weightType='variance', fitRMSF=False, noStokesI=False, phiNoise_radm2=1000000.0,
        nBits=32, showPlots=show_plots, debug=False, verbose=False, log=print,
        units='Jy/beam', prefixOut=os.path.join(outdir, "RM"), saveFigures=save,
        fit_function='log'
    )
    rm_clean_data = run_rmclean(
        rm_synth_data, rm_synth_ad, 0.1, maxIter=1000, gain=0.1, nBits=32,
        showPlots=show_plots, verbose=False, log=print
    )
    res = [
        rm_clean_data[0]['phiPeakPIfit_rm2'],
        rm_clean_data[0]['dPhiPeakPIfit_rm2'],
        rm_clean_data[0]['polAngle0Fit_deg'],
        rm_clean_data[0]['dPolAngle0Fit_deg'],
    ]
    return res


def estimate_rm(dspec, freq_mhz, time_ms, noisespec, phi_range, dphi, outdir, save, show_plots):
    left, right = boxcar_width(np.nansum(dspec[0], axis=0), frac=0.95)
    ispec   = np.nansum(dspec[0, :, left:right], axis=1)
    vspec   = np.nansum(dspec[3, :, left:right], axis=1)
    qspec0  = np.nansum(dspec[1, :, left:right], axis=1)
    uspec0  = np.nansum(dspec[2, :, left:right], axis=1)
    noispec = noisespec / np.sqrt(float(right + 1 - left))
    iquv  = (ispec, qspec0, uspec0, vspec)
    eiquv = (noispec[0], noispec[1], noispec[2], noispec[3])
    res_rmtool = rm_synth(freq_mhz / 1.0e3, iquv, eiquv, outdir, save, show_plots)
    return res_rmtool


def rm_correct_dspec(dspec, freq_mhz, rm0, ref_freq_mhz=None):
    dspec = np.asarray(dspec, dtype=float)
    new = np.zeros_like(dspec)
    new[0] = dspec[0]
    new[3] = dspec[3]
    lambda_m = 299.792458 / np.asarray(freq_mhz, dtype=float)
    lambda_sq = lambda_m**2
    if ref_freq_mhz is None:
        lambda_sq_ref = np.nanmedian(lambda_sq)
    else:
        if not np.isfinite(ref_freq_mhz) or ref_freq_mhz <= 0.0:
            lambda_sq_ref = 0.0
        else:
            lambda_ref = 299.792458 / float(ref_freq_mhz)
            lambda_sq_ref = lambda_ref**2
    rot = -2.0 * float(rm0) * (lambda_sq - lambda_sq_ref)
    c = np.cos(rot)
    s = np.sin(rot)
    Q = dspec[1]
    U = dspec[2]
    new[1] = Q * c[:, None] - U * s[:, None]
    new[2] = U * c[:, None] + Q * s[:, None]
    return new
