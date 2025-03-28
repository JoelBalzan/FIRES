#
#	Functions for FRB polarization analysis
#
#								AB, August 2024
#
#	Function list
#
#			estimate_rm(frbname, dm, nchan, ffac, avgfac, fmhz0, lwms, rwms, phirange, dphi, startchan, endchan, tpeakms):
#						Estimate rotation measure
#
#			unfarot(frbname, dm, nchan, ffac, avgfac, fmhz0, rm0):
#						Generate RM corrected dynamic spectrum 
#
#	--------------------------	Import modules	---------------------------

import os
import sys

import numpy as np
from RMtools_1D.do_RMclean_1D import run_rmclean
from RMtools_1D.do_RMsynth_1D import run_rmsynth
from ..utils.utils import *

#	---------------------------------------------------------------------------------

def pol_angle_diff(angle, ref_angle):
    """
    Calculate the difference between two angles, taking care of wrapping around (takes absolute diff).
    Inputs:
        - angle: Array of angles in degrees
        - ref_angle: Reference angle in degrees
    Returns:
        - dpang: Difference between angles in degrees
    """
    angle = np.deg2rad(angle)  # Convert angle to radians
    ref_angle = np.deg2rad(ref_angle)  # Convert reference angle to radians
    dpang = np.rad2deg(np.arcsin(np.sin(angle - ref_angle)))  # Calculate the difference and convert back to degrees
    return dpang


def rm_synth(freq_ghz, iquv, diquv):
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
                        nBits=32, showPlots=True, debug=False, verbose=False, log=print, units='Jy/beam', prefixOut='prefixOut', saveFigures=None, fit_function='log')
    
    # Run RM clean
    rm_clean_data = run_rmclean(rm_synth_data, rm_synth_ad, 0.1, maxIter=1000, gain=0.1, nBits=32, showPlots=False, verbose=False, log=print)
    
    print(rm_clean_data[0])
    
    # Extract results
    res = [rm_clean_data[0]['phiPeakPIfit_rm2'], rm_clean_data[0]['dPhiPeakPIfit_rm2'], rm_clean_data[0]['polAngle0Fit_deg'], rm_clean_data[0]['dPolAngle0Fit_deg']]
    
    return res


def estimate_noise(dynspec, time_ms, left_window_ms, right_window_ms):
    """
    Estimate noise spectra for IQUV.
    Inputs:
        - dynspec: Dynamic spectrum array
        - time_ms: Time array in ms
        - left_window_ms: Left window in ms for noise estimation
        - right_window_ms: Right window in ms for noise estimation
    Returns:
        - noisespec: Noise spectrum
    """
    
    # Find the start and end indices for the time range
    istart = np.argmin(np.abs(left_window_ms - time_ms))
    iend = np.argmin(np.abs(right_window_ms - time_ms))
    
    # Calculate the noise spectrum
    noisespec = np.nanstd(dynspec[:, :, istart:iend + 1], axis=2)
    
    return noisespec


def estimate_rm(dynspec, freq_mhz, time_ms, noisespec, left_window_ms, right_window_ms, phi_range, dphi, start_chan, end_chan):
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
    
    if end_chan <= 0:
        end_chan = len(freq_mhz) - 1
    
    res_rmtool = [0.0, 0.0, 0.0, 0.0]
        
    # Find the start and end indices for the time range
    istart = np.argmin(np.abs(left_window_ms - time_ms))
    iend = np.argmin(np.abs(right_window_ms - time_ms))
    
    # Calculate the mean spectra for each Stokes parameter
    ispec = np.nanmean(dynspec[0, start_chan:end_chan + 1, istart:iend + 1], axis=1)
    vspec = np.nanmean(dynspec[3, start_chan:end_chan + 1, istart:iend + 1], axis=1)
    qspec0 = np.nanmean(dynspec[1, start_chan:end_chan + 1, istart:iend + 1], axis=1)
    uspec0 = np.nanmean(dynspec[2, start_chan:end_chan + 1, istart:iend + 1], axis=1)
    noispec = noisespec / np.sqrt(float(iend + 1 - istart))	
        
    iqu = (ispec, qspec0, uspec0)
    eiqu = (noispec[0], noispec[1], noispec[2])
        
    iquv = (ispec, qspec0, uspec0, vspec)
    eiquv = (noispec[0], noispec[1], noispec[2], noispec[3])
        
    # Run RM synthesis
    res_rmtool = rm_synth(freq_mhz / 1.0e3, iquv, eiquv)
        
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


def est_profiles(dynspec, freq_mhz, time_ms, noisespec, start_chan, end_chan):
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
    if end_chan <= 0:
        end_chan = len(freq_mhz) - 1
    
    # Average the dynamic spectrum over the specified frequency channels
    iquvt = np.nanmean(dynspec[:, start_chan:end_chan], axis=1)					
    # Calculate the noise for each Stokes parameter
    noise_stokes = np.sqrt(np.nansum(noisespec[:, start_chan:end_chan] ** 2, axis=1)) / len(freq_mhz)
    
    # Extract the Stokes parameters
    itsub = iquvt[0]
    qtsub = iquvt[1]
    utsub = iquvt[2]
    vtsub = iquvt[3]
    
    # Calculate the linear polarization intensity
    lts = np.sqrt(utsub ** 2 + qtsub ** 2)
    lts = noise_stokes[0] * np.sqrt((lts / noise_stokes[0]) ** 2 - 1.0)						
    # Calculate the error in linear polarization intensity
    elts = np.sqrt((qtsub * noise_stokes[1]) ** 2 + (utsub * noise_stokes[2]) ** 2) / lts
    # Calculate the total polarization intensity
    pts = np.sqrt(lts ** 2 + vtsub ** 2)
    # Calculate the error in total polarization intensity
    epts = np.sqrt((qtsub * noise_stokes[1]) ** 2 + (utsub * noise_stokes[2]) ** 2 + (vtsub * noise_stokes[3]) ** 2) / pts

    # Calculate the polarization angles
    phits = np.rad2deg(0.5 * np.arctan2(utsub, qtsub))		
    dphits = np.rad2deg(0.5 * np.sqrt((utsub * noise_stokes[1]) ** 2 + (qtsub * noise_stokes[2]) ** 2) / (utsub ** 2 + qtsub ** 2))						
    psits = np.rad2deg(0.5 * np.arctan2(vtsub, lts))		
    dpsits = np.rad2deg(0.5 * np.sqrt((vtsub * elts) ** 2 + (lts * noise_stokes[3]) ** 2) / (vtsub ** 2 + lts ** 2))
    
    # Calculate the fractional polarizations
    vfrac = vtsub / itsub
    lfrac = lts / itsub
    pfrac = pts / itsub		
    qfrac = qtsub / itsub
    ufrac = utsub / itsub
    
    # Set large errors to NaN
    phits[dphits > 10.0] = np.nan
    dphits[dphits > 10.0] = np.nan
    psits[dpsits > 10.0] = np.nan
    dpsits[dpsits > 10.0] = np.nan

    # Calculate the errors in fractional polarizations
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
    
    # Find the start and end indices for the time range
    istart = np.argmin(np.abs(left_window_ms - time_ms))
    iend = np.argmin(np.abs(right_window_ms - time_ms))
    
    # Average the dynamic spectrum over the specified time range
    iquvspec = np.nanmean(dynspec[:, :, istart:iend + 1], axis=2)
    
    # Extract the Stokes parameters
    ispec = iquvspec[0]
    vspec = iquvspec[3]
    qspec = iquvspec[1]
    uspec = iquvspec[2]		
    
    # Calculate the noise for each Stokes parameter
    noispec0 = noisespec / np.sqrt(float(iend + 1 - istart))
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