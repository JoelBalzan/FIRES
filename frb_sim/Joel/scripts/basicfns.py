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

import os, sys
import numpy as np
from RMtools_1D.do_RMsynth_1D import run_rmsynth
from RMtools_1D.do_RMclean_1D import run_rmclean
from utils import *

#	---------------------------------------------------------------------------------

def pol_angle_diff(ang, ang0):
    """
    Calculate the difference between two angles, taking care of wrapping around (takes absolute diff).
    """
    ang			=	np.deg2rad(ang)  # Convert angle to radians
    ang0		=	np.deg2rad(ang0)  # Convert reference angle to radians
    dpang		=	np.rad2deg(np.arcsin(np.sin(ang-ang0)))  # Calculate the difference and convert back to degrees
    return(dpang)


def rm_synth(fghz, iquv, diquv):
    """
    Determine RM using RM synthesis with RMtool.
    Inputs:
        - fghz: Frequencies in GHz
        - iquv: I Q U V spectrum
        - diquv: I Q U V noise spectrum
    """
    
    # Prepare the data for RM synthesis
    rmtdata	=	np.array([fghz*1.0e9, iquv[0], iquv[1], iquv[2], diquv[0], diquv[1], diquv[2]])
    
    # Run RM synthesis
    rmd, rmad = run_rmsynth(rmtdata, polyOrd=3, phiMax_radm2=1.0e3, dPhi_radm2=1.0, nSamples=100.0, weightType='variance', fitRMSF=False, noStokesI=False, phiNoise_radm2=1000000.0, \
                        nBits=32, showPlots=True, debug=False, verbose=False, log=print, units='Jy/beam', prefixOut='prefixOut', saveFigures=None, fit_function='log')
    
    # Run RM clean
    rmc	=	run_rmclean(rmd, rmad, 0.1, maxIter=1000, gain=0.1, nBits=32, showPlots=False, verbose=False, log=print)
    
    print(rmc[0])
    
    # Extract results
    res	=	[rmc[0]['phiPeakPIfit_rm2'], rmc[0]['dPhiPeakPIfit_rm2'], rmc[0]['polAngle0Fit_deg'], rmc[0]['dPolAngle0Fit_deg']]
    
    return(res)	


def estimate_noise(dspec4, tmsarr, lwms, rwms):
    """
    Estimate noise spectra for IQUV.
    """
    
    # Find the start and end indices for the time range
    istart		=	np.argmin(np.abs(lwms-tmsarr))
    iend		=	np.argmin(np.abs(rwms-tmsarr))
    
    # Calculate the noise spectrum
    noisespec	=	np.nanstd(dspec4[:,:,istart:iend+1], axis=2)
    
    return(noisespec)


def estimate_rm(dspec4, fmhzarr, tmsarr, noisespec, lwms, rwms, phirange, dphi, startchan, endchan):
    """
    Estimate rotation measure.
    """ 
    
    if(endchan <= 0):
        endchan	=	len(fmhzarr) - 1
    
    res_rmtool	=	[0.0,0.0,0.0,0.0]
        
    # Find the start and end indices for the time range
    istart		=	np.argmin(np.abs(lwms-tmsarr))
    iend		=	np.argmin(np.abs(rwms-tmsarr))
    
    # Calculate the mean spectra for each Stokes parameter
    ispec		=	np.nanmean(dspec4[0,startchan:endchan+1,istart:iend+1], axis=1)
    vspec		=	np.nanmean(dspec4[3,startchan:endchan+1,istart:iend+1], axis=1)
    qspec0		=	np.nanmean(dspec4[1,startchan:endchan+1,istart:iend+1], axis=1)
    uspec0		=	np.nanmean(dspec4[2,startchan:endchan+1,istart:iend+1], axis=1)
    noispec		=	noisespec/np.sqrt(float(iend+1-istart))	
        
    iqu			=	(ispec, qspec0, uspec0)
    eiqu		=	(noispec[0], noispec[1], noispec[2])
        
    iquv		=	(ispec, qspec0, uspec0, vspec)
    eiquv		=	(noispec[0], noispec[1], noispec[2], noispec[3])
        
    # Run RM synthesis
    res_rmtool	=	rm_synth(fmhzarr/1.0e3, iquv, eiquv)
        
    print("\nResults from RMtool (RM synthesis) \n")
    print("RM = %.2f +/- %.2f rad/m2   PolAng0 = %.2f +/- %.2f deg\n"%(res_rmtool[0], res_rmtool[1], res_rmtool[2], res_rmtool[3]))
    
    return(res_rmtool)
#	-------------------------------------------------------------------------------

def rm_correct_dynspec(dspec4, fmhzarr, rm0):
    """
    Generate RM corrected dynamic spectrum.
    """
    
    # Initialize the new dynamic spectrum
    newdspec4	=	np.zeros(dspec4.shape, dtype=float)
    newdspec4[0]=	dspec4[0]
    newdspec4[3]=	dspec4[3]
    
    # Calculate the lambda squared array
    lm2arr		=	(c_cgs*1.0e-8 / fmhzarr)**2
    lm20		=	np.nanmedian(lm2arr)
        
    # Apply RM correction to Q and U spectra
    for ci in range(len(lm2arr)):
        rotang		=	-2*rm0*(lm2arr[ci]-lm20)
        newdspec4[1,ci]	=	dspec4[1,ci]*np.cos(rotang) - dspec4[2,ci]*np.sin(rotang) 
        newdspec4[2,ci]	=	dspec4[2,ci]*np.cos(rotang) + dspec4[1,ci]*np.sin(rotang) 	

    return(newdspec4)
#	-------------------------------------------------------------------------------

def est_profiles(dspec4, fmhzarr, tmsarr, noisespec, startchan, endchan):
    """
    Estimate time profiles.
    """
    if(endchan <= 0):
        endchan	=	len(fmhzarr) - 1
    
    # Average the dynamic spectrum over the specified frequency channels
    iquvt		=	np.nanmean(dspec4[:,startchan:endchan], axis=1)					
    # Calculate the noise for each Stokes parameter
    noistks		=	np.sqrt(np.nansum(noisespec[:,startchan:endchan]**2, axis=1))/len(fmhzarr)
    
    # Extract the Stokes parameters
    itsub		=	iquvt[0]
    qtsub		=	iquvt[1]
    utsub		=	iquvt[2]
    vtsub		=	iquvt[3]
    
    # Calculate the linear polarization intensity
    lts			=	np.sqrt(utsub**2 + qtsub**2)
    lts			=	noistks[0]*np.sqrt((lts/noistks[0])**2 - 1.0)						
    # Calculate the error in linear polarization intensity
    elts		=	np.sqrt((qtsub*noistks[1])**2 + (utsub*noistks[2])**2)/lts
    # Calculate the total polarization intensity
    pts			=	np.sqrt(lts**2 + vtsub**2)
    # Calculate the error in total polarization intensity
    epts		=	np.sqrt((qtsub*noistks[1])**2 + (utsub*noistks[2])**2 + (vtsub*noistks[3])**2)/pts

    # Calculate the polarization angles
    phits		=	np.rad2deg(0.5*np.arctan2(utsub, qtsub))		
    dphits		=	np.rad2deg(0.5*np.sqrt((utsub*noistks[1])**2 + (qtsub*noistks[2])**2) / (utsub**2 + qtsub**2))						
    psits		=	np.rad2deg(0.5*np.arctan2(vtsub, lts))		
    dpsits		=	np.rad2deg(0.5*np.sqrt((vtsub*elts)**2 + (lts*noistks[3])**2) / (vtsub**2 + lts**2))
    
    # Calculate the fractional polarizations
    vfrac		=	vtsub/itsub
    lfrac		=	lts/itsub
    pfrac		=	pts/itsub		
    qfrac		=	qtsub/itsub
    ufrac		=	utsub/itsub
    
    # Set large errors to NaN
    phits[dphits>10.0]	=	np.nan
    dphits[dphits>10.0]	=	np.nan
    psits[dpsits>10.0]	=	np.nan
    dpsits[dpsits>10.0]	=	np.nan

    # Calculate the errors in fractional polarizations
    evfrac		=	np.abs(vfrac)*np.sqrt((noistks[3]/vtsub)**2 + (noistks[0]/itsub)**2)
    eqfrac		=	np.abs(qfrac)*np.sqrt((noistks[1]/qtsub)**2 + (noistks[0]/itsub)**2)
    eufrac		=	np.abs(ufrac)*np.sqrt((noistks[2]/utsub)**2 + (noistks[0]/itsub)**2)
    elfrac		=	np.abs(lfrac)*np.sqrt((elts/lts)**2 + (noistks[0]/itsub)**2)
    epfrac		=	np.abs(pfrac)*np.sqrt((epts/pts)**2 + (noistks[0]/itsub)**2)
        
    # Return the time profiles as a frbts object
    return(frbts(iquvt, lts, elts, pts, epts, phits, dphits, psits, dpsits, qfrac, eqfrac, ufrac, eufrac, vfrac, evfrac, lfrac, elfrac, pfrac, epfrac))
#	-------------------------------------------------------------------------------

def est_spectra(dspec4, fmhzarr, tmsarr, noisespec, lwms, rwms):
    """
    Estimate spectra.
    """
    
    # Find the start and end indices for the time range
    istart		=	np.argmin(np.abs(lwms-tmsarr))
    iend		=	np.argmin(np.abs(rwms-tmsarr))
    
    # Average the dynamic spectrum over the specified time range
    iquvspec	=	np.nanmean(dspec4[:,:,istart:iend+1], axis=2)
    
    # Extract the Stokes parameters
    ispec		=	iquvspec[0]
    vspec		=	iquvspec[3]
    qspec		=	iquvspec[1]
    uspec		=	iquvspec[2]		
    
    # Calculate the noise for each Stokes parameter
    noispec0	=	noisespec/np.sqrt(float(iend+1-istart))
    # Calculate the linear polarization intensity
    lspec		=	np.sqrt(uspec**2 + qspec**2)
    # Calculate the error in linear polarization intensity
    dlspec		=	np.sqrt((uspec*noispec0[2])**2 + (qspec*noispec0[1])**2)/lspec
    # Calculate the total polarization intensity
    pspec		=	np.sqrt(lspec**2 + vspec**2)
    # Calculate the error in total polarization intensity
    dpspec		=	np.sqrt((vspec*dlspec)**2 + (lspec*noispec0[3])**2)/pspec

    # Calculate the fractional polarizations
    qfracspec	=	qspec/ispec
    ufracspec	=	uspec/ispec
    vfracspec	=	vspec/ispec
    # Calculate the errors in fractional polarizations
    dqfrac		=	np.sqrt((qspec*noispec0[0])**2 + (ispec*noispec0[1])**2)/(ispec**2)
    dufrac		=	np.sqrt((uspec*noispec0[0])**2 + (ispec*noispec0[2])**2)/(ispec**2)
    dvfrac		=	np.sqrt((vspec*noispec0[0])**2 + (ispec*noispec0[3])**2)/(ispec**2)

    # Calculate the fractional linear and total polarizations
    lfracspec	=	lspec/ispec
    dlfrac		=	np.sqrt((lspec*noispec0[0])**2 + (ispec*dlspec)**2)/(ispec**2)
    pfracspec	=	pspec/ispec
    dpfrac		=	np.sqrt((pspec*noispec0[0])**2 + (ispec*dpspec)**2)/(ispec**2)

    # Calculate the polarization angles
    phispec		=	np.rad2deg(0.5*np.arctan2(uspec, qspec))		
    dphispec	=	np.rad2deg(0.5*np.sqrt((uspec*noispec0[1])**2 + (qspec*noispec0[2])**2) / (uspec**2 + qspec**2))

    psispec		=	np.rad2deg(0.5*np.arctan2(vspec, lspec))		
    dpsispec	=	np.rad2deg(0.5*np.sqrt((vspec*dlspec)**2 + (lspec*noispec0[2])**2) / (vspec**2 + lspec**2))

    # Return the spectra as a frbspec object
    return(frbspec(iquvspec, noispec0, lspec, dlspec, pspec, dpspec, qfracspec, dqfrac, ufracspec, dufrac, vfracspec, dvfrac, lfracspec, dlfrac, pfracspec, dpfrac, phispec, dphispec, psispec, dpsispec))
#	-------------------------------------------------------------------------------