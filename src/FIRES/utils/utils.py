#
#	Functions for simulating scattering 
#
#								AB, May 2024
#
#	--------------------------	Import modules	---------------------------

import os
import pickle as pkl
import sys
from collections import namedtuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.optimize import curve_fit
from typing import NamedTuple


#    --------------------------	Define parameters	-------------------------------
def get_parameters(filename):
    parameters = {}
    with open(filename, 'r') as file:
        for line in file:
            # Skip empty lines or lines without '='
            if '=' not in line.strip():
                continue
            key, value = line.strip().split('=', 1)  # Use maxsplit=1 to handle extra '=' in values
            parameters[key.strip()] = value.strip()
    return parameters


#

def chi2_fit(x_data, y_data, y_err, model_func, initial_guess):
    """
    Perform chi-squared fitting on the data.

    Args:
        x_data (array): Independent variable (e.g., time or frequency).
        y_data (array): Observed data (e.g., flux density).
        y_err (array): Uncertainties in the observed data.
        model_func (callable): Model function to fit.
        initial_guess (array): Initial guess for the model parameters.

    Returns:
        tuple: Best-fit parameters and chi-squared value.
    """
    try:
        popt, pcov = curve_fit(model_func, x_data, y_data, sigma=y_err, p0=initial_guess, absolute_sigma=True)
        residuals = y_data - model_func(x_data, *popt)
        chi2 = np.sum((residuals / y_err) ** 2)
        return popt, chi2
    except Exception as e:
        print(f"Chi-squared fitting failed: {e}")
        return None, None

def gaussian_model(x, amp, mean, stddev):
    """
    Gaussian model function for fitting.

    Args:
        x (array): Independent variable.
        amp (float): Amplitude of the Gaussian.
        mean (float): Mean of the Gaussian.
        stddev (float): Standard deviation of the Gaussian.

    Returns:
        array: Gaussian function evaluated at x.
    """
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))




# Universal constants 
gravitational_constant_cgs	=	6.67430e-8					#	Universal gravitational constant in CGS
electron_charge_cgs		    =	4.8032047e-10				#	Absolute electronic charge in CGS
electron_mass_cgs			=	9.1093837e-28				#	Electron mass in CGS
speed_of_light_cgs		    =	2.99792458e10				#	Speed of light in CGS
parsec_cm				    =	3.0857e18					#	Parsec in cm							
omega_nu				    =	2 * np.pi					#	Omega / nu 
solar_mass_grams		    =	1.98847e33					#	Solar mass in grams
radian_to_arcsec		    =	180.0 * 3600 / np.pi		#	Radian in arcseconds
radian_to_picoarcsec	    =	180.0 * 3600 * 1.0e12 / np.pi	#	Radian in pico-arcseconds
solar_radius_cm			    =	6.957e10					#	Solar radius in cm
astronomical_unit_cm	    =	1.496e13					#	1 AU in cm
inch_to_cm				    =	2.54


# constants for scintillation application (SCINTOOLS)
#mb2							=	2							#mb2: Max Born parameter for strength of scattering
#rf							=	1							#rf: Fresnel scale
#ds							=	0.01						#ds (or dx,dy): Spatial step sizes with respect to rf
#alpha						=	5/3							#alpha: Structure function exponent (Kolmogorov = 5/3)
#ar							=	1							#ar: Anisotropy axial ratio
#psi							=	0							#psi: Anisotropy orientation
#inner						=	0.001						#inner: Inner scale w.r.t rf - should generally be smaller than ds
#ns							=	256							#ns (or nx,ny): Number of spatial steps
#nf							=	256							#nf: Number of frequency steps.
#dlam						=	0.25						#dlam: Fractional bandwidth relative to centre frequency
#lamsteps					=	False						#lamsteps: Boolean to choose whether steps in lambda or freq
#seed						=	1234 						#seed: Seed number, or use "-1" to shuffle
#nx							=	None
#ny							=	None
#dx							=	None
#dy							=	None
#plot						=	False
#verbose						=	False
#dt							=	30



simulated_frb	=	namedtuple('simulated_frb', ['frbname', 'freq_mhz', 'time_ms', 'tau_ms', 'sc_idx', 'gaussian_params', 'dynamic_spectrum'])

# time variation
frb_time_series	=	namedtuple('frbts',['iquvt','lts','elts','pts','epts','phits','dphits','psits','dpsits','qfrac','eqfrac','ufrac','eufrac','vfrac','evfrac','lfrac','elfrac','pfrac','epfrac'])

# spectra (anything varying with freq (hz))
frb_spectrum	=	namedtuple('frbspec',['iquvspec','diquvspec','lspec','dlspec','pspec','dpspec','qfracspec','dqfrac','ufracspec','dufrac','vfracspec','dvfrac',\
									  								'lfracspec','dlfrac','pfracspec','dpfrac','phispec','pshispec','psizpec','dpsispec'])



class DynspecParams(NamedTuple):
    freq_mhz: np.ndarray
    time_ms: np.ndarray
    time_res_ms: float
    spec_idx: np.ndarray
    peak_amp: np.ndarray
    width_ms: np.ndarray
    loc_ms: np.ndarray
    dm: np.ndarray
    pol_angle: np.ndarray
    lin_pol_frac: np.ndarray
    circ_pol_frac: np.ndarray
    delta_pol_angle: np.ndarray
    rm: np.ndarray
    seed: int
    noise: bool
    scatter: bool
    tau_ms: float
    sc_idx: float
    ref_freq_mhz: float
    num_micro_gauss: int
    width_range: float
    band_centre_mhz: float
    band_width_mhz: float






















































