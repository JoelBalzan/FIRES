# -----------------------------------------------------------------------------
# utils.py
# FIRES: The Fast, Intense Radio Emission Simulator
#
# This module provides utility functions and constants for the FIRES simulation
# pipeline, including parameter file parsing, chi-squared fitting, Gaussian
# modeling, and physical constants. It also defines namedtuples for FRB data
# structures used throughout the codebase.
#
# Author: JB
# Date: 2025-05-20
# -----------------------------------------------------------------------------

#	--------------------------	Import modules	---------------------------

import os
import numpy as np

from typing import NamedTuple
from collections import namedtuple
from scipy.optimize import curve_fit


# --------------------------	Define paths	-------------------------------
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
obs_params_path = os.path.join(parent_dir, "data/obsparams.txt")
gauss_params_path = os.path.join(parent_dir, "data/gparams.txt")


#    --------------------------	Define parameters	-------------------------------
def get_parameters(filename):
    parameters = {}
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            # Skip empty lines or comment lines
            if not line or line.startswith('#'):
                continue
            
            # Check for either '=' or ':' separator
            if '=' in line:
                key, value = line.split('=', 1)  # Use maxsplit=1 to handle extra '=' in values
            elif ':' in line:
                key, value = line.split(':', 1)  # Use maxsplit=1 to handle extra ':' in values
            else:
                continue  # Skip lines without either separator
            
            # Remove square brackets and their contents from the value
            import re
            value = re.sub(r'\[.*?\]', '', value).strip()
            
            parameters[key.strip()] = value
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


window_map = {
	'1q': 'lowest-quarter',
	'2q': 'lower-mid-quarter',
	'3q': 'upper-mid-quarter',
	'4q': 'highest-quarter',
	'full': 'full-band',

    'first': 'leading',
	'last': 'trailing',
	'all': 'total'
}

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



simulated_frb	=	namedtuple('simulated_frb', ['frbname', 'dynamic_spectrum', 'dspec_params', 'snr'])

# time variation
frb_time_series	=	namedtuple('frbts',['iquvt','lts','elts','pts','epts','phits','dphits','psits','dpsits','qfrac','eqfrac','ufrac','eufrac','vfrac','evfrac','lfrac','elfrac','pfrac','epfrac'])

# spectra (anything varying with freq (hz))
frb_spectrum	=	namedtuple('frbspec',['iquvspec','diquvspec','lspec','dlspec','pspec','dpspec','qfracspec','dqfrac','ufracspec','dufrac','vfracspec','dvfrac',\
									  								'lfracspec','dlfrac','pfracspec','dpfrac','phispec','pshispec','psizpec','dpsispec'])



class DynspecParams(NamedTuple):
    gdict          : dict
    var_dict       : dict
    freq_mhz       : np.ndarray
    freq_res_mhz   : float
    time_ms        : np.ndarray
    time_res_ms    : float
    seed           : int
    nseed          : int
    tsys           : float
    tau_ms         : float
    sc_idx         : float
    ref_freq_mhz   : float
    phase_window   : str
    freq_window    : str






















































