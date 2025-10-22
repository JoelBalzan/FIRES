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

import logging
import os
from collections import namedtuple
from typing import NamedTuple

import numpy as np
from scipy.optimize import curve_fit

#	--------------------------	Logging setup	---------------------------
LOG_NAME = "FIRES"

def init_logging(verbose: bool | None = None):
    """
    Initialise package logging once (idempotent).
    Precedence: explicit arg > env FIRES_VERBOSE > default (False).
    """
    if verbose is False:
        env = os.getenv("FIRES_VERBOSE", "").lower()
        verbose = env in ("1", "true", "yes", "on")
    level = logging.DEBUG if verbose else logging.INFO
    logger = logging.getLogger(LOG_NAME)
    if logger.handlers:  # already configured
        logger.setLevel(level)
        return logger
    class BracketFormatter(logging.Formatter):
        COLOURS = {
            "DEBUG": "\033[36m",
            "INFO": "\033[32m",
            "WARNING": "\033[33m",
            "ERROR": "\033[31m",
            "CRITICAL": "\033[41m",
        }
        RESET = "\033[0m"
        def format(self, record):
            base = super().format(record)
            if os.isatty(1):
                colour = self.COLOURS.get(record.levelname, "")
                if colour:
                    return f"{colour}{base}{self.RESET}"
            return base

    handler = logging.StreamHandler()
    fmt = "[%(levelname)s] %(name)s: %(message)s"
    # fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handler.setFormatter(BracketFormatter(fmt))
    logger.addHandler(handler)
    logger.setLevel(level)

    # Silence noisy libs
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    return logger

LOG = logging.getLogger(LOG_NAME)

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


# Canonical short aliases
_freq_alias = {
    '1q': '1q',
    '2q': '2q',
    '3q': '3q',
    '4q': '4q',
    'full': 'full',
    'full-band': 'full',
    'lowest-quarter': '1q',
    'lower-mid-quarter': '2q',
    'upper-mid-quarter': '3q',
    'highest-quarter': '4q',
}
_phase_alias = {
    'first': 'first',
    'leading': 'first',
    'last': 'last',
    'trailing': 'last',
    'all': 'all',
    'total': 'all',
}

def normalise_freq_window(name: str, target: str = 'dspec') -> str:
    """
    Normalise frequency window names.

    target = 'dspec'     -> long names used when slicing dspec:
                           {'1q'->'lowest-quarter', ..., 'full'->'full-band'}
    target = 'segments'  -> keys used in segments dict:
                           {'1q','2q','3q','4q','all'}
    """
    key = str(name).strip().lower()
    short = _freq_alias.get(key, 'full')
    if target == 'segments':
        return 'all' if short == 'full' else short
    # dspec (long names)
    if short == 'full':
        return 'full-band'
    return {
        '1q': 'lowest-quarter',
        '2q': 'lower-mid-quarter',
        '3q': 'upper-mid-quarter',
        '4q': 'highest-quarter',
    }[short]

def normalise_phase_window(name: str, target: str = 'dspec') -> str:
    """
    Normalise phase window names.

    target = 'dspec'     -> {'first'->'leading', 'last'->'trailing', 'all'->'total'}
    target = 'segments'  -> segment keys {'first','last','total'}
    """
    key = str(name).strip().lower()
    short = _phase_alias.get(key, 'all')
    if target == 'segments':
        return {'first': 'first', 'last': 'last', 'all': 'total'}[short]
    # dspec
    return {'first': 'leading', 'last': 'trailing', 'all': 'total'}[short]


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

# CELEBI 
bw_MHz = 336
nchan = 336
bw_hz = bw_MHz * 1e6
dt_ms = nchan*(1000/bw_hz)
df_MHz = bw_MHz/nchan


simulated_frb	=	namedtuple('simulated_frb', ['frbname', 'dynamic_spectrum', 'dspec_params', 'snr'])

# time variation
frb_time_series	=	namedtuple('frbts',['iquvt','Lts','eLts','Pts','ePts','phits','ephits','psits','epsits','qfrac','eqfrac','ufrac','eufrac','vfrac','evfrac','lfrac','elfrac','pfrac','epfrac'])

# spectra (anything varying with freq (hz))
frb_spectrum	=	namedtuple('frbspec',['iquvspec','diquvspec','lspec','dlspec','pspec','dpspec','qfracspec','dqfrac','ufracspec','dufrac','vfracspec','dvfrac',\
									  								'lfracspec','dlfrac','pfracspec','dpfrac','phispec','pshispec','psizpec','dpsispec'])



class dspecParams(NamedTuple):
    gdict          : dict
    sd_dict       : dict
    scint_dict     : dict
    freq_mhz       : np.ndarray
    freq_res_mhz   : float
    time_ms        : np.ndarray
    time_res_ms    : float
    seed           : int
    nseed          : int
    sefd           : float
    sc_idx         : float
    ref_freq_mhz   : float
    phase_window   : str
    freq_window    : str
    buffer_frac    : float
    sweep_mode     : str