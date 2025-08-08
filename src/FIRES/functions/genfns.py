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


import numpy as np

from FIRES.utils.utils import *
from FIRES.functions.basicfns import *
from FIRES.functions.plotfns import *

GAUSSIAN_FWHM_FACTOR = 2 * np.sqrt(2 * np.log(2))

#    --------------------------	Functions ---------------------------

def apply_faraday_rotation(pol_angle_arr, RM, lambda_sq, median_lambda_sq):
	return np.deg2rad(pol_angle_arr) + RM * (lambda_sq - median_lambda_sq)


def calculate_dispersion_delay(DM, freq, ref_freq):
	return 4.15 * DM * ((1.0e3 / freq) ** 2 - (1.0e3 / ref_freq) ** 2)


def calculate_stokes(temp_dynspec, lfrac, vfrac, faraday_rot_angle):
	stokes_q = temp_dynspec * lfrac * np.cos(2 * faraday_rot_angle)
	stokes_u = temp_dynspec * lfrac * np.sin(2 * faraday_rot_angle)
	stokes_v = temp_dynspec * vfrac
	return stokes_q, stokes_u, stokes_v



# -------------------------- FRB generator functions ---------------------------
def gauss_dynspec(freq_mhz, time_ms, time_res_ms, seed, gdict, noise, tau_ms, sc_idx, ref_freq_mhz, plot_multiple_frb):
	"""
	Generate dynamic spectrum for Gaussian pulses.
	
	Args:
		freq_mhz: Frequency array in MHz
		time_ms: Time array in ms
		time_res_ms: Time resolution in ms
		seed: Random seed for reproducibility
		gdict: Dictionary containing pulse parameters (t0, width_ms, peak_amp, etc.)
		noise: Boolean flag to add noise
		tau_ms: Scattering timescale in ms
		sc_idx: Scattering index
		ref_freq_mhz: Reference frequency in MHz
		plot_multiple_frb: Flag for plotting multiple FRBs
	
	Returns:
		tuple: (dynspec, snr, None) where dynspec is a 4D array [I, Q, U, V]
	"""


	if seed is not None:
		np.random.seed(seed)
		
	t0              = gdict['t0']
	width_ms        = gdict['width_ms']
	peak_amp        = gdict['peak_amp']
	spec_idx        = gdict['spec_idx']
	PA              = gdict['PA']
	DM              = gdict['DM']
	RM              = gdict['RM']
	lfrac           = gdict['lfrac']
	vfrac           = gdict['vfrac']
	dPA             = gdict['dPA']
	band_centre_mhz = gdict['band_centre_mhz']
	band_width_mhz  = gdict['band_width_mhz']

	# Initialize dynamic spectrum for all Stokes parameters
	dynspec = np.zeros((4, freq_mhz.shape[0], time_ms.shape[0]), dtype=float)  # [I, Q, U, V]
	lambda_sq = (speed_of_light_cgs * 1.0e-8 / freq_mhz) ** 2
	median_lambda_sq = np.nanmedian(lambda_sq)
	num_gauss = len(t0) 
	# Calculate frequency-dependent scattering timescale
	if tau_ms > 0:
		tau_cms = tau_ms * (freq_mhz / ref_freq_mhz) ** sc_idx


	for g in range(num_gauss):
		temp_dynspec = np.zeros_like(dynspec)
		norm_amp = peak_amp[g] * (freq_mhz / ref_freq_mhz) ** spec_idx[g]
		pol_angle_arr = PA[g] + (time_ms - t0[g]) * dPA[g]

		# Apply Gaussian spectral profile if band_centre_mhz and band_width_mhz are provided
		if band_width_mhz[g] != 0.:
			centre_freq = band_centre_mhz[g] if band_centre_mhz[g] != 0. else np.median(freq_mhz)
			spectral_profile = gaussian_model(freq_mhz, 1.0, centre_freq, band_width_mhz[g] / GAUSSIAN_FWHM_FACTOR)
			norm_amp *= spectral_profile

		for c in range(len(freq_mhz)):
			faraday_rot_angle = apply_faraday_rotation(pol_angle_arr, RM[g], lambda_sq[c], median_lambda_sq)
			temp_dynspec[0, c] = gaussian_model(time_ms, norm_amp[c], t0[g], width_ms[g] / GAUSSIAN_FWHM_FACTOR)
			
			if int(DM[g]) != 0:
				disp_delay_ms = calculate_dispersion_delay(DM[g], freq_mhz[c], ref_freq_mhz)
				temp_dynspec[0, c] = np.roll(temp_dynspec[0, c], int(np.round(disp_delay_ms / time_res_ms)))
			
			# Apply scattering if enabled
			if tau_ms > 0:
				temp_dynspec[0, c] = scatter_stokes_chan(temp_dynspec[0, c], time_res_ms, tau_cms[c])

			temp_dynspec[1, c], temp_dynspec[2, c], temp_dynspec[3, c] = calculate_stokes(
				temp_dynspec[0, c], lfrac[g], vfrac[g], faraday_rot_angle
			)  # Stokes Q, U, V

		dynspec += temp_dynspec
	snr = None
	if noise:
		dynspec, snr = add_noise(dynspec, 75, (freq_mhz[1] - freq_mhz[0]) * 1e6, (time_res_ms) / 1000, time_ms, plot_multiple_frb)

	return dynspec, snr, None




def m_gauss_dynspec(freq_mhz, time_ms, time_res_ms, seed, gdict, var_dict,
					noise, tau_ms, sc_idx, ref_freq_mhz, plot_multiple_frb, variation_parameter=None, xname=None):
	"""
	Generate dynamic spectrum for multiple main Gaussians, each with a distribution of micro-shots.
	Optionally apply a Gaussian spectral profile to create band-limited pulses.
	"""
	# Set the random seed for reproducibility
	if seed is not None:
		np.random.seed(seed)
	
	t0              = gdict['t0']
	width_ms        = gdict['width_ms']
	peak_amp        = gdict['peak_amp']
	spec_idx        = gdict['spec_idx']
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

	# check if varying scattering time scale or variable from gparams.txt
	if variation_parameter is not None:
		if xname in var_dict:
			var_dict[xname][0] = variation_parameter
		elif xname == "tau_ms":
			tau_ms = variation_parameter
			
	peak_amp_var        = var_dict['peak_amp_var'][0]
	pol_angle_var       = var_dict['PA_var'][0]
	lin_pol_frac_var    = var_dict['lfrac_var'][0]
	circ_pol_frac_var   = var_dict['vfrac_var'][0]
	delta_pol_angle_var = var_dict['dPA_var'][0]
	rm_var              = var_dict['RM_var'][0]
	dm_var              = var_dict['DM_var'][0]
	band_centre_mhz_var = var_dict['band_centre_mhz_var'][0]
	band_width_mhz_var  = var_dict['band_width_mhz_var'][0]
	

	if lin_pol_frac_var > 0.0 and circ_pol_frac_var > 0.0:
		raise ValueError("Both linear and circular polarization variations cannot be > 0.0. "
						"Set one to 0.0 before calling this function.")

			
	dynspec = np.zeros((4, freq_mhz.shape[0], time_ms.shape[0]), dtype=float)  # Initialize dynamic spectrum array
	lambda_sq = (speed_of_light_cgs * 1.0e-8 / freq_mhz) ** 2  # Lambda squared array
	median_lambda_sq = np.nanmedian(lambda_sq)  # Median lambda squared
	# Calculate frequency-dependent scattering timescale
	if tau_ms > 0:
		tau_cms = tau_ms * (freq_mhz / ref_freq_mhz) ** sc_idx
	else:
		tau_cms = np.zeros_like(freq_mhz)

	all_params = {
		'var_peak_amp'       : [],
		'var_width_ms'       : [],
		'var_t0'             : [],
		'var_PA'             : [],
		'var_lfrac'          : [],
		'var_vfrac'          : [],
		'var_dPA'            : [],
		'var_RM'             : [],
		'var_DM'             : [],
		'var_band_centre_mhz': [],
		'var_band_width_mhz' : []
	}

	num_main_gauss = len(t0) 
	for g in range(num_main_gauss):
		for _ in range(int(ngauss[g])):
			# Generate random variations for the micro-Gaussian parameters
			var_peak_amp        = np.random.normal(peak_amp[g], peak_amp_var)
			# Sample the micro width as a percentage of the main width
			var_width_ms        = width_ms[g] * np.random.uniform(width_range[g][0] / 100, width_range[g][1] / 100)
			var_t0              = np.random.normal(t0[g], width_ms[g] / GAUSSIAN_FWHM_FACTOR) 
			var_PA              = np.random.normal(PA[g], pol_angle_var)
			var_lfrac           = np.random.normal(lfrac[g], lin_pol_frac_var)
			var_vfrac           = np.random.normal(vfrac[g], circ_pol_frac_var)
			var_dPA             = np.random.normal(dPA[g], delta_pol_angle_var)
			var_RM              = np.random.normal(RM[g], rm_var)
			var_DM              = np.random.normal(DM[g], dm_var)
			var_band_centre_mhz = np.random.normal(band_centre_mhz[g], band_centre_mhz_var)
			var_band_width_mhz  = np.random.normal(band_width_mhz[g], band_width_mhz_var)

			if circ_pol_frac_var > 0.0:
				var_vfrac = np.clip(var_vfrac, 0.0, 1.0)
				var_lfrac = np.clip(1.0 - var_vfrac, 0.0, 1.0)

			elif lin_pol_frac_var > 0.0:
				var_lfrac = np.clip(var_lfrac, 0.0, 1.0)
				var_vfrac = np.clip(1.0 - var_lfrac, 0.0, 1.0)

			# Append values to the respective lists in `all_params`
			all_params['var_peak_amp'].append(var_peak_amp)
			all_params['var_width_ms'].append(var_width_ms)
			all_params['var_t0'].append(var_t0)
			all_params['var_PA'].append(var_PA)
			all_params['var_lfrac'].append(var_lfrac)
			all_params['var_vfrac'].append(var_vfrac)
			all_params['var_dPA'].append(var_dPA)
			all_params['var_RM'].append(var_RM)
			all_params['var_DM'].append(var_DM)
			all_params['var_band_centre_mhz'].append(var_band_centre_mhz)
			all_params['var_band_width_mhz'].append(var_band_width_mhz)

			# Initialize a temporary array for the current micro-shot
			temp_dynspec = np.zeros_like(dynspec)

			# Calculate the normalized amplitude for each frequency
			norm_amp = var_peak_amp * (freq_mhz / ref_freq_mhz) ** spec_idx[g]
			
			# Apply Gaussian spectral profile if band_centre_mhz and band_width_mhz are provided
			if band_width_mhz[g] != 0.:
				centre_freq = band_centre_mhz[g] if band_centre_mhz[g] != 0. else np.median(freq_mhz)
				spectral_profile = gaussian_model(freq_mhz, 1.0, centre_freq, band_width_mhz[g] / GAUSSIAN_FWHM_FACTOR)
				norm_amp *= spectral_profile

			pol_angle_arr = var_PA + (time_ms - var_t0) * var_dPA

			for c in range(len(freq_mhz)):
				# Apply Faraday rotation
				faraday_rot_angle = apply_faraday_rotation(pol_angle_arr, var_RM, lambda_sq[c], median_lambda_sq)
				# Add the Gaussian pulse to the temporary dynamic spectrum
				temp_dynspec[0, c] = gaussian_model(time_ms, norm_amp[c], var_t0, var_width_ms / GAUSSIAN_FWHM_FACTOR)

				# Calculate the dispersion delay
				if int(var_DM) != 0:
					disp_delay_ms = calculate_dispersion_delay(var_DM, freq_mhz[c], ref_freq_mhz)
					temp_dynspec[0, c] = np.roll(temp_dynspec[0, c], int(np.round(disp_delay_ms / time_res_ms)))

				# Apply scattering if enabled
				if tau_ms > 0:
					temp_dynspec[0, c] = scatter_stokes_chan(temp_dynspec[0, c], time_res_ms, tau_cms[c])

				# Calculate Stokes Q, U, V
				temp_dynspec[1, c], temp_dynspec[2, c], temp_dynspec[3, c] = calculate_stokes(
					temp_dynspec[0, c], var_lfrac, var_vfrac, faraday_rot_angle
				)

			# Accumulate the contributions from the current micro-shot
			dynspec += temp_dynspec

	# Calculate variance for each parameter in var_params
	var_params = {key: np.var(values) for key, values in all_params.items()}

	snr = None
	if noise:
		dynspec, snr = add_noise(dynspec, 75, (freq_mhz[1] - freq_mhz[0])*1e6, (time_res_ms)/1000, time_ms, plot_multiple_frb)

	return dynspec, snr, var_params