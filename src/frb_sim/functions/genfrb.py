#
#	Simulating scattering 
#
#								AB, Sep 2024
#                                                               
#	--------------------------	Import modules	---------------------------
from importlib.resources import files

import matplotlib as mpl
import numpy as np

from .genfns import *
from ..utils.utils import *

import os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
obs_params_path = os.path.join(parent_dir, "utils/obsparams.txt")
gauss_params_path = os.path.join(parent_dir, "utils/gparams.txt")


#	-------------------------	Execute steps	-------------------------------
def generate_frb(scattering_timescale_ms, frb_identifier, data_dir, mode, num_micro_gauss, seed, width_range, write, 
				 obs_params, gauss_params, noise, scatter, plot, startms, stopms, startchan, endchan
				 ):
	"""
	Generate a simulated FRB with a dispersed and scattered dynamic spectrum
	"""

	# Load observation parameters from obsparams.txt
	obsparams = get_parameters(obs_params)

	start_frequency_mhz = float(obsparams['f0'])
	end_frequency_mhz   = float(obsparams['f1'])
	channel_width_mhz   = float(obsparams['f_res'])
	start_time_ms       = float(obsparams['t0'])
	end_time_ms         = float(obsparams['t1'])
	time_resolution_ms  = float(obsparams['t_res'])
	scattering_index    = float(obsparams['scattering_index'])
	reference_frequency_mhz = float(obsparams['reference_freq'])

	# Array of frequency channels
	frequency_mhz_array = np.arange(
		start_frequency_mhz,
		end_frequency_mhz+ channel_width_mhz,
		channel_width_mhz,
		dtype=float
	)

	# Array of time bins
	time_ms_array = np.arange(
		start_time_ms,
		end_time_ms+ time_resolution_ms,
		time_resolution_ms,
		dtype=float
	)

	# Load Gaussian parameters from gparams.txt
	gaussian_params = np.loadtxt(gauss_params)
	t0              = gaussian_params[:, 0]  # Time of the first Gaussian component
	width           = gaussian_params[:, 1]  # Width of the Gaussian component
	peak_amp        = gaussian_params[:, 2]  # Peak amplitude of the Gaussian component
	spec_idx        = gaussian_params[:, 3]  # Spectral index of the Gaussian component
	dm              = gaussian_params[:, 4]  # Dispersion measure of the Gaussian component
	rm              = gaussian_params[:, 5]  # Rotation measure of the Gaussian component
	pol_angle       = gaussian_params[:, 6]  # Polarization angle of the Gaussian component
	lin_pol_frac    = gaussian_params[:, 7]  # Linear polarization fraction of the Gaussian component
	circ_pol_frac   = gaussian_params[:, 8]  # Circular polarization fraction of the Gaussian component
	delta_pol_angle = gaussian_params[:, 9]  # Change in polarization angle with time of the Gaussian component

	# Check if linear and circular polarization fractions sum to more than 1.0
	if (lin_pol_frac + circ_pol_frac).any() > 1.0:
		print("WARNING: Linear and circular polarization fractions sum to more than 1.0")

	if plot != 'pa_rms':
		if mode=='gauss':
			# Generate initial dispersed dynamic spectrum with Gaussian components
			dynspec = gauss_dynspec(
				frequency_mhz_array,
				time_ms_array,
				channel_width_mhz,
				time_resolution_ms,
				spec_idx,
				peak_amp,
				width,
				t0,
				dm,
				pol_angle,
				lin_pol_frac,
				circ_pol_frac,
				delta_pol_angle,
				rm,
				seed,
				noise,
				scatter,
				scattering_timescale_ms,
				scattering_index,
				reference_frequency_mhz
			)
		elif mode=='sgauss':
			dynspec = sub_gauss_dynspec(
				frequency_mhz_array,
				time_ms_array,
				channel_width_mhz,
				time_resolution_ms,
				spec_idx,
				peak_amp,
				width,
				t0,
				dm,
				pol_angle,
				lin_pol_frac,
				circ_pol_frac,
				delta_pol_angle,
				rm,
				num_micro_gauss,
				seed,
				width_range,
				noise,
				scatter,
				scattering_timescale_ms,
				scattering_index,
				reference_frequency_mhz
			)

		# 'Pickle' the simulated FRB and save it to the disk
		simulated_frb_data = simulated_frb(
			frb_identifier,
			frequency_mhz_array,
			time_ms_array,
			scattering_timescale_ms,
			scattering_index,
			gaussian_params,
			dynspec
		)

		if write:
			# Create the data directory, keep all simulated FRBs
			output_filename = "{}{}_sc_{:.2f}.pkl".format(data_dir, frb_identifier, scattering_timescale_ms)
			with open(output_filename, 'wb') as frbfile:
				pkl.dump(simulated_frb_data, frbfile)

			return simulated_frb_data, rm

		else:
			return simulated_frb_data, rm
		
	elif plot == 'pa_rms':
		# If multiple scattering timescales are provided, generate dynamic spectra for each
		pa_rms_values = []
		pa_rms_errors = []
		if mode == 'gauss':
			for s in scattering_timescale_ms:
				dynspec = gauss_dynspec(
					frequency_mhz_array,
					time_ms_array,
					channel_width_mhz,
					time_resolution_ms,
					spec_idx,
					peak_amp,
					width,
					t0,
					dm,
					pol_angle,
					lin_pol_frac,
					circ_pol_frac,
					delta_pol_angle,
					rm,
					seed,
					noise,
					scatter,
					s,
					scattering_index,
					reference_frequency_mhz
				)

				max_rm = rm[np.argmax(np.abs(rm))]
				# Correct for RM and estimate profiles
				max_rm = rm[np.argmax(np.abs(rm))]
				corrdspec = rm_correct_dynspec(dynspec, frequency_mhz_array, max_rm)
				noisespec	=	estimate_noise(dynspec, time_ms_array, startms, stopms)  
				tsdata = est_profiles(corrdspec, frequency_mhz_array, time_ms_array, noisespec, startchan, endchan)

				# Calculate PA RMS and error
				pa_rms = np.sqrt(np.nanmean(tsdata.phits**2))
				pa_rms_error = np.sqrt(np.nansum((2 * tsdata.phits * tsdata.dphits)**2)) / (2 * len(tsdata.phits))

				pa_rms_values.append(pa_rms)
				pa_rms_errors.append(pa_rms_error)

			return pa_rms_values, pa_rms_errors
		elif mode == 'sgauss':
			for s in scattering_timescale_ms:
				dynspec = sub_gauss_dynspec(
					frequency_mhz_array,
					time_ms_array,
					channel_width_mhz,
					time_resolution_ms,
					spec_idx,
					peak_amp,
					width,
					t0,
					dm,
					pol_angle,
					lin_pol_frac,
					circ_pol_frac,
					delta_pol_angle,
					rm,
					num_micro_gauss,
					seed,
					width_range,
					noise,
					scatter,
					s,
					scattering_index,
					reference_frequency_mhz
				)

				max_rm = rm[np.argmax(np.abs(rm))]
				# Correct for RM and estimate profiles
				max_rm = rm[np.argmax(np.abs(rm))]
				corrdspec = rm_correct_dynspec(dynspec, frequency_mhz_array, max_rm)
				noisespec	=	estimate_noise(dynspec, time_ms_array, startms, stopms)  
				tsdata = est_profiles(corrdspec, frequency_mhz_array, time_ms_array, noisespec, startchan, endchan)

				# Calculate PA RMS and error
				pa_rms = np.sqrt(np.nanmean(tsdata.phits**2))
				pa_rms_error = np.sqrt(np.nansum((2 * tsdata.phits * tsdata.dphits)**2)) / (2 * len(tsdata.phits))

				pa_rms_values.append(pa_rms)
				pa_rms_errors.append(pa_rms_error)

			return np.array(pa_rms_values), np.array(pa_rms_errors)
		else:
			print("Invalid mode specified. Please use 'gauss' or 'sgauss'.")
	


























































