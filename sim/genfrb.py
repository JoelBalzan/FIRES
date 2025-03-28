#
#	Simulating scattering 
#
#								AB, Sep 2024
#                                                               
#	--------------------------	Import modules	---------------------------
import matplotlib as mpl
import numpy as np
from functions.genfns import *
from utils.utils import *


#	-------------------------	Execute steps	-------------------------------


# Check if linear and circular polarization fractions sum to more than 1.0
if (lin_pol_frac + circ_pol_frac).any() > 1.0:
    print("WARNING: Linear and circular polarization fractions sum to more than 1.0")

if (t0.any()<(-time_window_ms)) or (t0.any()>time_window_ms):
    print("WARNING: Gaussian component(s) outside the time window")


def generate_frb(scattering_timescale_ms, frb_identifier, write=True):
    """
    Generate a simulated FRB with a dispersed and scattered dynamic spectrum
    """
    # Generate initial dispersed dynamic spectrum with Gaussian components
    initial_dynspec = gauss_dynspec(
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
        rm
    )

    # Scatter the dynamic spectrum
    scattered_dynspec = scatter_dynspec(
        initial_dynspec,
        frequency_mhz_array,
        time_ms_array,
        channel_width_mhz,
        time_resolution_ms,
        scattering_timescale_ms,
        scattering_index,
        rm=np.max(rm)
    )

    # 'Pickle' the simulated FRB and save it to the disk
    simulated_frb_data = simulated_frb(
        frb_identifier,
        frequency_mhz_array,
        time_ms_array,
        scattering_timescale_ms,
        scattering_index,
        gaussian_params,
        scattered_dynspec
    )

    if write:
        # Create the data directory, keep all simulated FRBs
        output_filename = "{}{}_sc_{:.2f}.pkl".format(data_directory, frb_identifier, scattering_timescale_ms)
        with open(output_filename, 'wb') as frbfile:
            pkl.dump(simulated_frb_data, frbfile)
    else:
        return simulated_frb_data


























































