import logging

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import circvar

from fires.core.rm import estimate_rm, rm_correct_dspec, rm_synth
from fires.core.dspec import (compute_segments, format_global_stats,
                               pa_variance_deg2, print_global_stats,
                               process_dspec, scatter_dspec,
                               scatter_loaded_dspec, stokes_consistency_diagnostics,
                               wrap_pa_deg)
from fires.utils.profiles import (boxcar_width, make_offpulse_mask,
                                   make_onpulse_mask,
                                   on_off_pulse_masks_from_profile)
from fires.core.noise import (add_noise, apply_baseline_correction,
                               baseline_stats_from_offpulse, boxcar_snr,
                               compute_required_sefd, correct_baseline,
                               estimate_noise_with_offpulse_mask,
                               scale_dspec_to_target_snr, snr_onpulse)
from fires.utils.utils import frb_spectrum, frb_time_series

logging.basicConfig(level=logging.INFO)

__all__ = [
    'rm_synth', 'estimate_rm', 'rm_correct_dspec',
    'boxcar_width', 'make_onpulse_mask', 'make_offpulse_mask',
    'on_off_pulse_masks_from_profile', 'estimate_noise_with_offpulse_mask',
    'process_dspec', 'compute_segments', 'pa_variance_deg2',
    'format_global_stats', 'print_global_stats', 'wrap_pa_deg',
    'scatter_dspec', 'scatter_loaded_dspec', 'stokes_consistency_diagnostics',
    'add_noise', 'compute_required_sefd', 'boxcar_snr',
    'baseline_stats_from_offpulse', 'apply_baseline_correction',
    'snr_onpulse', 'scale_dspec_to_target_snr', 'correct_baseline',
    'frb_spectrum', 'frb_time_series',
]
