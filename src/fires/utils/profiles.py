import numpy as np


def boxcar_width(profile, frac=0.95):
    prof = np.nan_to_num(np.squeeze(profile))
    n = len(prof)
    target_flux = frac * np.sum(prof)
    cumsum = np.cumsum(prof)
    min_width = n
    best_start, best_end = 0, n-1
    for start in range(n):
        start_flux = cumsum[start-1] if start > 0 else 0
        target_end_flux = start_flux + target_flux
        end_indices = np.where(cumsum >= target_end_flux)[0]
        if len(end_indices) > 0:
            end = end_indices[0]
            width = end - start + 1
            if width < min_width:
                min_width = width
                best_start, best_end = start, end
    return best_start, best_end


def make_onpulse_mask(n_time, left, right):
    on_mask = np.zeros(int(n_time), dtype=bool)
    l = max(0, int(left))
    r = min(int(n_time) - 1, int(right))
    if r >= l:
        on_mask[l:r+1] = True
    return on_mask


def make_offpulse_mask(n_time, left, right, buffer_bins=0):
    n = int(n_time)
    l_on = max(0, int(left))
    r_on = min(n - 1, int(right))
    buf = max(0, int(buffer_bins))
    l_excl = max(0, l_on - buf)
    r_excl = min(n - 1, r_on + buf)
    off_mask = np.ones(n, dtype=bool)
    if r_excl >= l_excl:
        off_mask[l_excl:r_excl+1] = False
    return off_mask


def on_off_pulse_masks_from_profile(profile, intrinsic_width_bins, frac=0.95, buffer_frac=None):
    prof = np.asarray(profile, dtype=float)
    n = prof.size
    left, right = boxcar_width(prof, frac=frac)
    buffer_bins = int(float(buffer_frac) * intrinsic_width_bins) if buffer_frac is not None else 0
    on_mask = make_onpulse_mask(n, left, right)
    off_mask = np.zeros(n, dtype=bool)
    end_off = max(0, left - buffer_bins - 1)
    if end_off >= 0:
        off_mask[0:end_off + 1] = True
    return on_mask, off_mask, (left, right)
