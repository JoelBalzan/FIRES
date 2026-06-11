import logging
import os
import pickle as pkl

import numpy as np

from fires.utils.utils import simulated_frb


def write_stokes_cube(dspec, freq_mhz, time_ms, out_dir, frb_id):
    if dspec is None:
        return None
    if dspec.ndim != 3 or dspec.shape[0] != 4:
        raise ValueError("Expected dspec with shape (4, nfreq, ntime) to save Stokes cube.")
    dspec_dir = os.path.join(out_dir, f"{frb_id}_dspec")
    os.makedirs(dspec_dir, exist_ok=True)
    for idx, stokes in enumerate(["I", "Q", "U", "V"]):
        out_path = os.path.join(dspec_dir, f"out_{stokes}.npy")
        np.save(out_path, dspec[idx])
    full_path = os.path.join(dspec_dir, "dspec.npy")
    np.save(full_path, dspec)
    if freq_mhz is not None:
        np.save(os.path.join(dspec_dir, "freq.npy"), freq_mhz)
    if time_ms is not None:
        np.save(os.path.join(dspec_dir, "time.npy"), time_ms)
    logging.info("Saved Stokes dspec cube to %s", dspec_dir)
    return dspec_dir


def build_sweep_output_path(out_dir, frb_id, mode, xvals, nseed, plot_mode, xname,
                           sweep_idx, mean_override_parts=None, sd_override_parts=None):
    parts = [f"sweep_{sweep_idx}", f"n{nseed}", f"plot_{plot_mode.name}", f"xname_{xname}"]
    if len(xvals) > 0:
        parts.append(f"xvals_{min(xvals):.2f}-{max(xvals):.2f}")
    parts.append(f"mode_{mode}")
    if mean_override_parts:
        parts.extend(mean_override_parts)
    if sd_override_parts:
        parts.extend(sd_override_parts)
    fname = "_".join(parts) + ".pkl"
    return os.path.join(out_dir, fname)


def write_frb_dict(frb_dict, fpath):
    with open(fpath, "wb") as f:
        pkl.dump(frb_dict, f)
    logging.info(f"Saved results to {fpath}")


def build_single_output_path(out_dir, frb_id, mode, tau, seed, nseed, pa, mean_override_parts=None):
    parts = [
        frb_id,
        f"mode_{mode}",
        f"sc_{tau:.2f}",
        f"seed_{seed}",
    ]
    if nseed is not None:
        parts.append(f"nseed_{nseed}")
    parts.append(f"PA_{pa:.2f}")
    return os.path.join(out_dir, "_".join(parts) + ".pkl")


def write_single_frb(frb_data, out_dir, frb_id, mode, tau, seed, nseed, pa):
    out_file = build_single_output_path(out_dir, frb_id, mode, tau, seed, nseed, pa)
    with open(out_file, 'wb') as frb_file:
        pkl.dump(frb_data, frb_file)


def build_override_parts(param_overrides, gdict, sd_dict):
    mean_override_parts = []
    sd_override_parts = []
    if not param_overrides:
        return mean_override_parts, sd_override_parts
    for key, value in param_overrides.items():
        if key.startswith("sd_"):
            base_key = key[3:].replace("_", "")
            append_to = sd_override_parts
        else:
            base_key = key
            append_to = mean_override_parts
        if isinstance(value, (int, np.integer)):
            append_to.append(f"sd{base_key}{value}" if append_to is sd_override_parts else f"{base_key}{value}")
        elif isinstance(value, (float, np.floating)):
            if value.is_integer():
                append_to.append(f"sd{base_key}{int(value)}" if append_to is sd_override_parts else f"{base_key}{int(value)}")
            else:
                append_to.append(f"sd{base_key}{value:.2f}" if append_to is sd_override_parts else f"{base_key}{value:.2f}")
        else:
            prefix = "sd" if append_to is sd_override_parts else ""
            append_to.append(f"{prefix}{base_key}{value}")
    return mean_override_parts, sd_override_parts
