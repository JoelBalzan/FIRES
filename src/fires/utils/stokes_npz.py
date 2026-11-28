"""
Single-file Stokes dynamic spectrum archive shared by FIRES and FRBop.

One ``.npz`` holds the (4, nchan, nsamp) cube, its axes and metadata. It is a superset of the
``stokes`` + ``pol_order`` + ``freq_mhz`` + ``time_ms`` archive that FRBop's dmop already reads,
and contains no pickled objects, so ``np.load(path)`` works with ``allow_pickle=False``.

Required keys: stokes, pol_order, freq_mhz, time_ms, nchan, nsamp, format, format_version.
Optional keys: label, source, flux_unit, dm, rm_applied, ref_freq_mhz, chan_mask, params_json.
Frequency order is whatever ``freq_mhz`` says; readers must not assume one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FORMAT_NAME = "stokes-dspec"
FORMAT_VERSION = 1
_OPTIONAL_SCALARS = ("label", "source", "flux_unit", "dm", "rm_applied", "ref_freq_mhz")


def write_stokes_npz(path, stokes, freq_mhz, time_ms, *, pol_order="IQUV", chan_mask=None,
                     params=None, compress=False, dtype=np.float32, **meta):
    """
    Write a Stokes cube and its axes to ``path`` (``.npz`` is appended by NumPy if missing).

    ``meta`` takes the optional scalar keys (label, source, flux_unit, dm, rm_applied,
    ref_freq_mhz); ``params`` is any JSON-serialisable dict, stored as ``params_json``.
    The cube is stored as ``dtype`` (float32 by default, ~7 significant digits); axes stay float64.
    """
    stokes = np.asarray(stokes, dtype=dtype)
    freq_mhz = np.asarray(freq_mhz, dtype=float)
    time_ms = np.asarray(time_ms, dtype=float)
    if stokes.ndim != 3 or stokes.shape[0] != len(pol_order):
        raise ValueError(f"stokes must have shape ({len(pol_order)}, nchan, nsamp), got {stokes.shape}")
    if stokes.shape[1:] != (freq_mhz.size, time_ms.size):
        raise ValueError(
            f"axes ({freq_mhz.size} channels, {time_ms.size} samples) do not match stokes {stokes.shape}"
        )
    unknown = set(meta) - set(_OPTIONAL_SCALARS)
    if unknown:
        raise ValueError(f"unknown metadata keys: {sorted(unknown)}")

    arrays = {
        "stokes": stokes,
        "pol_order": np.array(pol_order),
        "freq_mhz": freq_mhz,
        "time_ms": time_ms,
        "nchan": np.array(stokes.shape[1]),
        "nsamp": np.array(stokes.shape[2]),
        "format": np.array(FORMAT_NAME),
        "format_version": np.array(FORMAT_VERSION),
    }
    arrays.update({k: np.array(v) for k, v in meta.items() if v is not None})
    if chan_mask is not None:
        arrays["chan_mask"] = np.asarray(chan_mask, dtype=bool)
    if params is not None:
        arrays["params_json"] = np.array(json.dumps(params, default=_json_default))
    (np.savez_compressed if compress else np.savez)(path, **arrays)
    return Path(path) if str(path).endswith(".npz") else Path(f"{path}.npz")


def read_stokes_npz(path):
    """
    Read a Stokes archive. Returns (stokes, freq_mhz, time_ms, meta) with stokes as (4, nchan, nsamp)
    in IQUV order (a missing V is filled with zeros). ``meta`` holds every other key as plain
    Python values; ``params_json`` is decoded into ``meta["params"]``.

    Also reads dmop-style archives that only hold ``stokes`` (+ optional ``pol_order`` and axes).
    """
    with np.load(path, allow_pickle=False) as archive:
        if "stokes" not in archive.files:
            raise ValueError(f"{path} has no 'stokes' array")
        fmt = str(archive["format"]) if "format" in archive.files else FORMAT_NAME
        if fmt != FORMAT_NAME:
            raise ValueError(f"{path}: unsupported format {fmt!r}")
        stokes = np.asarray(archive["stokes"])
        meta = {k: _to_python(archive[k]) for k in archive.files if k != "stokes"}

    order = str(meta.get("pol_order", "IQUV"[: stokes.shape[0]])).upper()
    if stokes.ndim != 3 or stokes.shape[0] != len(order) or "I" not in order:
        raise ValueError(f"{path}: cannot interpret stokes {stokes.shape} with pol_order {order!r}")
    cube = np.zeros((4,) + stokes.shape[1:], dtype=stokes.dtype)
    for idx, name in enumerate(order):
        cube["IQUV".index(name)] = stokes[idx]

    nchan, nsamp = cube.shape[1:]
    freq_mhz = np.asarray(meta.pop("freq_mhz", meta.pop("freq", np.arange(nchan))), dtype=float)
    time_ms = np.asarray(meta.pop("time_ms", meta.pop("time", np.arange(nsamp))), dtype=float)
    if freq_mhz.size != nchan or time_ms.size != nsamp:
        raise ValueError(f"{path}: axes do not match stokes shape {stokes.shape}")
    if "params_json" in meta:
        meta["params"] = json.loads(meta.pop("params_json"))
    return cube, freq_mhz, time_ms, meta


def _to_python(value):
    value = np.asarray(value)
    if value.ndim == 0:
        return value.item()
    return value


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)
