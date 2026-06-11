from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# Canonical column mapping: parameter name/alias -> column index in gauss_params matrix
COL_MAP = {
    't0': 0, 't0_ms': 0,
    'width': 1, 'width_ms': 1,
    'a': 2, 'amplitude': 2, 'amplitude_jy': 2,
    'spec_idx': 3, 'spectral_index': 3,
    'tau': 4, 'tau_ms': 4,
    'dm': 5,
    'rm': 6,
    'pa': 7, 'pa_deg': 7,
    'lfrac': 8,
    'vfrac': 9,
    'dpa': 10, 'dpa_deg_per_ms': 10,
    'band_centre_mhz': 11, 'band_centre': 11,
    'band_width_mhz': 12, 'band_width': 12,
    'sd_t0': 0, 't0_sigma_ms': 0,
    'sd_width': 1, 'width_sigma_ms': 1,
    'sd_a': 2, 'amplitude_sigma': 2,
    'sd_spec_idx': 3, 'spectral_index_sigma': 3,
    'sd_tau': 4, 'tau_sigma_ms': 4,
    'sd_dm': 5, 'dm_sigma': 5,
    'sd_rm': 6, 'rm_sigma': 6,
    'sd_pa': 7, 'pa_sigma_deg': 7,
    'sd_lfrac': 8, 'lfrac_sigma': 8,
    'sd_vfrac': 9, 'vfrac_sigma': 9,
    'sd_dpa': 10, 'dpa_sigma': 10,
    'sd_band_centre_mhz': 11, 'band_centre_sigma': 11,
    'sd_band_width_mhz': 12, 'band_width_sigma': 12,
}

EMISSION_KEYS = {
    't0', 'width', 'A', 'spec_idx', 'tau', 'DM', 'RM', 'PA',
    'lfrac', 'vfrac', 'dPA', 'band_centre_mhz', 'band_width_mhz',
    'N', 'mg_width_low', 'mg_width_high', 'amp_sampling',
    'sd_t0', 'sd_width', 'sd_A', 'sd_spec_idx', 'sd_tau', 'sd_DM', 'sd_RM',
    'sd_PA', 'sd_lfrac', 'sd_vfrac', 'sd_dPA', 'sd_band_centre_mhz', 'sd_band_width_mhz',
}

MEAN_ALIASES = {
    't0': 't0', 't0_ms': 't0',
    'width': 'width', 'width_ms': 'width',
    'a': 'A', 'amplitude': 'A', 'amplitude_jy': 'A',
    'spec_idx': 'spec_idx', 'spectral_index': 'spec_idx',
    'tau': 'tau', 'tau_ms': 'tau',
    'dm': 'DM',
    'rm': 'RM',
    'pa': 'PA', 'pa_deg': 'PA',
    'lfrac': 'lfrac',
    'vfrac': 'vfrac',
    'dpa': 'dPA', 'dpa_deg_per_ms': 'dPA',
    'band_centre': 'band_centre_mhz', 'band_centre_mhz': 'band_centre_mhz',
    'band_width': 'band_width_mhz', 'band_width_mhz': 'band_width_mhz',
    'n': 'N', 'microshots_n': 'N',
    'mg_width_low': 'mg_width_low', 'width_frac_low': 'mg_width_low',
    'mg_width_high': 'mg_width_high', 'width_frac_high': 'mg_width_high',
    'amp_sampling': 'amp_sampling',
}

SD_ALIASES = {
    'sd_t0': 'sd_t0', 't0_sigma_ms': 'sd_t0',
    'sd_width': 'sd_width', 'width_sigma_ms': 'sd_width',
    'sd_a': 'sd_A', 'amplitude_sigma': 'sd_A',
    'sd_spec_idx': 'sd_spec_idx', 'spectral_index_sigma': 'sd_spec_idx',
    'sd_tau': 'sd_tau', 'tau_sigma_ms': 'sd_tau',
    'sd_dm': 'sd_DM', 'dm_sigma': 'sd_DM',
    'sd_rm': 'sd_RM', 'rm_sigma': 'sd_RM',
    'sd_pa': 'sd_PA', 'pa_sigma_deg': 'sd_PA',
    'sd_lfrac': 'sd_lfrac', 'lfrac_sigma': 'sd_lfrac',
    'sd_vfrac': 'sd_vfrac', 'vfrac_sigma': 'sd_vfrac',
    'sd_dpa': 'sd_dPA', 'dpa_sigma': 'sd_dPA',
    'sd_band_centre': 'sd_band_centre_mhz', 'sd_band_centre_mhz': 'sd_band_centre_mhz',
    'band_centre_sigma': 'sd_band_centre_mhz',
    'sd_band_width': 'sd_band_width_mhz', 'sd_band_width_mhz': 'sd_band_width_mhz',
    'band_width_sigma': 'sd_band_width_mhz',
}

# Canonical ordering of gdict keys (matching legacy matrix column layout)
GDICT_KEYS = [
    't0', 'width', 'A', 'spec_idx', 'tau', 'DM', 'RM', 'PA',
    'lfrac', 'vfrac', 'dPA', 'band_centre_mhz', 'band_width_mhz',
    'N', 'mg_width_low', 'mg_width_high',
]

# Per-component mean parameters (replaces one row of gauss_params matrix)
@dataclass
class ComponentParams:
    t0: float = 0.0
    width: float = 1.0
    A: float = 1.0
    spec_idx: float = 0.0
    tau: float = 0.0
    DM: float = 0.0
    RM: float = 0.0
    PA: float = 0.0
    lfrac: float = 0.0
    vfrac: float = 0.0
    dPA: float = 0.0
    band_centre_mhz: float = 0.0
    band_width_mhz: float = 0.0
    N: float = 0.0
    mg_width_low: float = 0.0
    mg_width_high: float = 0.0

# Per-parameter standard deviations (replaces the stddev row of gauss_params matrix)
@dataclass
class StdDevParams:
    sd_t0: float = 0.0
    sd_width: float = 0.0
    sd_A: float = 0.0
    sd_spec_idx: float = 0.0
    sd_tau: float = 0.0
    sd_DM: float = 0.0
    sd_RM: float = 0.0
    sd_PA: float = 0.0
    sd_lfrac: float = 0.0
    sd_vfrac: float = 0.0
    sd_dPA: float = 0.0
    sd_band_centre_mhz: float = 0.0
    sd_band_width_mhz: float = 0.0


@dataclass
class SweepSpec:
    param_name: str = ""
    start: float = 0.0
    stop: float = 0.0
    step: float = 0.0
    log_steps: Optional[int] = None

    @property
    def active(self) -> bool:
        return bool(self.param_name)

    @property
    def is_sweep(self) -> bool:
        return self.active and self.step != 0.0

    @property
    def is_single_point(self) -> bool:
        return self.active and self.step == 0.0


def canonical_emission_key(raw_key: str) -> str:
    key_l = raw_key.strip().lower()
    if key_l.startswith("sd_"):
        return SD_ALIASES.get(key_l, raw_key)
    if key_l.endswith("_sd"):
        base = key_l[:-3]
        if base in MEAN_ALIASES:
            return f"sd_{MEAN_ALIASES[base]}"
        return raw_key
    if key_l in SD_ALIASES:
        return SD_ALIASES[key_l]
    return MEAN_ALIASES.get(key_l, raw_key)

# Base parameter symbols and units
_BASE_INFO = {
    "t0"              : (r"t_0", r"\mathrm{ms}"),
    "width"        : (r"W_0", r"\mathrm{ms}"),
    "A"               : (r"A_0", r"\mathrm{Jy}"),
    "spec_idx"        : (r"\alpha_0", ""),
    "DM"              : (r"\mathrm{DM}_0", r"\mathrm{pc\,cm^{-3}}"),
    "RM"              : (r"\mathrm{RM}_0", r"\mathrm{rad\,m^{-2}}"),
    "PA"              : (r"\psi_0", r"\mathrm{deg}"),
    "lfrac"           : (r"\Pi_{L,0}", ""),
    "vfrac"           : (r"\Pi_{V,0}", ""),
    "dPA"             : (r"\Delta\psi_0", r"\mathrm{deg}"),
    "band_centre_mhz" : (r"\nu_{\mathrm{c},0}", r"\mathrm{MHz}"),
    "band_width_mhz"  : (r"\Delta \nu_0", r"\mathrm{MHz}"),
    "tau"          : (r"\tau_0", r"\mathrm{ms}"),
    "N"               : (r"N_{\mathrm{gauss},0}", ""),
    "mg_width_low"    : (r"W_{\mathrm{low},0}", r"\mathrm{ms}"),
    "mg_width_high"   : (r"W_{\mathrm{high},0}", r"\mathrm{ms}"),
}

def base_param_name(key: str) -> str:
    if key.startswith("sd_"):
        return key[3:]
    if key.startswith("meas_var_"):
        return key[len("meas_var_"):]
    if key.startswith("meas_std_"):
        return key[len("meas_std_"):]
    if key.startswith("meas_mean_"):
        return key[len("meas_mean_"):]
    return key

def is_measured_key(key: str) -> bool:
    return key.startswith("meas_")

def is_sd_key(key: str) -> bool:
    return key.startswith("sd_")

def base_symbol_unit(base: str) -> tuple[str, str]:
    return _BASE_INFO.get(base, (base, ""))

def param_info(name: str) -> tuple[str, str]:
    base = base_param_name(name)
    sym, unit = base_symbol_unit(base)
    if name.startswith("sd_") or name.startswith("meas_std_"):
        sym_core = sym.replace(",0", "").replace("_0", "")
        return rf"\sigma_{{{sym_core}}}", unit
    if name.startswith("meas_var_"):
        u = f"{unit}^2" if unit not in ("",) else ""
        if base == "PA":
            return r"\mathbb{V}(\psi)", r"\mathrm{deg}^2"
        return rf"\mathbb{{V}}({sym})", u
    return sym, unit
