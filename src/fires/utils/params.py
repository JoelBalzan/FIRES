from __future__ import annotations
import re

# Base parameter symbols and units
_BASE_INFO = {
    "t0"              : (r"t_0", r"\mathrm{ms}"),
    "width_ms"        : (r"W_0", r"\mathrm{ms}"),
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
    "tau_ms"          : (r"\tau_0", r"\mathrm{ms}"),
    "N"               : (r"N_{\mathrm{gauss},0}", ""),
    "mg_width_low"    : (r"W_{\mathrm{low},0}", r"\mathrm{ms}"),
    "mg_width_high"   : (r"W_{\mathrm{high},0}", r"\mathrm{ms}"),
}

def base_param_name(key: str) -> str:
    """
    Strip role prefixes/suffixes to get the base parameter name.
    Examples:
      'sd_PA' -> 'PA'
      'meas_var_PA' -> 'PA'
      'meas_std_width_ms' -> 'width_ms'
    """
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
    # Fallback to raw name if unknown
    return _BASE_INFO.get(base, (base, ""))


def param_info(name: str) -> tuple[str, str]:
    """
    Return (latex_symbol, unit) for any of:
      - base key, e.g. 'PA'
      - 'sd_<base>'
      - 'meas_std_<base>'
      - 'meas_var_<base>'
    """
    base = base_param_name(name)
    sym, unit = base_symbol_unit(base)

    if name.startswith("sd_") or name.startswith("meas_std_"):
        # σ_{symbol core}, unit unchanged
        # remove trailing _0 if present in symbol for sd label
        sym_core = sym.replace(",0", "").replace("_0", "")
        return rf"\sigma_{{{sym_core}}}", unit

    if name.startswith("meas_var_"):
        # mathbb{V}(symbol), unit squared (if any)
        u = f"{unit}^2" if unit not in ("",) else ""
        # Pretty special-case: PA variance uses mathbb{V}(\psi)
        if base == "PA":
            return r"\mathbb{V}(\psi)", r"\mathrm{deg}^2"
        return rf"\mathbb{{V}}({sym})", u

    # Base
    return sym, unit