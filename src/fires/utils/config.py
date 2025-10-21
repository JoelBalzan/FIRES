from __future__ import annotations

import logging
import numpy as np
import os
import shutil
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Optional

from platformdirs import AppDirs

try:                 # Python 3.11+
    import tomllib  # type: ignore
except ModuleNotFoundError:  # Python 3.10 fallback
    import tomli as tomllib  # type: ignore

logging.basicConfig(level=logging.INFO)

APP_NAME = "fires"
DIRS = AppDirs(APP_NAME, APP_NAME)

# Map logical kinds to candidate filenames (first existing wins)
DEFAULT_FILES = {
    "gparams"  : ("gparams.toml", "gparams.txt"),
    "simparams": ("simparams.toml", "simparams.txt"),
    "scparams" : ("scparams.toml", "scparams.txt"),  # scintillation params
}

# ---------------- Core helpers ----------------
def user_config_dir() -> Path:
    return Path(DIRS.user_config_dir)

def default_package_path(kind: str) -> Optional[Path]:
    """
    Return a packaged default file path for the given kind (if any).
    Searches fires.data/ for the first candidate that exists.
    """
    for fname in DEFAULT_FILES[kind]:
        try:
            with resources.as_file(resources.files("fires.data") / fname) as p:
                if p.exists():
                    return p
        except FileNotFoundError:
            continue
    return None

def ensure_user_config(update: bool = False, backup: bool = True) -> Path:
    """
    Ensure the user config dir exists and contains default files.
    update=True forces overwrite from packaged defaults.
    """
    cfg_dir = user_config_dir()
    cfg_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    for kind, candidates in DEFAULT_FILES.items():
        target = cfg_dir / candidates[0]  # prefer primary (.toml)
        if target.exists() and not update:
            continue
        pkg = default_package_path(kind)
        if not pkg:
            continue
        if target.exists() and backup:
            bak = target.with_suffix(target.suffix + f".{timestamp}.bak")
            shutil.copy2(target, bak)
        shutil.copy2(pkg, target)
    return cfg_dir

def init_user_config(overwrite: bool = True, backup: bool = True) -> Path:
    return ensure_user_config(update=overwrite, backup=backup)

# --------------- Parsing helpers ---------------
def _coerce(s: str) -> Any:
    sl = s.strip()
    if sl.lower() in ("true", "false"):
        return sl.lower() == "true"
    try:
        if "." in sl or "e" in sl.lower():
            return float(sl)
        return int(sl)
    except ValueError:
        return sl

def _parse_txt(p: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sep = "=" if "=" in line else (":" if ":" in line else None)
        if not sep:
            continue
        k, v = line.split(sep, 1)
        out[k.strip()] = _coerce(v.strip())
    return out

def _read_toml(p: Path) -> Dict[str, Any]:
    with p.open("rb") as f:
        return tomllib.load(f)

# --------------- File resolution ---------------
def find_config_file(kind: str,
                     config_dir: Optional[Path] = None,
                     override_path: Optional[Path] = None) -> Path:
    """
    Resolve the path to a config file.
    Priority:
    1. override_path (explicit)
    2. Existing file (or explicit file) in provided config_dir
    3. Existing file in user config dir
    4. Packaged default (copied into user dir)
    Always returns the primary candidate name path (even if .txt exists) unless
    an explicit file path was given.
    """
    if override_path:
        p = Path(override_path).expanduser().resolve()
        return p

    if config_dir:
        cfg_in = Path(config_dir).expanduser()
        cfg = cfg_in.resolve()
        # If user passed a direct file path (not a directory)
        if cfg.is_file():
            return cfg
        if cfg.is_dir():
            # Search for candidate names inside provided dir
            for fname in DEFAULT_FILES[kind]:
                cand = cfg / fname
                if cand.exists():
                    return cand
            # Nothing found; warn (optional)
            logging.warning(f"No {kind} candidates found in {cfg}; falling back to user config.")
        else:
            logging.warning(f"Provided --config-dir path does not exist: {cfg_in}")

    # Ensure user config contains defaults
    ensure_user_config(update=False)
    u_dir = user_config_dir()
    for fname in DEFAULT_FILES[kind]:
        cand = u_dir / fname
        if cand.exists():
            return cand

    # Fallback: copy packaged default if missing
    pkg = default_package_path(kind)
    if pkg:
        target = u_dir / DEFAULT_FILES[kind][0]
        if not target.exists():
            shutil.copy2(pkg, target)
        return target

    # Last resort: return primary candidate (non-existent) in user dir
    return u_dir / DEFAULT_FILES[kind][0]

# --------------- Public loading API ---------------
def load_params(kind: str,
                config_dir: Optional[Path] = None,
                name: Optional[str] = None,
                override_path: Optional[Path] = None) -> Dict[str, Any]:
    p = find_config_file(kind, config_dir=config_dir, override_path=override_path)
    if not p.exists():
        raise FileNotFoundError(f"No config file found for kind '{kind}' at {p}")
    if p.suffix.lower() == ".toml":
        data = _read_toml(p)
        # If a table name is requested, return that sub-table when present
        if name and isinstance(data, dict) and name in data and isinstance(data[name], dict):
            out = data[name].copy()
            out["_table"] = name
            return out
        return data
    d = _parse_txt(p)
    if isinstance(d, dict) and name in d and isinstance(d[name], dict):
        out = d[name].copy()
        out["_table"] = name
        return out
    return d

# --------------- Legacy TXT parsing ---------------
# used when loading CELEBI param files
def get_parameters(filepath):
    """
    Parse a parameters.txt file with key = value format.
    
    Extracts relevant parameters for FRB analysis and converts to FIRES format.
    
    Parameters:
    -----------
    filepath : str
        Path to parameters.txt file
        
    Returns:
    --------
    dict
        Dictionary with parameter arrays (e.g., 'DM', 'RM', 'width_ms', 'tau_ms')
    """
    params = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and section headers
            if not line or line.startswith('****') or line.startswith('#'):
                continue
            
            # Parse key = value
            if '=' in line:
                parts = line.split('=', 1)
                key = parts[0].strip()
                value = parts[1].strip()
                
                # Remove trailing comments
                if '#' in value:
                    value = value.split('#')[0].strip()
                
                params[key] = value
    
    # Convert to FIRES format
    gdict = {}
    
    # DM
    if 'dm_frb' in params:
        try:
            gdict['DM'] = np.array([float(params['dm_frb'])])
        except ValueError:
            gdict['DM'] = np.array([0.0])
    else:
        gdict['DM'] = np.array([0.0])
    
    # RM (not in this format, default to 0)
    gdict['RM'] = np.array([0.0])
    
    # Width (estimate from data or use default)
    gdict['width_ms'] = np.array([1.0])  # Will be updated from data if available
    
    # Tau (scattering timescale, not in this format)
    gdict['tau_ms'] = np.array([0.0])
    
    # Center frequency
    if 'centre_freq_frb' in params:
        try:
            gdict['band_centre_mhz'] = np.array([float(params['centre_freq_frb'])])
        except ValueError:
            pass
    
    # Bandwidth
    if 'bw' in params:
        try:
            gdict['band_width_mhz'] = np.array([float(params['bw'])])
        except ValueError:
            pass
    
    # RA/Dec for label
    if 'label' in params:
        gdict['label'] = params['label']
    elif 'ra_frb' in params and 'dec_frb' in params:
        gdict['label'] = f"RA={params['ra_frb']}, Dec={params['dec_frb']}"
    else:
        gdict['label'] = "FRB"
    
    logging.info(
        f"Parsed parameters: DM={gdict.get('DM', [0])[0]:.2f} pc/cm³, "
        f"center_freq={gdict.get('band_centre_mhz', ['N/A'])[0]} MHz, "
        f"bandwidth={gdict.get('band_width_mhz', ['N/A'])[0]} MHz"
    )
    
    return gdict


# --------------- Saving / editing ---------------
def save_params(kind: str,
                data: Dict[str, Any],
                config_dir: Optional[Path] = None) -> Path:
    cfg_dir = Path(config_dir) if config_dir else user_config_dir()
    cfg_dir.mkdir(parents=True, exist_ok=True)
    target = cfg_dir / DEFAULT_FILES[kind][0]

    # Write TOML (simple manual emitter to avoid extra dependency)
    def _emit(obj, indent=0):
        lines = []
        for k, v in obj.items():
            if isinstance(v, dict):
                lines.append(f"\n[{k}]")
                lines.extend(_emit(v, indent + 1))
            else:
                if isinstance(v, str):
                    lines.append(f"{k} = \"{v}\"")
                elif isinstance(v, bool):
                    lines.append(f"{k} = {str(v).lower()}")
                else:
                    lines.append(f"{k} = {v}")
        return lines

    content = "\n".join(_emit(data)) + "\n"
    target.write_text(content)
    return target

def open_in_editor(p: Path) -> None:
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "nano"
    os.system(f"{editor} {p}")

def edit_params(kind: str, config_dir: Optional[Path] = None) -> Path:
    p = find_config_file(kind, config_dir=config_dir)
    if not p.exists():
        raise FileNotFoundError(p)
    open_in_editor(p)
    return p