from __future__ import annotations

import logging
import os
import re
import shutil
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

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
    "fires"    : ("fires.toml",),
    "plotparams": ("plotparams.toml", "plotparams.txt"),
}

# ---------------- Core helpers ----------------
def user_config_dir() -> Path:
    return Path(DIRS.user_config_dir)

def default_package_path(kind: str) -> Optional[Path]:
    """
    Return a packaged default file path for the given kind (if any).
    Searches fires.config/ for the first candidate that exists.
    """
    for fname in DEFAULT_FILES[kind]:
        try:
            with resources.as_file(resources.files("fires.config") / fname) as p:
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

def _read_toml_lenient_none(p: Path) -> Dict[str, Any]:
    """
    Allow unquoted None (case-insensitive) as a Python None.
    Converts lines like:
      key = None
    into a sentinel then back to None after parsing.
    """
    txt = p.read_text(encoding="utf-8")
    pat = re.compile(r'^(\s*[\w.\-]+\s*=\s*)(None)(\s*(#.*)?)$', re.IGNORECASE | re.MULTILINE)

    def _repl(m):
        prefix, comment = m.group(1), m.group(3) or ""
        return f'{prefix}"__FIRES_SENTINEL_NONE__"{comment}'

    txt = pat.sub(_repl, txt)
    data = tomllib.loads(txt)

    def _restore(d):
        for k, v in d.items():
            if isinstance(v, dict):
                _restore(v)
            elif v == "__FIRES_SENTINEL_NONE__":
                d[k] = None
    _restore(data)
    return data

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
        if kind == "plotparams":
            data = _read_toml_lenient_none(p)
        else:
            data = _read_toml(p)
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


def apply_config_overrides(raw_config: dict, overrides: dict) -> None:
    import logging
    for key_path, value_str in overrides.items():
        keys = key_path.split(".")
        current = raw_config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        value_str = str(value_str).strip()
        if value_str.lower() in ("true", "false"):
            final_value = value_str.lower() == "true"
        elif value_str.lower() in ("null", "none"):
            final_value = None
        else:
            try:
                if "." in value_str or "e" in value_str.lower():
                    final_value = float(value_str)
                else:
                    final_value = int(value_str)
            except ValueError:
                final_value = value_str
        current[keys[-1]] = final_value
        logging.info(f"Config override applied: {key_path} = {final_value}")


def apply_emission_overrides(gdict, sd_dict, param_overrides):
    mean_override_parts = []
    sd_override_parts = []
    if not param_overrides:
        return gdict, sd_dict, mean_override_parts, sd_override_parts
    for key, value in param_overrides.items():
        if key in gdict:
            original_shape = gdict[key].shape
            gdict[key] = np.full(original_shape, value, dtype=float)
            logging.info(f"Override applied: {key} = {value} (shape: {original_shape})")
        elif key in sd_dict:
            sd_dict[key] = float(value)
            logging.info(f"Override applied: {key} = {value} (std dev)")
        else:
            raise ValueError(f"Override key '{key}' not found in gdict or sd_dict.")
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
    return gdict, sd_dict, mean_override_parts, sd_override_parts