from __future__ import annotations

import logging
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
    "obsparams": ("obsparams.toml", "obsparams.txt"),
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
                override_path: Optional[Path] = None) -> Dict[str, Any]:
    p = find_config_file(kind, config_dir=config_dir, override_path=override_path)
    if not p.exists():
        raise FileNotFoundError(f"No config file found for kind '{kind}' at {p}")
    if p.suffix.lower() == ".toml":
        return _read_toml(p)
    return _parse_txt(p)


def load_obs_params(config_dir: Optional[Path] = None) -> Dict[str, Any]:
    d = load_params("obsparams", config_dir)
    # Unwrap table if present
    if isinstance(d, dict) and "observation" in d and isinstance(d["observation"], dict):
        out = d["observation"].copy()
        out["_table"] = "observation"
        return out
    return d

def load_scint_params(config_dir: Optional[Path] = None) -> Dict[str, Any]:
    d = load_params("scparams", config_dir)
    if isinstance(d, dict) and "scintillation" in d and isinstance(d["scintillation"], dict):
        out = d["scintillation"].copy()
        out["_table"] = "scintillation"
        return out
    return d

def get_parameters(filename):
    parameters = {}
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            # Skip empty lines or comment lines
            if not line or line.startswith('#'):
                continue
            
            # Check for either '=' or ':' separator
            if '=' in line:
                key, value = line.split('=', 1)  # Use maxsplit=1 to handle extra '=' in values
            elif ':' in line:
                key, value = line.split(':', 1)  # Use maxsplit=1 to handle extra ':' in values
            else:
                continue  # Skip lines without either separator
            
            # Remove square brackets and their contents from the value
            import re
            value = re.sub(r'\[.*?\]', '', value).strip()
            
            parameters[key.strip()] = value
    return parameters


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