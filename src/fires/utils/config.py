from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from platformdirs import AppDirs
from importlib import resources

try:
    import tomllib  # type: ignore # py311+
except ModuleNotFoundError:
    import tomli as tomllib  # py310

APP_NAME = "fires"
DIRS = AppDirs(APP_NAME, APP_NAME)

DEFAULT_FILES = {
    "gparams": ("gparams.toml", "gparams.txt"),
    "obsparams": ("obsparams.toml", "obsparams.txt"),
}

def user_config_dir() -> Path:
    return Path(DIRS.user_config_dir)

def default_package_path(kind: str) -> Optional[Path]:
    for fname in DEFAULT_FILES[kind]:
        try:
            ref = resources.files("fires.data").joinpath(fname)
            if ref.is_file():
                return Path(ref)
        except Exception:
            continue
    return None

def ensure_user_config() -> Path:
    cfg_dir = user_config_dir()
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for kind, _candidates in DEFAULT_FILES.items():
        pkg = default_package_path(kind)
        if not pkg:
            continue
        target = cfg_dir / pkg.name
        if not target.exists():
            shutil.copy2(pkg, target)
    return cfg_dir

def _coerce(s: str) -> Any:
    sl = s.lower()
    if sl in ("true", "false"):
        return sl == "true"
    try:
        if any(c in s for c in (".", "e", "E")):
            return float(s)
        return int(s)
    except ValueError:
        return s

def _parse_txt(p: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            k, v = parts[0], " ".join(parts[1:])
        out[k.strip()] = _coerce(v.strip())
    return out

def _read_toml(p: Path) -> Dict[str, Any]:
    with p.open("rb") as f:
        return tomllib.load(f)

def find_config_file(kind: str, config_dir: Optional[Path] = None, override_path: Optional[Path] = None) -> Path:
    if override_path:
        return Path(override_path)
    cfg_dir = Path(config_dir) if config_dir else user_config_dir()
    for fname in DEFAULT_FILES[kind]:
        cand = cfg_dir / fname
        if cand.exists():
            return cand
    pkg = default_package_path(kind)
    if pkg:
        return pkg
    return cfg_dir / DEFAULT_FILES[kind][0]

def load_params(kind: str, config_dir: Optional[Path] = None, override_path: Optional[Path] = None) -> Dict[str, Any]:
    p = find_config_file(kind, config_dir, override_path)
    if not p.exists():
        ensure_user_config()
        p = find_config_file(kind, config_dir, override_path)
    if p.suffix.lower() == ".toml":
        return _read_toml(p)
    return _parse_txt(p)

def save_params(kind: str, data: Dict[str, Any], config_dir: Optional[Path] = None) -> Path:
    cfg_dir = Path(config_dir) if config_dir else ensure_user_config()
    target = cfg_dir / DEFAULT_FILES[kind][0]
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import tomli_w
    except Exception as e:
        raise RuntimeError("tomli-w is required to write TOML configs") from e
    with target.open("wb") as f:
        tomli_w.dump(data, f)
    return target

def open_in_editor(p: Path) -> None:
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "nano"
    subprocess.call([editor, str(p)])

def edit_params(kind: str, config_dir: Optional[Path] = None, override_path: Optional[Path] = None) -> Path:
    ensure_user_config()
    p = find_config_file(kind, config_dir, override_path)
    cfg_dir = Path(config_dir) if config_dir else user_config_dir()
    if not str(p).startswith(str(cfg_dir)):
        local = cfg_dir / (p.name if p.suffix else DEFAULT_FILES[kind][0])
        if not local.exists() and p.exists():
            shutil.copy2(p, local)
        p = local
    if not p.exists():
        p.write_text("# Edit parameters here (TOML)\n")
    open_in_editor(p)
    return p