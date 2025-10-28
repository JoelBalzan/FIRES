# FIRES: Fast, Intense Radio Emission Simulator

FIRES is a Python package to simulate Fast Radio Bursts (FRBs) with scattering and polarisation effects. It can generate dynamic spectra for Gaussian pulses or micro-shot ensembles, apply scattering, scintillation, add noise, and provide plotting and simple fitting utilities.

## Installation

From GitHub:
```bash
git clone https://github.com/JoelBalzan/FIRES.git
cd FIRES
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quickstart

Show help:
```bash
fires --help
```

Single FRB, plot IQUV, add noise:
```bash
fires --plot iquv --sefd 50
```

Micro-shot mode with LV+PA plot, save plots:
```bash
fires -m psn --plot lvpa --save-plots
```

Sweep scattering times (see configs):
```bash
fires -m psn --plot pa_var l_frac --plot-scale log
```

## Configuration (TOML)

Defaults ship with the package and are copied to your user config on first run.

- Package defaults:
  - `src/fires/data/simparams.toml`
  - `src/fires/data/gparams.toml`
  - `src/fires/data/scparams.toml`

Manage config:
```bash
# Create editable defaults in ~/.config/fires
fires --init-config

# Edit in your $EDITOR (VISUAL/EDITOR), falls back to nano
fires --edit-config simparams
fires --edit-config gparams
fires --edit-config scparams
```

- User config directory (Linux, XDG): `~/.config/fires/`
  - Editable copies: `~/.config/fires/simparams.toml`, `~/.config/fires/gparams.toml`, `~/.config/fires/scparams.toml`

Override file locations at runtime:
- `--config-dir /path/to/my/configdir` (to use a different config folder)

Notes
- The tool searches in order: explicit file path → `--config-dir` (or `~/.config/fires`) → packaged defaults.
- Check the packaged defaults in `src/fires/data/*.toml` for descriptions

## Command-line options (core)

- `-f, --frb_identifier <str>`: Simulation identifier.
- `-o, --output-dir <dir>`: Output directory (default: `simfrbs/`).
- `--write`: Save simulated data to disk.
- `-p, --plot <modes...>`: One or more of: `all`, `None`, `iquv`, `lvpa`, `dpa`, `RM`, `pa_var`, `l_frac`.
- `-s, --save-plots`: Save plots to disk.
- `-d, --data <dir>`: Directory containing Stokes cube (.npy).
- `--phase-window <name>`: `first`, `last`, `all` (aka `leading`, `trailing`, `total`).
- `--freq-window <name>`: `1q`, `2q`, `3q`, `4q`, `full` (aka `lowest-quarter`, `lower-mid-quarter`, `upper-mid-quarter`, `highest-quarter`, `full-band`).
- `--buffer <float>`: Window buffer between on- and off-pulse regions as a function of FRB width for noise calculation. (default: `0.25`)
- `--verbose`: Verbose output.

Config management:
- `--config-dir <dir>`: Override user config directory.
- `--init-config`: Create editable defaults in the user config directory.
- `--edit-config {gparams,simparams}`: Open the chosen config in your editor.

Tip
- Some plot modes aggregate across multiple tau values (`pa_var`, `l_frac`).


## Project structure

```
src/
  fires/
    __init__.py
    __main__.py          # console entry: fires
    core/                # simulation core
      basicfns.py
      genfns.py
      genfrb.py
    plotting/            # plotting API and modes
      plotfns.py
      plotmodes.py
    scint/               # apply scintillation to dynamic spectrum using ScintillationMaker
      ScintillationMaker.py
      ScintillationMaker.cpp
      ScintillationMaker.so
    utils/               # helpers, config loader
      utils.py
      config.py
    data/                # packaged default configs (TOML)
      simparams.toml
      gparams.toml
      scparams.toml
```

## Development

```bash
# In repo root
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Citation

This project includes adapted code from:

Sprenger T. (2025). ScintillationMaker. GitHub repository: https://github.com/SprengerT/ScintillationMaker (commit e33a4ca).

Please cite both this project and the original ScintillationMaker repository if you use the scintillation simulation components.
