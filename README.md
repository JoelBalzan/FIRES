# FIRES: Fast, Intense Radio Emission Simulator

FIRES is a Python package to simulate Fast Radio Bursts (FRBs) with scattering and polarisation effects. It can generate dynamic spectra for Gaussian pulses or micro-shot ensembles, apply scattering, add noise, and provide plotting and simple fitting utilities.

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
fires -t 0.5 --plot iquv --tsys 50
```

Micro-shot mode with LV+PA plot, save plots:
```bash
fires -m psn -t 0.05 --plot lvpa --save-plots
```

Sweep scattering times, aggregate plots:
```bash
fires -t 0.1,10,0.5 --plot pa_var l_var --plot-scale loglog
```

## Configuration (TOML)

Defaults ship with the package and are copied to your user config on first run.

- Package defaults:
  - `src/fires/data/obsparams.toml`
  - `src/fires/data/gparams.toml`
- User config directory (Linux, XDG): `~/.config/fires/`
  - Editable copies: `~/.config/fires/obsparams.toml`, `~/.config/fires/gparams.toml`

Manage config:
```bash
# Create editable defaults in ~/.config/fires
fires --init-config

# Edit in your $EDITOR (VISUAL/EDITOR), falls back to nano
fires --edit-config obsparams
fires --edit-config gparams
```

Override file locations at runtime:
- `--obs_params /path/to/obsparams.toml`
- `--gauss_params /path/to/gparams.toml`
- `--config-dir /path/to/my/configdir` (to use a different config folder)

Notes
- TOML is preferred; legacy `.txt` configs are still read for compatibility.
- The tool searches in order: explicit file path → `--config-dir` (or `~/.config/fires`) → packaged defaults.

## Command-line options (core)

- `-f, --frb_identifier <str>`: Simulation identifier.
- `-o, --obs_params <file>`: Path to observation parameters (TOML or legacy TXT).
- `-g, --gauss_params <file>`: Path to Gaussian/micro-shot parameters (TOML or legacy TXT).
- `-d, --output-dir <dir>`: Output directory (default: `simfrbs/`).
- `--write`: Save simulated data to disk.
- `-p, --plot <modes...>`: One or more of: `all`, `None`, `iquv`, `lvpa`, `dpa`, `RM`, `pa_var`, `l_var`.
- `-s, --save-plots`: Save plots to disk.
- `--phase-window <name>`: `first`, `last`, `all` (aka `leading`, `trailing`, `total`).
- `--freq-window <name>`: `1q`, `2q`, `3q`, `4q`, `full` (aka `lowest-quarter`, `lower-mid-quarter`, `upper-mid-quarter`, `highest-quarter`, `full-band`).
- `--buffer <float>`: Window buffer between on- and off-pulse regions as a function of FRB width for noise calculation. (default: `0.25`)
- `--verbose`: Verbose output.

Config management:
- `--config-dir <dir>`: Override user config directory.
- `--init-config`: Create editable defaults in the user config directory.
- `--edit-config {gparams,obsparams}`: Open the chosen config in your editor.

Tip
- Some plot modes aggregate across multiple tau values (`pa_var`, `l_var`).

## Example TOML snippets

Check the packaged defaults in `src/fires/data/*.toml` 

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
    utils/               # helpers, config loader
      utils.py
      config.py
    data/                # packaged default configs (TOML)
      obsparams.toml
      gparams.toml
```

## Development

```bash
# In repo root
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

##