# FIRES: Fast, Intense Radio Emission Simulator

FIRES is a Python toolkit for simulating Fast Radio Bursts (FRBs) including:
- Micro-shot (micro-Gaussian) ensemble pulse generation
- Scattering (pulse broadening) and dispersion
- Polarisation (I, Q, U, V; PA evolution; linear/circular fractions)
- Optional scintillation (multiplicative gain field) adapted from ScintillationMaker
- Noise injection with system temperature (via SEFD) or direct S/N targeting
- Analytical parameter sweeps (e.g. PA variance vs scattering timescale)
- Flexible plotting with a TOML-driven styling/configuration layer

It is designed for experimentation with intrinsic + propagation effects and for producing publication-quality figures.

## Installation

```bash
git clone https://github.com/JoelBalzan/FIRES.git
cd FIRES
python -m venv .venv
source .venv/bin/activate
pip install -e .
fires --help
```

## Quickstart Examples

On first use, initialise your local configuration (only needs to be done once):
```bash
fires --init-config
```

This creates:
```text
~/.config/fires/
  fires.toml
  plotparams.toml
```

Run with a master config (`--config-dir` accepts either a directory containing `fires.toml` or the file path itself):
```bash
fires --config-dir paper/191001 --plot lvpa
```

Override a parameter at runtime:
```bash
fires --config-dir paper/191001 --override-param tau=0.8 sd_tau=0.15 --plot lvpa
```

Change plot config values inline:
```bash
fires --config-dir paper/191001 --override-plot styling.font_size=18 general.extension=png --plot iquv
```

Compare multiple windows from a single run:
```bash
fires --config-dir paper/191001 --compare-windows full-band:leading full-band:trailing full-band:total --plot lvpa
```

Use precomputed simulation data for analytical plotting:
```bash
fires --config-dir paper/191001/PA_sweep/L0.95 \
      --plot l_frac \
      --sim-data /path/to/precomputed/sweep \
      --obs-data /path/to/obs \
      --obs-params /path/to/parameters.txt
```

## Configuration System

On first explicit initialization:
```bash
fires --init-config
```
Creates editable copies in:
```
~/.config/fires/
  fires.toml
  plotparams.toml
```

Edit a config (respects $VISUAL / $EDITOR; falls back to nano):
```bash
fires --edit-config fires
fires --edit-config plotparams
```

Override config location:
```bash
fires --config-dir /path/to/custom/config-dir
fires --config-dir /path/to/custom/fires.toml
```

Search order per file:
1. Explicit override (e.g. via --config-dir)
2. User config (~/.config/fires/)
3. Packaged defaults (src/fires/config/*.toml)

## File Roles

- fires.toml: Master simulation configuration (meta, grid, propagation, emission, sweep, observation, numerics, output).
- plotparams.toml: Plotting + style configuration (see below).

## Plot Configuration (plotparams.toml)

Controllable entirely via TOML and runtime overrides.

Example (abridged from current default):
```toml
[general]
extension   = "pdf"
use_latex   = true
show_plots  = true
save_plots  = true
legend      = false
xlim        = [-2.75,-1.5]
ylim        = [-40, 50]

[analytical]
plot_scale    = "log"
draw_style    = "line-param"
nbins         = 15
weight_x_by   = "width"
weight_y_by   = "meas_var_PA_i"
ylim          = [2e-4,0.4]
equal_value_lines = 3

[styling]
font_size       = 16
axes_labelsize  = 22
xtick_labelsize = 22
ytick_labelsize = 22
color_cycle     = ["#1f77b4","#ff7f0e","#2ca02c","#d62728"]
```

Runtime override examples:
```bash
fires --config-dir paper/191001 --override-plot general.extension=png general.save_plots=true --plot lvpa
fires --config-dir paper/191001 --override-plot styling.font_size=20 analytical.plot_scale=log --plot pa_var
```

Nested keys use dot notation; lists use Python literal syntax: figsize=[6,4].

## Pulse / Micro-shot Model

A macro-Gaussian envelope is specified by:
- Arrival time t0 (ms)
- Width (FWHM, ms)
- Amplitude A (Jy)
- Scattering timescale tau (ms)
- Dispersion measure DM (pc cm^-3) at reference frequency
- Rotation measure RM (rad m^-2)
- Polarisation angle PA (deg)
- Linear/Circular fractions (lfrac, vfrac)
- dPA (deg/ms) temporal gradient
- Band centre & width (MHz) for frequency localization
- N micro-Gaussians with uniform FWHM fraction range [mg_width_low, mg_width_high] %

Micro-shots: arrival_time_i ~ Normal(t0, sigma_macro) with sigma_macro = width/(2*sqrt(2 ln 2)). Individual micro widths sampled uniformly from given percentage range.

## Windows

Phase windows (synonyms accepted):
- leading (first), trailing (last), total (all)

Frequency windows:
- 1q, 2q, 3q, 4q, full, full-band
- Verbose aliases: lowest-quarter, lower-mid-quarter, upper-mid-quarter, highest-quarter

Noise estimation uses on/off-pulse segmentation plus `observation.buffer_fraction` from `fires.toml`.

## Analytical Sweeps

Sweeps are configured in `fires.toml` under `[analysis.sweep]` and `[analysis.sweep.parameter]`:
- `enable = true|false`
- `mode = "none" | "mean" | "sd"`
- `parameter.name`, `start`, `stop`, `step`
- Optional `log_steps` for logarithmic spacing

Analytical plots (`pa_var`, `l_frac`) aggregate realisations using `numerics.nseed`.

## Baseline Correction

Set `observation.baseline_correct` in `fires.toml`:
- `false` or `null`: disable baseline correction
- `median`: subtract median off-pulse
- `mean`: subtract mean off-pulse
- `z`: convert to z-score globally
- `z_i`: per-frequency-channel z-score

## Scintillation

Configure under `[propagation.scintillation]` in `fires.toml`.
Gain is applied multiplicatively to all Stokes prior to noise.

## Chi-squared Fitting

--chi2-fit applies a Gaussian chi-squared fit to the final Stokes I profile (time-collapsed, simple initial guess).

## Observational Overlay

Provide measured dynamic spectrum for analytical comparison:
```bash
fires --config-dir paper/191001 --plot pa_var --obs-data path/to/obs.npy --obs-params path/to/params.toml
```

## Command-Line Reference

```text
Configuration:
  --config-dir <path>     Path to fires.toml or directory containing fires.toml (required for runs)
  --init-config           Copy packaged defaults to user config
  --edit-config {fires,plotparams}

Core I/O:
  -f, --frb_identifier <str>   FRB identifier (default FRB)
  -d, --sim-data <path>        Existing simulation data (use instead of generating)
  -o, --output-dir <dir>       Output directory (default simfrbs/)
  -v, --verbose                Verbose logging

Generation:
  -m, --mode psn               Micro-shot ensemble (only mode at present)
  --override-param PARAM=VAL [PARAM=VAL ...]
                               Override mean or std dev (use sd_<param> or <param>_sd)

Windows & Noise:
  --phase-window {leading,trailing,total,first,last,all}
  --freq-window  {1q,2q,3q,4q,full,full-band,...}

Plotting:
  -p, --plot <modes...>         any of: all None iquv lvpa dpa RM pa_var l_frac pa
  --plot-config <path>          Custom plotparams.toml
  --override-plot KEY=VALUE [...]  Nested via dot notation

Analytical:
  --compare-windows FREQ:PHASE [..]  Multi-window overlay (single-run)

Overlay:
  --obs-data <path>
  --obs-params <path>
```

## Outputs

Depending on options:
- Pickled simulation objects (`output.write = true` in `fires.toml`)
- Plot PDFs/PNGs as configured (plotparams.toml)

## Examples and Paper Data

- Example Jupyter notebooks are provided in [examples/](examples/):
  - [FRB_20191001A_example.ipynb](examples/FRB_20191001A_example.ipynb)
  - [FRB_20191001A_sweep.ipynb](examples/FRB_20191001A_sweep.ipynb)
- Parameter files and supporting data used for the paper figures are in [paper/](paper/), with FRB-specific setups under [paper/191001](paper/191001) and [paper/240318A](paper/240318A). The example notebooks reference these configurations to reproduce the paper-style analyses.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest   # (if tests added)
```

## Compatibility

FIRES has been tested on Linux with Python 3.12.

## Citation

Scintillation routines adapted from:
Sprenger T. (2025). ScintillationMaker. https://github.com/SprengerT/ScintillationMaker (commit e33a4ca).

Please cite FIRES (this page and/or https://arxiv.org/abs/2601.19254) and ScintillationMaker if scintillation functionality is used.

## Acknowledgements

This project is based on the work by Tehya Conroy and Apurba Bera.

## License

See repository for licensing details.
