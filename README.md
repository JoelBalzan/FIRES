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

Simulate a single FRB and plot linear polarisation + PA:
```bash
fires --plot lvpa
```

Add noise with SEFD (Jy):
```bash
fires --sefd 50 --plot iquv
```

Force a target peak S/N (overrides --sefd):
```bash
fires --snr 25 --plot lvpa
```

Enable scintillation (requires scparams.toml):
```bash
fires --scint --plot iquv
```

Sweep scattering timescale analytically (uses gparams sweep rows):
```bash
fires --plot pa_var l_frac --sweep-mode sd
```

Override a Gaussian parameter mean and std dev at runtime:
```bash
fires --override-param tau=0.8 sd_tau=0.15 --plot lvpa
```

Change plot config values inline:
```bash
fires --override-plot styling.font_size=18 general.extension=png --plot iquv
```

Compare multiple windows from a single run:
```bash
fires --compare-windows full-band:leading full-band:trailing full-band:total --plot lvpa
```

## Configuration System

On first explicit initialization:
```bash
fires --init-config
```
Creates editable copies in:
```
~/.config/fires/
  simparams.toml
  gparams.toml
  scparams.toml
  plotparams.toml
```

Edit a config (respects $VISUAL / $EDITOR; falls back to nano):
```bash
fires --edit-config simparams
fires --edit-config gparams
fires --edit-config scparams
fires --edit-config plotparams
```

Override base directory:
```bash
fires --config-dir /path/to/custom/config
```

Search order per file:
1. Explicit override (e.g. via --config-dir)
2. User config (~/.config/fires/)
3. Packaged defaults (src/fires/data/*.toml)

## File Roles

- simparams.toml: Dynamic spectrum axes (frequency/time ranges), scattering index, reference frequency.
- gparams.toml: Gaussian ensemble specification (mean row + std dev row + sweep rows). Only one parameter may be swept per run.
- scparams.toml: Scintillation maker parameters (timescale, bandwidth, number of phasor components).
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
fires --override-plot general.extension=png general.save_plots=true --plot lvpa
fires --override-plot styling.font_size=20 analytical.plot_scale=log --plot pa_var
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

Noise estimation uses on/off-pulse segmentation plus buffer fraction (--buffer). Default buffer: 1 × intrinsic width.

## Analytical Sweeps

gparams last three rows (start, stop, step) define a sweep range for exactly one parameter (non-zero step). Modes:
- --sweep-mode none: use mean row + std dev row only
- --sweep-mode mean: sweep mean; force its std dev to zero
- --sweep-mode sd: fixed mean; sweep its std dev
Optional logarithmic stepping: --logstep N (replaces linear stepping count).

Analytical plots (pa_var, l_frac) may aggregate multiple FRB realisations (--nseed).

## Baseline Correction

--baseline choices:
- median: subtract median off-pulse
- mean: subtract mean off-pulse
- z: convert to z-score globally
- z_i: per-frequency-channel z-score
Omit for none.

## Scintillation

Enable with --scint (loads scparams.toml). Gain applied multiplicatively to all Stokes prior to noise. Parameters:
- t_s (characteristic timescale, s)
- nu_s (decorrelation bandwidth, Hz)
- N_im (number of phasor components)

## Chi-squared Fitting

--chi2-fit applies a Gaussian chi-squared fit to the final Stokes I profile (time-collapsed, simple initial guess).

## Observational Overlay

Provide measured dynamic spectrum for analytical comparison:
```bash
fires --plot pa_var --obs-data path/to/obs.npy --obs-params path/to/params.toml
```

## Command-Line Reference

```text
Configuration:
  --config-dir <dir>      Override user config directory
  --init-config           Copy packaged defaults to user config
  --edit-config {gparams,simparams,scparams,plotparams}

Core I/O:
  -f, --frb_identifier <str>   FRB identifier (default FRB)
  -d, --sim-data <path>        Existing simulation data (use instead of generating)
  -o, --output-dir <dir>       Output directory (default simfrbs/)
  --write                      Persist simulation products (pickle/arrays)
  -v, --verbose                Verbose logging

Generation:
  -m, --mode psn               Micro-shot ensemble (only mode at present)
  --seed <int>                 RNG seed
  --nseed <int>                Number of realisations (analytical)
  --ncpu <int>                 Parallel threads (analytical multi-runs)
  --sefd <float>               System equivalent flux density (Jy)
  --snr <float>                Target peak S/N (overrides --sefd)
  --scint                      Enable scintillation
  --override-param PARAM=VAL [PARAM=VAL ...]
                               Override mean or std dev (use sd_<param> or <param>_sd)
  -b, --baseline {median,mean,z,z_i}

Windows & Noise:
  --phase-window {leading,trailing,total,first,last,all}
  --freq-window  {1q,2q,3q,4q,full,full-band,...}
  --buffer <float>              Buffer multiplier (default 1)

Plotting:
  -p, --plot <modes...>         any of: all None iquv lvpa dpa RM pa_var l_frac pa
  --plot-config <path>          Custom plotparams.toml
  --override-plot KEY=VALUE [...]  Nested via dot notation

Analytical:
  --sweep-mode {none,mean,sd}
  --logstep <int>               Logarithmic step count (optional)
  --compare-windows FREQ:PHASE [..]  Multi-window overlay (single-run)

Fitting / Overlay:
  --chi2-fit
  --obs-data <path>
  --obs-params <path>
```

## Outputs

Depending on options:
- Pickled simulation objects (--write)
- Plot PDFs/PNGs as configured (plotparams.toml)

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest   # (if tests added)
```

## Citation

Scintillation routines adapted from:
Sprenger T. (2025). ScintillationMaker. https://github.com/SprengerT/ScintillationMaker (commit e33a4ca).

Please cite FIRES and ScintillationMaker if scintillation functionality is used.

## License

See repository for licensing details.
