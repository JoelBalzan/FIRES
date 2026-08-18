# FIRES: Fast, Intense Radio Emission Simulator

FIRES simulates polarised Fast Radio Burst (FRB) dynamic spectra using a micro-shot (micro-Gaussian) ensemble model with full polarimetric propagation physics.

- **Emission:** Polarised shot-noise (`psn`) or pulsar-fold mode (`fold`) --- stacks N independent realisations
- **Propagation:** Scattering (thin/thick/uniform screens), dispersion, Faraday rotation, diffractive scintillation
- **Noise:** SEFD-based or target-SNR injection with configurable baseline correction
- **Analysis:** Parameter sweeps across emission/propagation parameters with multi-FRB analytic plotting
- **Output:** Stokes I/Q/U/V cubes, pickle files, publication-quality figures (TOML-driven styling)

---

## Quickstart

Skip downloading the simulated data pack in `examples/sim_data.tar.gz`:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/JoelBalzan/FIRES.git
```

Or if you want the data pack:

```bash
git clone https://github.com/JoelBalzan/FIRES.git
```

```bash
cd FIRES
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Initialise user config (one-time)
fires --init-config

# Run with default config
fires --plot lvpa
# or
fires --config-dir ~/.config/fires --plot lvpa

# Run with an example config
fires --config-dir examples/20191001A --plot lvpa
```

---

## Configuration

All simulation parameters are set in a single `fires.toml` file. Search order:

1. Explicit path via `--config-dir`
2. User config: `~/.config/fires/fires.toml`
3. Packaged defaults (shipped with the package)

```bash
fires --config-dir /path/to/fires.toml          # direct file path
fires --config-dir /path/to/config-dir/          # or directory containing fires.toml
fires --edit-config fires                        # open in $EDITOR
fires --override-param tau=0.8 sd_tau=0.15       # inline overrides
```

### Full Configuration Reference

See **[docs/config_wiki.md](docs/config_wiki.md)** for a complete parameter-by-parameter reference with units, defaults, valid options, and code flow explanations.

### Plot Configuration (plotparams.toml)

Plot styling is controlled separately via `plotparams.toml`:

```toml
[general]
extension   = "pdf"
show_plots  = true
save_plots  = true

[analytical]
plot_scale  = "log"
draw_style  = "line-param"

[styling]
font_size   = 16
color_cycle = ["#1f77b4","#ff7f0e","#2ca02c","#d62728"]
```

Override inline:

```bash
fires --override-plot general.extension=png styling.font_size=20 --plot lvpa
```

---

## Emission Models

### `psn` — Polarised Shot Noise (default)

Each Gaussian component spawns N microshots with statistically scattered parameters. Microshots are synthesised in full Stokes and coherently summed into the final dynamic spectrum. The `[emission.components.microshot_scatter]` section controls per-microshot variance — `pa_sigma_deg` is the primary driver of PA variance and depolarisation.

### `fold` — Pulsar Fold Mode

Set `model = "fold"` and `nfold = 10` in `fires.toml`:

```toml
[emission]
model = "fold"

[emission.fold]
nfold = 10
```

Generates `nfold` independent PSN realisations with staggered seeds and averages them into one folded dspec. Signal sums coherently; noise averages down as `sqrt(nfold)`.

---

## Examples

```bash
# Single burst
fires --config-dir examples/20191001A -f FRB191001

# Analytical sweep (PA variance vs scattering timescale)
fires --config-dir examples/20191001A --override-param tau=0.1 --plot pa_var

# Sweep with inline overrides
fires --config-dir examples/20191001A --override-param tau=0.8 sd_tau=0.15 --plot lvpa

# Fold mode with 10 stacks
fires --config-dir my_config --override-param emission.model=fold emission.fold.nfold=10 -f folded_burst

# Multi-window comparison
fires --config-dir examples/20191001A --compare-windows full-band:leading full-band:trailing --plot lvpa

# Observational overlay
fires --config-dir examples/20191001A --plot l_frac --sim-data /path/to/sweep --obs-data /path/to/obs.npy

# Save Stokes cube
fires --config-dir examples/20191001A --save-dspec -f myburst
```

---

## Key Concepts

### Micro-shot Ensemble

Each `[[emission.components]]` defines a Gaussian envelope and a population of microshots:

- Macro envelope: `t0_ms`, `width_ms` define the burst envelope
- Micro population: `N` microshots per component with fractional widths drawn from `Uniform(width_frac_low, width_frac_high)` and amplitudes drawn from `amplitude_distribution`
- Per-microshot scatter: `*_sigma` fields add Gaussian jitter to each microshot parameter

### Windows

- Phase: `leading` / `trailing` / `total`
- Frequency: `1q` / `2q` / `3q` / `4q` / `full`

### Sweeps

Configured under `[analysis.sweep]`:

```toml
[analysis.sweep]
enable = true
mode   = "sd"               # "mean" or "sd"

[analysis.sweep.parameter]
component = 0
name      = "pa_sigma_deg"
start     = 0
stop      = 45
step      = 1
log_steps = 10              # optional log-spaced points
```

Each sweep point generates `nseed` realisations, aggregated over the full on-pulse window.

---

## Command-Line Reference

```
Configuration:
  --config-dir <path>              Path to fires.toml or directory containing it
  --init-config                    Copy packaged defaults to ~/.config/fires/
  --edit-config {fires,plotparams} Open config in $EDITOR

Core I/O:
  -f, --frb_identifier <str>       FRB name (default: FRB)
  -d, --sim-data <path>            Use existing simulation data instead of generating
  -o, --output-dir <dir>           Output directory (default: from config)
  -v, --verbose                    Debug-level logging
  --sd, --save-dspec               Save Stokes I/Q/U/V cube as .npy

Generation:
  --override-param KEY=VAL [...]   Override any emission or config parameter
  -m, --mode {psn}                 Ensemble mode (legacy, output path only)

Windows & Noise:
  --phase-window {leading,trailing,total,first,last,all}
  --freq-window  {1q,2q,3q,4q,full,full-band,...}

Plotting:
  -p, --plot <modes...>      iquv lv dpa RM pa_var l_frac pa pads pali
  --plot-config <path>             Custom plotparams.toml
  --override-plot KEY=VAL [...]    Inline plot overrides (dot notation)

Analytical:
  --compare-windows FREQ:PHASE [FREQ:PHASE ...]
  --obs-data <path>                Observed dspec for overlay
  --obs-params <path>              Observed parameters
  -ts N, --tscrunch N              Average input observed data over time  (factor N)
  -fs N, --fscrunch N              Average input observed data over frequency (factor N)
```

---

## Outputs

| Product | Condition | Format |
|---------|-----------|--------|
| FRB pickle | Single-run mode, `--no-write` not set | `.pkl` (frb data + metadata) |
| Sweep dictionary | Multi-FRB mode, `write = true` in config | `.pkl` (xvals × measures) |
| Stokes cube | Single-run mode, `--save-dspec` | `.npy` (4 × nfreq × ntime) |
| Plots | `save_plots = true` in plotparams, or `--plot` | PDF/PNG per plot mode |

---

## Compatibility

Tested on Linux with Python 3.12.

## Citation

If you use FIRES in your work, please cite <https://ui.adsabs.harvard.edu/abs/2026PASA...43...74B/abstract>.

Scintillation routines adapted from [ScintillationMaker](https://github.com/SprengerT/ScintillationMaker) (Sprenger 2025).

## Acknowledgements

Based on the work of Tehya Conroy and Apurba Bera.

## License

See [LICENSE](LICENSE) for details.

## Contact

If you have any questions please feel free to sent me an email at <joel.balzan@icrar.org>.
