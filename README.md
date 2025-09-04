# FIRES: The Fast, Intense Radio Emission Simulator

FIRES is a Python package to simulate Fast Radio Bursts (FRBs) with scattering and polarisation effects. It can generate dynamic spectra for Gaussian pulses, apply scattering, add noise, and provides tools for visualisation and simple fitting.

## Features

- Customisable FRB simulations
  - Single Gaussian pulse or micro-shot ensembles (psn)
  - Sweep over multiple scattering timescales in one run
  - Add system noise via system temperature (K)
- Analysis and plotting
  - Plot IQUV, L and PA, dPA/dt, RM, PA variance, and L/I
  - Windowing by phase and frequency (quarters or full band)
  - Optional chi-squared Gaussian fit to final profiles
- Output
  - Save simulated data and plots to disk

## Project Structure

- Core
  - `src/FIRES/main.py` — CLI entry point
  - `src/FIRES/functions/genfrb.py` — FRB generation
  - `src/FIRES/functions/plotmodes.py` — Plot mode registry and plot functions
- Utilities
  - `src/FIRES/utils/utils.py` — helpers, defaults, fitting
  - `src/FIRES/utils/obsparams.txt` — observation parameters
  - `src/FIRES/utils/gparams.txt` — Gaussian/micro-shot parameters

## Installation

From GitHub
```bash
git clone https://github.com/JoelBalzan/FIRES.git
cd FIRES
pip install -r requirements.txt
pip install -e .
```

## Usage

After installation, run:
```bash
FIRES --help
```

If the console entry is unavailable:
```bash
python -m FIRES.main --help
```

### Command-Line Options (current)

| Flag | Type | Default | Description |
|---|---|---|---|
| -t, --tau_ms | str, nargs+ | 0.0 | Scattering timescale(s) in ms. Accepts single values, lists, and ranges. Examples: `-t 0.5`, `-t 0.1 0.3 1.0`, `-t 0.1,2.0,0.1` (start,stop,step). |
| -f, --frb_identifier | str | FRB | Identifier for the simulation. |
| -o, --obs_params | str | utils default | Path to observation parameters file. |
| -g, --gauss_params | str | utils default | Path to Gaussian/micro-shot parameters file. |
| -d, --output-dir | str | simfrbs/ | Output directory. |
| --write | flag | False | Save simulated data to disk. |
| -p, --plot | str, nargs+ | lvpa | Plot modes: `all`, `None`, `iquv`, `lvpa`, `dpa`, `RM`, `pa_var`, `l_var`. Note: `pa_var` and `l_var` require multiple tau values. |
| -s, --save-plots | flag | False | Save plots to disk. |
| --show-plots | bool | True | Show plots interactively. |
| --figsize | float float | None | Figure size (inches): width height. |
| -e, --extension | str | pdf | File extension for saved plots. |
| --plot-scale | str | linear | Plot scale for `pa_var`/`l_var`: `linear`, `logx`, `logy`, `loglog`. |
| --fit | str, nargs+ | None | Fit for `pa_var`/`l_var`: `exp`, `power`, `log`, `linear`, `constant`, `broken-power`, or `power,N` / `poly,N`. |
| --phase-window | str | all | Phase window: `first`, `last`, `all`, `leading`, `trailing`, `total` (synonyms accepted). |
| --freq-window | str | full | Frequency window: `1q`, `2q`, `3q`, `4q`, `full` or `lowest-quarter`, `lower-mid-quarter`, `upper-mid-quarter`, `highest-quarter`, `full-band`. |
| -m, --mode | str | gauss | Pulse mode: `gauss`, `psn`. |
| --seed | int | None | RNG seed (psn). |
| --nseed | int | 1 | Number of realisations per tau (psn). |
| --tsys | float | 0.0 | System temperature in K (noise level). |
| --ncpu | int | 1 | CPUs for parallel processing. |
| --chi2-fit | flag | False | Chi-squared Gaussian fit on final profiles (non-`pa_var` runs). |
| --data | str | None | Use existing data file instead of generating. |

Notes
- Windows: long names map to short codes internally.
- Some plot modes operate on single FRBs, while `pa_var`/`l_var` aggregate across many tau values.

### Examples

- Single FRB, plot IQUV, add noise, show interactively
```bash
FIRES -t 0.5 --plot iquv --tsys 50
```

- Micro-shot mode with standard dynamic spectrum, pulse profile and PA profile
```bash
FIRES -m psn -t 0.05 --plot lvpa --save-plots
```

- Micro-shot mode with fixed seed and multiple realisations
```bash
FIRES --mode psn -t 0,10,1 --seed 42 --nseed 10 --plot pa_var l_var --phase-window leading --freq-window 4q --plot-scale loglog
```


## Acknowledgements

Based on work by Tehya Conroy and Apurba Bera.

## License

MIT