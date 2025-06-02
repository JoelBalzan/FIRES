# FIRES: The Fast, Intense Radio Emission Simulator

FIRES is a Python package designed to simulate Fast Radio Bursts (FRBs) with scattering and polarization effects. The simulation generates dynamic spectra for Gaussian pulses, applies scattering, and provides tools for visualization and analysis.

## Features

- **Customizable FRB Simulations**:
  - Simulate FRBs with adjustable scattering timescales.
  - Generate Gaussian or micro-shot pulse distributions.
  - Add noise and apply scattering effects.

- **Data Output**:
  - Save simulated FRB data to disk in `.pkl` format.
  - Generate plots for visualizing FRB properties.

- **Plotting Options**:
  - Visualize Stokes parameters (`IQUV`), linear polarization position angle (`PA`), and other properties.
  - Generate plots for dynamic spectra, polarization angle, and more.


## Project Structure

- **Core Scripts**:
  - `src/FIRES/main.py`: Entry point for the FRB simulation package.
  - `src/FIRES/functions/genfrb.py`: Main script for generating and saving simulated FRB data.
  - `src/FIRES/functions/processfrb.py`: Functions for analyzing and visualizing FRB data.
  - `src/FIRES/functions/plotfns.py`: Plotting functions for FRB data.

- **Utilities**:
  - `src/FIRES/utils/obsparams.txt`: Observation parameters for simulations.
  - `src/FIRES/utils/gparams.txt`: Gaussian parameters for pulse generation.

## Installation
### From PyPi: CURRENTLY UNAVAILABLE
```bash
pip install FIRES
```

### From GitHub
1. Clone the repository:
    ```bash
    git clone https://github.com/JoelBalzan/FIRES.git
    cd FIRES
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Install the package in editable mode:
    ```bash
    pip install -e .
    ```

## Usage

The `FIRES` command-line tool provides several options to customize the simulation of Fast Radio Bursts (FRBs):

### Command-Line Options

| **Flag**                  | **Type**   | **Default**       | **Description**                                                                                     |
|---------------------------|------------|-------------------|-----------------------------------------------------------------------------------------------------|
| `-t`, `--tau_ms`          | `float`    | `0.0`             | Scattering time scale(s) in milliseconds. Provide single values or ranges `(start,stop,step)`.      |
| `-f`, `--frb_identifier`  | `str`      | `FRB`             | Identifier for the simulated FRB.                                                                   |
| `-o`, `--obs_params`      | `str`      | obs_params_path   | Path to observation parameters file.                                                                |
| `-g`, `--gauss_params`    | `str`      | gauss_params_path | Path to Gaussian parameters file.                                                                   |
| `-d`, `--output-dir`      | `str`      | `simfrbs/`        | Directory to save the simulated FRB data.                                                           |
| `--write`                 | `flag`     | `False`           | Save the simulation to disk.                                                                        |
| `-p`, `--plot`            | `str`      | `lvpa`            | Generate plots. Options: `all`, `None`, `iquv`, `lvpa`, `dpa`, `RM`, `pa_var`, `lfrac`.             |
| `-s`, `--save-plots`      | `flag`     | `False`           | Save plots to disk.                                                                                 |
| `--show-plots`            | `bool`     | `True`            | Display plots. Set to `False` to disable plot display.                                              |
| `--figsize`               | `float`    | `[6, 10]`         | Figure size for plots (width and height in inches).                                                 |
| `--plot-scale`            | `str`      | `linear`          | Scale for plots: `linear`, `logx`, `logy`, `loglog`.                                                |
| `--fit`                   | `str`      | `None`            | Fit function for pa_var and lfrac plots. Options: `power`, `exp`, or `power N`.                     |
| `--phase-window`          | `str`      | `all`             | Window for plotting PA variance and L fraction: `first`, `last`, or `all`.                          |
| `--freq-window`           | `str`      | `all`             | Frequency window for plotting PA variance and L fraction: `1q`, `2q`, `3q`, `4q`, or `all`.         |
| `--tz`                    | `float`    | `[0, 0]`          | Time zoom range for plots (start and end in ms).                                                    |
| `--fz`                    | `float`    | `[0, 0]`          | Frequency zoom range for plots (start and end in MHz).                                              |
| `-m`, `--mode`            | `str`      | `gauss`           | Mode for generating pulses: `gauss` or `mgauss`.                                                    |
| `--n-gauss`               | `int`      | *Required*        | Number of micro-shots for each main Gaussian (if `--mode` is `mgauss`).                             |
| `--seed`                  | `int`      | `None`            | Seed for repeatability in `mgauss` mode.                                                            |
| `--nseed`                 | `int`      | `1`               | Number of realizations to generate at each scattering timescale for `mgauss` mode.                  |
| `--mg-width`              | `float`    | `[10, 50]`        | Min and max percentage of the main Gaussian width for micro-shots.                                  |
| `--snr`                   | `float`    | `0`               | Signal-to-noise ratio (SNR) for the simulated FRB. Set to `0` for no noise.                         |
| `--scatter`               | `flag`     | `True`            | Enable scattering.                                                                                  |
| `--no-scatter`            | `flag`     | `False`           | Disable scattering. Overrides `--scatter`.                                                          |
| `--ncpu`                  | `int`      | `1`               | Number of CPUs to use for parallel processing.                                                      |
| `--chi2-fit`              | `flag`     | `False`           | Enable chi-squared fitting on the final profiles.                                                   |
| `--data`                  | `str`      | `None`            | Path to data file. Use existing data instead of generating new.                                     |

### Examples

#### Basic Simulation
Simulate an FRB with a scattering timescale of 0.5 ms and SNR of 2:
```bash
FIRES -t 0.5 --plot iquv --mode gauss --snr 10
```

#### Simulate an FRB with micro-shots:
```bash
FIRES -t 0.5 --plot lvpa --mode mgauss --n-gauss 30 20 --mg-width 10 40
```

#### Generate and save all plots for the simulated FRB:
```bash
FIRES -t 0.5 --plot all --save-plots
```

For more detailed instructions, see the [Wiki](https://github.com/JoelBalzan/FIRES/wiki).

## Acknowledgements

This project is based on the work by Tehya Conroy and Apurba Bera.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.