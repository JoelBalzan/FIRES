# FRB_SIM

FRB_SIM is a Python package designed to simulate Fast Radio Bursts (FRBs) with scattering and polarization effects. The simulation generates dynamic spectra for Gaussian pulses, applies scattering, and saves the simulated FRB data to disk.

## Features
- Simulate FRBs with customizable scattering timescales.
- Save simulated data to disk or return it directly.
- Generate plots for visualizing FRB properties, including:
  - `iquv`: Stokes parameters.
  - `lvpa`: Linear polarization position angle.
  - `dpa`: Differential polarization angle.
  - `rm`: Rotation measure.
- Zoom into specific time or frequency ranges for detailed analysis.

## Project Structure

- `src/frb_sim/genfrb.py`: Main script to generate and save simulated FRB data.
- `src/frb_sim/main.py`: Entry point for the FRB simulation package.
- `src/frb_sim/plotfns.py`: Functions for plotting FRB data.
- `src/frb_sim/processfrb.py`: Script for analyzing and visualizing FRB data.
- `src/functions/`: Contains helper functions for generating and processing FRB data.
- `src/utils/`: Utility functions and constants used throughout the project.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/JoelBalzan/FRB_SIM.git
    cd FRB_SIM
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Install the package in editable mode:
    ```sh
    pip install -e .
    ```

## Usage
The `frb-sim` command-line tool provides several options to customize the simulation of Fast Radio Bursts (FRBs). Below is a detailed explanation of each option:

| **Flag**                  | **Type**   | **Default**       | **Description**                                                                                     |
|---------------------------|------------|-------------------|-----------------------------------------------------------------------------------------------------|
| `-t`, `--scattering_timescale_ms` | `float`   | **Required**    | Scattering time scale in milliseconds.                                                             |
| `-f`, `--frb_identifier`  | `str`      | `FRB`             | Identifier for the simulated FRB.                                                                  |
| `-d`, `--output-dir`       | `str`      | `simfrbs/`        | Directory to save the simulated FRB data.                                                          |
| `-o`, `--obs_params`       | `str`      | `obs_params_path` | Observation parameters for the simulated FRB.                                                      |
| `-g`, `--gauss_params`     | `str`      | `gauss_params_path` | Gaussian parameters for the simulated FRB.                                                         |
| `--write`                  | `flag`     | `False`           | If set, the simulation will be saved to disk. Default is False.                                      |
| `-p`, `--plot`             | `str`      | `lvpa`            | Generate plots. Options: `all`, `None`, `iquv`, `lvpa`, `dpa`, `rm`.                               |
| `-s`, `--save-plots`       | `flag`     | `False`           | Save plots to disk.                                                                                |
| `--tz`                    | `float`    | `[0, 0]`          | Time zoom range for plots. Provide two values: start time and end time (in milliseconds).          |
| `--fz`                    | `float`    | `[0, 0]`          | Frequency zoom range for plots. Provide two values: start frequency and end frequency (in MHz).    |
| `-m`, `--mode`             | `str`      | `gauss`           | Mode for generating pulses: `gauss` or `sgauss`.                                                   |
| `--n-gauss`               | `int`      | **Required**      | Number of sub-Gaussians to generate for each main Gaussian (required if `--mode` is `sgauss`).      |
| `--seed`                  | `int`      | `None`            | Set seed for repeatability in `sgauss` mode.                                                       |
| `--sg-width`              | `float`    | `[10, 50]`        | Minimum and maximum percentage of the main Gaussian width to generate micro-Gaussians.             |
| `--noise`                 | `float`    | `2`               | Noise scale in the dynamic spectrum. Multiplied by the standard deviation of each frequency channel.|
| `--scatter`               | `flag`     | `True`            | Enable scattering.                                                                                 |
| `--no-scatter`            | `flag`     | `False`           | Disable scattering. Overrides `--scatter` if both are provided.                                    |
| `--figsize`               | `float`    | `[6, 10]`         | Figure size for plots. Provide two values: width and height (in inches).                           |

### Examples

#### Basic Simulation
1. Basic simulation with scattering:
```sh
frb-sim -t 0.5 --mode gauss --noise 2
```
2. Simulation with 2 main gaussians comprised of (30 and 20) sub-gaussians with widths of ~10% and ~40% of their respective main gaussians:
```sh
frb-sim -t 0.5 --mode sgauss --n-gauss 30 20 --sg-width 10 40
```
For more detailed instructions, see the [Wiki](https://github.com/JoelBalzan/FRB_SIM/wiki).

# Acknowledgements
This project is based on the work by Tehya Conroy and Apurba Bera.
