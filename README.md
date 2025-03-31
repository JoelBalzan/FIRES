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

| Flag                          | Type       | Default Value          | Description                                                                 |
|-------------------------------|------------|------------------------|-----------------------------------------------------------------------------|
| `-t`, `--scattering_timescale_ms` | `float`    | **Required**           | Scattering time scale in milliseconds.                                      |
| `-f`, `--frb_identifier`          | `str`      | `FRB`                  | Identifier for the simulated FRB.                                           |
| `-d`, `--output-dir`              | `str`      | `simfrbs/`             | Directory to save the simulated FRB data.                                   |
| `-o`, `--obs_params`              | `str`      | `utils/obsparams.txt`  | Path to the observation parameters file.                                    |
| `-g`, `--gauss_params`            | `str`      | `utils/gparams.txt`    | Path to the Gaussian parameters file.                                       |
| `--no-write`                      | `flag`     | `False`                | If set, the simulation will not be saved to disk and will return the data instead. |
| `--plot`                          | `str`      | `all`                  | Generate plots. Pass `all` for all plots or specify a plot name (`iquv`, `lvpa`, `dpa`, `rm`). |
| `--save-plots`                    | `bool`     | `False`                | Save plots to disk. Default is False.                                       |
| `--tz`                            | `float` x2 | `[0, 0]`               | Time zoom range for plots. Provide two values: start time and end time (ms).|
| `--fz`                            | `float` x2 | `[0, 0]`               | Frequency zoom range for plots. Provide two values: start frequency and end frequency (MHz). |
| `--rm`                            | `int`      | `-1`                   | Gaussian component to use for RM correction. Default is `-1` (last component). |

### Examples

#### Basic Simulation
Simulate an FRB with a scattering timescale of 0.5 ms and save the output to the default directory:
```sh
frb-sim -t 0.5
```
For more detailed instructions, see the [Wiki](https://github.com/JoelBalzan/FRB_SIM/wiki).

# Acknowledgements
This project is based on the work by Tehya and Apurba Bera.
