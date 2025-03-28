# FRB_SIM

FRB_SIM is a Python package designed to simulate Fast Radio Bursts (FRBs) with scattering and polarization effects. The simulation generates dynamic spectra for Gaussian pulses, applies scattering, and saves the simulated FRB data to disk.

## Features

- Simulate FRBs with scattering and polarization effects.
- Generate dynamic spectra for Gaussian pulses.
- Save simulated FRB data in `.pkl` format.
- Analyze and visualize FRB data using provided tools.

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

| Option                     | Type    | Default Value               | Description                                                                 |
|----------------------------|---------|-----------------------------|-----------------------------------------------------------------------------|
| `-t`, `--scattering_timescale_ms` | `float` | **Required**              | Scattering time scale in milliseconds.                                      |
| `-f`, `--frb_identifier`   | `str`   | `"FRB"`                     | Identifier for the simulated FRB.                                           |
| `-d`, `--output-dir`       | `str`   | `"../../simfrbs/"`          | Directory to save the simulated FRB data.                                   |
| `-o`, `--obs_params`       | `str`   | `"../utils/obsparams.txt"`  | Path to the observation parameters file.                                    |
| `-g`, `--gauss_params`     | `str`   | `"../utils/gparams.txt"`    | Path to the Gaussian parameters file.                                       |
| `--no-write`               | `flag`  | `False`                     | If set, the simulation will not be saved to disk and will return the data instead. |

### Examples

#### Basic Simulation
Simulate an FRB with a scattering timescale of 0.5 ms and save the output to the default directory:
```sh
frb-sim -t 0.5
```

## CURRENTLY UNAVAILABLE
```
To Analyse the simulated FRB data, use the following command:
```sh
python processfrb.py <Name> <mode> <taums> <startms> <stopms> <startchan> <endchan> <rm0>
```
Supported Modes:
 - calcrm: Estimate Rotation Measure (RM).
 - iquv: Plot IQUV dynamic spectra.
 - lvpa: Plot L, V, and PA profiles.
 - dpa: Find PA change.

e.g.
```sh
python processfrb.py FRB_123 iquv 1.0 -1.5 2.5 0 0 0.0
```
For more detailed instructions, see the [Wiki](https://github.com/JoelBalzan/FRB_SIM/wiki).

# Acknowledgements
This project is based on the work by Tehya and Apurba Bera.
