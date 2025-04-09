# FIRES: The Fast, Intense Radio Emission Simulator

FIRES is a Python package designed to simulate Fast Radio Bursts (FRBs) with scattering and polarization effects. The simulation generates dynamic spectra for Gaussian pulses, applies scattering, and provides tools for visualization and analysis.

## Features

- **Customizable FRB Simulations**:
  - Simulate FRBs with adjustable scattering timescales.
  - Generate Gaussian or sub-Gaussian pulse distributions.
  - Add noise and apply scattering effects.

- **Data Output**:
  - Save simulated FRB data to disk in `.pkl` format.
  - Generate plots for visualizing FRB properties.

- **Plotting Options**:
  - Visualize Stokes parameters (`IQUV`), linear polarization position angle (`PA`), and other properties.
  - Generate plots for dynamic spectra, polarization angle RMS, and more.

- **Zoom and Customize**:
  - Zoom into specific time or frequency ranges.
  - Customize figure sizes and save plots to disk.

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
### From PyPi
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

The `FIRES` command-line tool provides several options to customize the simulation of Fast Radio Bursts (FRBs). Below is a summary of the available options:

### Command-Line Options

| **Flag**                  | **Type**   | **Default**       | **Description**                                                                                     |
|---------------------------|------------|-------------------|-----------------------------------------------------------------------------------------------------|
| `-t`, `--scattering_timescale_ms` | `float`   | `0.0`            | Scattering time scale(s) in milliseconds. Provide single values or ranges `(start,stop,step)`.      |
| `-f`, `--frb_identifier`  | `str`      | `FRB`             | Identifier for the simulated FRB.                                                                  |
| `-d`, `--output-dir`       | `str`      | `simfrbs/`        | Directory to save the simulated FRB data.                                                          |
| `-o`, `--obs_params`       | `str`      | `obs_params_path` | Path to observation parameters file.                                                               |
| `-g`, `--gauss_params`     | `str`      | `gauss_params_path` | Path to Gaussian parameters file.                                                                  |
| `--write`                  | `flag`     | `False`           | Save the simulation to disk.                                                                       |
| `-p`, `--plot`             | `str`      | `lvpa`            | Generate plots. Options: `all`, `None`, `iquv`, `lvpa`, `dpa`, `rm`, `pa_rms`.                     |
| `-s`, `--save-plots`       | `flag`     | `False`           | Save plots to disk.                                                                                |
| `--tz`                    | `float`    | `[0, 0]`          | Time zoom range for plots (start and end in milliseconds).                                         |
| `--fz`                    | `float`    | `[0, 0]`          | Frequency zoom range for plots (start and end in MHz).                                             |
| `-m`, `--mode`             | `str`      | `gauss`           | Mode for generating pulses: `gauss` or `sgauss`.                                                   |
| `--n-gauss`               | `int`      | **Required**      | Number of sub-Gaussians for each main Gaussian (required if `--mode` is `sgauss`).                 |
| `--seed`                  | `int`      | `None`            | Seed for repeatability in `sgauss` mode.                                                           |
| `--sg-width`              | `float`    | `[10, 50]`        | Min and max percentage of the main Gaussian width for sub-Gaussians.                               |
| `--noise`                 | `float`    | `0`               | Noise scale in the dynamic spectrum.                                                               |
| `--scatter`               | `flag`     | `True`            | Enable scattering.                                                                                 |
| `--no-scatter`            | `flag`     | `False`           | Disable scattering. Overrides `--scatter`.                                                        |
| `--figsize`               | `float`    | `[6, 10]`         | Figure size for plots (width and height in inches).                                                |

### Examples

#### Basic Simulation
1. Simulate an FRB with a scattering timescale of 0.5 ms:
    ```bash
    FIRES -t 0.5 --mode gauss --noise 2
    ```

2. Simulate an FRB with sub-Gaussians:
    ```bash
    FIRES -t 0.5 --mode sgauss --n-gauss 30 20 --sg-width 10 40
    ```

3. Generate plots for the simulated FRB:
    ```bash
    FIRES -t 0.5 --plot all --save-plots
    ```

For more detailed instructions, see the [Wiki](https://github.com/JoelBalzan/FIRES/wiki).

## Acknowledgements

This project is based on the work by Tehya Conroy and Apurba Bera.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.