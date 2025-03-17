# FRB Simulation

This project simulates Fast Radio Bursts (FRBs) with scattering and polarization effects. The simulation generates dynamic spectra for Gaussian pulses, applies scattering, and saves the simulated FRB data to disk.

## Project Structure

- `genfrb.py`: Main script to generate and save simulated FRB data.
- `genfns.py`: Contains functions for generating dynamic spectra and applying scattering.
- `utils.py`: Utility functions used in the project.
- `gparams.txt`: File containing Gaussian parameters for the simulation.
- `obsparams.txt`: File containing observation parameters for the simulation.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/FRB_SIM.git
    cd FRB_SIM
    ```

2. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To run the simulation, use the following command:
```sh
python [genfrb.py](http://_vscodecontentref_/0) <scattering_time_scale> <frb_identifier>
```
