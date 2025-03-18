# FRB Simulation

This project simulates Fast Radio Bursts (FRBs) with scattering and polarization effects. The simulation generates dynamic spectra for Gaussian pulses, applies scattering, and saves the simulated FRB data to disk.

## Project Structure

- `genfrb.py`: Main script to generate and save simulated FRB data.
- `genfns.py`: Contains functions for generating dynamic spectra and applying scattering.
- `utils.py`: Utility functions used in the project.
- `gparams.txt`: File containing Gaussian parameters for the simulation.
- `obsparams.txt`: File containing observation parameters for the simulation.
- `processfrb.py`: Main script to generate plots.
- `plotfns.py`: Contains functions for plotting the FRBs.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/JoelBalzan/FRB_SIM.git
    cd FRB_SIM
    ```
## Usage

To run the simulation, use the following command:
```sh
python genfrb.py <scattering_time_scale> <frb_identifier>
```
e.g.
```sh
python genfrb.py 1.0 FRB_123
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
# Acknowledgements
This project is based on the work by Tehya and Apurba Bera.
