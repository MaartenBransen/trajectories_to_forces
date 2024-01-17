# Trajectories to forces

## Overview
This python package provides a set of functions for extracting pairwise interaction forces from particle trajectories under overdamped (Brownian) or inertial (molecular) dynamics. The codes for this package were developed as part of the work in reference [1], based on the concepts in reference [2].
Detailed descriptions of the theory behind this method and its implementation can be found in reference [1].

### References
[1] Bransen, M. (2024). Measuring interactions between colloidal (nano)particles. PhD thesis, Utrecht University.

[2] Jenkins, I. C., Crocker, J. C., & Sinno, T. (2015). [Interaction potentials from arbitrary multi-particle trajectory data](https://doi.org/10.1039/C5SM01233C). Soft Matter, 11(35), 6948–6956. 


## Installation
### PIP
This package can be installed directly from GitHub using pip:
```
pip install git+https://github.com/MaartenBransen/trajectories_to_forces
```
### Anaconda
When using the Anaconda distribution, it is safer to run the conda version of pip as follows:
```
conda install pip
conda install git
pip install git+https://github.com/MaartenBransen/trajectories_to_forces
```
### Updating
To update to the most recent version, use `pip install` with the `--upgrade` flag set:
```
pip install --upgrade git+https://github.com/MaartenBransen/trajectories_to_forces
```

## Usage
For a complete API reference see [the documentation](https://maartenbransen.github.io/trajectories_to_forces/).
There are two main functions, `run_overdamped` and `run_inertial` for overdamped (brownian) and fully inertial (molecular) dynamics respectively. The main input to either version is a list of pandas.DataFrame objects where each dataframe contains coordinates of all particles in a single timestep, and a list of timestamps corresponding to the dataframes. For interaction forces with cylindrical (anisotropic) symmetry, there is `run_overdamped_cylindrical`, which solves for the full anisotropic force vectors.
Additionally, there are some utility functions for easily saving and loading the analysis results to and from disk: `save_forceprofile` and `load_forceprofile`.