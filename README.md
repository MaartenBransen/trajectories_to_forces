# Trajectories to forces

Based on [1]

## Info
* Created by Maarten Bransen
* email: m.bransen@uu.nl
* written for Python 3.7

## Installation
Downoad the master branch and make sure that the folder `trajectories_to_forces` is added to your python path, or that it is placed in your default package directory. Under Anaconda this is the `site-packages` folder.

## Useage
there are two main functions, `run_overdamped` and `run_inertial` for overdamped (brownian) and fully inertial (molecular) dynamics respectively. The main input to either version is a list of pandas.DataFrame objects where each dataframe contains coordinates of all particles in a single timestep, and a list of timestamps corresponding to the dataframes.

Additionally, there are two utility functions for easily saving and loading the analysis results to and from disk: `save_forceprofile` and `load_forceprofile'.

## References

[1] I. C. Jenkins, J. C. Crocker and T. Sinno. (2015). Soft Matter 35(11), 6948-6956. https://doi.org/10.1039/C5SM01233C.
