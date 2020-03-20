# Trajectories to forces

Based on [1]

## Info
* Created by Maarten Bransen
* email: m.bransen@uu.nl

## Installation
Installation with anaconda is recommended. Download the `trajectories_to_forces` folder and place it in your `site-packages` location of your Anaconda installation. If you are unsure where this is located you can find the path of any already installed package, e.g. using numpy:
```
import numpy
print(numpy.__file__)
```
which may print something like
```
'<current user>\\AppData\\Local\\Continuum\\anaconda3\\lib\\site-packages\\numpy\\__init__.py'
```

Alternatively, you can install the master branch directly from github to the site packages folder using anaconda prompt and the git package.

## Useage
there are two main functions, `run_overdamped` and `run_inertial` for overdamped (brownian) and fully inertial (molecular) dynamics respectively. 

## References

[1] I. C. Jenkins, J. C. Crocker and T. Sinno. (2015). Soft Matter 35(11), 6948-6956. https://doi.org/10.1039/C5SM01233C.
