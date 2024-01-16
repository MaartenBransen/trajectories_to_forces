"""
Description
-----------
Python package with a set of functions for extracting pairwise interaction
forces from particle trajectories under overdamped (Brownian) or inertial
(molecular) dynamics.

The codes for this package were developed as part of the work in reference [1],
based on the concepts in reference [2].

References
----------
[1] Bransen, M. (2024). Measuring interactions between colloidal 
(nano)particles. PhD thesis, Utrecht University.

[2] Jenkins, I. C., Crocker, J. C., & Sinno, T. (2015). Interaction potentials 
from arbitrary multi-particle trajectory data. Soft Matter, 11(35), 6948–6956. 
https://doi.org/10.1039/C5SM01233C


License
-------
MIT license

Copyright (c) 2024 Maarten Bransen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

__version__ = '1.0.0'

from .trajectories_to_forces import (
    run_overdamped,run_inertial,
    run_overdamped_cylindrical,
    save_forceprofile,
    load_forceprofile,
    save_forceprofile_cylindrical,
    load_forceprofile_cylindrical,
    convert_cylindrical_force_axes,
    filter_msd,
)

__all__ = [
    'run_overdamped',
    'run_overdamped_cylindrical',
    'run_inertial',
    'save_forceprofile',
    'load_forceprofile',
    'save_forceprofile_cylindrical',
    'load_forceprofile_cylindrical',
    'convert_cylindrical_force_axes',
    'filter_msd'
]

#add submodules to pdoc ignore list for generated documentation
__pdoc__ = {
    'trajectories_to_forces' : False,
}