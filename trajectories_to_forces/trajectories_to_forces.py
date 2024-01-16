"""
-------------------------------------------------------------------------------
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
-------------------------------------------------------------------------------
"""

#%% imports
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from itertools import tee,islice,repeat
import numba as nb
from warnings import warn

#%% private definitions

def _check_inputs(coordinates,times,boundary,pos_cols,eval_particles):
    """checks the format of the input, i.e. if coordinates/times match and give 
    a single timeseries (nested=False) or multiple sequences of consecutive 
    steps (nested=True), if eval_particles are used and if the boundary 
    conditions match the data."""
    #check if one list or nested list, if not make it nested
    if isinstance(times[0],(list,np.ndarray)):
        nested = True
        nsteps = len(times)
        
        #perform checks to see if shapes/lengths of inputs match
        if not isinstance(coordinates[0],(list,np.ndarray)):
            raise ValueError('`coordinates` must be nested list if `times` is')
        if len(coordinates) != nsteps:
            raise ValueError('number of subsets in `times` and `coordinates` '
                             'must match')
        if any([len(coord)!=len(t) for coord,t in zip(coordinates,times)]):
            raise ValueError('length of each subset in `times` and '
                             '`coordinates` must match')
        if any([len(t) <= 1 for t in times]):
            raise ValueError('each subset in `times` and `coordinates` must '
                             'have at least 2 elements')
        if not eval_particles is None:
            if any(isinstance(ep,(list,set,np.ndarray)) \
                   for ep in eval_particles):
                if len(eval_particles) != nsteps:
                    raise ValueError('length of `times` and `eval_particles'
                                     ' must match')
            elif len(eval_particles)!=nsteps or \
                any(not ep is None for ep in eval_particles):
                eval_particles = repeat(eval_particles)
        else:
            eval_particles = repeat(None)
        
    else:
        nested = False
        if len(times) <= 1:
            raise ValueError('must be at least 2 time steps in a series')
        times = [times]
        coordinates = [coordinates]
        eval_particles = [eval_particles]
    
    #get dimensionality from pos_cols, check names
    pos_cols = list(pos_cols)
    ndims = len(pos_cols)
    
    #get default boundaries from min and max values in any coordinate set
    if boundary is None:
        boundary = [
            [
                repeat([
                    min([c[dim].min() for c in coords]),
                    max([c[dim].max() for c in coords])
                ]) for dim in pos_cols
            ] for coords in coordinates
        ]
        
    #otherwise check inputs
    elif nested:
        #if single boundary for whole set given, make boundary iterable on a 
        #per timestep per coordinate set basis
        if np.array(boundary[0]).ndim == 1:
            if len(boundary) != ndims:
                raise ValueError('number of `pos_cols` does not match '
                                 '`boundary`')
            boundary = repeat(repeat(boundary))
        
        #else check if length matches
        elif len(boundary) != nsteps:
            raise ValueError('length of `boundary` and `coordinates` must '
                             'match')
        
        #if single boundary per coordinate set given
        elif np.array(boundary[0]).ndim == 2:
            #check dimensionality of each item
            if any([len(bounds) != ndims for bounds in boundary]):
                raise ValueError('number of `pos_cols` does not match '
                                 '`boundary`')
            #make iterable on a per timestep basis
            boundary = [repeat(bounds) for bounds in boundary]
            
        #if boundary for each timestep check lengths and dimensionality
        elif any([len(bounds) != len(time) \
                  for bounds,time in zip(boundary,times)]):
            raise ValueError('number of items in each item in `boundary` must '
                             'match that in `times`')
        elif any([np.array(bounds).shape[1] != ndims for bounds in boundary]):
            raise ValueError('number of `pos_cols` does not match `boundary`')
    
    #if not nested, make nested with single item per list for later iterating
    else:
        if np.array(boundary[0]).ndim == 2:
            if any([len(bounds) != ndims for bounds in boundary]):
                raise ValueError('number of `pos_cols` does not match '
                                 '`boundary`')
            boundary = [boundary]
        elif len(boundary) != ndims:
            raise ValueError('number of `pos_cols` does not match `boundary`')
        else:
            boundary = [repeat(boundary)]
    
    return coordinates,times,boundary,eval_particles,nested

def _nwise(iterable, n=2):
    """
    Loops over sets of n subsequent items of an iterable. For example n=2:
        
        s -> (s0,s1), (s1,s2), (s2, s3), ..., (sn-1,sn)
    
    or n=3:
    
        s -> (s0,s1,s2), (s1,s2,s3), ... ,(sn-2,sn-1,sn)

    Parameters
    ----------
    iterable : iterable
        list of values
    n : int, optional
        size of the tuples returned. The default is 2.

    Returns
    -------
    iterable of tuples
        iterable of tuples in which each tuple contains the set of `n`
        subsequent values from the input

    """                                    
    iters = tee(iterable, n)                                                     
    for i, it in enumerate(iters):                                               
        next(islice(it, i, i), None)                                               
    return zip(*iters)

def _calculate_displacement_periodic(x0,x1,xmin,xmax):
    """calulate displacement under periodic boundary conditions in 1D"""
    if x1-x0 > (xmax-xmin)/2:
        return x1-x0-(xmax-xmin)
    elif x1-x0 <= -(xmax-xmin)/2:
        return x1-x0+(xmax-xmin)
    else:
        return x1-x0

def _calculate_accelerations_periodic(x0,x1,x2,xmin,xmax):
    """calulate acceleration under periodic boundary conditions in 1D"""
    modifier = 0
    if x1-x0 > (xmax-xmin)/2:
        modifier += (xmax-xmin)
    elif x1-x0 < -(xmax-xmin)/2:
        modifier -= (xmax-xmin)
    if x2-x1 > (xmax-xmin)/2:
        modifier -= (xmax-xmin)
    elif x2-x1 < -(xmax-xmin)/2:
        modifier += (xmax-xmin)
    return x0 - 2*x1 + x2 + modifier

@nb.njit()
def _calculate_force_overdamped_simple(coords0,coords1,boundary,
                                       periodic_boundary,dt,gamma):
    """optimized force calculation with constant particles"""
    forces = coords1-coords0
    dims = len(boundary)
    if periodic_boundary:
        bb = boundary[:,1]-boundary[:,0]
        for i in range(len(forces)):
            for d in range(dims):
                if forces[i,d]>bb[d]/2:
                    forces[i,d] -= bb[d]
                elif forces[i,d]<=-bb[d]/2:
                    forces[i,d] += bb[d]
    return forces*gamma/dt

def _calculate_forces_overdamped(coords0,coords1,dt,boundary,gamma=1,
                                 periodic_boundary=False):
    """
    calculate forces acting on each particle for an overdamped system (particle
    velocity times damping parameter) between two points in time. Removes
    particles which occur only in one of the two sets of coordinates from the
    result.

    Parameters
    ----------
    coords0,coords1 : pandas.DataFrame
        pandas DataFrames with coordinates of first and second timestep of a
        set of two snapshots of the particles. Columns with position data must 
        have names/headers matching `pos_cols` and the DataFrames must be 
        indexed by particle ID where the same index corresponds to the same 
        particle in either DataFrame.
    dt : float
        time difference between two snapshots, i.e. the delta time
    boundary : list or tuple
        boundaries of the box in which the coordinates are defined in the form
        ((d0_min,d0_max),(d1_min,d1_max),...) with a length (number) and order 
        of dimensions matching `pos_cols`.
    pos_cols : list of str
        names of the columns in `coords0` and `coords1` containing the particle 
        coordinates along each dimension. The length (i.e. number of 
        dimensions) must match len(boundary) and the number of columns in 
        `coords0` and `coords1.
    gamma : float, optional
        damping coefficient kT/D where D is the diffusion coefficient. The
        default is 1.
    periodic_boundary : bool, optional
        Whether the coordinates are in a periodic box. The default is False.
        
    Returns
    -------
    pandas.Dataframe
        pandas Dataframe indexed by particle number where each column contains
        the net force experienced by the particles along those cartesian axes.

    """
    if periodic_boundary:
        cols = coords0.columns
        forces = pd.DataFrame()
         
        #loop over dimensions
        for col in cols:
            
            #get boundaries for the dimension
            colmin,colmax = boundary[list(cols).index(col)]
            
            #create dataframe with extra columns to call 'apply' on
            forces[col+'0'] = coords0[col]
            forces[col+'1'] = coords1[col]
            
            forces[col] = forces.apply(
                lambda x: _calculate_displacement_periodic(
                    x[col+'0'],
                    x[col+'1'],
                    colmin,
                    colmax
                ),
                axis=1
            )
        
        return forces[cols].dropna()*gamma/dt
    else:
        return (coords1 - coords0).dropna()*gamma/dt

def _calculate_forces_inertial(coords0,coords1,coords2,dt,boundary,pos_cols,
                               mass=1,periodic_boundary=False):
    """
    calculate forces acting on each particle for an inertial system (molecular
    dynamics-like) between two points in time. Removes particles which occur
    only in one of the three sets of coordinates from the result.

    Parameters
    ----------
    coords0,coords1,coords2 : pandas.DataFrame
        pandas DataFrames with coordinates of first, second and third timestep
        of a set of three snapshots of the particles. Columns with position
        data must have names/headers matching `pos_cols` and the DataFrames 
        must be indexed by particle ID where the same index corresponds to the 
        same particle in either DataFrame.
    dt : float
        time difference between two snapshots, i.e. the delta time
    boundary : list or tuple
        boundaries of the box in which the coordinates are defined in the form
        ((d0_min,d0_max),(d1_min,d1_max),...) with a length (number) and order 
        of dimensions matching `pos_cols`.
    pos_cols : list of str
        names of the columns in `coords0`, `coords1` and `coords2` containing 
        the particle coordinates along each dimension. The length (i.e. number 
        of dimensions) must match len(boundary) and the number of columns in 
        the coordinate DataFrames.
    gamma : float, optional
        damping coefficient kT/D where D is the diffusion coefficient. The
        default is 1.
    periodic_boundary : bool, optional
        Whether the coordinates are in a periodic box. The default is False.
        
    Returns
    -------
    pandas.Dataframe
        pandas Dataframe indexed by particle number where each column contains
        the net force experienced by the particles along those cartesian axes.

    """
    if periodic_boundary:
        
        forces = pd.DataFrame()
        
        #loop over dimensions (DataFrame columns)
        cols = coords0.columns
        for col in cols:
            
            #get dimension boundaries
            colmin,colmax = boundary[list(pos_cols).index(col)]
            
            #create dataframe with extra columns to call 'apply' on
            forces[col+'0'] = coords0[col]
            forces[col+'1'] = coords1[col]
            forces[col+'2'] = coords2[col]
            
            forces[col] = forces.apply(lambda dim: _calculate_accelerations_periodic(
                    dim[col+'0'],
                    dim[col+'1'],
                    dim[col+'2'],
                    colmin,
                    colmax
                    ),axis=1)
        
        return forces[cols].dropna()*mass/dt**2
    else:
        return (coords0 - 2*coords1 + coords2).dropna()*mass/dt**2


@nb.njit(parallel=False)
def _coefficient_loop(particles,queryparticles,ndims,dist,indices,mask,
                              rmax,M):
    """loop over all pairs found by KDTree.query and calculate coefficients"""
    #allocate memory for coefficient matrix
    coefficients = np.zeros((ndims*len(queryparticles),M))
    counter = np.zeros(M)
    binmeanpos = np.zeros(M)
    imax,jmax = dist.shape
    
    #loop over pairs in distance/indices array
    for i in nb.prange(imax):
        for j in range(jmax):
            if not mask[i,j]:
                d = dist[i,j]
                counter[int(d/rmax*M)] += 1
                binmeanpos[int(d/rmax*M)] += d
                for dim in range(ndims):
                    coefficients[ndims*i+dim,int(d/rmax*M)] += \
                        (queryparticles[i,dim]-particles[indices[i,j],dim])/d
    
    return coefficients,counter,binmeanpos

@nb.njit(parallel=False)
def _coefficient_loop_linear(particles,queryparticles,ndims,dist,
                                     indices,mask,rmax,M):
    """loop over all pairs found by KDTree.query and calculate coefficients
    using linear basis functions"""
    #allocate memory for coefficient matrix
    coefficients = np.zeros((ndims*len(queryparticles),M+1))
    counter = np.zeros(M+1)
    binmeanpos = np.zeros(M+1)
    imax,jmax = dist.shape
    
    #loop over pairs in distance/indices array
    for i in nb.prange(imax):
        for j in range(jmax):
            if not mask[i,j]:
                d = dist[i,j]
                lb = int(d/rmax*M) #floor division gives nearest bin on left
                phi = np.array([1-d/rmax*M+lb, 1+d/rmax*M-(lb+1)])
                counter[lb] += phi[0]
                counter[lb+1] += phi[1]
                binmeanpos[lb] += d*phi[0]
                binmeanpos[lb+1] += d*phi[1]
                for dim in range(ndims):
                    coefficients[ndims*i+dim,lb] += phi[0]*\
                        (queryparticles[i,dim]-particles[indices[i,j],dim])/d
                    coefficients[ndims*i+dim,lb+1] += phi[1]*\
                        (queryparticles[i,dim]-particles[indices[i,j],dim])/d
    
    return coefficients[:,:-1],counter[:-1],binmeanpos[:-1]

@nb.njit(parallel=False)
def _bruteforce_pair_loop(particles,queryparticles,ndims,rmax,M):
    """loop over all pairs with i from queryparticles and j from particles, and
    calculate coefficients"""
    #allocate memory for coefficient matrix
    coefficients = np.zeros((ndims*len(queryparticles),M))
    counter = np.zeros(M)
    binmeanpos = np.zeros(M)
    
    #loop over pairs in distance/indices array
    for i in nb.prange(len(queryparticles)):
        for j in range(len(particles)):
            d = np.sum((queryparticles[i]-particles[j])**2)**0.5
            if d < rmax and d != 0:
                counter[int(d/rmax*M)] += 1
                binmeanpos[int(d/rmax*M)] += d
                for dim in range(ndims):
                    coefficients[ndims*i+dim,int(d/rmax*M)] += \
                        (queryparticles[i,dim]-particles[j,dim])/d

    return coefficients,counter,binmeanpos

@nb.njit()
def _distance_periodic_wrap(ci,cj,boxmin,boxmax):
    """calculates element-wise distance between two n-dimensional coordinates 
    while wrapping around boundaries of periodic box with bounds boxmin, boxmax
    along each dimension"""
    distances = np.empty(len(ci))
    #note the somewhat weird convention here. Since we define an attractive 
    #force to be negative and repulsive force positive, we want the distance
    #from j to i, not i to j
    for dim,(i,j,mi,ma) in enumerate(zip(ci,cj,boxmin,boxmax)):
        if i-j > (ma-mi)/2:
            distances[dim] = i-j-ma+mi
        elif i-j <= (mi-ma)/2:
            distances[dim] = i-j+ma-mi
        else:
            distances[dim] = i-j
    return distances


@nb.njit(parallel=False)
def _coefficient_loop_periodic(particles,queryparticles,ndims,dist,
                                       indices,mask,rmax,M,boxmin,boxmax):
    """loop over all pairs found by KDTree.query and calculate coefficients in 
    periodic boundary conditions"""
    #allocate memory for coefficient matrix
    coefficients = np.zeros((ndims*len(queryparticles),M))
    counter = np.zeros(M)
    binmeanpos = np.zeros(M)
    imax,jmax = dist.shape
    
    #loop over pairs in distance/indices array
    for i in nb.prange(imax):
        for j in range(jmax):
            if not mask[i,j]:
                d_zyx = _distance_periodic_wrap(
                    queryparticles[i],particles[indices[i,j]],boxmin,boxmax
                )
                d = dist[i,j]
                counter[int(d/rmax*M)] += 1
                binmeanpos[int(d/rmax*M)] += d
                for dim in range(ndims):
                    coefficients[ndims*i+dim,int(d/rmax*M)] += d_zyx[dim]/d

    return coefficients,counter,binmeanpos

@nb.njit(parallel=False)
def _coefficient_loop_periodic_linear(particles,queryparticles,ndims,
        dist,indices,mask,rmax,M,boxmin,boxmax):
    """loop over all pairs found by KDTree.query and calculate coefficients
    in periodic boundary conditions using linear basis functions"""
    #allocate memory for coefficient matrix
    coefficients = np.zeros((ndims*len(queryparticles),M+1))
    counter = np.zeros(M+1)
    binmeanpos = np.zeros(M+1)
    imax,jmax = dist.shape
    
    #loop over pairs in distance/indices array
    for i in nb.prange(imax):
        for j in range(jmax):
            if not mask[i,j]:
                d_zyx = _distance_periodic_wrap(
                    queryparticles[i],particles[indices[i,j]],boxmin,boxmax
                )
                d = dist[i,j]
                lb = int(d/rmax*M) #floor division gives nearest bin on left
                phi = [1-d/rmax*M+lb, d/rmax*M-lb]
                counter[lb] += phi[0]
                counter[lb+1] += phi[1]
                binmeanpos[lb] += d*phi[0]
                binmeanpos[lb+1] += d*phi[1]
                for dim in range(ndims):
                    coefficients[ndims*i+dim,lb] += phi[0]*d_zyx[dim]/d
                    coefficients[ndims*i+dim,lb+1] += phi[1]*d_zyx[dim]/d
    
    return coefficients[:,:-1],counter[:-1],binmeanpos[:-1]

@nb.njit(parallel=False)
def _bruteforce_pair_loop_periodic(particles,queryparticles,ndims,rmax,M,
                                      boxmin,boxmax):
    """loop over all pairs with i from queryparticles and j from particles, and
    calculate coefficients in periodic boundary conditions"""
    #allocate memory for coefficient matrix
    coefficients = np.zeros((ndims*len(queryparticles),M))
    counter = np.zeros(M)
    binmeanpos = np.zeros(M)
    
    #loop over pairs in distance/indices array
    for i in nb.prange(len(queryparticles)):
        for j in range(len(particles)):
            d_zyx = _distance_periodic_wrap(
                queryparticles[i],particles[j],boxmin,boxmax
            )
            d = np.sum(d_zyx**2)**0.5
            if d < rmax and d != 0:
                counter[int(d/rmax*M)] += 1
                binmeanpos[int(d/rmax*M)] += d
                for dim in range(ndims):
                    coefficients[ndims*i+dim,int(d/rmax*M)] += d_zyx[dim]/d

    return coefficients,counter,binmeanpos

def _calculate_coefficients(coords,query_indices,rmax,M,boundary,pos_cols,
        periodic_boundary=False,bruteforce=False,basis_function='constant',
        neighbour_upper_bound=None):
    """
    Uses a brute force method for finding all neighbours of particles within 
    a cutoff radius and then calculates the coefficient matrix C* from the
    paper [1]. Can be combined with periodic boundary conditions

    Parameters
    ----------
    coordinates : list of pandas.DataFrame
        A pandas dataframe containing coordinates for each timestep. Must be
        indexed by particle (with each particle having a unique identifyer that
        matches between different time steps) and contain coordinates along 
        each dimension in a separate column, with column names matching those 
        given in `pos_cols`.
    coords : pandas.DataFrame
        all particle coordinates in the system to consider in evaluating the
        coefficients. Must have columns headers matching `pos_cols` and indices
        corresponding to those in query_indices
    query_indices : list
        indices corresponding to the particles to calculate the coefficients 
        for, while accounting for all particles in the coordinate set (also
        those not in `query_indices`).
    rmax : float
        cut-off radius up to which to calculate the coefficients.
    M : int
        number of discretization steps to bin the matrix into. The bin with
        will be rmax/M.
    boundary : list or tuple
        boundaries of the box in which the coordinates are defined in the form
        ((d0_min,d0_max),(d1_min,d1_max),...) with a length (number) and order 
        of dimensions matching `pos_cols`. The default is `None`, which uses 
        the min and max value found in the entire set of coordinates along each
        axis.
    pos_cols : list of str
        names of the columns in `coords` containing the particle coordinates 
        along each dimension. The length (i.e. number of dimensions) must match
        len(boundary) and the number of columns in `coords`.
    periodic_boundary : bool, optional
        whether the box has periodic boundary conditions. The default is False.
    bruteforce : bool, optional
        if True and when using nonperiodic boundary conditions, this switches
        to a brute-force approach where every possible particle pair is
        evaluated instead of (the default) efficient KDtree approach for pair
        finding. When periodic_boundary=True, bruteforcing is always used. The
        default is False.
    basis_function : ['constant','linear'], optional
        which type of basis function to use for the discritization of the 
        pairwise force

    Returns
    -------
    coefficients : array
        an array containing the 3n by M matric of coefficients, where n is
        the number of particles
    counter : list of length M
        total number of pair counts for each column in the matrix

    """
    #get dimensionality
    ndims = len(pos_cols)
    
    #convert to numpy array with axes (particle,dim) and dim=[x,y,z]
    coords.sort_index(inplace=True)
    particles = coords[pos_cols].to_numpy()
    queryparticles = coords.loc[sorted(query_indices)][pos_cols].to_numpy()
    
    #set maximum number of neighbours 1 particle may have within rmax
    if neighbour_upper_bound is None:
        neighbour_upper_bound = len(particles)
    else:
        neighbour_upper_bound = min([neighbour_upper_bound,len(particles)])
    
    #coefficient calculation in periodic boundary conditions
    if periodic_boundary:
        
        boundary = np.array(boundary)
        boxmin = boundary[:,0]
        boxmax = boundary[:,1]
        
        #optionally use (inefficient) brute-force search through all pairs
        if bruteforce:
            if basis_function == 'linear':
                raise NotImplementedError(
                    'bruteforcing with linear basis functions is not '
                    'implemented'
                )
            coefficients,counter,binmeanpos = _bruteforce_pair_loop_periodic(
                particles,queryparticles,ndims,rmax,M,boxmin,boxmax)
        
        #else use KDTree based efficient neighbour searching algorithm
        else:
            #correct box and coordinates to have lower lim at 0 for cKDTree
            particles -= boxmin
            queryparticles -= boxmin
            boxmax -= boxmin
            boxmin -= boxmin
            
            #initialize and query periodic KDTree for pairs within rmax
            tree = cKDTree(particles,boxsize=boxmax)
            dist,indices = tree.query(
                queryparticles,
                k=neighbour_upper_bound,
                distance_upper_bound=rmax,
            )
            
            #remove pairs with self and np.inf fill values
            dist, indices = dist[:,1:],indices[:,1:]
            mask = np.isinf(dist)
            
            #perform numba-optimized loop over particle pairs
            if basis_function == 'constant':
                coefficients,counter,binmeanpos = \
                    _coefficient_loop_periodic(
                        particles,queryparticles,ndims,dist,indices,mask,rmax,
                        M,boxmin,boxmax
                    )
            elif basis_function == 'linear':
                coefficients,counter,binmeanpos = \
                    _coefficient_loop_periodic_linear(
                        particles,queryparticles,ndims,dist,indices,mask,rmax,
                        M,boxmin,boxmax
                    )
    
    #no periodic boundary conditions
    else:
        #optionally use (inefficient) brute-force search through all pairs
        if bruteforce:
            coefficients,counter,binmeanpos = _bruteforce_pair_loop(
                particles,queryparticles,rmax,M
            )
        
        #use KDTree based efficient neighbour searching algorithm
        else:
            #initialize and query KDTree for fast pairfinding
            tree = cKDTree(particles)
            dist,indices = tree.query(
                queryparticles,
                k=neighbour_upper_bound,
                distance_upper_bound=rmax,
            )
            
            #remove pairs with self and np.inf fill values
            dist, indices = dist[:,1:], indices[:,1:]
            mask = np.isinf(dist)

            #perform numba-optimized loop over particle pairs
            if basis_function == 'constant':
                coefficients,counter,binmeanpos = \
                    _coefficient_loop(
                        particles,queryparticles,ndims,dist,indices,mask,rmax,M
                    )
            elif basis_function == 'linear':
                coefficients,counter,binmeanpos = \
                    _coefficient_loop_linear(
                        particles,queryparticles,ndims,dist,indices,mask,rmax,M
                    )

    return coefficients,counter,binmeanpos




#%% public definitions
def save_forceprofile(
        filename, rmax, M, basis_function, gamma, periodic_boundary, binmean_r,
        G, G_err, counts
    ):
    """Saves the results from trajectory analysis to a text file

    Parameters
    ----------
    filename : string
        filename to use for results file
    binmean_r : list of float
        mean value of all pairs in each bin, i.e. a 'count weighted' bin center
    forces : list of float
        list of force values as obtained from the trajectory analysis
    counts : list of int
        the number of evaluations used for each bin
    M : int
        number of bins
    rmax : float
        cut-off radius for force
    gamma : float
        the friction coefficient used in the calculation
    """
    header =  f'rmax {rmax}\n'
    header += f'M {M}\n'
    header += f'basis_function {basis_function}\n'
    header += f'gamma {gamma}\n'
    header += f'periodic {periodic_boundary}\n'
    
    m = range(M)

    np.savetxt(
        filename,
        np.array([m,G,G_err,binmean_r,counts]).T,
        header=header+'\nm G G_err binmean_r counts',
        fmt = ['%03d','% .7e','% .7e','% .7e','% .7e']
    )
    
def load_forceprofile(filename):
    """load results file as stored by `save_forceprofile`

    Parameters
    ----------
    filename : str
        filename with extension to store data in

    Returns
    -------
    m : np.ndarray of int
        bin indices
    r_cent : np.ndarray of float
        center r position of each bin.
    G : np.ndarray of float
        force coefficients.
    G_err : np.ndarray of float
        residual error in G.
    binmean_r : np.ndarray of float
        count weighted mean distance for all pairs in each bin.
    counts : np.ndarray of int or float
        number of pair counts in each bin weighted for the basis functions.
    M : int
        number of discretisation steps
    rmax : float
        cut-off distance for the force.
    basis_function : str
        type of basis functions used.
    gamma : float
        friction factor used for the calculations.
    periodic_boundary : bool
        whether periodic boundary conditions were used.
    """
    #read header
    with open(filename,'r') as f:
        rmax = float(f.readline().split()[-1])
        M = int(f.readline().split()[-1])
        bf = f.readline().split()[-1]
        gamma = float(f.readline().split()[-1])
        pb = bool(f.readline().split()[-1])
    
    d_r = rmax/M
    
    #read data
    m, G, G_err, r_mean, counts = np.loadtxt(filename,unpack=True)
    
    m = m.astype(int)

    #bin cent values
    r_cent = np.arange(0,rmax,d_r)+0.5*d_r,

    return m, r_cent, G, G_err, r_mean, counts, M, rmax, bf, gamma, pb

def save_forceprofile_legacy(
        filename,
        binmean_r,
        forces,
        counts,
        M,
        rmax,
        gamma
    ):
    """
    !DEPRECATED!
    
    Saves the results to a text file in the __version__ <= 0.2.0 format

    Parameters
    ----------
    filename : string
        filename to use for results file
    binmean_r : list of float
        mean value of all pairs in each bin, i.e. a 'count weighted' bin center
    forces : list of float
        list of force values as obtained from the trajectory analysis
    counts : list of int
        the number of evaluations used for each bin
    M : int
        number of bins
    rmax : float
        cut-off radius for force
    gamma : float
        the friction coefficient used in the calculation
    """
    with open(filename,'w+') as file:
        #write input parameters
        file.write("gamma:\t{:.5f}\n".format(gamma))
        file.write("M:\t{}\n".format(M))
        file.write("rmax:\t{}\n".format(rmax))
        file.write('\n')

        #write table headers
        file.write('r\tforce\tcounts\n')

        #write table
        for r,f,c in zip(binmean_r,forces,counts):
            file.write(f'{r:.3f}\t{f: 5f}\t{int(c):d}\n')

    print('saved results as "'+filename+'"')

def load_forceprofile_legacy(filename):
    """
    loads the results from a text file in the __version__ <= 0.2.0 format

    Parameters
    ----------
    filename : string
        name of the file to load.

    Returns
    -------
    binmean_r : list of float
        count-weighted mean r value of all pairs contributing to the bin
    forces : list
        the mean force in each bin
    counts : list
        the number of particle pairs counted for each bin
    M : int
        number of discretization steps
    rmax : float
        cut-off radius for force
    gamma : float
        the friction coefficient used in the calculation

    """
    with open(filename,'r') as file:
        filedata = [line[:-1] for line in file.readlines()]

    #load input parameters
    s = 0
    if 'gamma' in filedata[0]:
        gamma = float(filedata[0].split()[1])
        s=1
    else:
        gamma = None
    M = int(filedata[0+s].split()[1])
    rmax = float(filedata[1+s].split()[1])

    #load data table
    binmean_r = []
    forces = []
    counts = []

    for line in filedata[4+s:]:
        line = line.split()
        binmean_r.append(float(line[0]))
        forces.append(float(line[1]))
        counts.append(int(line[2]))

    return binmean_r,forces,counts,M,rmax,gamma

def save_forceprofile_cylindrical(
        filename,rmax,M_z,M_rho,basis_function,gamma,periodic_boundary,
        m_z,m_rho,G_z,G_rho,G_z_err,G_rho_err,binmean_z,binmean_rho,counts
    ):
    """
    save results from cylindrical trajectory analysis to a file

    Parameters
    ----------
    filename : str
        filename with extension to store data in.
    rmax : float
        cut-off distance for the force.
    M_z : int
        number of discretisation steps along the z axis
    M_rho : int
        number of discretisation steps along the rho axis.
    basis_function : str
        type of basis functions used.
    gamma : float
        friction factor used for the calculations.
    periodic_boundary : bool
        whether periodic boundary conditions were used.
    m_z : np.ndarray of int
        z bin indices
    m_rho : np.ndarray of int
        rho bin indices.
    G_z : np.ndarray of float
        force coefficients along z.
    G_rho : np.ndarray of float
        force coefficients along rho.
    G_z_err : np.ndarray of float
        residual error in G_z.
    G_rho_err : np.ndarray of float
        residual error in G_rho.
    binmean_z : np.ndarray of float
        count weighted mean z distance for all pairs in each bin.
    binmean_rho : np.ndarray of float
        count weighted mean rho distance for all pairs in each bin.
    counts : np.ndarray of int or float
        number of pair counts in each bin weighted for the basis functions.

    """
    header =  f'rmax {rmax}\n'
    header += f'M_z {M_z}\n'
    header += f'M_rho {M_rho}\n'
    header += f'basis_function {basis_function}\n'
    header += f'gamma {gamma}'
    header += f'periodic {periodic_boundary}\n'
    
    data = [a.flatten() for a in \
            [m_z,m_rho,G_z,G_rho,G_z_err,G_rho_err,binmean_z,binmean_rho,counts]]
    
    np.savetxt(
        filename,
        np.array(data).T,
        header=header+'\nm_z m_rho G_z G_rho G_z_err G_rho_err binmean_z binmean_rho counts',
        fmt = ['%03d','%03d','% .7e','% .7e','% .7e','% .7e','% .7e','% .7e','%.7e']
    )

def load_forceprofile_cylindrical(filename):
    """
    load results file as stored by `save_forceprofile_cylindrical`

    Parameters
    ----------
    filename : str
        filename with extension to store data in

    Returns
    -------
    m_z : np.ndarray of int
        z bin indices
    m_rho : np.ndarray of int
        rho bin indices.
    z_cent : np.ndarray of float
        z position of center of each bin.
    rho_cent : np.ndarray of float
        rho position of center of each bin.
    G_z : np.ndarray of float
        force coefficients along z.
    G_rho : np.ndarray of float
        force coefficients along rho.
    G_z_err : np.ndarray of float
        residual error in G_z.
    G_rho_err : np.ndarray of float
        residual error in G_rho.
    binmean_z : np.ndarray of float
        count weighted mean z distance for all pairs in each bin.
    binmean_rho : np.ndarray of float
        count weighted mean rho distance for all pairs in each bin.
    counts : np.ndarray of int or float
        number of pair counts in each bin weighted for the basis functions.
    M_z : int
        number of discretisation steps along the z axis
    M_rho : int
        number of discretisation steps along the rho axis.
    rmax : float
        cut-off distance for the force.
    basis_function : str
        type of basis functions used.
    gamma : float
        friction factor used for the calculations.
    periodic_boundary : bool
        whether periodic boundary conditions were used.
    """
    #read header
    with open(filename,'r') as f:
        rmax = float(f.readline().split()[-1])
        M_z = int(f.readline().split()[-1])
        M_rho = int(f.readline().split()[-1])
        bf = f.readline().split()[-1]
        gamma = float(f.readline().split()[-1])
        pb = bool(f.readline().split()[-1])
    
    d_z = rmax/M_z
    d_rho = rmax/M_rho
    
    #read data
    m_z, m_rho, G_z, G_rho, G_z_err, G_rho_err, z_mean, rho_mean, counts \
        = np.loadtxt(filename,unpack=True)
    
    #typecast and reshape
    if bf=='linear':
        shape = (M_z+1,M_rho+1)
    else:
        shape = (M_z,M_rho)
    m_z = m_z.astype(int).reshape(shape)
    m_rho = m_rho.astype(int).reshape(shape)
    G_z.shape = shape
    G_rho.shape = shape
    G_z_err.shape = shape
    G_rho_err.shape = shape
    z_mean.shape = shape
    rho_mean.shape = shape
    counts.shape = shape
    
    #bin cent values
    rho_cent,z_cent = np.meshgrid(
        np.arange(0,rmax,d_rho)+0.5*d_z,
        np.arange(0,rmax,d_z)+0.5*d_rho
    )

    return m_z, m_rho, z_cent, rho_cent, G_z, G_rho, G_z_err, G_rho_err, \
        z_mean, rho_mean, counts, M_z, M_rho, rmax, bf, gamma, pb

def convert_cylindrical_force_axes(binmean_z,binmean_rho,G_z,G_rho,):
    """
    convert the force coefficients along rho and z axes to the absolute value
    and force as well as the coefficient values parallel and perpendicular to
    the particle-particle vector.

    Parameters
    ----------
    binmean_z : np.ndarray of float
        count weighted mean z distance for all pairs in each bin.
    binmean_rho : np.ndarray of float
        count weighted mean rho distance for all pairs in each bin.
    G_z : np.ndarray of float
        force coefficients along z.
    G_rho : np.ndarray of float
        force coefficients along rho.

    Returns
    -------
    G_abs : np.ndarray of float
        absolute value of the force coefficients, i.e. net force scalar.
    G_ang : np.ndarray of float
        (clockwise) angle of the force vector w.r.t. the z axis in radians.
    G_para : np.ndarray of float
        force coefficients for the force parallel to the particle-particle 
        vector.
    G_perp : np.ndarray of float
        force coefficients for the force perpendicular to the particle-particle 
        vector.

    """
    G_abs = np.sqrt(G_z**2+G_rho**2)
    binmean_r = np.sqrt(binmean_z**2+binmean_rho**2)
    binmean_ang = np.arccos(binmean_z/binmean_r)
    G_ang = np.sign(G_rho)*np.arccos(G_z/G_abs)
    G_para = G_abs*np.cos(G_ang-binmean_ang)
    G_perp = G_abs*np.sin(G_ang-binmean_ang)
    return G_abs,G_ang,G_para,G_perp


def filter_msd(coordinates, times=None, pos_cols=('z','y','x'),
                      msd_min=0.01, msd_max=1, interval=1):
    """calculates the squared displacement between subsequent frames for each
    particle individually, meaned over all intervals, and filters the indices
    by some threshold displacement
    
    Parameters
    ----------
    coordinates : list of list of pandas.DataFrame
        A pandas dataframe containing coordinates for each timestep. Must be
        indexed by particle (with each particle having a unique identifyer that
        matches between different time steps) and contain coordinates along 
        each dimension in a separate column, with column names matching those 
        given in `pos_cols`.
    times : list, optional
        timestamps for the sets of coordinates
    pos_cols : tuple of str, optional
        names of the columns of the DataFrames in `coordinates` containing the
        particle coordinates along each dimension. The length (i.e. number of
        dimensions) must match len(boundary) and the number of columns in 
        `coordinates`. The default is `('z','y','x')`.
    msd_min : float, optional
        Lower limit for the mean squared displacement between two  time steps 
        below which a particle is considered stationary. The default is 0.01.
    msd_max : float, optional
        Upper limit on the MSD between two timesteps. The default is 1.
    interval : int, optional
        time interval in integer index units used for the calculations, e.g.
        averaging over neigbouring steps or over larger intervals than the 
        sampling rate in the data. The default is 1.

    Returns
    -------
    indices : set
        particle identifiers for the particles which are NOT considered 
        stationary, i.e. those whose MSD is LARGER than the threshold.
    msds : list of float
        full list of the MSD values for all particles in all timesteps, useful
        e.g. for plotting a histogram to estimate a value for `threshold`.
    """
    #default min and max
    if msd_min is None:
        msd_min = 0
    if msd_max is None:
        msd_max = np.inf
    
    #init lists
    indices = []
    msds = []
    
    #in case of no timesteps return squared displacement / interval
    if times is None:
        for coord in coordinates:
            sd = []
            for i,(coord0,coord1) \
                in enumerate(zip(coord[:-interval],coord[interval:])):
                sd.append(sum([(coord0[col]-coord1[col])**2/interval \
                               for col in pos_cols]))
            sd = pd.concat(sd,axis=1)
            msd = sd.mean(axis=1,skipna=True)
            mask = (msd_min <= msd) & (msd < msd_max) & ~np.isnan(msd)
            msds.extend(msd)
            indices.append(set(mask.loc[mask].index))
    
    #when using timesteps return squared displacement  / dt
    else:
        for time,coord in zip(times,coordinates):
            sd = []
            for i,((coord0,t0,coord1,t1)) in enumerate(
                    zip(
                        coord[:-interval],
                        time[:-interval],
                        coord[interval:],
                        time[interval:]
                        )
                ):
                sd.append(sum([(coord0[col]-coord1[col])**2/(t1-t0) \
                               for col in pos_cols]))
            sd = pd.concat(sd,axis=1)
            msd = sd.mean(axis=1,skipna=True)
            mask = (msd_min <= msd) & (msd < msd_max) & ~np.isnan(msd)
            msds.extend(msd)
            indices.append(set(mask.loc[mask].index))
        
    return indices,msds

def run_overdamped_legacy(coordinates,times,boundary=None,gamma=1,rmax=1,M=20,
                   pos_cols=('z','y','x'),eval_particles=None,
                   periodic_boundary=False,bruteforce=False,
                   remove_near_boundary=True,solve_per_dim=False,
                   return_data = False):
    """
    Run the analysis for overdamped dynamics (brownian dynamics like), iterates
    over all subsequent sets of two timesteps and obtains forces from the 
    velocity of the particles as a function of the distribution of the
    particles around eachother.

    Parameters
    ----------
    coordinates : list of pandas.DataFrame
        A pandas dataframe containing coordinates for each timestep. Must be
        indexed by particle (with each particle having a unique identifyer that
        matches between different time steps) and contain coordinates along 
        each dimension in a separate column, with column names matching those 
        given in `pos_cols`.
    times : list of float
        list timestamps corresponding to the coordinates
    boundary : list or tuple, optional
        boundaries of the box in which the coordinates are defined in the form
        ((d0_min,d0_max),(d1_min,d1_max),...) with a length (number) and order 
        of dimensions matching `pos_cols`. The default is `None`, which uses 
        the min and max value found in the entire set of coordinates along each
        axis.
    gamma : float, optional
        damping/friction coefficient (kT/D) for calculation of F=V*kT/D. The
        default is 1.
    rmax : float, optional
        cut-off radius for calculation of the pairwise forces. The default is
        1.
    M : int, optional
        The number of discretization steps for the force profile, i.e. the
        number of bins from 0 to rmax into which the data will be sorted. The
        default is 20.
    pos_cols : tuple of str, optional
        names of the columns of the DataFrames in `coordinates` containing the
        particle coordinates along each dimension. The length (i.e. number of
        dimensions) must match len(boundary) and the number of columns in 
        `coordinates`. The default is `('z','y','x')`.
    periodic_boundary : bool, optional
        Whether the box has periodic boundary conditions. If True, the boundary
        must be given. The default is False.
    bruteforce : bool, optional
        If True, the coefficients are calculated in a naive brute-force 
        approach with a nested loop over all particles. The default is False,
        which uses a scipy.spatial.cKDTree based approach to only evaluate 
        pairs which are <rmax apart.
    remove_near_boundary : bool, optional
        If true, particles which are closer than rmax from any of the
        boundaries are not analyzed, but still accounted for when analyzing
        particles closer to the center of the box in order to only analyze 
        particles for which the full spherical shell up to rmax is within the 
        box of coordinates, and to prevent erroneous handling of particles
        which interact with other particles outside the measurement volume.
        Only possible when periodic_boundary=False. The default is True.
    solve_per_dim : bool, optional
        if True, the matrix is solved for each dimension separately, and a 
        force vector and error are returned for each dimension.
    return_data : bool, optional
        If True, the full coefficient matrix and force vector are returned 
        together with the force vector, error and the list of bincounts. The
        default is False.

    Returns
    -------
    G : numpy.array of length M
        discretized force vector, the result of the computation.
    G_err : numpy.array of length M
        errors in G based on the least_squares solution of the matrix equation
    counts : numpy.array of length M
        number of individual force evaluations contributing to the result in
        each bin.
    coefficients : numpy.array of M by 3n*(len(times)-1)
        coefficient matrix of the full dataset as specified in [1]. This is 
        only returned when `return_data=True`
    forces : numpy.array of length 3n*(len(times)-1)
        vector of particle forces of form [t0p0z,t0p0y,t0p0x,t0p1z,t0p1y, ...,
        tn-1pnx]. This is only returned when `return_data=True`
        
    References
    ----------
    [1] Jenkins, I. C., Crocker, J. C., & Sinno, T. (2015). Interaction
    potentials from arbitrary multi-particle trajectory data. Soft Matter, 11
    (35), 6948–6956. https://doi.org/10.1039/C5SM01233C

    """
    #check if one list or nested list, if not make it nested
    if isinstance(times[0],(list,np.ndarray)):
        nested = True
        nsteps = len(times)
        
        #perform checks to see if shapes/lengths of inputs match
        if not isinstance(coordinates[0],(list,np.ndarray)):
            raise ValueError('`coordinates` must be nested list if `times` is')
        if len(coordinates) != nsteps:
            raise ValueError('number of subsets in `times` and `coordinates` '
                             'must match')
        if any([len(coord)!=len(t) for coord,t in zip(coordinates,times)]):
            raise ValueError('length of each subset in `times` and '
                             '`coordinates` must match')
        if any([len(t) <= 1 for t in times]):
            raise ValueError('each subset in `times` and `coordinates` must '
                             'have at least 2 elements')
        if not eval_particles is None:
            if any(isinstance(ep,(list,set,np.ndarray)) \
                   for ep in eval_particles):
                if len(eval_particles) != nsteps:
                    raise ValueError('length of `times` and `eval_particles'
                                     ' must match')
            elif len(eval_particles)!=nsteps or \
                any(not ep is None for ep in eval_particles):
                eval_particles = repeat(eval_particles)
        else:
            eval_particles = repeat(None)
                     
        
    else:
        nested = False
        times = [times]
        coordinates = [coordinates]
        eval_particles = [eval_particles]
    
    #get dimensionality from pos_cols, check names
    pos_cols = list(pos_cols)
    ndims = len(pos_cols)
    
    #get default boundaries from min and max values in any coordinate set
    if boundary is None:
        if periodic_boundary:
            raise ValueError('when periodic_boundary=True, boundary must be '
                             'given')
        boundary = [
            [
                repeat([
                    min([c[dim].min() for c in coords]),
                    max([c[dim].max() for c in coords])
                ]) for dim in pos_cols
            ] for coords in coordinates
        ]
        
    #otherwise check inputs
    elif nested:
        #if single boundary for whole set given, make boundary iterable on a 
        #per timestep per coordinate set basis
        if np.array(boundary[0]).ndim == 1:
            if len(boundary) != ndims:
                raise ValueError('number of `pos_cols` does not match '
                                 '`boundary`')
            boundary = repeat(repeat(boundary))
        
        #else check if length matches
        elif len(boundary) != nsteps:
            raise ValueError('length of `boundary` and `coordinates` must '
                             'match')
        
        #if single boundary per coordinate set given
        elif np.array(boundary[0]).ndim == 2:
            #check dimensionality of each item
            if any([len(bounds) != ndims for bounds in boundary]):
                raise ValueError('number of `pos_cols` does not match '
                                 '`boundary`')
            #make iterable on a per timestep basis
            boundary = [repeat(bounds) for bounds in boundary]
            
        #if boundary for each timestep check lengths and dimensionality
        elif any([len(bounds) != len(time) \
                  for bounds,time in zip(boundary,times)]):
            raise ValueError('number of items in each item in `boundary` must '
                             'match that in `times`')
        elif any([np.array(bounds).shape[1] != ndims for bounds in boundary]):
            raise ValueError('number of `pos_cols` does not match `boundary`')
    
    #if not nested, make nested with single item per list for later iterating
    else:
        if np.array(boundary[0]).ndim == 2:
            if any([len(bounds) != ndims for bounds in boundary]):
                raise ValueError('number of `pos_cols` does not match '
                                 '`boundary`')
            boundary = [boundary]
        elif len(boundary) != ndims:
            raise ValueError('number of `pos_cols` does not match `boundary`')
        else:
            boundary = [repeat(boundary)]
        
    #initialize variables
    forces = []
    coefficients = []
    counts = []
    
    #loop over separate sets of coordinates
    for i,(coords,bounds,tsteps,eval_parts) in \
        enumerate(zip(coordinates,boundary,times,eval_particles)):
    
        #get number of timestep
        nt = len(tsteps)
        
        #check data
        if nt != len(coords):
            raise ValueError('length of timesteps does not match coordinate '
                             'data')
        
        #loop over all sets of two particles
        for j,((coords0,bound0,t0),(coords1,_,t1)) in \
            enumerate(_nwise(zip(coords,bounds,tsteps),n=2)):
            
            #print progress
            if nested:
                print(('\revaluating set {:d} of {:d}, step {:d} of {:d} '
                       '(time: {:.5f} to {:.5f})').format(i+1,nsteps,j+1,nt-1,
                                                    t0,t1),end='',flush=True)
            else:
                print(('\revaluating step {:d} of {:d} (time: {:.5f} to '
                       '{:.5f})').format(j+1,nt-1,t0,t1),end='',flush=True)
            
            #assure boundary is array
            bound0 = np.array(bound0)
            
            #find the particles which are far enough from boundary
            if remove_near_boundary:
                if rmax > min(bound0[:,1]-bound0[:,0])/2:
                    raise ValueError(
                        'when remove_near_boundary=True, rmax cannot be more '
                        'than half the smallest box dimension. Use rmax < '
                        '{:}'.format(min(bound0[:,1]-bound0[:,0])/2)
                    )
                
                selected = coords0.loc[(
                    (coords0[pos_cols] >= bound0[:,0]+rmax).all(axis=1) &
                    (coords0[pos_cols] <  bound0[:,1]-rmax).all(axis=1)
                )].index
                
            else:
                selected = coords0.index
            
            if not eval_parts is None:
                selected = selected.intersection(set(eval_parts))
            
            #check inputs
            if periodic_boundary:
                if rmax > min(bound0[:,1]-bound0[:,0])/2:
                    raise ValueError('when periodic_boundary=True, rmax '
                        'cannot be more than half the smallest box dimension')
                
                #remove any items outside of boundaries
                mask = (coords0[pos_cols] < bound0[:,0]).any(axis=1) | \
                    (coords0[pos_cols] >= bound0[:,1]).any(axis=1)
                if mask.any():
                    print('\n[WARNING] trajectories_to_forces.run_overdamped:'
                          ' some coordinates are outside of boundary and will'
                          ' be removed')
                    coords0 = coords0.loc[~mask]
                        
    
            #calculate the force vector containin the total force acting on 
            #each particle
            f = _calculate_forces_overdamped(
                    coords0.loc[selected],
                    coords1,
                    t1-t0,
                    bound0,
                    pos_cols,
                    gamma=gamma,
                    periodic_boundary=periodic_boundary,
                ).sort_index()
            
            #reshape f to 3n vector and add to total vector
            f = f[pos_cols].to_numpy().ravel()
            forces.append(f)
    
            #find neighbours and coefficients at time t0 for all particles 
            #present in t0 and t1
            C,c,bmp = _calculate_coefficients(
                    coords0.loc[set(coords0.index).intersection(coords1.index)],
                    set(selected).intersection(coords1.index),
                    rmax,
                    M,
                    bound0,
                    pos_cols,
                    bruteforce=bruteforce,
                    periodic_boundary=periodic_boundary,
                    )
            coefficients.append(C)
            counts.append(c)
        
        #newline between steps
        if nested:
            print()
    
    if nested:
        print('solving matrix equation')
    else:
        print('\nsolving matrix equation')

    #create one big array of coefficients and one of forces
    coefficients = np.concatenate(coefficients,axis=0)
    forces  = np.concatenate(forces,axis=0)
    counts = np.sum(counts,axis=0)
    
    #remove rows with only zeros
    mask = (np.sum(coefficients,axis=1) != 0)
    coefficients = coefficients[mask]
    forces = forces[mask]
    
    #solve eq. 15 from the paper in x, y and z separately
    if solve_per_dim:
        G = []
        G_err = []
        for dim in range(ndims):
            coef = coefficients[dim::ndims]
            G_dim,G_dim_err,_,_ = np.linalg.lstsq(
                np.dot(coef.T,coef),
                np.dot(coef.T,forces[dim::ndims]),
                rcond=None
            )
            G_dim[counts==0] = np.nan
            G.append(G_dim)
            G_err.append(G_dim_err)
        G,G_err = tuple(G),tuple(G_err)
    
    #solve eq. 15 from the paper for all dimensions together
    else:
        #G = sp.dot(sp.dot(1/sp.dot(C.T,C),C.T),f)
        G,G_err,_,_ = np.linalg.lstsq(
            np.dot(coefficients.T,coefficients),
            np.dot(coefficients.T,forces),
            rcond=None
        )
        G[counts==0] = np.nan
    
    print('done')
    if return_data:
        return G,G_err,counts,coefficients,forces
    else:
        return G,G_err,counts
    
def run_overdamped(coordinates,times,boundary=None,gamma=1,rmax=1,M=20,
    pos_cols=('z','y','x'),eval_particles=None,periodic_boundary=False,
    basis_function='constant',bruteforce=False,remove_near_boundary=True,
    constant_particles=False,solve_per_dim=False,neighbour_upper_bound=None,
    newline=False,use_gpu=False,
    ):
    """
    Run the analysis for overdamped dynamics (brownian dynamics like), iterates
    over all subsequent sets of two timesteps and obtains forces from the 
    velocity of the particles as a function of the distribution of the
    particles around eachother. Based on ref. [1].

    Parameters
    ----------
    coordinates : (list of) list of pandas.DataFrame
        A pandas dataframe containing coordinates for each timestep as a series
        of consecutive timesteps of at least 2 items (i.e. 1 time interval). 
        Multiple nonconsecutive series may be given as list of lists of 
        DataFrames. DataFrames must be indexed by particle (with each particle 
        having a unique identifyer that matches between different time steps) 
        and contain coordinates along each dimension in a separate column, with
        column names matching those given in `pos_cols`.
    times : (list of) list of float
        list(s) of timestamps corresponding to the coordinate sets
    boundary : tuple, list of tuple or list of list of tuple, optional
        boundaries of the box in which the coordinates are defined in the form
        ((d0_min,d0_max),(d1_min,d1_max),...) with a length (number) and order 
        of dimensions matching `pos_cols`. A single set of boundaries may be 
        given for all timesteps, or a (list of) list of boundaries for each 
        timestep may be specified. The default is `None`, which uses the min 
        and max value found in the entire set of coordinates along each axis. 
    gamma : float, optional
        damping/friction coefficient (kT/D) for calculation of F=V*kT/D. The
        default is 1.
    rmax : float, optional
        cut-off radius for calculation of the pairwise forces. The default is
        1.
    M : int, optional
        The number of discretization steps for the force profile, i.e. the
        number of bins from 0 to rmax into which the data will be sorted. The
        default is 20.
    pos_cols : tuple of str, optional
        names of the columns of the DataFrames in `coordinates` containing the
        particle coordinates along each dimension. The length (i.e. number of
        dimensions) must match len(boundary) and the number of columns in 
        `coordinates`. The default is `('z','y','x')`.
    eval_particles : set
        set of particle id's (matching the indices in `coordinates`) to use in 
        the force evaluation, such that forces are not calculated for any 
        particle not in the set of eval_particles. Note that all particles are
        always used to calculate the coefficients.
    periodic_boundary : bool, optional
        Whether the box has periodic boundary conditions. If True, the boundary
        must be given. The default is False.
    basis_function : ['constant', 'linear']
        the type of basis functions to use, where `'constant'` uses square wave
        basis functions which assume the force is constant over each bin, and 
        `'linear'` uses linear wave basis functions where each pair contributes
        to the nearby bins with linearly interpolated weights. The default is 
        `'constant'`.
    bruteforce : bool, optional
        If True, the coefficients are calculated in a naive brute-force 
        approach with a nested loop over all particles. The default is False,
        which uses a scipy.spatial.cKDTree based approach to only evaluate 
        pairs which are <rmax apart.
    remove_near_boundary : bool, optional
        If true, particles which are closer than rmax from any of the
        boundaries are not analyzed, but still accounted for when analyzing
        particles closer to the center of the box in order to only analyze 
        particles for which the full spherical shell up to rmax is within the 
        box of coordinates, and to prevent erroneous handling of particles
        which interact with other particles outside the measurement volume.
        Only possible when periodic_boundary=False. The default is True.
    constant_particles : bool, optional
        when the same set of particles is present in each timestep, i.e. the 
        indices of coordinates are identical for all time steps after selecting
        `eval_particles`, more efficient (indexing) algorithms can be used
    solve_per_dim : bool, optional
        if True, the matrix is solved for each dimension separately, and a 
        force vector and error are returned for each dimension.
    neighbour_upper_bound : int, optional
        upper bound on the number of neighbours within rmax a particle may have
        to limit memory use and computing time in the pair finding step. The
        default is the total number of particles in each time step.
    newline : bool, optional
        whether to print output for each series on a new line. The default is 
        True.
    use_gpu : bool, optional
        if True, matrix operations for the least squares solver are offloaded 
        to the gpu via CuPy, which requires a cuda-compatible gpu. For large
        numbers of bins (>>100) this may result in significantly better 
        performance but there is considerable overhead in moving data between
        cpu and gpu, so for most use cases this is not faster.

    Returns
    -------
    G : numpy.array of length M
        discretized force vector, the result of the computation.
    G_err : numpy.array of length M
        errors in G based on the least_squares solution of the matrix equation
    counts : numpy.array of length M
        number of individual force evaluations contributing to the result in
        each bin.
    mean_rho : numpy.array of length M
        the average interparticle distance of all particle pairs contributing 
        to each bin, i.e. the mean of the distances for which the forces were 
        evaluated.
        
    References
    ----------
    [1] Jenkins, I. C., Crocker, J. C., & Sinno, T. (2015). Interaction
    potentials from arbitrary multi-particle trajectory data. Soft Matter, 11
    (35), 6948–6956. https://doi.org/10.1039/C5SM01233C

    """
    #warn against inaccurate bounds
    if periodic_boundary and boundary is None:
        raise ValueError('when periodic_boundary=True, boundary must be '
                             'given')
        
    if remove_near_boundary and constant_particles:
        warn('`constant_particles` is not compatible with '
             '`remove_near_boundary`, falling back to standard implementation',
             stacklevel=2)
        constant_particles = False
    if use_gpu:
        import cupy as cp
    
    #check the inputs
    pos_cols = list(pos_cols)
    ndims = len(pos_cols)
    coordinates, times, boundary, eval_particles, nested = \
        _check_inputs(coordinates, times, boundary, pos_cols, eval_particles)
    nsteps = len(times)
        
    #initialize matrices for least squares solving
    if solve_per_dim:
        if use_gpu:
            X = [np.zeros((M,M)) for _ in range(ndims)]
            Y = [np.zeros((M)) for _ in range(ndims)]
        else:
            X = [cp.zeros((M,M)) for _ in range(ndims)]
            Y = [cp.zeros((M)) for _ in range(ndims)]
    else:
        if use_gpu:
            X = cp.zeros((M,M)) #C dot C.T
            Y = cp.zeros((M)) #C.T dot f
        else:
            X = np.zeros((M,M)) #C dot C.T
            Y = np.zeros((M)) #C.T dot f
    counts = np.zeros((M))
    binmeanpos = np.zeros((M))
    
    #loop over separate sets of coordinates
    for i,(coords,bounds,tsteps,eval_parts) in \
        enumerate(zip(coordinates,boundary,times,eval_particles)):
        
        #get number of timestep
        nt = len(tsteps)
        
        #make sure eval_parts is a set
        if not eval_parts is None and type(eval_parts) != set:
            eval_parts = set(eval_parts)
        
        #check data
        if nt != len(coords):
            raise ValueError('length of timesteps does not match coordinate '
                             'data')
        
        #loop over all sets of two coordinate arrays
        for j,((coords0,bound0,t0),(coords1,_,t1)) in \
            enumerate(_nwise(zip(coords,bounds,tsteps),n=2)):
                
            #print progress
            if nested:
                print(('\revaluating set {:d} of {:d}, step {:d} of {:d} '
                       '(time: {:.5f} to {:.5f})').format(i+1,nsteps,j+1,nt-1,
                                                    t0,t1),end='')
            else:
                print(('\revaluating step {:d} of {:d} (time: {:.5f} to '
                       '{:.5f})').format(j+1,nt-1,t0,t1),end='')
            
            #assure boundary is array, coords are only pos_cols
            bound0 = np.array(bound0)
            if not pos_cols is None:
                coords0 = coords0[pos_cols]
                coords1 = coords1[pos_cols]
            
            #check inputs
            if periodic_boundary:
                if rmax > min(bound0[:,1]-bound0[:,0])/2:
                    raise ValueError('when periodic_boundary=True, rmax '
                        'cannot be more than half the smallest box dimension')
                
            #remove any items outside of boundaries
            mask = (coords0 <= bound0[:,0]).any(axis=1) | \
                (coords0 > bound0[:,1]).any(axis=1)
            if mask.any():
                print()
                warn('trajectories_to_forces.run_overdamped: some '
                     'coordinates are outside of boundary and will be '
                     f'removed from series {i} set {j}')
                coords0 = coords0.loc[~mask]
            
            #find the particles which are far enough from boundary
            if remove_near_boundary:
                if rmax > min(bound0[:,1]-bound0[:,0])/2:
                    raise ValueError(
                        'when remove_near_boundary=True, rmax cannot be more '
                        'than half the smallest box dimension. Use rmax < '
                        '{:}'.format(min(bound0[:,1]-bound0[:,0])/2)
                    )
                
                selected = set(coords0.loc[(
                    (coords0 >= bound0[:,0]+rmax).all(axis=1) &
                    (coords0 <  bound0[:,1]-rmax).all(axis=1)
                )].index)
                
            else:
                selected = set(coords0.index)
            
            if not eval_parts is None:
                selected = selected.intersection(eval_parts)       
    
            #calculate the force vector containin the total force acting on 
            #each particle and reshape from (N,3) to (3N,) numpy list
            if constant_particles:#if indices are not needed, fast numpy math
                f = _calculate_force_overdamped_simple(
                    coords0.to_numpy(),
                    coords1.to_numpy(),
                    bound0,
                    periodic_boundary,
                    t1-t0,
                    gamma
                ).reshape((-1,))
            
            else:
                f = _calculate_forces_overdamped(
                    coords0.loc[sorted(selected)],
                    coords1,
                    t1-t0,
                    bound0,
                    gamma=gamma,
                    periodic_boundary=periodic_boundary,
                ).to_numpy().reshape((-1,))
    
            #find neighbours and coefficients at time t0 for all particles 
            #present in t0 and t1
            C,c,bmp = _calculate_coefficients(
                coords0 if constant_particles else \
                    coords0.loc[
                        list(set(coords0.index).intersection(coords1.index))],
                set(selected).intersection(coords1.index),
                rmax,
                M,
                bound0,
                pos_cols,
                bruteforce=bruteforce,
                periodic_boundary=periodic_boundary,
                basis_function=basis_function,
                neighbour_upper_bound=neighbour_upper_bound,
            )
            
            #precalculate dot products to avoid having entire matrix in memory
            if solve_per_dim:
                if use_gpu:
                    C, f = cp.asarray(C), cp.asarray(f)
                    for dim in range(ndims):
                        X[dim] += cp.dot(C[dim::ndims].T,C[dim::ndims])
                        Y[dim] += cp.dot(C[dim::ndims].T,f[dim::ndims])
                else:
                    for dim in range(ndims):
                        X[dim] += np.dot(C[dim::ndims].T,C[dim::ndims])
                        Y[dim] += np.dot(C[dim::ndims].T,f[dim::ndims])
            else:
                if use_gpu:
                    C, f = cp.asarray(C), cp.asarray(f)
                    X += cp.dot(C.T,C)
                    Y += cp.dot(C.T,f)
                else:
                    X += np.dot(C.T,C)
                    Y += np.dot(C.T,f)
            
            #update counter and mean bin positions
            counts += c
            binmeanpos += bmp
        
        #newline between steps
        if nested and newline:
            print()
    
    if nested and newline:
        print('solving matrix equation')
    else:
        print('\nsolving matrix equation')

    #mask for zero data
    mask = counts>0

    #solve eq. 15 from the paper per dim
    if solve_per_dim:
        G, G_err = [], []
        for dim in range(ndims):
            
            #remove zero data
            X[dim] = X[dim][mask][:,mask]
            Y[dim] = Y[dim][mask]
            
            #initialize result and error vectors
            G_dim, G_dim_err = np.empty(M),np.empty(M)
            G_dim[~mask] = np.nan
            G_dim_err[~mask] = np.nan
            
            #solve and calculate error
            if use_gpu:
                G_dim[mask] = cp.dot(cp.linalg.inv(X[dim]),Y[dim]).get()
                G_dim_err[mask] = ((Y - cp.dot(X,cp.asarray(G_dim[mask])))**2).get()
            else:
                G_dim[mask] = np.dot(np.linalg.inv(X[dim]),Y[dim])
                G_dim_err[mask] = (Y - np.dot(X,G_dim[mask]))**2
            G.append(G_dim)
            G_err.append(G_dim_err)
        G, G_err = tuple(G), tuple(G_err)
    
    #solve eq. 15 from the paper for all dimensions together
    else:
        #remove zero data
        X = X[mask][:,mask]
        Y = Y[mask]
        
        #initialize result and error vectors
        G,G_err = np.empty(M),np.empty(M)
        G[~mask] = np.nan
        G_err[~mask] = np.nan
        
        #solve for lowest error solution and calculate error
        if use_gpu:
            G[mask] = cp.dot(cp.linalg.inv(X),Y).get()
            G_err[mask] = ((Y - cp.dot(X,cp.asarray(G[mask])))**2).get()
        else:
            G[mask] = np.dot(np.linalg.inv(X),Y)
            G_err[mask] = (Y - np.dot(X,G[mask]))**2
    
    #calculate mean distance in each bin
    binmeanpos[~mask] = np.nan
    binmeanpos /= counts
    
    print('done')
    return G,G_err,counts,binmeanpos


def run_inertial(coordinates,times,boundary=None,mass=1,rmax=1,M=20,
               pos_cols=['z','y','x'],periodic_boundary=False,bruteforce=False,
               remove_near_boundary=False,solve_per_dim=False,
               return_data=False,neighbour_upper_bound=None,):
    """
    Run the analysis for inertial dynamics (molecular dynamics like), iterates
    over all subsequent sets of three timesteps and obtains forces from the 
    accellerations of the particles. Based on [1].

    Parameters
    ----------
    coordinates : list of pandas.DataFrame
        A pandas dataframe containing coordinates for each timestep. Must be
        indexed by particle (with each particle having a unique identifyer that
        matches between different time steps) and contain coordinates along 
        each dimension in a separate column, with column names matching those 
        given in `pos_cols`.
    times : list of float
        list timestamps corresponding to the coordinates with length matching 
        `len(coordinates)`.
    boundary : list or tuple, optional
        boundaries of the box in which the coordinates are defined in the form
        ((d0_min,d0_max),(d1_min,d1_max),...) with a length (number) and order 
        of dimensions matching `pos_cols`. The default is `None`, which uses 
        the min and max value found in the entire set of coordinates along each
        axis.
    mass : float, optional
        particle mass for calculation of F=m*a. The default is 1.
    pos_cols : list of str, optional
        names of the columns of the DataFrames in `coordinates` containing the
        particle coordinates along each dimension. The length (i.e. number of
        dimensions) must match len(boundary) and the number of columns in 
        `coordinates`. The default is `['z','y','x'].
    rmax : float, optional
        cut-off radius for calculation of the pairwise forces. The default is
        1.
    M : int, optional
        The number of discretization steps for the force profile, i.e. the
        number of bins from 0 to rmax into which the data will be sorted. The
        default is 20.
    periodic_boundary : bool, optional
        Whether the box has periodic boundary conditions. If True, the boundary
        must be given. The default is False.
    bruteforce : bool, optional
        If True, the coefficients are calculated in a naive brute-force 
        approach with a nested loop over all particles. The default is False,
        which uses a scipy.spatial.cKDTree based approach to only evaluate 
        pairs which are <rmax apart.
    remove_near_boundary : bool, optional
        If true, particles which are closer than rmax from any of the
        boundaries are not analyzed, but still accounted for when analyzing
        particles closer to the center of the box in order to only analyze 
        particles for which the full spherical shell up to rmax is within the 
        box of coordinates, and to prevent erroneous handling of particles
        which interact with other particles outside the measurement volume.
        Only possible when periodic_boundary=False. The default is True.
    solve_per_dim : bool, optional
        if True, the matrix is solved for each dimension separately, and a 
        force vector and error are returned for each dimension.
    return_data : bool, optional
        If True, the full coefficient matrix and force vector are returned 
        together with the force vector, error and the list of bincounts. The
        default is False.
    neighbour_upper_bound : int
        upper bound on the number of neighbours within rmax a particle may have
        to limit memory use and computing time in the pair finding step. The
        default is the total number of particles in each time step.

    Returns
    -------
    G : numpy.array of length M
        discretized force vector, the result of the computation.
    G_err : numpy.array of length M
        errors in G based on the least_squares solution of the matrix equation
    counts : numpy.array of length M
        number of individual force evaluations contributing to the result in
        each bin.
    coefficients : numpy.array of M by 3n*(len(times)-1)
        coefficient matrix of the full dataset as specified in [1]. This is 
        only returned when `return_data=True`
    forces : numpy.array of length 3n*(len(times)-1)
        vector of particle forces of form [t0p0z,t0p0y,t0p0x,t0p1z,t0p1y, ...,
        tn-1pnx]. This is only returned when `return_data=True`
    
    References
    ----------
    [1] Jenkins, I. C., Crocker, J. C., & Sinno, T. (2015). Interaction 
    potentials from arbitrary multi-particle trajectory data. Soft Matter, 
    11(35), 6948–6956. https://doi.org/10.1039/C5SM01233C
    """
    
    #check if one list or nested list, if not make it nested
    if isinstance(times[0],list):
        nested = True
        nsteps = len(times)
        
        if not isinstance(coordinates[0],list):
            raise ValueError('`coordinates` must be nested list if `times` is')
        if len(coordinates) != nsteps:
            raise ValueError('length of `times` and `coordinates` must match')
        
    else:
        nested = False
        times = [times]
        coordinates = [coordinates]
    
    #get dimensionality from pos_cols, check names
    ndims = len(pos_cols)
    
    #get default boundaries from min and max values in any coordinate set
    if boundary is None:
        if periodic_boundary:
            raise ValueError('when periodic_boundary=True, boundary must be '+
                             'given')
        boundary = [
            [
                [
                    min([c[dim].min() for c in coords]),
                    max([c[dim].max() for c in coords])
                ] for dim in pos_cols
            ] for coords in coordinates
        ]
        
    #otherwise check inputs
    elif nested:
        if len(boundary) != nsteps:
            raise ValueError('length of `boundary` and `coordinates` must '+
                             'match')
        elif any([len(bounds) != ndims for bounds in boundary]):
            raise ValueError('number of `pos_cols` does not match `boundary`')
    else:
        boundary = [boundary]
        
    #initialize variables
    forces = []
    coefficients = []
    counts = []
    
    #loop over separate sets of coordinates
    for i,(coords,bounds,tsteps) in enumerate(zip(coordinates,boundary,times)):
    
        #set boundaries, get number of timestep
        bounds = np.array(bounds)
        nt = len(tsteps)
    
        #loop over triplets of time steps, i.e. (t0,t1,t2),(t1,t2,t3),
        #   (t2,t3,t4), ..., (tn-2,tn-1,tn)
        print('starting calculation')
        for j,((coords0,t0),(coords1,t1),(coords2,t2)) in \
            enumerate(_nwise(zip(coords,tsteps),n=3)):
            
            #print progress
            if nested:
                print(('\revaluating set {:d} of {:d}, step {:d} of {:d} '+
                       '(time: {:.5f} to {:.5f})').format(i+1,nsteps,j+1,nt-2,
                                                    t0,t2),end='',flush=True)
            else:
                print(('\revaluating step {:d} of {:d} (time: {:.5f} to '+
                       '{:.5f})').format(j+1,nt-2,t0,t2),end='',flush=True)
            
            #find the particles which are far enough from boundary
            if remove_near_boundary:
                if rmax > min(bounds[:,1]-bounds[:,0])/2:
                    raise ValueError(
                        'when remove_near_boundary=True, rmax cannot be more '+
                        'than half the smallest box dimension. Use rmax < '+
                        '{:}'.format(min(bounds[:,1]-bounds[:,0])/2)
                    )
                selected = coords1.loc[(
                    (coords1[pos_cols] >= bounds[:,0]+rmax).all(axis=1) &
                    (coords1[pos_cols] <  bounds[:,1]-rmax).all(axis=1)
                )].index
            
            #otherwise take all particles
            else:
                selected = coords1.index
            
            #check inputs
            if periodic_boundary:
                if rmax > min(bounds[:,1]-bounds[:,0])/2:
                    raise ValueError('when periodic_boundary=True, rmax '+
                                     'cannot be more than half the smallest '+
                                     'box dimension')
    
                #remove any items outside of boundaries
                mask = (coords1[pos_cols] < bounds[:,0]).any(axis=1) | \
                    (coords1[pos_cols] >= bounds[:,1]).any(axis=1)
                if mask.any():
                    print('\n[WARNING] trajectories_to_forces.run_inertial: '+
                          'some coordinates are outside of boundary and will '+
                          'be removed')
                    coords1 = coords1.loc[~mask]
            
            #calculate the force vector containing the total force acting on 
            #each particle
            f = _calculate_forces_inertial(
                    coords0[pos_cols],
                    coords1.loc[selected],
                    coords2,
                    (t2-t0)/2,
                    bounds,
                    pos_cols,
                    mass = mass,
                    periodic_boundary=periodic_boundary
                ).sort_index()
            
            #reshape f to 3n vector and append to total result
            f = f[pos_cols].to_numpy().ravel()
            forces.append(f)
            
            #find neighbours and coefficients 
            C,c,bmp = _calculate_coefficients(
                coords1.loc[
                    list(set(coords1.index).intersection(
                        coords0.index).intersection(coords2.index))
                ],
                set(selected).intersection(coords0.index).intersection(
                    coords2.index),
                rmax,
                M,
                bounds,
                pos_cols,
                bruteforce=bruteforce,
                periodic_boundary=periodic_boundary,
                neighbour_upper_bound=neighbour_upper_bound,
            )
            coefficients.append(C)
            counts.append(c)
        
    print('\nsolving matrix equation')

    #create one big array of coefficients and one of forces
    C = np.concatenate(coefficients,axis=0)
    f  = np.concatenate(forces,axis=0)
    counts = np.sum(counts,axis=0)

    #solve eq. 15 from the paper in x, y and z separately
    if solve_per_dim:
        G = []
        G_err = []
        for dim in range(ndims):
            coef = coefficients[dim::ndims]
            G_dim,G_dim_err,_,_ = np.linalg.lstsq(
                np.dot(coef.T,coef),
                np.dot(coef.T,forces[dim::ndims]),
                rcond=None
            )
            G_dim[counts==0] = np.nan
            G.append(G_dim)
            G_err.append(G_dim_err)
        G,G_err = tuple(G),tuple(G_err)
    
    #solve eq. 15 from the paper for all dimensions together
    else:
        #G = sp.dot(sp.dot(1/sp.dot(C.T,C),C.T),f)
        G,G_err,_,_ = np.linalg.lstsq(
            np.dot(coefficients.T,coefficients),
            np.dot(coefficients.T,forces),
            rcond=None
        )
        G[counts==0] = np.nan

    if return_data:
        return G,G_err,counts,C,f
    else:
        return G,G_err,counts


#%% cylindrical

@nb.njit(parallel=False)
def _coefficient_loop_cylindrical(
        particles,queryparticles,indices,mask,rmax,M,M_rho,M_z
    ):
    """loop over all pairs found by KDTree.query and calculate coefficients"""
    #allocate memory for coefficient matrix
    coefficients = np.zeros((3*len(queryparticles),M))
    counter = np.zeros(M,np.uint64)
    binmean_rho = np.zeros(M)
    binmean_z = np.zeros(M)
    imax,jmax = indices.shape
    
    #loop over pairs in distance/indices array
    for i in nb.prange(imax):
        for j in range(jmax):
            if not mask[i,j]:
                d_zyx = queryparticles[i]-particles[indices[i,j]]
                d_rho = (d_zyx[1]**2+d_zyx[2]**2)**0.5
                m = int(d_rho*M_rho/rmax) + M_rho*int(abs(d_zyx[0])*M_z/rmax)
                counter[m] += 1
                binmean_rho[m] += d_rho
                binmean_z[m] += abs(d_zyx[0])
                
                coefficients[3*i,m] += np.sign(d_zyx[0])#z is always +1 or -1
                coefficients[3*i+1,m] += d_zyx[1]/d_rho#y
                coefficients[3*i+2,m] += d_zyx[2]/d_rho#x
    
    return coefficients,counter,binmean_z,binmean_rho

@nb.njit(parallel=False)
def _coefficient_loop_cylindrical_linear(
        particles,queryparticles,indices,mask,rmax,M,M_rho,M_z
    ):
    """loop over all pairs found by KDTree.query and calculate coefficients in 
    periodic boundary conditions"""
    #allocate memory for coefficient matrix, note that M=(M_z+1)(M_rho+1)
    coefficients = np.zeros((3*len(queryparticles),M))
    counter = np.zeros(M)
    binmean_z = np.zeros(M)
    binmean_rho = np.zeros(M)
    imax,jmax = indices.shape
    
    #loop over pairs in distance/indices array
    for i in nb.prange(imax):
        for j in range(jmax):
            if not mask[i,j]:
                #calculate distances and bins
                d_zyx = queryparticles[i]-particles[indices[i,j]]
                d_rho = (d_zyx[1]**2+d_zyx[2]**2)**0.5
                m_rho = int(d_rho*M_rho/rmax)
                m_z = int(abs(d_zyx[0])*M_z/rmax)
                #calculate weights for the four corners of each bin
                phi = [
                    (1-d_rho*M_rho/rmax+m_rho)*(1-abs(d_zyx[0])*M_z/rmax+m_z),
                    (d_rho*M_rho/rmax-m_rho)*(1-abs(d_zyx[0])*M_z/rmax+m_z),
                    (1-d_rho*M_rho/rmax+m_rho)*(abs(d_zyx[0])*M_z/rmax-m_z),
                    (d_rho*M_rho/rmax-m_rho)*(abs(d_zyx[0])*M_z/rmax-m_z)
                ]
                #convert rho and z bins to M bins
                bins = [
                    m_rho   + (M_rho+1)*m_z,
                    m_rho+1 + (M_rho+1)*m_z,
                    m_rho   + (M_rho+1)*(m_z+1),
                    m_rho+1 + (M_rho+1)*(m_z+1)
                ]
                #assign values weighted by basis functions
                for k in range(4):
                    counter[bins[k]] += phi[k]
                    binmean_rho[bins[k]] += d_rho*phi[k]
                    binmean_z[bins[k]] += abs(d_zyx[0])*phi[k]
                    coefficients[3*i,bins[k]] += phi[k]*np.sign(d_zyx[0])#z
                    coefficients[3*i+1,bins[k]] += phi[k]*d_zyx[1]/d_rho#y
                    coefficients[3*i+2,bins[k]] += phi[k]*d_zyx[2]/d_rho#x
    
    return coefficients,counter,binmean_z,binmean_rho

@nb.njit(parallel=False)
def _coefficient_loop_cylindrical_periodic(
        particles,queryparticles,indices,mask,rmax,M,M_rho,M_z,
        boxmin,boxmax
    ):
    """loop over all pairs found by KDTree.query and calculate coefficients in 
    periodic boundary conditions"""
    #allocate memory for coefficient matrix
    coefficients = np.zeros((3*len(queryparticles),M))
    counter = np.zeros(M,np.uint64)
    binmean_z = np.zeros(M)
    binmean_rho = np.zeros(M)
    imax,jmax = indices.shape
    
    #loop over pairs in distance/indices array
    for i in nb.prange(imax):
        for j in range(jmax):
            if not mask[i,j]:
                d_zyx = _distance_periodic_wrap(
                    queryparticles[i],particles[indices[i,j]],boxmin,boxmax
                )
                d_rho = (d_zyx[1]**2+d_zyx[2]**2)**0.5
                m = int(d_rho*M_rho/rmax) + M_rho*int(abs(d_zyx[0])*M_z/rmax)
                counter[m] += 1
                binmean_rho[m] += d_rho
                binmean_z[m] += abs(d_zyx[0])

                coefficients[3*i,m] += np.sign(d_zyx[0])#z is always +1 or -1
                coefficients[3*i+1,m] += d_zyx[1]/d_rho#y
                coefficients[3*i+2,m] += d_zyx[2]/d_rho#x
                
    return coefficients,counter,binmean_z,binmean_rho

@nb.njit(parallel=False)
def _coefficient_loop_cylindrical_periodic_linear(
        particles,queryparticles,indices,mask,rmax,M,M_rho,M_z,
        boxmin,boxmax
    ):
    """loop over all pairs found by KDTree.query and calculate coefficients in 
    periodic boundary conditions"""
    #allocate memory for coefficient matrix, note that M=(M_z+1)(M_rho+1)
    coefficients = np.zeros((3*len(queryparticles),M))
    counter = np.zeros(M)
    binmean_z = np.zeros(M)
    binmean_rho = np.zeros(M)
    imax,jmax = indices.shape
    
    #loop over pairs in distance/indices array
    for i in nb.prange(imax):
        for j in range(jmax):
            if not mask[i,j]:
                #calculate distances and bins
                d_zyx = _distance_periodic_wrap(
                    queryparticles[i],particles[indices[i,j]],boxmin,boxmax
                )
                d_rho = (d_zyx[1]**2+d_zyx[2]**2)**0.5

                m_rho = int(d_rho*M_rho/rmax)
                m_z = int(abs(d_zyx[0])*M_z/rmax)
                #calculate weights for the four corners of each bin
                phi = [
                    (1-d_rho*M_rho/rmax+m_rho)*(1-abs(d_zyx[0])*M_z/rmax+m_z),
                    (d_rho*M_rho/rmax-m_rho)*(1-abs(d_zyx[0])*M_z/rmax+m_z),
                    (1-d_rho*M_rho/rmax+m_rho)*(abs(d_zyx[0])*M_z/rmax-m_z),
                    (d_rho*M_rho/rmax-m_rho)*(abs(d_zyx[0])*M_z/rmax-m_z)
                ]
                #convert rho and z bins to M bins
                bins = [
                    m_rho   + (M_rho+1)*m_z,
                    m_rho+1 + (M_rho+1)*m_z,
                    m_rho   + (M_rho+1)*(m_z+1),
                    m_rho+1 + (M_rho+1)*(m_z+1)
                ]
                #assign values weighted by basis functions
                for k in range(4):
                    counter[bins[k]] += phi[k]
                    binmean_rho[bins[k]] += d_rho*phi[k]
                    binmean_z[bins[k]] += abs(d_zyx[0])*phi[k]
                    coefficients[3*i,bins[k]] += phi[k]*np.sign(d_zyx[0])#z
                    coefficients[3*i+1,bins[k]] += phi[k]*d_zyx[1]/d_rho#y
                    coefficients[3*i+2,bins[k]] += phi[k]*d_zyx[2]/d_rho#x
    
    return coefficients,counter,binmean_z,binmean_rho

def _calculate_coefficients_cylindrical(
        coords,query_indices,rmax,M,M_z,M_rho,boundary,periodic_boundary=False,
        basis_function='constant',neighbour_upper_bound=None
    ):
    """calculate the coefficient matric in cylindrical coordinates where the 
    first dimension is the cylinder axis and the second and third define the 
    circularly symmetric plane
    """
    #convert to numpy array with axes (particle,dim) and dim=[x,y,z]
    coords.sort_index(inplace=True)
    particles = coords.to_numpy()
    queryparticles = coords.loc[sorted(query_indices)].to_numpy()
    
    #set maximum number of neighbours 1 particle may have within rmax
    if neighbour_upper_bound is None:
        neighbour_upper_bound = len(particles)
    else:
        neighbour_upper_bound = min([neighbour_upper_bound,len(particles)])
    
    #coefficient calculation in periodic boundary conditions
    if periodic_boundary:
        
        boundary = np.array(boundary)
        boxmin = boundary[:,0]
        boxmax = boundary[:,1]
        
        #correct box and coordinates to have lower lim at 0 for cKDTree
        particles -= boxmin
        queryparticles -= boxmin
        boxmax -= boxmin
        boxmin -= boxmin
        
        #initialize and query periodic KDTree for pairs within rmax
        tree = cKDTree(particles,boxsize=boxmax)
        dist,indices = tree.query(
            queryparticles,
            k=neighbour_upper_bound,
            distance_upper_bound=rmax,
        )
        
        #remove pairs with self and np.inf fill values
        dist, indices = dist[:,1:],indices[:,1:]
        mask = np.isinf(dist)
        
        #perform numba-optimized loop over particle pairs
        if basis_function == 'constant':
            coefficients,counter,binmean_z,binmean_rho = \
                _coefficient_loop_cylindrical_periodic(
                    particles,queryparticles,indices,mask,rmax,M,M_rho,M_z,
                    boxmin,boxmax
                )
        elif basis_function == 'linear':
            coefficients,counter,binmean_z,binmean_rho = \
                _coefficient_loop_cylindrical_periodic_linear(
                    particles,queryparticles,indices,mask,rmax,M,M_rho,M_z,
                    boxmin,boxmax
                )
    
    #no periodic boundary conditions
    else:

        #initialize and query KDTree for fast pairfinding
        tree = cKDTree(particles)
        dist,indices = tree.query(
            queryparticles,
            k=neighbour_upper_bound,
            distance_upper_bound=rmax,
        )
        
        #remove pairs with self and np.inf fill values
        dist, indices = dist[:,1:], indices[:,1:]
        mask = np.isinf(dist)

        #perform numba-optimized loop over particle pairs
        if basis_function == 'constant':
            coefficients,counter,binmean_z,binmean_rho = \
                _coefficient_loop_cylindrical(
                    particles,queryparticles,indices,mask,rmax,M,M_rho,M_z
                )
        elif basis_function == 'linear':
            coefficients,counter,binmean_z,binmean_rho = \
                _coefficient_loop_cylindrical_linear(
                    particles,queryparticles,indices,mask,rmax,M,M_rho,M_z
                )
                
    return coefficients,counter,binmean_z,binmean_rho

def run_overdamped_cylindrical(coordinates,times,boundary=None,gamma=1,rmax=1,
        M_z=20,M_rho=20,pos_cols=('z','y','x'),eval_particles=None,
        periodic_boundary=False,basis_function='constant',
        remove_near_boundary=True,constant_particles=False,
        neighbour_upper_bound=None,use_gpu=False,check_coordinates=True,
        newline=True):
    """
    Run the analysis for overdamped dynamics (brownian dynamics like) in a 
    cylindrical coordinate system where interactions are radially averaged in 
    the xy plane (dimensions 2 and 1) but not along the cylinder (z) axis.

    Parameters
    ----------
    coordinates : (list of) list of pandas.DataFrame
        A pandas dataframe containing coordinates for each timestep as a series
        of consecutive timesteps of at least 2 items (i.e. 1 time interval). 
        Multiple nonconsecutive series may be given as list of lists of 
        DataFrames. DataFrames must be indexed by particle (with each particle 
        having a unique identifyer that matches between different time steps) 
        and contain coordinates along each dimension in a separate column, with
        column names matching those given in `pos_cols`.
    times : (list of) list of float
        list(s) of timestamps corresponding to the coordinate sets
    boundary : tuple, list of tuple or list of list of tuple, optional
        boundaries of the box in which the coordinates are defined in the form
        ((d0_min,d0_max),(d1_min,d1_max),...) with a length (number) and order 
        of dimensions matching `pos_cols`. A single set of boundaries may be 
        given for all timesteps, or a (list of) list of boundaries for each 
        timestep may be specified. The default is `None`, which uses the min 
        and max value found in the entire set of coordinates along each axis. 
    gamma : float, optional
        damping/friction coefficient (kT/D) for calculation of F=V*kT/D. The
        default is 1.
    rmax : float, optional
        cut-off radius for calculation of the pairwise forces. The default is
        1.
    M_z : int, optional
        The number of discretization steps for the force profile along the z
        (cylinder) direction, i.e. the number of bins from 0 to rmax into which
        the data will be sorted. The default is 20.
    M_rho : int, optional
        The number of discretization steps along the axial (rho) direction. The
        default is 20.
    pos_cols : tuple of str, optional
        names of the columns of the DataFrames in `coordinates` containing the
        particle coordinates along each dimension. The first is assumed to be 
        the axial (z) direction and the second and third define the plane for 
        the rho component. The default is `('z','y','x')`.
    eval_particles : set, optional
        set of particle id's (matching the indices in `coordinates`) to use in 
        the force evaluation, such that forces are not calculated for any 
        particle not in the set of eval_particles. Note that all particles are
        always used to calculate the coefficients. The default is to evaluate
        all particles in `coordinates`.
    periodic_boundary : bool, optional
        Whether the box has periodic boundary conditions. If True, the boundary
        must be given. The default is False.
    basis_function : ['constant', 'linear']
        the type of basis functions to use, where `'constant'` uses square wave
        basis functions which assume the force is constant over each bin, and 
        `'linear'` uses linear wave basis functions where each pair contributes
        to the nearby bins with linearly interpolated weights. The default is 
        `'constant'`.
    remove_near_boundary : bool, optional
        If true, particles which are closer than rmax from any of the
        boundaries are not analyzed, but still accounted for when analyzing
        particles closer to the center of the box in order to only analyze 
        particles for which the full spherical shell up to rmax is within the 
        box of coordinates, and to prevent erroneous handling of particles
        which interact with other particles outside the measurement volume.
        Only possible when periodic_boundary=False. The default is True.
    constant_particles : bool, optional
        when the same set of particles is present in each timestep, i.e. the 
        indices of coordinates are identical for all time steps after selecting
        `eval_particles`, more efficient (indexing) algorithms can be used
    neighbour_upper_bound : int
        upper bound on the number of neighbours within rmax a particle may have
        to limit memory use and computing time in the pair finding step. The
        default is the total number of particles in each time step.
    use_gpu : bool, optional
        if True, matrix operations for the least squares solver are offloaded 
        to the gpu via CuPy, which requires a cuda-compatible gpu. For large
        numbers of bins (M_z×M_rho>>100) this may result in significantly 
        better performance. There is considerable overhead in moving data 
        between cpu and gpu, so for small numbers of bins this is not faster.
    check_coordinates : bool, optional
        by default, coordinates are checked for being within boundaries. This 
        can be bypassed by setting this to False.
    newline : bool, optional
        whether to print output for each series on a new line. The default is 
        True.

    Returns
    -------
    m_z : numpy.array of ints of shape (M_z,M_rho)
        z axis bin indices corresponding to the data arrays.
    m_rho : numpy.array of ints of shape (M_z,M_rho)
        rho axis bin indices corresponding to the data arrays.
    G_z : numpy.array of floats of shape (M_z,M_rho)
        values for the axial (z) component of the force in each bin, i.e. the
        magnitude associated with each basis function.
    G_rho : numpy.array of floats of shape (M_z,M_rho)
        values for the in-plane (rho) component of the force in each bin, i.e. 
        the magnitude associated with each basis function.
    G_z_err : numpy.array of floats of shape (M_z,M_rho)
        residual least-squares error in each bin in G_z
    G_rho_err : numpy.array of floats of shape (M_z,M_rho)
        residual least-squares error in each bin in G_rho
    mean_z : numpy.array of floats of shape (M_z,M_rho)
        the average absolute z distance of all particle pairs contributing to 
        each bin, i.e. the mean of the distances for which the forces were 
        evaluated.
    mean_rho : numpy.array of floats of shape (M_z,M_rho)
        the average absolute planar (rho) distance of all particle pairs 
        contributing to each bin, i.e. the mean of the distances for which the 
        forces were evaluated.
    counts : numpy.array of floats of shape (M_z,M_rho)
        number of individual force evaluations contributing to the result in
        each bin. When `basis_function='linear'` this is a total weight rather
        than an integer count, since each particle pair has its unity weight
        divided over the 4 surrounding bins.
    """
    if use_gpu:
        import cupy as cp
    if periodic_boundary and boundary is None:
        raise ValueError('when periodic_boundary=True, boundary must be '
                             'given')
    if remove_near_boundary and constant_particles:
        warn('`constant_particles` is not compatible with '
             '`remove_near_boundary`, falling back to standard implementation',
             stacklevel=2)
        constant_particles = False
    
    #get dimensionality from pos_cols, must be 3D for cylindrical
    pos_cols = list(pos_cols)
    ndims = len(pos_cols)
    if ndims != 3:
        raise NotImplementedError('cylindrical coordinates only implemented in'
                                  '3 dimensions')
    
    #check the inputs
    coordinates, times, boundary, eval_particles, nested = \
        _check_inputs(coordinates, times, boundary, pos_cols, eval_particles)
    nsteps = len(times)
        
    #for linear basis functions, need extra point at the right side of last bin
    if basis_function == 'linear':
        M = (M_rho+1)*(M_z+1)
    else:
        M = M_rho*M_z
        
    #initialize matrices for least squares solving
    if use_gpu:
        X_z,X_rho = cp.zeros((M,M)),cp.zeros((M,M)) #C dot C.T
        Y_z,Y_rho = cp.zeros((M)),cp.zeros((M)) #C.T dot f
    else:
        X_z,X_rho = np.zeros((M,M)),np.zeros((M,M)) #C dot C.T
        Y_z,Y_rho = np.zeros((M)),np.zeros((M)) #C.T dot f
    counts = np.zeros((M))
    binmean_z = np.zeros((M))
    binmean_rho = np.zeros((M))
    
    #loop over separate sets of coordinates
    for i,(coords,bounds,tsteps,eval_parts) in \
        enumerate(zip(coordinates,boundary,times,eval_particles)):
        
        #get number of timestep
        nt = len(tsteps)
        
        #make sure eval_parts is a set
        if not eval_parts is None and type(eval_parts) != set:
            eval_parts = set(eval_parts)
        
        #check data
        if nt != len(coords):
            raise ValueError('length of timesteps does not match coordinate '
                             'data')
        
        #loop over all sets of two coordinate arrays
        for j,((coords0,bound0,t0),(coords1,_,t1)) in \
            enumerate(_nwise(zip(coords,bounds,tsteps),n=2)):
                
            #print progress
            if nested:
                print(('\revaluating set {:d} of {:d}, step {:d} of {:d} '
                       '(time: {:.5f} to {:.5f})').format(i+1,nsteps,j+1,nt-1,
                                                    t0,t1),end='',flush=True)
            else:
                print(('\revaluating step {:d} of {:d} (time: {:.5f} to '
                       '{:.5f})').format(j+1,nt-1,t0,t1),end='',flush=True)
            
            #assure boundary is array, coords are only pos_cols
            bound0 = np.array(bound0)
            coords0 = coords0[pos_cols]
            coords1 = coords1[pos_cols]
            
            #check inputs
            if periodic_boundary:
                if rmax > min(bound0[:,1]-bound0[:,0])/2:
                    raise ValueError('when periodic_boundary=True, rmax '
                        'cannot be more than half the smallest box dimension')

            #remove any items outside of boundaries
            if check_coordinates:
                mask = (coords0 < bound0[:,0]).any(axis=1) | \
                    (coords0 >= bound0[:,1]).any(axis=1)
                if mask.any():
                    mask = coords0.index[mask]
                    warn('\ntrajectories_to_forces.run_overdamped: some '
                         'coordinates are outside of boundary and will be '
                         f'removed: indices {mask.tolist()}',stacklevel=2)
                    coords0 = coords0.drop(mask)
                    if constant_particles:
                        coords1 = coords1.drop(mask,errors='ignore')
            
            #find the particles which are far enough from boundary
            if remove_near_boundary:
                if rmax > min(bound0[:,1]-bound0[:,0])/2:
                    raise ValueError(
                        'when remove_near_boundary=True, rmax cannot be more '
                        'than half the smallest box dimension. Use rmax < '
                        '{:}'.format(min(bound0[:,1]-bound0[:,0])/2)
                    )
                
                selected = set(coords0.loc[(
                    (coords0 >= bound0[:,0]+rmax).all(axis=1) &
                    (coords0 <  bound0[:,1]-rmax).all(axis=1)
                )].index)
                
            else:
                selected = set(coords0.index)
            
            if not eval_parts is None:
                selected = selected.intersection(eval_parts)
    
            #calculate the force vector containin the total force acting on 
            #each particle and reshape from (N,3) to (3N,) numpy list
            if constant_particles:#if indices are not needed, fast numpy math
                f = _calculate_force_overdamped_simple(
                    coords0.to_numpy(),
                    coords1.to_numpy(),
                    bound0,
                    periodic_boundary,
                    t1-t0,
                    gamma
                ).reshape((-1,))
            
            else:#if particles can change we need the pandas indices
                f = _calculate_forces_overdamped(
                    coords0.loc[sorted(selected)],
                    coords1,
                    t1-t0,
                    bound0,
                    gamma=gamma,
                    periodic_boundary=periodic_boundary,
                ).to_numpy().reshape((-1,))
    
            #find neighbours and coefficients at time t0 for all particles 
            #present in t0 and t1
            C,c,bmp_z,bmp_rho = _calculate_coefficients_cylindrical(
                coords0 if constant_particles else \
                    coords0.loc[
                        list(set(coords0.index).intersection(coords1.index))],
                set(selected).intersection(coords1.index),
                rmax,
                M,
                M_z,
                M_rho,
                bound0,
                periodic_boundary=periodic_boundary,
                basis_function=basis_function,
                neighbour_upper_bound=neighbour_upper_bound,
            )

            #precalculate dot products per force dimension
            mask = np.arange(len(f)) %3 == 0#true for z dim, false for y and x
            if use_gpu:
                C, f  = cp.asarray(C), cp.asarray(f)
                X_z += cp.dot(C[mask].T, C[mask])
                Y_z += cp.dot(C[mask].T, f[mask])
                mask = ~mask
                X_rho += cp.dot(C[mask].T, C[mask])
                Y_rho += cp.dot(C[mask].T, f[mask])
            else:
                X_z += np.dot(C[mask].T, C[mask])
                Y_z += np.dot(C[mask].T, f[mask])
                mask = ~mask
                X_rho += np.dot(C[mask].T, C[mask])
                Y_rho += np.dot(C[mask].T, f[mask])
            
            #update counter and mean bin positions
            counts += c
            binmean_rho += bmp_rho
            binmean_z += bmp_z
        
        #newline between steps
        if nested and newline:
            print()
    
    if nested and newline:
        print('solving matrix equation')
    else:
        print('\nsolving matrix equation')
    
    #remove bins with zero data
    mask = counts>0
    X_z = X_z[mask][:,mask]
    Y_z = Y_z[mask]
    X_rho = X_rho[mask][:,mask]
    Y_rho = Y_rho[mask]
    
    #initialize result vector and solve for least squares solution
    G_z = np.empty(M)
    G_z[~mask] = np.nan
    G_rho = G_z.copy()
    if use_gpu:
        G_z[mask] = cp.dot(cp.linalg.inv(X_z),Y_z).get()
        G_rho[mask] = cp.dot(cp.linalg.inv(X_rho),Y_rho).get()
    else:
        G_z[mask] = np.dot(np.linalg.inv(X_z),Y_z)
        G_rho[mask] = np.dot(np.linalg.inv(X_rho),Y_rho)
    
    #calculate error
    G_z_err = np.empty(M)
    G_z_err[~mask] = np.nan
    G_rho_err = G_z_err.copy()
    if use_gpu:
        G_z_err[mask] = ((Y_z - cp.dot(X_z,cp.asarray(G_z[mask])))**2).get()
        G_rho_err[mask] = ((Y_rho - cp.dot(X_rho,cp.asarray(G_rho[mask])))**2).get()
    else:
        G_z_err[mask] = (Y_z - np.dot(X_z,G_z[mask]))**2
        G_rho_err[mask] = (Y_rho - np.dot(X_rho,G_rho[mask]))**2
    
    #calculate mean position in each bin
    binmean_rho[~mask] = np.nan
    binmean_z[~mask] = np.nan
    binmean_rho /= counts
    binmean_z /= counts
    
    #provide bin indices and reshape for convenience
    if basis_function == 'linear':
        M_z += 1
        M_rho += 1
    m_rho,m_z = np.meshgrid(range(M_rho),range(M_z))
    G_z.shape = (M_z,M_rho)
    G_rho.shape = (M_z,M_rho)
    G_z_err.shape = (M_z,M_rho)
    G_rho_err.shape = (M_z,M_rho)
    binmean_z.shape = (M_z,M_rho)
    binmean_rho.shape = (M_z,M_rho)
    counts.shape = (M_z,M_rho)
    
    print('done')
    return m_z,m_rho,G_z,G_rho,G_z_err,G_rho_err,binmean_z,binmean_rho,counts