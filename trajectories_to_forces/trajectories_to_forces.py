
#%% imports
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from itertools import tee,islice,repeat
import numba as nb

#%% private definitions

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
    elif x1-x0 < -(xmax-xmin)/2:
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

def _calculate_forces_overdamped(coords0,coords1,dt,boundary,pos_cols,gamma=1,
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
            colmin,colmax = boundary[list(pos_cols).index(col)]
            
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


@nb.njit(parallel=True)
def _coefficient_pair_loop_nb(particles,queryparticles,ndims,dist,indices,mask,
                              rmax,m):
    """loop over all pairs found by KDTree.query and calculate coefficients"""
    #allocate memory for coefficient matrix
    coefficients = np.zeros((ndims*len(queryparticles),m))
    counter = np.zeros(m)
    
    #loop over pairs in distance/indices array
    for i in nb.prange(len(queryparticles)):
        for j in range(len(particles)):
            if not mask[i,j]:
                d = dist[i,j]
                counter[int(d/rmax*m)] += 1
                for dim in range(ndims):
                    coefficients[ndims*i+dim,int(d/rmax*m)] += \
                        (queryparticles[i,dim]-particles[indices[i,j],dim])/d

    return coefficients,counter

@nb.njit(parallel=True)
def _bruteforce_pair_loop_nb(particles,queryparticles,ndims,rmax,m):
    """loop over all pairs with i from queryparticles and j from particles, and
    calculate coefficients"""
    #allocate memory for coefficient matrix
    coefficients = np.zeros((ndims*len(queryparticles),m))
    counter = np.zeros(m)
    
    #loop over pairs in distance/indices array
    for i in nb.prange(len(queryparticles)):
        for j in range(len(particles)):
            d = np.sum((queryparticles[i]-particles[j])**2)**0.5
            if d < rmax and d != 0:
                counter[int(d/rmax*m)] += 1
                for dim in range(ndims):
                    coefficients[ndims*i+dim,int(d/rmax*m)] += \
                        (queryparticles[i,dim]-particles[j,dim])/d

    return coefficients,counter

@nb.njit()
def _distance_periodic_wrap(ci,cj,boxmin,boxmax):
    """calculates element-wise distances between two sets of n-dimensional 
    coordinates while wrapping around boundaries of periodic box with bounds
    boxmin, boxmax along each dimension"""
    distances = np.empty(len(ci))
    for dim,(i,j,mi,ma) in enumerate(zip(ci,cj,boxmin,boxmax)):
        if i-j > (ma-mi)/2:
            distances[dim] = i-j-ma+mi
        elif i-j <= (mi-ma)/2:
            distances[dim] = i-j+ma-mi
        else:
            distances[dim] = i-j
    return distances

@nb.njit(parallel=True)
def _coefficient_pair_loop_periodic_nb(particles,queryparticles,ndims,dist,
                                       indices,mask,rmax,m,boxmin,boxmax):
    """loop over all pairs found by KDTree.query and calculate coefficients in 
    periodic boundary conditions"""
    #allocate memory for coefficient matrix
    coefficients = np.zeros((ndims*len(queryparticles),m))
    counter = np.zeros(m)
    
    #loop over pairs in distance/indices array
    for i in nb.prange(len(queryparticles)):
        for j in range(len(particles)):
            if not mask[i,j]:
                d_xyz = _distance_periodic_wrap(
                    queryparticles[i],particles[indices[i,j]],boxmin,boxmax
                )
                d = dist[i,j]
                counter[int(d/rmax*m)] += 1
                for dim in range(ndims):
                    coefficients[ndims*i+dim,int(d/rmax*m)] += d_xyz[dim]/d

    return coefficients,counter

@nb.njit(parallel=True)
def _bruteforce_pair_loop_periodic_nb(particles,queryparticles,ndims,rmax,m,
                                      boxmin,boxmax):
    """loop over all pairs with i from queryparticles and j from particles, and
    calculate coefficients in periodic boundary conditions"""
    #allocate memory for coefficient matrix
    coefficients = np.zeros((ndims*len(queryparticles),m))
    counter = np.zeros(m)
    
    #loop over pairs in distance/indices array
    for i in nb.prange(len(queryparticles)):
        for j in range(len(particles)):
            d_xyz = _distance_periodic_wrap(
                queryparticles[i],particles[j],boxmin,boxmax
            )
            d = np.sum(d_xyz**2)**0.5
            if d < rmax and d != 0:
                counter[int(d/rmax*m)] += 1
                for dim in range(ndims):
                    coefficients[ndims*i+dim,int(d/rmax*m)] += d_xyz[dim]/d

    return coefficients,counter

def _calculate_coefficients(coords,query_indices,rmax,m,boundary,pos_cols,
                            periodic_boundary=False,bruteforce=False):
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
    m : int
        number of discretization steps to bin the matrix into. The bin with
        will be rmax/m.
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

    Returns
    -------
    coefficients : array
        an array containing the 3n by m matric of coefficients, where n is
        the number of particles
    counter : list of length m
        total number of pair counts for each column in the matrix

    """
    #get dimensionality
    ndims = len(pos_cols)
    
    #convert to numpy array with axes (particle,dim) and dim=[x,y,z]
    coords.sort_index(inplace=True)
    particles = coords[pos_cols].to_numpy()
    queryparticles = coords.loc[sorted(query_indices)][pos_cols].to_numpy()
    
    #coefficient calculation in periodic boundary conditions
    if periodic_boundary:
        
        boundary = np.array(boundary)
        boxmin = boundary[:,0]
        boxmax = boundary[:,1]
        
        #optionally use (inefficient) brute-force search through all pairs
        if bruteforce:
            coefficients,counter = _bruteforce_pair_loop_periodic_nb(
                particles,queryparticles,ndims,rmax,m,boxmin,boxmax)
        
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
                k=len(particles),
                distance_upper_bound=rmax
            )
            
            #remove pairs with self and np.inf fill values
            mask = (~np.isfinite(dist)) | (dist==0)
            
            #calculate the coefficients using the numba-compiled function
            coefficients,counter = _coefficient_pair_loop_periodic_nb(
                particles,queryparticles,ndims,dist,indices,mask,rmax,m,boxmin,
                boxmax
            )
    
    #no periodic boundary conditions
    else:
        #optionally use (inefficient) brute-force search through all pairs
        if bruteforce:
            coefficients,counter = _bruteforce_pair_loop_nb(
                particles,queryparticles,rmax,m
            )
        
        #use KDTree based efficient neighbour searching algorithm
        else:
            #initialize and query KDTree for fast pairfinding
            tree = cKDTree(particles)
            dist,indices = tree.query(
                queryparticles,k=len(particles),distance_upper_bound=rmax)
            
            #remove pairs with self and np.inf fill values
            mask = (~np.isfinite(dist)) | (dist==0)
            
            #perform numba-optimized loop over particle pairs
            coefficients,counter = _coefficient_pair_loop_nb(
                particles,queryparticles,ndims,dist,indices,mask,rmax,m
            )

    return coefficients,counter

#%% public definitions

def save_forceprofile(
        filename,
        rsteps,
        rmax,
        forces,
        counts):
    """
    Saves the results to a text file

    Parameters
    ----------
    filename : string
        filename to use for results file
    rsteps : int
        number of bins
    rmax : float
        cut-off radius for force
    forces : list of float
        list of force values as obtained from the trajectory analysis
    counts : list of int
        the number of evaluations used for each bin

    Returns
    -------
    None.

    """
    with open(filename,'w+') as file:
        #write input parameters
        file.write("rsteps:\t{}\n".format(rsteps))
        file.write("rmax:\t{}\n".format(rmax))
        file.write('\n')

        #write table headers
        file.write('r\tforce\tcounts\n')

        #write table
        for i,r in enumerate(np.linspace(
                0+rmax/2/rsteps,
                rmax+rmax/2/rsteps,
                rsteps,
                endpoint=False
            )):
            file.write(f'{r:.3f}\t{forces[i]:5f}\t{int(counts[i]):d}\n')

    print('saved results as "'+filename+'"')

def load_forceprofile(filename):
    """
    loads the results from a text file

    Parameters
    ----------
    filename : string
        name of the file to load.

    Returns
    -------
    rvals : list
        bin edges of pairwise interparticle distance
    forces : list
        the mean force in each bin
    counts : list
        the number of particle pairs counted for each bin
    rsteps : int
        number of discretization steps
    rmax : float
        cut-off radius for force

    """
    with open(filename,'r') as file:
        filedata = [line[:-1] for line in file.readlines()]

    #load input parameters
    rsteps = int(filedata[0].split()[1])
    rmax = float(filedata[1].split()[1])

    #load data table
    rvals = []
    forces = []
    counts = []

    for line in filedata[4:]:
        line = line.split()
        rvals.append(float(line[0]))
        forces.append(float(line[1]))
        counts.append(int(line[2]))

    return rvals,forces,counts,rsteps,rmax

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

def run_overdamped(coordinates,times,boundary=None,gamma=1,rmax=1,m=20,
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
    m : int, optional
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
    G : numpy.array of length m
        discretized force vector, the result of the computation.
    G_err : numpy.array of length m
        errors in G based on the least_squares solution of the matrix equation
    counts : numpy.array of length m
        number of individual force evaluations contributing to the result in
        each bin.
    coefficients : numpy.array of m by 3n*(len(times)-1)
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
            C,c = _calculate_coefficients(
                    coords0.loc[set(coords0.index).intersection(coords1.index)],
                    set(selected).intersection(coords1.index),
                    rmax,
                    m,
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


def run_inertial(coordinates,times,boundary=None,mass=1,rmax=1,m=20,
               pos_cols=['z','y','x'],periodic_boundary=False,bruteforce=False,
               remove_near_boundary=False,solve_per_dim=False,
               return_data=False):
    """
    Run the analysis for inertial dynamics (molecular dynamics like), iterates
    over all subsequent sets of three timesteps and obtains forces from the 
    accellerations of the particles.

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
    m : int, optional
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

    Returns
    -------
    G : numpy.array of length m
        discretized force vector, the result of the computation.
    G_err : numpy.array of length m
        errors in G based on the least_squares solution of the matrix equation
    counts : numpy.array of length m
        number of individual force evaluations contributing to the result in
        each bin.
    coefficients : numpy.array of m by 3n*(len(times)-1)
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
            C,c = _calculate_coefficients(
                coords1.loc[set(coords1.index).intersection(
                    coords0.index).intersection(coords2.index)],
                set(selected).intersection(coords0.index).intersection(
                    coords2.index),
                rmax,
                m,
                bounds,
                pos_cols,
                bruteforce=bruteforce,
                periodic_boundary=periodic_boundary,
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

