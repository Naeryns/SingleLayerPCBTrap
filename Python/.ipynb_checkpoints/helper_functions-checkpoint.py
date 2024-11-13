import numpy as np
from scipy.interpolate import griddata

import time
import warnings
import re

###################################
###    Data Readout Functions   ###
###################################

def grid_from_header(header=None, filename=None, mode='grid'):
    
    """
    Generates coordinate grids or retrieves grid configuration from the header of an Ansys .fld file.

    Parameters:
        header (str, optional): The header line from the .fld file containing grid metadata.
                                If not provided, it will be read from the file specified by 'filename'.
        filename (str, optional): Path to the .fld file from which to read the header, if 'header' is not provided.
        mode (str, optional): Determines the output of the function.
                              - If 'grid', returns coordinate grids and grid sizes.
                              - If 'conf', returns grid boundaries and spacing.
                              Default is 'grid'.

    Returns:
        If mode == 'grid':
            tuple: A tuple containing:
                - xs (ndarray): 3D array of x-coordinates.
                - ys (ndarray): 3D array of y-coordinates.
                - zs (ndarray): 3D array of z-coordinates.
                - nx (int): Number of grid points along the x-axis.
                - ny (int): Number of grid points along the y-axis.
                - nz (int): Number of grid points along the z-axis.

        If mode == 'conf':
            tuple: A tuple containing:
                - min_bounds (tuple): Minimum coordinates (x_min, y_min, z_min).
                - max_bounds (tuple): Maximum coordinates (x_max, y_max, z_max).
                - grid_spacing (tuple): Grid spacing along each axis (dx, dy, dz).

    Raises:
        ValueError: If the header cannot be parsed to extract grid information.
        ValueError: If 'mode' is not 'grid' or 'conf'.

    Example:
        # Using header string
        xs, ys, zs, nx, ny, nz = grid_from_header(header=header_line, mode='grid')

        # Reading header from file
        xs, ys, zs, nx, ny, nz = grid_from_header(filename='path_to_file.fld', mode='grid')

        # Getting grid configuration
        min_bounds, max_bounds, grid_spacing = grid_from_header(filename='path_to_file.fld', mode='conf')
    """
    # Read header from file if not provided.
    if header is None:
        if filename is None:
            raise ValueError("Either 'header' or 'filename' must be provided.")
        with open(filename) as f:
            header = f.readline()

    # Use regular expressions to extract grid parameters from the header
    min_match = re.search(r'Min: \[([^\]]+)\]', header)
    max_match = re.search(r'Max: \[([^\]]+)\]', header)
    size_match = re.search(r'Grid Size: \[([^\]]+)\]', header)

    if min_match and max_match and size_match:
        # Extract minimum coordinates and convert units from mm to meters
        min_vals = [float(val.replace('mm', 'e-3')) for val in min_match.group(1).split()]
        # Extract maximum coordinates and convert units from mm to meters
        max_vals = [float(val.replace('mm', 'e-3')) for val in max_match.group(1).split()]
        # Extract grid spacing and convert units from mm to meters
        grid_size_vals = [float(val.replace('mm', 'e-3')) for val in size_match.group(1).split()]
    else:
        raise ValueError('Unable to analyze the header for grid information.')

    x_min, y_min, z_min = min_vals  # Minimum coordinates
    x_max, y_max, z_max = max_vals  # Maximum coordinates
    dx, dy, dz = grid_size_vals     # Grid spacing along each axis

    if mode == 'grid':
        # Generate coordinate arrays for each axis
        X = np.arange(x_min, x_max + dx / 2, dx)
        Y = np.arange(y_min, y_max + dy / 2, dy)
        Z = np.arange(z_min, z_max + dz / 2, dz)

        # Create 3D meshgrids for coordinates
        xs, ys, zs = np.meshgrid(X, Y, Z, indexing='ij')

        # Get the number of points along each axis
        nx = len(X)
        ny = len(Y)
        nz = len(Z)

        return xs, ys, zs, nx, ny, nz

    elif mode == 'conf':
        # Return the grid configuration: boundaries and spacing
        return (x_min, y_min, z_min), (x_max, y_max, z_max), (dx, dy, dz)

    else:
        raise ValueError("'mode' must be 'grid' or 'conf'.")


def vec_from_file(filename):
    """
    Reads vector field data (e.g., electric field components) from an Ansys .fld file
    and returns the data reshaped into 3D grid arrays.

    Parameters:
        filename (str): Path to the .fld file containing vector field data.

    Returns:
        tuple: A tuple containing:
            - xs, ys, zs (ndarray): 3D arrays of x, y, z coordinates.
            - E (dict): Dictionary containing the vector field components with keys 'x', 'y', 'z'.

    Raises:
        ValueError: If the header cannot be parsed to extract grid information.
    """
    # Open the file and read all lines
    with open(filename) as f:
        lines = f.readlines()

    # Generate coordinate grids from the header
    header = lines[0]
    xs, ys, zs, nx, ny, nz = grid_from_header(header)

    # Initialize arrays to hold vector components
    Ex = np.zeros(len(lines) - 2)
    Ey = np.zeros(len(lines) - 2)
    Ez = np.zeros(len(lines) - 2)

    # Read vector data from the file
    for i, line in enumerate(lines[2:]):
        try:
            # Extract the last three elements in the line (Ex, Ey, Ez)
            E_components = line.split(' ')[-3:]
            Ex[i] = float(E_components[0])
            Ey[i] = float(E_components[1])
            Ez[i] = float(E_components[2])
        except Exception as e:
            print(f'Error parsing line {i + 2}: {e}')

    # Reshape the vector components into 3D grids
    E = {
        'x': Ex.reshape((nx, ny, nz)),
        'y': Ey.reshape((nx, ny, nz)),
        'z': Ez.reshape((nx, ny, nz)),
    }

    return xs, ys, zs, E


def scalar_from_file(filename):
    """
    Reads scalar field data (e.g., electric potential, magnitude of electric field)
    from an Ansys .fld file and returns the data reshaped into a 3D grid array.

    Parameters:
        filename (str): Path to the .fld file containing scalar field data.

    Returns:
        tuple: A tuple containing:
            - xs, ys, zs (ndarray): 3D arrays of x, y, z coordinates.
            - data_grid (ndarray): 3D array of the scalar field data.

    Raises:
        ValueError: If the header cannot be parsed to extract grid information.
    """
    # Open the file and read all lines
    with open(filename) as f:
        lines = f.readlines()

    # Generate coordinate grids from the header
    header = lines[0]
    xs, ys, zs, nx, ny, nz = grid_from_header(header)

    # Initialize an array to hold scalar data
    data = np.zeros(len(lines) - 2)

    # Read scalar data from the file
    for i, line in enumerate(lines[2:]):
        try:
            # Extract the last element in the line (scalar value)
            data[i] = float(line.split(' ')[-1])
        except Exception as e:
            print(f'Error parsing line {i + 2}: {e}')

    # Reshape the scalar data into a 3D grid
    data_grid = data.reshape((nx, ny, nz))

    return xs, ys, zs, data_grid


def read_field_data(conf):
    """
    Reads field data and grid configurations for multiple zones from provided configurations.

    Parameters:
        conf (dict): A dictionary containing configuration data for each zone.
                     Each key is a zone name, and each value is a dictionary with keys:
                         - 'file': Path to the field data file for the zone.
                         - 'min', 'max', 'step': (Optional) Will be populated with grid boundaries and spacing.

    Returns:
        tuple:
            - conf (dict): Updated configuration dictionary with 'min', 'max', 'step' for each zone.
            - grid (dict): Dictionary containing grid coordinate arrays for each zone.
                           Structure:
                           grid[zone]['x'], grid[zone]['y'], grid[zone]['z']: 3D arrays of coordinates.
                           grid[zone]['unique_x'], grid[zone]['unique_y'], grid[zone]['unique_z']: 1D arrays of unique coordinates.
            - field (dict): Dictionary containing field data arrays for each zone.
                            field[zone]: 3D array of field data for the zone.
    """
    import time
    import numpy as np

    grid = {}   # Dictionary to store grid data for each zone
    field = {}  # Dictionary to store field data for each zone

    # Iterate over each zone in the configuration
    for zone in conf:
        print(f'Reading zone {zone}')
        start_time = time.time()  # Start timing the file readout

        grid[zone] = {}  # Initialize a dictionary to store grid data for this zone

        # Read grid configuration from the file header and update 'conf' with 'min', 'max', 'step'
        conf[zone]['min'], conf[zone]['max'], conf[zone]['step'] = grid_from_header(
            filename=conf[zone]['file'], mode='conf'
        )

        # Read scalar field data and grid coordinates from the file
        xs, ys, zs, fld = scalar_from_file(conf[zone]['file'])

        # Store the grid coordinate arrays in the 'grid' dictionary
        grid[zone]['x'] = xs
        grid[zone]['y'] = ys
        grid[zone]['z'] = zs

        # Store unique coordinate values for each axis
        grid[zone]['unique_x'] = np.unique(xs)
        grid[zone]['unique_y'] = np.unique(ys)
        grid[zone]['unique_z'] = np.unique(zs)

        # Store the field data array in the 'field' dictionary
        field[zone] = fld

        end_time = time.time()  # End timing the file readout
        print(f'File Readout Time: {end_time - start_time:.4f} seconds.')

    # Return the updated configuration, grid data, and field data
    return conf, grid, field


###################################
###    Data process Functions   ###
###################################

def fill_NaN_nearest(data, X, Y, Z):
    """
    Fills NaN values in a dataset by interpolating using the nearest valid data points.

    Parameters:
        data (ndarray): The data array containing NaN values to be filled.
        X (ndarray): 3D array of x-coordinates corresponding to the data.
        Y (ndarray): 3D array of y-coordinates corresponding to the data.
        Z (ndarray): 3D array of z-coordinates corresponding to the data.

    Returns:
        ndarray: The data array with NaN values filled using nearest neighbor interpolation.
    """
    # Create a mask for valid (non-NaN) data points
    valid_mask = ~np.isnan(data)
    # Extract coordinates and values of valid data points
    valid_points = np.column_stack((X[valid_mask], Y[valid_mask], Z[valid_mask]))
    valid_values = data[valid_mask]

    # Prepare all points for interpolation
    all_points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

    # Perform nearest neighbor interpolation to fill NaNs
    filled_data = griddata(
        valid_points, valid_values, all_points, method='nearest'
    ).reshape(data.shape)

    return filled_data


###############################
###    Sampling Functions   ###
###############################

def random_sampling(x_range, y_range, z_range, num_points):
    """
    Generates random sampling points within the specified ranges.

    Parameters:
        x_range (tuple): A tuple (x_min, x_max) specifying the range for the x-axis.
        y_range (tuple): A tuple (y_min, y_max) specifying the range for the y-axis.
        z_range (tuple): A tuple (z_min, z_max) specifying the range for the z-axis.
        num_points (int): The number of random points to generate.

    Returns:
        ndarray: An array of shape (num_points, 3) containing the random sampling points.
    """
    x_rand = np.random.uniform(x_range[0], x_range[1], num_points)
    y_rand = np.random.uniform(y_range[0], y_range[1], num_points)
    z_rand = np.random.uniform(z_range[0], z_range[1], num_points)
    points = np.column_stack((x_rand, y_rand, z_rand))
    return points


def grid_sampling(x_range, y_range, z_range, x_steps, y_steps, z_steps):
    """
    Generates grid sampling points within the specified ranges and number of steps.

    Parameters:
        x_range (tuple): A tuple (x_min, x_max) specifying the range for the x-axis.
        y_range (tuple): A tuple (y_min, y_max) specifying the range for the y-axis.
        z_range (tuple): A tuple (z_min, z_max) specifying the range for the z-axis.
        x_steps (int): Number of steps along the x-axis.
        y_steps (int): Number of steps along the y-axis.
        z_steps (int): Number of steps along the z-axis.

    Returns:
        ndarray: An array of shape (N, 3) containing the grid sampling points,
                 where N = x_steps * y_steps * z_steps.
    """
    x_vals = np.linspace(x_range[0], x_range[1], x_steps)
    y_vals = np.linspace(y_range[0], y_range[1], y_steps)
    z_vals = np.linspace(z_range[0], z_range[1], z_steps)
    x_grid, y_grid, z_grid = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    points = np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel()))
    return points


##############################################
###    Interpolation Analyzing Functions   ###
##############################################

def find_large_errors(err, threshold=10):
    """
    Identifies the indices of points where the relative error exceeds a specified threshold.

    Parameters:
        err (ndarray): Array of relative errors between interpolation methods.
        threshold (float): The error threshold above which points are considered to have large errors.

    Returns:
        ndarray: Array of indices where the relative error exceeds the threshold.
    """
    # Identify indices where the error exceeds the threshold
    large_error_indices = np.where(err > threshold)[0]
    return large_error_indices


def find_adjacent_points(point, offset=5e-5, x_range=None, y_range=None, z_range=None):
    """
    Finds the 6 adjacent sample points with specified offsets to a given sample point.
    If grid boundaries are provided, ensures that the adjacent points are within the grid boundaries.
    If grid boundaries are not provided, a warning is issued and boundary checks are skipped.
    
    Parameters:
        point (array-like): The coordinate [x, y, z] of the point of interest.
        offset (float or list/tuple of floats, optional): The offset distance(s) to adjacent points.
            - If a single float is provided, the same offset is used in all directions.
            - If a list or tuple of three floats is provided, they specify the offsets
              in the x, y, and z directions, respectively.
            Defaults to 5e-5.
        x_range (tuple, optional): The (min, max) values of the x-axis grid.
        y_range (tuple, optional): The (min, max) values of the y-axis grid.
        z_range (tuple, optional): The (min, max) values of the z-axis grid.
            If any of these are not provided and the corresponding global variables
            'xrange', 'yrange', 'zrange' do not exist, boundary checks are skipped.

    Returns:
        list: A list of coordinates of the adjacent points. If grid boundaries are provided,
              only points within the boundaries are included.

    Raises:
        ValueError: If any of the adjacent points fall outside the grid boundaries (when boundaries are provided).
        ValueError: If 'offset' is not a single float or a list/tuple of three floats.
    """
    # Ensure that offset is in the correct format
    if isinstance(offset, (int, float)):
        # Single value provided; use the same offset in all directions
        offset_x = offset_y = offset_z = offset
    elif isinstance(offset, (list, tuple, np.ndarray)) and len(offset) == 3:
        # Individual offsets provided for each axis
        offset_x, offset_y, offset_z = offset
    else:
        raise ValueError("Offset must be a single float or a list/tuple of three floats.")

    # Unpack the coordinates of the given point
    x, y, z = point

    # Define the six possible directions with their respective offsets
    directions = [
        np.array([offset_x, 0, 0]),    # Positive x-direction
        np.array([-offset_x, 0, 0]),   # Negative x-direction
        np.array([0, offset_y, 0]),    # Positive y-direction
        np.array([0, -offset_y, 0]),   # Negative y-direction
        np.array([0, 0, offset_z]),    # Positive z-direction
        np.array([0, 0, -offset_z])    # Negative z-direction
    ]

    # If any range is None, skip boundary checks
    boundaries_provided = True
    if x_range is None or y_range is None or z_range is None:
        boundaries_provided = False
        warnings.warn("Grid boundaries are not provided. Boundary checks will be skipped.")

    adjacent_points = []
    for direction in directions:
        adj_point = point + direction
        # Check if the adjacent point is within the grid boundaries if boundaries are provided
        if boundaries_provided:
            if (x_range[0] <= adj_point[0] <= x_range[1] and
                y_range[0] <= adj_point[1] <= y_range[1] and
                z_range[0] <= adj_point[2] <= z_range[1]):
                adjacent_points.append(adj_point)
            else:
                raise ValueError(f"Adjacent point {adj_point} is outside the grid boundaries.")
        else:
            # If boundaries are not provided, include all adjacent points
            adjacent_points.append(adj_point)

    return adjacent_points


def find_adjacent_grids(point, x_grid, y_grid, z_grid):
    """
    Finds the 8 nearest grid points in the field data to a given sample point.
    The points are sorted from small to large, altering z first, y second, x last.
    
    Parameters:
        point (ndarray): The coordinate [x, y, z] of the point of interest.
        x_grid (ndarray): 1D array of x-coordinates from the field data grid.
        y_grid (ndarray): 1D array of y-coordinates from the field data grid.
        z_grid (ndarray): 1D array of z-coordinates from the field data grid.
    
    Returns:
        tuple:
            indices_list (list of tuples): Sorted list of indices (ix, iy, iz) of the adjacent grid points.
            grid_points_list (list of arrays): List of grid point coordinates corresponding to the indices.
    
    Raises:
        ValueError: If the point is outside the grid boundaries.
    """
    x, y, z = point
    
    # Check if the point is within the grid boundaries
    if not (x_grid[0] <= x <= x_grid[-1] and
            y_grid[0] <= y <= y_grid[-1] and
            z_grid[0] <= z <= z_grid[-1]):
        raise ValueError("Point is outside the grid boundaries.")
    
    # Find indices of the grid points just below and above the point for each axis
    ix0 = np.searchsorted(x_grid, x, side='right') - 1
    iy0 = np.searchsorted(y_grid, y, side='right') - 1
    iz0 = np.searchsorted(z_grid, z, side='right') - 1
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    iz1 = iz0 + 1
    
    # Ensure indices are within the valid range
    ix0 = max(min(ix0, len(x_grid) - 1), 0)
    iy0 = max(min(iy0, len(y_grid) - 1), 0)
    iz0 = max(min(iz0, len(z_grid) - 1), 0)
    ix1 = max(min(ix1, len(x_grid) - 1), 0)
    iy1 = max(min(iy1, len(y_grid) - 1), 0)
    iz1 = max(min(iz1, len(z_grid) - 1), 0)
    
    # Generate unique indices for x, y, z
    ix_list = sorted(set([ix0, ix1]))
    iy_list = sorted(set([iy0, iy1]))
    iz_list = sorted(set([iz0, iz1]))
    
    # Generate all combinations of indices, sorting to alter z first, y second, x last
    indices_list = []
    for ix in ix_list:
        for iy in iy_list:
            for iz in iz_list:
                indices_list.append((ix, iy, iz))
    
    # Sort the indices to alter z first, y second, x last (z changes fastest)
    indices_list.sort(key=lambda idx: (idx[0], idx[1], idx[2]))
    
    # Retrieve the corresponding grid points
    grid_points_list = [
        np.array([x_grid[ix], y_grid[iy], z_grid[iz]]) for (ix, iy, iz) in indices_list
    ]
    
    return indices_list, grid_points_list


#############################
###    Zoning Functions   ###
#############################

def zone_identifier(point, zones):
    """
    Determines which zone a given point belongs to, with zones of smaller stepsize taking priority.
    
    Parameters:
        point (array-like): The (x, y, z) coordinates of the point as a list, tuple, or NumPy array.
        zones (dict): A dictionary of zones with their configurations. Each key is the zone name,
            and each value is a dictionary containing:
                - 'file' (optional for this function): The file path associated with the zone.
                - 'min': A tuple of minimum (x, y, z) coordinates defining the zone boundary.
                - 'max': A tuple of maximum (x, y, z) coordinates defining the zone boundary.
                - 'step': A tuple of stepsizes (dx, dy, dz) for the zone grid.
            Defaults to `rf_conf`, which should be defined elsewhere in your code.
    
    Returns:
        str: The name of the zone to which the point belongs.
    
    Raises:
        ValueError: If the point does not belong to any of the defined zones.
    
    Example:
        >>> rf_conf = {
        ...     'rf': {'file': '...', 'min': (...), 'max': (...), 'step': (...)},
        ...     'c1': {'file': '...', 'min': (...), 'max': (...), 'step': (...)},
        ...     # Add other zones as needed
        ... }
        >>> point = [0.0003, 0.0003, 0.0003]
        >>> zone = zone_identifier(point, zones=rf_conf)
        >>> print(f'Point {point} belongs to zone: {zone}')
        Point [0.0003, 0.0003, 0.0003] belongs to zone: c3
    """
    import numpy as np

    # Convert the input point to a NumPy array for easy manipulation
    point = np.array(point)

    # Function to calculate the maximum stepsize of a zone
    def max_stepsize(zone):
        return max(zone['step'])

    # Sort zones by ascending maximum stepsize (smaller stepsize has higher priority)
    sorted_zones = sorted(zones.items(), key=lambda item: max_stepsize(item[1]))

    # Iterate over the zones in order of priority
    for zone_name, zone_data in sorted_zones:
        min_bounds = np.array(zone_data['min'])
        max_bounds = np.array(zone_data['max'])
        # Check if the point lies within the bounds of the zone
        if np.all(point >= min_bounds) and np.all(point <= max_bounds):
            return zone_name  # Return the name of the zone

    # If the point does not belong to any zone, raise an error
    raise ValueError(f"Point {point} does not belong to any zone.")