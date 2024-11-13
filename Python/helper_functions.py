import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.integrate import solve_ivp

import time
import warnings
import re
import sys
#from tqdm import tqdm
#from tqdm.notebook import tqdm

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


def read_field_data(conf, rf_file=None):
    """
    Reads field data and grid configurations for multiple zones from provided configurations.
    Optionally reads RF field data from a separate file.

    Parameters:
        conf (dict): A dictionary containing configuration data for each zone.
                     Each key is a zone name, and each value is a dictionary with keys:
                         - 'file': Path to the field data file for the zone.
                         - 'min', 'max', 'step': (Optional) Will be populated with grid boundaries and spacing.

        rf_file (str, optional): Path to the RF field data file. If provided, the function will read the RF data
                                 and return additional outputs related to the RF field. Defaults to None.

    Returns:
        If rf_file is None:
            tuple:
                - conf (dict): Updated configuration dictionary with 'min', 'max', 'step' for each zone.
                - grid (dict): Dictionary containing grid coordinate arrays for each zone.
                               Structure:
                                   grid[zone]['x'], grid[zone]['y'], grid[zone]['z']: 3D arrays of coordinates.
                                   grid[zone]['unique_x'], grid[zone]['unique_y'], grid[zone]['unique_z']: 1D arrays of unique coordinates.
                - field (dict): Dictionary containing field data arrays for each zone.
                                field[zone]: 3D array of field data for the zone.
        If rf_file is provided:
            tuple:
                - conf (dict): Updated configuration dictionary with 'min', 'max', 'step' for each zone.
                - grid (dict): Dictionary as described above.
                - field (dict): Dictionary as described above.
                - grid_rf (dict): Dictionary containing grid coordinate arrays for the RF data.
                                  Structure similar to 'grid' but for RF data.
                - magE (ndarray): 3D array of RF field magnitude data.

    Example:
        # Without RF data
        conf, grid, field = read_field_data(conf)

        # With RF data
        conf, grid, field, grid_rf, magE = read_field_data(conf, rf_file='path_to_rf_file.fld')
    """

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

    # If an RF file is provided, read the RF data
    if rf_file is not None:
        print('Reading RF data...')
        start_time = time.time()

        grid_rf = {}  # Dictionary to store grid data for the RF field

        # Read scalar field data and grid coordinates from the RF file
        xrf, yrf, zrf, magE = scalar_from_file(rf_file)

        # Store the grid coordinate arrays in the 'grid_rf' dictionary
        grid_rf['x'] = xrf
        grid_rf['y'] = yrf
        grid_rf['z'] = zrf

        # Store unique coordinate values for each axis
        grid_rf['unique_x'] = np.unique(xrf)
        grid_rf['unique_y'] = np.unique(yrf)
        grid_rf['unique_z'] = np.unique(zrf)

        end_time = time.time()  # End timing the RF file readout
        print(f'RF File Readout Time: {end_time - start_time:.4f} seconds.')

        # Return the updated configuration, grid data, field data, and RF data
        return conf, grid, field, grid_rf, magE

    # Return the updated configuration, grid data, and field data (without RF data)
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


def calculate_pseudopotential(magE, freq, gradient=False, stepsize=None, particle='e'):
    """
    Calculates the pseudopotential for a given particle in an RF electric field.

    Parameters:
        magE (ndarray): Magnitude of the electric field (in V/m). Can be a scalar or ndarray.
        freq (float): Frequency of the RF field (in Hz).
        gradient (bool, optional): If True, calculates the gradient of the pseudopotential to obtain the RF force.
                                   Defaults to False.
        stepsize (float or tuple, optional): The spacing between points in each dimension (in meters).
                                             Required if gradient is True.
        particle (str, optional): The type of particle. Supported particles are:
                                  - 'e' for electron (default)
                                  - 'ca40' for Calcium-40 ion
                                  If additional particles are needed, they can be added by editing the function.

    Returns:
        ndarray or tuple:
            - If gradient is False:
                - Ups (ndarray): The pseudopotential (in joules).
            - If gradient is True:
                - Ups (ndarray): The pseudopotential (in joules).
                - rf_force (dict): Dictionary containing the RF force components:
                    - 'x': Force component in the x-direction (in newtons).
                    - 'y': Force component in the y-direction (in newtons).
                    - 'z': Force component in the z-direction (in newtons).

    Raises:
        ValueError: If an unsupported particle is specified.
        ValueError: If stepsize is not provided when gradient is True.

    Example:
        # Calculate pseudopotential for an electron without gradient
        Ups = calculate_pseudopotential(magE, freq)

        # Calculate pseudopotential and RF force for a Calcium-40 ion with gradient
        Ups, rf_force = calculate_pseudopotential(magE, freq, gradient=True, stepsize=1e-4, particle='ca40')
    """

    # Define physical constants for supported particles
    if particle == 'e':
        # Electron
        m = 9.1093837e-31           # Mass of an electron (kg)
        q = -1.60217663e-19         # Charge of an electron (C)
    elif particle == 'ca40':
        # Calcium-40 ion (assuming singly ionized Ca+)
        m = 6.642156e-26            # Mass of Calcium-40 ion (kg) [40 * atomic mass unit]
        q = 1.60217663e-19          # Charge of a singly ionized ion (C)
    else:
        # Unsupported particle
        raise ValueError('Only particles "e" (electron) and "ca40" (Calcium-40 ion) are supported. '
                         'To add more particles, edit the function accordingly.')

    # Calculate the pseudopotential (in joules)
    # Ups = (q * |E|)^2 / (4 * m * (omega)^2)
    Ups = (q * magE) ** 2 / (4 * m * (2 * np.pi * freq) ** 2)

    if gradient:
        # Check if stepsize is provided
        if stepsize is None:
            raise ValueError('stepsize must be provided when gradient is True')

        # Initialize dictionary to hold RF force components
        rf_force = {}

        # Calculate the gradient of the pseudopotential
        # F = -∇Ups
        F = np.gradient(Ups, stepsize)

        # Store the negative gradient as the force components
        rf_force['x'] = -F[0]  # Force in x-direction (N)
        rf_force['y'] = -F[1]  # Force in y-direction (N)
        rf_force['z'] = -F[2]  # Force in z-direction (N)

        # Return both the pseudopotential and the RF force
        return Ups, rf_force

    else:
        # Return only the pseudopotential
        return Ups

    
def calculate_E_field(conf, voltages):
    """
    Calculates the electric field components for each zone based on voltage data.

    Parameters:
        conf (dict): Configuration dictionary for each zone.
            Each key is a zone name, and the value is a dictionary containing:
                - 'step': A tuple (dx, dy, dz) representing the step sizes along each axis for the zone.
        voltages (dict): Dictionary of voltage data arrays for each zone.
            Each key should match a zone in 'conf', and the value is a NumPy array representing the voltage field data for that zone.

    Returns:
        dict: A dictionary containing the electric field components for each zone.
            Structure:
                E[zone]['x']: Electric field component in the x-direction (ndarray).
                E[zone]['y']: Electric field component in the y-direction (ndarray).
                E[zone]['z']: Electric field component in the z-direction (ndarray).
    """

    E = {}  # Dictionary to hold electric field components for each zone

    # Iterate over each zone in the configuration
    for zone in conf:
        E[zone] = {}  # Initialize dictionary for the current zone

        # Extract the voltage data and step sizes for the current zone
        voltage_data = voltages[zone]          # Voltage field data array for the zone
        step_sizes = conf[zone]['step']        # Tuple of step sizes (dx, dy, dz) for the zone

        # Calculate the gradient of the voltage field (i.e., the electric field components)
        # np.gradient returns a list of arrays representing the gradient along each axis
        # We pass the step sizes to ensure accurate calculation of the derivatives
        field = np.gradient(voltage_data, *step_sizes)

        # The electric field components are the negative gradients of the potential
        E[zone]['x'] = -field[0]  # Electric field component in the x-direction
        E[zone]['y'] = -field[1]  # Electric field component in the y-direction
        E[zone]['z'] = -field[2]  # Electric field component in the z-direction

    return E  # Return the dictionary containing the electric field components for each zone


def construct_interp_funcs(data, grid, data_rf=None, grid_rf=None, fill_NaNs=False, method='cubic', particle='e', test_set=False):
    """
    Constructs interpolation functions for field data across multiple zones.

    Parameters:
        data (dict): Dictionary containing field data for each zone.
                     Structure: data[zone][coord], where 'coord' is 'x', 'y', or 'z'.
        grid (dict): Dictionary containing grid coordinates for each zone.
                     Structure:
                         grid[zone]['x'], grid[zone]['y'], grid[zone]['z']: 3D arrays of grid coordinates.
                         grid[zone]['unique_x'], grid[zone]['unique_y'], grid[zone]['unique_z']: 1D arrays of unique coordinates.
        data_rf (dict, optional): Dictionary containing RF field data (force, not field).
                                  Structure: data_rf[coord], where 'coord' is 'x', 'y', or 'z'.
        grid_rf (dict, optional): Dictionary containing grid coordinates for RF data.
                                  Structure: similar to 'grid' but for RF data.
        fill_NaNs (bool, optional): If True, fills NaN values in the field data using nearest neighbor interpolation.
                                    Defaults to False.
        method (str, optional): Interpolation method to use ('linear', 'nearest', 'cubic').
                                Defaults to 'cubic'.
        particle (str, optional): The type of particle, affecting the charge used in calculations.
                                  Supported particles are:
                                      - 'e' for electron (default)
                                      - 'ca40' for Calcium-40 ion
        test_set (bool, optional): If True, creates reference interpolation functions using method 'nearest' for testing.
                                   Defaults to False.

    Returns:
        Depending on the inputs and 'test_set' parameter, the function returns:

        - If 'data_rf' is None:
            - If 'test_set' is False:
                interp_funcs
            - If 'test_set' is True:
                interp_funcs, ref_funcs
        - If 'data_rf' is provided:
            - If 'test_set' is False:
                interp_funcs, rf_interp
            - If 'test_set' is True:
                interp_funcs, ref_funcs, rf_interp, ref_rf

        Where:
            interp_funcs (dict): Interpolation functions for each zone and coordinate.
                                 Structure: interp_funcs[zone][coord].
            ref_funcs (dict): Reference interpolation functions for each zone and coordinate (if 'test_set' is True).
            rf_interp (dict): Interpolation functions for RF data (field, not force).
            ref_rf (dict): Reference interpolation functions for RF data (if 'test_set' is True).

    Raises:
        ValueError: If an unsupported particle is specified.
        ValueError: If 'grid_rf' is not provided when 'data_rf' is provided.

    Example:
        # Construct interpolation functions without RF data
        interp_funcs = construct_interp_funcs(data, grid)

        # Construct interpolation functions with RF data and test set
        interp_funcs, ref_funcs, rf_interp, ref_rf = construct_interp_funcs(
            data, grid, data_rf=data_rf, grid_rf=grid_rf, fill_NaNs=True, test_set=True
        )
    """

    # Set particle charge based on the specified particle type
    if particle == 'e':
        q = -1.60217663e-19           # Charge of an electron (C)
    elif particle == 'ca40':
        q = 1.60217663e-19            # Charge of a singly ionized Calcium-40 ion (C)
    else:
        raise ValueError('Only particles "e" (electron) and "ca40" (Calcium-40 ion) are supported. '
                         'To add more particles, edit the function accordingly.')

    # Initialize dictionaries to hold interpolated data and functions
    interp_data = {}       # Holds field data after NaN filling (if applied)
    interp_funcs = {}      # Holds interpolation functions for each zone
    if test_set:
        ref_funcs = {}     # Holds reference interpolation functions for testing

    ############################################################
    # Step 1: Fill NaN values in the field data (if requested) #
    ############################################################
    if fill_NaNs:
        for zone in data:
            print(f'Filling NaNs in data for zone {zone} ...')
            interp_data[zone] = {}  # Initialize dictionary for this zone
            for coord in ('x', 'y', 'z'):
                start_time = time.time()
                # Fill NaN values using nearest neighbor interpolation
                interp_data[zone][coord] = fill_NaN_nearest(
                    data[zone][coord],
                    grid[zone]['x'],
                    grid[zone]['y'],
                    grid[zone]['z']
                )
                end_time = time.time()
                print(f'Time consumed for filling NaNs in {coord} component: {end_time - start_time:.4f} seconds.')
    else:
        interp_data = data  # Use the original data if no NaN filling is needed

    ###########################################################
    # Step 2: Create interpolation functions for each zone    #
    ###########################################################
    for zone in data:
        print(f'Creating interpolation functions for zone {zone} ...')
        interp_funcs[zone] = {}      # Initialize dictionary for this zone
        if test_set:
            ref_funcs[zone] = {}     # Initialize dictionary for reference functions

        for coord in ('x', 'y', 'z'):
            start_time = time.time()
            # Define the grid points as a tuple of 1D arrays
            grid_points = (
                grid[zone]['unique_x'],
                grid[zone]['unique_y'],
                grid[zone]['unique_z']
            )
            # Extract the field data for the current coordinate component
            values = interp_data[zone][coord]
            # Create the interpolation function using the specified method
            interp_funcs[zone][coord] = RegularGridInterpolator(
                grid_points, values, method=method
            )
            end_time = time.time()
            print(f'Time consumed for constructing interpolation function in {coord} component: {end_time - start_time:.4f} seconds.')

            # Create reference interpolation function if test_set is True
            if test_set:
                start_time = time.time()
                ref_funcs[zone][coord] = RegularGridInterpolator(
                    grid_points, values, method='nearest'
                )
                end_time = time.time()
                print(f'Time consumed for constructing reference function in {coord} component: {end_time - start_time:.4f} seconds.')

    ######################################################
    # Step 3: Handle RF data if provided                #
    ######################################################
    if data_rf is not None:
        if grid_rf is None:
            raise ValueError('Grid for RF must be provided if RF interpolation is desired.')

        data_rf_interp = {}   # Holds RF field data after NaN filling (if applied)
        rf_interp = {}        # Holds interpolation functions for RF data
        if test_set:
            ref_rf = {}       # Holds reference interpolation functions for RF data

        # Fill NaN values in RF data if requested
        if fill_NaNs:
            print('Filling NaNs in RF data ...')
            for coord in ('x', 'y', 'z'):
                start_time = time.time()
                # Fill NaN values and adjust for particle charge
                data_rf_interp[coord] = fill_NaN_nearest(
                    data_rf[coord] / q,
                    grid_rf['x'],
                    grid_rf['y'],
                    grid_rf['z']
                )
                end_time = time.time()
                print(f'Time consumed for filling NaNs in RF data ({coord} component): {end_time - start_time:.4f} seconds.')
        else:
            # Adjust RF data for particle charge
            for coord in ('x', 'y', 'z'):
                data_rf_interp[coord] = data_rf[coord] / q

        # Create interpolation functions for RF data
        print('Creating interpolation functions for RF data ...')
        for coord in ('x', 'y', 'z'):
            start_time = time.time()
            # Define the grid points for RF data
            grid_points_rf = (
                grid_rf['unique_x'],
                grid_rf['unique_y'],
                grid_rf['unique_z']
            )
            # Extract the RF field data for the current coordinate component
            values_rf = data_rf_interp[coord]
            # Create the interpolation function for RF data
            rf_interp[coord] = RegularGridInterpolator(
                grid_points_rf, values_rf, method=method
            )
            end_time = time.time()
            print(f'Time consumed for constructing RF interpolation function ({coord} component): {end_time - start_time:.4f} seconds.')

            # Create reference interpolation function for RF data if test_set is True
            if test_set:
                start_time = time.time()
                ref_rf[coord] = RegularGridInterpolator(
                    grid_points_rf, values_rf, method='nearest'
                )
                end_time = time.time()
                print(f'Time consumed for constructing RF reference function ({coord} component): {end_time - start_time:.4f} seconds.')

        # Return appropriate values based on test_set
        if test_set:
            return interp_funcs, ref_funcs, rf_interp, ref_rf
        else:
            return interp_funcs, rf_interp

    #############################################################
    # Step 4: Return interpolation functions (without RF data)  #
    #############################################################
    if test_set:
        return interp_funcs, ref_funcs
    else:
        return interp_funcs


#########################################################
###    Sampling Functions for Interpolation Testing   ###
#########################################################

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


#######################################
###    Field Evaluation Functions   ###
#######################################

def zone_identifier(point, zones, rf_range=None):
    """
    Determines which zone a given point belongs to, with zones of smaller stepsize taking priority.
    Optionally checks whether the point is within a specified RF range.

    Parameters:
        point (array-like): The (x, y, z) coordinates of the point as a list, tuple, or NumPy array.
        zones (dict): A dictionary of zones with their configurations. Each key is the zone name,
            and each value is a dictionary containing:
                - 'file' (optional for this function): The file path associated with the zone.
                - 'min': A tuple of minimum (x, y, z) coordinates defining the zone boundary.
                - 'max': A tuple of maximum (x, y, z) coordinates defining the zone boundary.
                - 'step': A tuple of stepsizes (dx, dy, dz) for the zone grid.
        rf_range (dict, optional): A dictionary specifying the RF range boundaries with keys:
                - 'min': A tuple of minimum (x, y, z) coordinates of the RF range.
                - 'max': A tuple of maximum (x, y, z) coordinates of the RF range.
            If provided, the function also checks whether the point is within this RF range.

    Returns:
        str or tuple:
            - If rf_range is None:
                Returns the name of the zone to which the point belongs.
            - If rf_range is provided:
                Returns a tuple (zone_name, in_rf_range), where 'in_rf_range' is a boolean
                indicating whether the point is within the RF range.

    Raises:
        ValueError: If the point does not belong to any of the defined zones.

    Example:
        >>> zones = {
        ...     'c1': {'min': (-0.002, -0.002, -0.001), 'max': (0.002, 0.002, 0.002), 'step': (2e-05, 2e-05, 2e-05)},
        ...     'c2': {'min': (-0.001, -0.001, -0.001), 'max': (0.001, 0.001, 0.001), 'step': (1e-05, 1e-05, 1e-05)},
        ...     'c3': {'min': (-0.0005, -0.0005, -0.0005), 'max': (0.0005, 0.0005, 0.0005), 'step': (5e-06, 5e-06, 5e-06)},
        ...     # Add other zones as needed
        ... }
        >>> point = [0.0003, 0.0003, 0.0003]
        >>> zone = zone_identifier(point, zones)
        >>> print(f'Point {point} belongs to zone: {zone}')
        Point [0.0003, 0.0003, 0.0003] belongs to zone: c3

        >>> rf_range = {
        ...     'min': (-0.001, -0.001, -0.001),
        ...     'max': (0.001, 0.001, 0.001)
        ... }
        >>> zone, in_rf_range = zone_identifier(point, zones, rf_range=rf_range)
        >>> print(f'Point {point} belongs to zone: {zone}, within RF range: {in_rf_range}')
        Point [0.0003, 0.0003, 0.0003] belongs to zone: c3, within RF range: True
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
            if rf_range is None:
                return zone_name  # Return the name of the zone
            else:
                # Check if the point lies within the RF range
                min_rf = np.array(rf_range['min'])
                max_rf = np.array(rf_range['max'])
                if np.all(point >= min_rf) and np.all(point <= max_rf):
                    return zone_name, True  # Point is within the RF range
                else:
                    return zone_name, False  # Point is outside the RF range

    # If the point does not belong to any zone, raise an error
    raise ValueError(f"Point {point} does not belong to any zone.")


def field_at_point(point, zones, interp, rf_interp=None, rf_range=None, ref=None):
    """
    Evaluates the interpolated field values at a given point.

    Parameters:
        point (array-like): The (x, y, z) coordinates of the point where the field is to be evaluated.
        zones (dict): Configuration dictionary containing zone definitions. Used by 'zone_identifier'
                      to determine which zone the point belongs to.
        interp (dict): Dictionary of interpolation functions for each zone and coordinate component.
                       Structure: interp[zone]['x'], interp[zone]['y'], interp[zone]['z'],
                       where each is a callable interpolation function.
        rf_interp (dict, optional): Dictionary of RF interpolation functions for coordinate components.
                                    Structure: rf_interp['x'], rf_interp['y'], rf_interp['z'],
                                    where each is a callable interpolation function. Defaults to None.
        rf_range (dict, optional): Dictionary specifying the RF range boundaries with keys 'min' and 'max'.
                                   Required if rf_interp is provided. Defaults to None.
        ref (dict, optional): Dictionary of reference interpolation functions for comparison.
                              Has the same structure as 'interp'. Defaults to None.

    Returns:
        numpy.ndarray or tuple:
            - If only 'interp' is provided:
                field_values (numpy.ndarray): The interpolated field values at the point, as a NumPy array
                                              [value_x, value_y, value_z]. If the point is outside all zones,
                                              returns an array of zeros [0.0, 0.0, 0.0].
            - If 'rf_interp' is provided:
                (field_values, rf_field_values): Tuple containing two NumPy arrays:
                    - field_values (numpy.ndarray): The interpolated field values at the point.
                    - rf_field_values (numpy.ndarray): The RF field values at the point if within RF range;
                                                       otherwise, an array of zeros [0.0, 0.0, 0.0].
            - If 'ref' is provided (and 'rf_interp' is not provided):
                (field_values, ref_values): Tuple containing two NumPy arrays:
                    - field_values (numpy.ndarray): The interpolated field values at the point.
                    - ref_values (numpy.ndarray): The reference interpolated field values at the point.

    Note:
        - If the point is outside all zones, the function returns zero field values instead of raising an error.
        - If both 'rf_interp' and 'ref' are provided, the function will return 'field_values' and 'rf_field_values',
          and will not evaluate or return 'ref_values'.

    Raises:
        ValueError: If 'rf_range' is not provided when 'rf_interp' is provided.

    Example:
        # Assuming 'interp_funcs' and 'rf_interp_funcs' are dictionaries of interpolation functions
        point = [0.001, 0.002, 0.003]

        # Without RF or reference functions
        field_values = field_at_point(point, zones_conf, interp_funcs)
        # field_values is a NumPy array: [value_x, value_y, value_z]

        # With RF interpolation functions
        field_values, rf_field_values = field_at_point(
            point, zones_conf, interp_funcs, rf_interp=rf_interp_funcs, rf_range=rf_range
        )
        # rf_field_values is a NumPy array: [value_rf_x, value_rf_y, value_rf_z] or [0.0, 0.0, 0.0]

        # With reference functions
        field_values, ref_values = field_at_point(point, zones_conf, interp_funcs, ref=ref_funcs)
        # ref_values is a NumPy array: [ref_x, ref_y, ref_z]
    """
    import numpy as np

    # Determine the zone to which the point belongs
    if rf_interp is None:
        # No RF interpolation; determine the zone normally
        try:
            zone = zone_identifier(point, zones)
        except ValueError:
            # Point is outside all zones; return zero field values
            field_values = np.array([0.0, 0.0, 0.0])
            return field_values
    else:
        # RF interpolation is provided; ensure rf_range is also provided
        if rf_range is None:
            raise ValueError('rf_range must be provided if you wish to evaluate RF field.')
        # Determine the zone and whether the point is within the RF range
        try:
            zone, in_rf_range = zone_identifier(point, zones, rf_range)
        except ValueError:
            # Point is outside all zones; return zero field values for both DC and RF fields
            field_values = np.array([0.0, 0.0, 0.0])
            rf_field_values = np.array([0.0, 0.0, 0.0])
            return field_values, rf_field_values

    # Evaluate the interpolated field values at the point for each coordinate component
    # Each interpolation function returns an array; we extract the first element [0]
    value_x = interp[zone]['x'](point)[0]  # Field value in x-direction
    value_y = interp[zone]['y'](point)[0]  # Field value in y-direction
    value_z = interp[zone]['z'](point)[0]  # Field value in z-direction

    field_values = np.array((value_x, value_y, value_z))

    # If RF interpolation functions are provided
    if rf_interp is not None:
        if in_rf_range:
            # Evaluate the RF interpolated field values at the point
            value_rf_x = rf_interp['x'](point)[0]
            value_rf_y = rf_interp['y'](point)[0]
            value_rf_z = rf_interp['z'](point)[0]
            rf_field_values = np.array((value_rf_x, value_rf_y, value_rf_z))
        else:
            # Point is outside the RF range; RF field values are zero
            rf_field_values = np.array((0.0, 0.0, 0.0))
        return field_values, rf_field_values

    # If reference interpolation functions are provided (and RF interpolation is not provided)
    if ref is not None:
        # Evaluate the reference field values at the point for comparison
        ref_x = ref[zone]['x'](point)[0]  # Reference field value in x-direction
        ref_y = ref[zone]['y'](point)[0]  # Reference field value in y-direction
        ref_z = ref[zone]['z'](point)[0]  # Reference field value in z-direction

        ref_values = np.array((ref_x, ref_y, ref_z))
        return field_values, ref_values

    # Return only the field values if no RF or reference functions are provided
    return field_values


############################################
###    Trajectory Simulation Functions   ###
############################################

def solve_traj(t, y, dc, rf, zones, rf_range, particle='e'):
    """
    Computes the derivative of the state vector y at time t for use with solve_ivp.

    Parameters:
        t (float): The current time (not used in this function but required by solve_ivp).
        y (array-like): The current state vector, where y[:3] are position coordinates (x, y, z),
                        and y[3:] are velocity components (vx, vy, vz).
        dc (dict): Dictionary of DC interpolation functions.
        rf (dict): Dictionary of RF interpolation functions.
        zones (dict): Configuration dictionary containing zone definitions.
        rf_range (dict): Dictionary specifying the RF range boundaries with keys 'min' and 'max'.
        particle (str, optional): The type of particle. Supported particles are:
                                  - 'e' for electron (default)
                                  - 'ca40' for Calcium-40 ion
                                  Defaults to 'e'.

    Returns:
        ndarray: The derivative of the state vector y, where dy[:3] = velocity,
                 and dy[3:] = acceleration.

    Raises:
        ValueError: If an unsupported particle is specified.
        ValueError: If 'rf_range' is not provided when 'rf' is provided.
    """
    import numpy as np

    # Set charge-to-mass ratio based on the specified particle type
    if particle == 'e':
        q = -1.60217663e-19         # Charge of an electron (C)
        m = 9.1093837e-31           # Mass of an electron (kg)
    elif particle == 'ca40':
        q = 1.60217663e-19          # Charge of a singly ionized Calcium-40 ion (C)
        m = 6.642156e-26            # Mass of Calcium-40 ion (kg) [40 * atomic mass unit]
    else:
        raise ValueError('Only particles "e" (electron) and "ca40" (Calcium-40 ion) are supported. '
                         'To add more particles, edit the function accordingly.')

    qtom = q / m  # Charge-to-mass ratio

    # Initialize derivative vector
    dy = np.zeros(len(y))
    position = y[:3]
    velocity = y[3:]
    #print(position)

    # The derivative of position is velocity
    dy[:3] = velocity

    # Get the electric fields at the current position
    # Corrected 'rf_ramge' to 'rf_range'
    Edc, Erf = field_at_point(position, zones, dc, rf_interp=rf, rf_range=rf_range)

    #print(Edc, Erf)
    # The derivative of velocity is acceleration: a = (q/m) * E
    dy[3:] = qtom * (Edc + Erf)

    return dy


def hit_boundary(t, y, dc, rf, zones, rf_range, particle='e'):
    """
    Event function to detect when a particle hits the boundary of the simulation domain.

    Parameters:
        t (float): Current time.
        y (array-like): State vector [x, y, z, vx, vy, vz].
        dc, rf, rf_range: Additional arguments passed via `args`, not used in this function.
        zones (dict): Dictionary containing zone configurations.

    Returns:
        float: Minimum distance to the simulation domain boundary.
    
    Note:
        - This event function has `terminal` attribute set to `True` by default, meaning
          the integration will stop when this event is detected.
        - The `direction` attribute can also be set here if needed.
    """
    import numpy as np

    # Extract the particle's current position
    xpos, ypos, zpos = y[:3]

    # Initialize global minimum and maximum coordinates
    min_coords = np.array([np.inf, np.inf, np.inf])
    max_coords = np.array([-np.inf, -np.inf, -np.inf])

    # Compute global minimum and maximum coordinates across all zones
    for zone_data in zones.values():
        min_coords = np.minimum(min_coords, zone_data['min'])
        max_coords = np.maximum(max_coords, zone_data['max'])

    # Calculate distances to each boundary
    dx1 = xpos - min_coords[0]
    dx2 = max_coords[0] - xpos
    dy1 = ypos - min_coords[1]
    dy2 = max_coords[1] - ypos
    dz1 = zpos - min_coords[2]
    dz2 = max_coords[2] - zpos

    # Determine the minimum distance to any boundary
    min_distance = min(dx1, dx2, dy1, dy2, dz1, dz2)

    return min_distance

# Set the terminal attribute
hit_boundary.terminal = True


def hit_mcp(t, y, dc, rf, zones, rf_range, particle='e'):
    """
    Event function to detect when a particle crosses the MCP plane at z = 20 mm,
    within x and y ranges of [-5 mm, 5 mm].

    Parameters:
        t (float): Current time (required by solve_ivp but not used here).
        y (array-like): Current state vector [x, y, z, vx, vy, vz].
        dc, rf, rf_range: Additional arguments passed via `args` (not used in this function).
        zones (dict): Zones configuration dictionary (not used here).

    Returns:
        float: Difference between the particle's z-position and the MCP plane z-position.

    Note:
        - This function returns a positive value when the particle is below the MCP plane,
          zero when it is exactly at z = 20 mm, and negative when it is above the MCP plane.
        - The event is only considered valid when the particle's x and y positions are within
          the specified ranges of [-5 mm, 5 mm]. Otherwise, the function returns a positive
          value to prevent triggering the event.
    """
    import numpy as np

    # Extract position components
    x_pos, y_pos, z_pos = y[0], y[1], y[2]

    # Convert positions to millimeters for comparison
    x_mm = x_pos * 1e3
    y_mm = y_pos * 1e3
    z_mm = z_pos * 1e3

    # Define MCP plane z-position in millimeters
    z_mcp = 20.0  # z = 20 mm

    # Check if x and y are within the MCP area
    if -5.0 <= x_mm <= 5.0 and -5.0 <= y_mm <= 5.0:
        # Particle is within the MCP area; return the difference in z
        return z_mm - z_mcp
    else:
        # Particle is outside the MCP area; return a positive value to prevent event triggering
        # The positive value ensures the event function does not cross zero
        return 1.0  # Arbitrary positive value

# Set event attributes
hit_mcp.terminal = False  # Do not terminate the integration when this event occurs
hit_mcp.direction = 1     # Detect only when z increases through z_mcp (from below to above)


def sample_initial_conditions(N, sigma_pos, sigma_velo):
    """
    Generates initial conditions for N particles with positions and velocities sampled from normal distributions.

    Parameters:
        N (int): Number of particles.
        sigma_pos (float or array-like): Standard deviation(s) for initial positions.
            - If float, the same standard deviation is used for all axes.
            - If array-like (tuple, list, or NumPy array) of length 3, specifies (sigma_x, sigma_y, sigma_z).
        sigma_velo (float or array-like): Standard deviation(s) for initial velocities.
            - Same format as sigma_pos.

    Returns:
        numpy.ndarray: An array of shape (N, 6) where:
            - particles[:, :3] are positions (x, y, z).
            - particles[:, 3:] are velocities (vx, vy, vz).

    Raises:
        ValueError: If sigma_pos or sigma_velo are not a float or array-like of length 3.
    """
    import numpy as np

    # Handle sigma_pos
    if isinstance(sigma_pos, (tuple, list, np.ndarray)):
        if len(sigma_pos) != 3:
            raise ValueError("sigma_pos must be a float or an array-like of length 3.")
        sigma_x, sigma_y, sigma_z = sigma_pos
    elif isinstance(sigma_pos, (int, float)):
        sigma_x = sigma_y = sigma_z = sigma_pos
    else:
        raise ValueError("sigma_pos must be a float or an array-like of length 3.")

    # Handle sigma_velo
    if isinstance(sigma_velo, (tuple, list, np.ndarray)):
        if len(sigma_velo) != 3:
            raise ValueError("sigma_velo must be a float or an array-like of length 3.")
        sigma_vx, sigma_vy, sigma_vz = sigma_velo
    elif isinstance(sigma_velo, (int, float)):
        sigma_vx = sigma_vy = sigma_vz = sigma_velo
    else:
        raise ValueError("sigma_velo must be a float or an array-like of length 3.")

    # Initialize the particles array
    particles = np.zeros((N, 6))

    # Sample positions from normal distributions
    particles[:, 0] = np.random.normal(0, sigma_x, N)  # x positions
    particles[:, 1] = np.random.normal(0, sigma_y, N)  # y positions
    particles[:, 2] = np.random.normal(0, sigma_z, N)  # z positions

    # Sample velocities from normal distributions
    particles[:, 3] = np.random.normal(0, sigma_vx, N)  # x velocities
    particles[:, 4] = np.random.normal(0, sigma_vy, N)  # y velocities
    particles[:, 5] = np.random.normal(0, sigma_vz, N)  # z velocities

    return particles
    

def sample_and_solve_trajectories(N, sp, sv, dc, rf, zones, rf_range, particle='e', tf=1e-8, max_step=5e-11, method='RK45', dense_output=True, save_trajectories=True):
    """
    Simulates the trajectories of N particles with initial conditions sampled from normal distributions.

    Parameters:
        N (int): Number of particles to simulate.
        sp (float or array-like): Standard deviation(s) for initial positions.
            - If float, the same standard deviation is used for all axes.
            - If array-like (tuple, list, or NumPy array) of length 3, specifies (sigma_x, sigma_y, sigma_z).
        sv (float or array-like): Standard deviation(s) for initial velocities.
            - Same format as sp.
        dc: DC field interpolation functions.
        rf: RF field interpolation functions.
        zones (dict): Zones configuration dictionary.
        rf_range (dict): RF field range dictionary.
        particle (str): Particle type ('e' for electron). Default is 'e'.
        tf (float): Final time for the simulation. Default is 1e-8 seconds.
        max_step (float): Maximum step size for the solver. Default is 5e-11 seconds.
        method (str): Integration method for solve_ivp. Default is 'RK45'.
        dense_output (bool): Whether to compute a continuous solution. Default is True.
        save_trajectories (bool): Whether to save the entire trajectory for each particle. Default is True.

    Returns:
        dict: A dictionary containing simulation results with keys:
            - 'trajectories': List of dictionaries for each particle containing:
                - 't': Array of time points.
                - 'y': Array of state vectors at each time point.
              (Only included if save_trajectories is True)
            - 'initial_conditions': Array of initial positions and velocities for each particle.
            - 'final_positions': Array of final positions for each particle.
            - 'final_velocities': Array of final velocities for each particle.
            - 't_mcp': List of times when particles hit the MCP (or None if they did not).
            - 'state_at_mcp': List of state vectors when particles hit the MCP (or None if they did not).
            - 't_boundary': List of times when particles hit the boundary (or None if they did not).
            - 'num_hit_mcp': Number of particles that hit the MCP.
            - 'conf': Dictionary of the parameters passed into the function.
    """

    print('Generating initial conditions for particles ...')
    # Generate initial positions and velocities for N particles
    particles = sample_initial_conditions(N, sp, sv)

    print('Simulating particle trajectories ...')
    start_time = time.time()
    t0 = 0.0

    # Initialize lists and arrays to store results
    if save_trajectories:
        trajectories = []  # To store the trajectory (t and y) for each particle
    else:
        trajectories = None  # Not storing trajectories

    initial_conditions = particles.copy()  # Store initial conditions
    final_positions = np.zeros((N, 3))     # To store final positions
    final_velocities = np.zeros((N, 3))    # To store final velocities
    t_mcp = [None] * N                     # To store times when particles hit the MCP
    state_at_mcp = [None] * N              # To store states when particles hit the MCP
    t_boundary = [None] * N                # To store times when particles hit the boundary

    num_hit_mcp = 0                        # Counter for particles that hit the MCP

    #for i in tqdm(range(N)):
    for i in range(N):
        y0 = particles[i]  # Initial condition for particle i

        # Print progress every 100 particles, refreshing the same line
        #if i % 100 == 0:
        print(f'\rSolving particles {i+1}/{N} ...', end='')
        sys.stdout.flush()

        # Solve the trajectory using solve_ivp
        sol = solve_ivp(
            solve_traj,
            t_span=(t0, tf),
            y0=y0,
            args=(dc, rf, zones, rf_range, particle),
            max_step=max_step,
            events=(hit_boundary, hit_mcp),
            method=method,
            dense_output=dense_output
        )

        # Store the trajectory or initial/final states
        if save_trajectories:
            # Store the entire trajectory for particle i
            trajectory = {
                't': sol.t,        # Array of time points
                'y': sol.y         # Array of state vectors [x, y, z, vx, vy, vz] at each time point
            }
            trajectories.append(trajectory)
        else:
            # Store only final positions and velocities
            final_positions[i] = sol.y[:3, -1]    # Final position
            final_velocities[i] = sol.y[3:, -1]   # Final velocity

        # Check for events (hit_boundary and hit_mcp)
        t_events = sol.t_events  # List of arrays of event times
        # hit_boundary is the first event function, hit_mcp is the second
        if t_events[0].size > 0:
            # Particle hit the boundary
            t_boundary[i] = t_events[0][0]    # Record time when boundary was hit

        if t_events[1].size > 0:
            # Particle hit the MCP
            t_mcp[i] = t_events[1][0]         # Record time when MCP was hit
            num_hit_mcp += 1                  # Increment counter

            # Get the state at the time when particle hits MCP
            state_at_mcp[i] = sol.sol(t_mcp[i])  # State vector at MCP hit time
        else:
            state_at_mcp[i] = None  # Particle did not hit the MCP

    # Ensure we move to the next line after progress
    print('')

    end_time = time.time()
    print(f'Time consumed for calculating {N} trajectories: {end_time - start_time:.4f} seconds.')

    # Save configuration parameters
    conf = {
        'N': N,
        'sp': sp,
        'sv': sv,
        'dc': dc,
        'rf': rf,
        'zones': zones,
        'rf_range': rf_range,
        'particle': particle,
        'tf': tf,
        'max_step': max_step,
        'method': method,
        'dense_output': dense_output,
        'save_trajectories': save_trajectories
    }

    # Return the results as a dictionary
    results = {
        'initial_conditions': initial_conditions,
        'final_positions': final_positions,
        'final_velocities': final_velocities,
        't_mcp': t_mcp,
        'state_at_mcp': state_at_mcp,
        't_boundary': t_boundary,
        'num_hit_mcp': num_hit_mcp,
        'conf': conf
    }

    if save_trajectories:
        results['trajectories'] = trajectories

    return results
