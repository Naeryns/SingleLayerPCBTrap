import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
import time

#####################################################
#####          Setup Basic Parameters          #####
#####################################################

if __name__ == '__main__':
    # File paths for field data, starting from 'SingleLayerPCBTrap\Ansys' folder
    DC_file = 'Outputs_DC\\Mesh_MCP_Global\\Mesh.fld'  # DC potential field data file
    RF_file = 'Outputs_RF\\TrajSimTest.fld'           # RF magnitude of electric field data file

    # Grid positions for error sampling (ranges in meters)
    xrange = [-0.00795, 0.00795]  # x-axis range
    yrange = [-0.00645, 0.00645]  # y-axis range
    zrange = [-0.00095, 0.00995]  # z-axis range
    xstep = 500                   # Number of steps along x-axis
    ystep = 300                   # Number of steps along y-axis
    zstep = 300                   # Number of steps along z-axis

    # Threshold for identifying large interpolation errors
    threshold = 100

    # Printing setup for error analysis output
    print_points = True           # Whether to print detailed error information
    print_lines = 5               # Number of error points to print (set to a high number to print all)
    
    # Plotting setup
    plot_large_errors = True      # Whether to plot the large error points
    plot_sample_size = np.iinfo(np.int64).max       # Number of points to plot (to avoid overcrowding)

##########################################
#####          Data Readout          #####
##########################################

def Grid_from_header(header):
    """
    Generates coordinate grids from the header of an Ansys .fld file.

    Parameters:
        header (str): The header line from the .fld file containing grid metadata.

    Returns:
        tuple: A tuple containing:
            - xs (ndarray): 3D array of x-coordinates.
            - ys (ndarray): 3D array of y-coordinates.
            - zs (ndarray): 3D array of z-coordinates.
            - nx (int): Number of grid points along the x-axis.
            - ny (int): Number of grid points along the y-axis.
            - nz (int): Number of grid points along the z-axis.

    Raises:
        ValueError: If the header cannot be parsed to extract grid information.
    """
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

def Vec_from_file(filename):
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
    xs, ys, zs, nx, ny, nz = Grid_from_header(header)

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

def Scalar_from_file(filename):
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
    xs, ys, zs, nx, ny, nz = Grid_from_header(header)

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

if __name__ == '__main__':
    # Data Readout
    prefix = '..\\Ansys\\'
    # Read DC potential data and coordinate grids
    xdc, ydc, zdc, Udc = Scalar_from_file(prefix + DC_file)
    # Read RF magnitude of electric field data and coordinate grids
    xrf, yrf, zrf, magE = Scalar_from_file(prefix + RF_file)

#########################################################
#####          Calculate Equivalent Fields          #####
#########################################################

'''
This section performs calculations to obtain the pseudopotential from the RF
electric field magnitude and computes the equivalent electric field by taking
the gradient of the pseudopotential. It also computes the DC electric field
by taking the gradient of the DC potential.
'''

if __name__ == '__main__':
    # Physical constants and simulation parameters
    m = 9.1093837e-31         # Mass of an electron (kg)
    q = -1.60217663e-19       # Charge of an electron (C)
    freq = 1.36e+09           # RF frequency (Hz)
    stepsize = 1e-4           # Grid spacing for gradient calculation (m)

    # Calculate RF pseudopotential (in joules)
    Ups = (q * magE) ** 2 / (4 * m * (2 * np.pi * freq) ** 2)

    # Calculate the gradient of the pseudopotential to obtain the RF force
    F_rf = np.gradient(Ups, stepsize)
    # Compute the equivalent RF electric field components (E = -F/q)
    Ex_rf = -F_rf[0] / q
    Ey_rf = -F_rf[1] / q
    Ez_rf = -F_rf[2] / q

    # Calculate the gradient of the DC potential to obtain the DC electric field
    E_dc = np.gradient(Udc, stepsize)
    Ex_dc = -E_dc[0]
    Ey_dc = -E_dc[1]
    Ez_dc = -E_dc[2]

    # Note: The DC electric field components are already in the correct form since E = -∇V

#############################################################
#####          Create Interpolation for Fields          #####
#############################################################

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

if __name__ == '__main__':
    # Generate unique coordinate arrays for interpolation
    xrf_unique = np.unique(xrf)
    yrf_unique = np.unique(yrf)
    zrf_unique = np.unique(zrf)
    xdc_unique = np.unique(xdc)
    ydc_unique = np.unique(ydc)
    zdc_unique = np.unique(zdc)

    # Fill NaN values in the RF electric field data
    Ex_rf_filled = fill_NaN_nearest(Ex_rf, xrf, yrf, zrf)

    # Create interpolation functions for the RF electric field component Ex_rf
    # Linear interpolation function
    Exrf_interp_func = RegularGridInterpolator(
        (xrf_unique, yrf_unique, zrf_unique), Ex_rf_filled, method='linear'
    )
    # Nearest neighbor interpolation function (for comparison)
    test_interp_func = RegularGridInterpolator(
        (xrf_unique, yrf_unique, zrf_unique), Ex_rf_filled, method='nearest'
    )

    # Generate grid points where the interpolation will be evaluated
    x_vals = np.linspace(xrange[0], xrange[1], xstep)
    y_vals = np.linspace(yrange[0], yrange[1], ystep)
    z_vals = np.linspace(zrange[0], zrange[1], zstep)
    x_grid, y_grid, z_grid = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    # Flatten the grid arrays to create a list of points
    points = np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel()))

    # Evaluate the interpolation functions at the grid points
    start_time = time.time()
    linear_values = Exrf_interp_func(points)
    end_time = time.time()
    print(f'Linear interpolation computation time: {end_time - start_time} seconds.')

    start_time = time.time()
    nearest_values = test_interp_func(points)
    end_time = time.time()
    print(f'Nearest interpolation computation time: {end_time - start_time} seconds.')

####################################################
#####          Analyze Interpolations          #####
####################################################

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
    Ensures that the adjacent points are within the grid boundaries.
    
    Parameters:
        point (ndarray): The coordinate [x, y, z] of the point of interest.
        offset (float or list/tuple of floats): The offset distance(s) to adjacent points.
            - If a single float is provided, the same offset is used in all directions.
            - If a list or tuple of three floats is provided, they specify the offsets
              in the x, y, and z directions, respectively.
        x_range (tuple, optional): The (min, max) values of the x-axis grid.
        y_range (tuple, optional): The (min, max) values of the y-axis grid.
        z_range (tuple, optional): The (min, max) values of the z-axis grid.
    
    Returns:
        list: A list of coordinates of the adjacent points within the grid boundaries.
    
    Raises:
        ValueError: If any of the adjacent points fall outside the grid boundaries.
    """
    # Default grid ranges from the 'Setup Basic Parameters' section
    if x_range is None:
        x_range = xrange  # xrange defined in the 'Setup Basic Parameters'
    if y_range is None:
        y_range = yrange  # yrange defined in the 'Setup Basic Parameters'
    if z_range is None:
        z_range = zrange  # zrange defined in the 'Setup Basic Parameters'
    
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
    
    adjacent_points = []
    for direction in directions:
        adj_point = point + direction
        # Check if the adjacent point is within the grid boundaries
        if (x_range[0] <= adj_point[0] <= x_range[1] and
            y_range[0] <= adj_point[1] <= y_range[1] and
            z_range[0] <= adj_point[2] <= z_range[1]):
            adjacent_points.append(adj_point)
        else:
            raise ValueError(f"Adjacent point {adj_point} is outside the grid boundaries.")
    
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

if __name__ == '__main__':
    # Calculate relative errors between linear and nearest neighbor interpolation
    errors = np.abs(linear_values - nearest_values) / np.abs(nearest_values)
    print(f'Number of NaNs in error array: {np.sum(np.isnan(errors))}')

    # Filter out NaN errors to compute statistics
    valid_mask = ~np.isnan(errors)
    valid_errors = errors[valid_mask]
    valid_points = points[valid_mask]

    # Compute error statistics
    max_error = np.max(valid_errors)
    mean_error = np.mean(valid_errors)
    std_error = np.std(valid_errors)
    print(f'Maximum Relative Error: {max_error}')
    print(f'Average Relative Error: {mean_error}')
    print(f'Standard Deviation of Relative Error: {std_error}')

    # Identify points with large relative errors
    large_err_points = find_large_errors(valid_errors, threshold)
    print(f'Number of points with relative error greater than {threshold}: {len(large_err_points)}\n')

    # Optionally print detailed information about points with large errors
    if print_points:
        for idx in large_err_points[:print_lines]:
            point = valid_points[idx]
            print(f'Point {idx}: Coordinate {point}, '
                  f'Relative Error {errors[idx]}, '
                  f'Linear Interpolation {linear_values[idx]}, '
                  f'Nearest Value {nearest_values[idx]}')

    # Investigate the adjacent of a picked large error point, change index of the point of interest if getting a ValueError
    pt = points[large_err_points[1000]]
    print(f'\nInvestigating Point {pt}, Linear Interpolation: {Exrf_interp_func(pt)[0]}, Nearest Value: {test_interp_func(pt)[0]}')
    adj_pts = find_adjacent_points(pt, offset=2e-5)
    adj_linear = Exrf_interp_func(adj_pts)
    adj_nearest = test_interp_func(adj_pts)
    for ind, p in enumerate(adj_pts):
        print(f'Adjacent Point: {p}, Linear Interpolation: {adj_linear[ind]}, Nearest Value: {adj_nearest[ind]}')
    
    print('\n')
    # Investigate the adjacent grid points of a picked large error point
    adj_inds, adj_grids = find_adjacent_grids(pt, xrf_unique, yrf_unique, zrf_unique)
    for i in range(len(adj_inds)):
        print(f'Adjacent Grid Point: {adj_grids[i]} (Index {adj_inds[i]})\n\tField Data: {Ex_rf[adj_inds[i]]}, Filled: {Ex_rf_filled[adj_inds[i]]}\n')
        
############################################################
#####          Plot Large Error Points in 3D          #####
############################################################

#import plotly.graph_objs as go  # Import plotly for interactive plotting
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting in matplotlib

if __name__ == '__main__':
    
    '''
    if plot_large_errors and len(large_err_points) > 0:
        # Prepare data for plotting
        # Limit the number of points to plot to avoid performance issues
        sample_size = min(plot_sample_size, len(large_err_points))
        sampled_indices = np.random.choice(large_err_points, size=sample_size, replace=False)
        sampled_points = valid_points[sampled_indices]
        sampled_errors = valid_errors[sampled_indices]

        # Create a scatter plot using plotly
        trace = go.Scatter3d(
            x=sampled_points[:, 0],
            y=sampled_points[:, 1],
            z=sampled_points[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=sampled_errors,          # Color by error magnitude
                colorscale='Viridis',
                colorbar=dict(title='Relative Error'),
                opacity=0.8
            )
        )

        layout = go.Layout(
            title='Points with Large Interpolation Errors',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)'
            ),
            margin=dict(l=0, r=0, b=0, t=50)
        )

        fig = go.Figure(data=[trace], layout=layout)
        fig.show()

    else:
        print('No large error points to plot or plotting disabled.')
    '''
    
    if plot_large_errors and len(large_err_points) > 0:
        # Prepare data for plotting
        # Limit the number of points to plot to avoid performance issues
        sample_size = min(plot_sample_size, len(large_err_points))
        sampled_indices = np.random.choice(large_err_points, size=sample_size, replace=False)
        sampled_points = valid_points[sampled_indices]
        sampled_errors = valid_errors[sampled_indices]

        # Create a 3D scatter plot using matplotlib
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Normalize errors for coloring
        norm = plt.Normalize(sampled_errors.min(), sampled_errors.max())
        colors = plt.cm.viridis(norm(sampled_errors))

        scatter = ax.scatter(
            sampled_points[:, 0],
            sampled_points[:, 1],
            sampled_points[:, 2],
            c=sampled_errors,
            cmap='viridis',
            marker='.',
            s=20,
            alpha=0.8
        )

        # Add a color bar to show error magnitude
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('Relative Error')

        # Set labels
        ax.set_title('Points with Large Interpolation Errors')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

        # Show the plot
        plt.show()
    else:
        print('No large error points to plot or plotting disabled.')
        
############################################################
#####          Plot 3D Histograms of Projections      #####
############################################################

if plot_large_errors and len(large_err_points) > 0:
    # Prepare data for plotting
    # Limit the number of points to plot to avoid performance issues
    sample_size = min(plot_sample_size, len(large_err_points))
    sampled_indices = np.random.choice(large_err_points, size=sample_size, replace=False)
    sampled_points = valid_points[sampled_indices]
    sampled_errors = valid_errors[sampled_indices]
    
    # Define number of bins for the histograms
    num_bins = 30  # Adjust as needed for resolution
    
    # Create 3D histograms for XY, YZ, and ZX planes
    projections = [('XY', 0, 1), ('YZ', 1, 2), ('ZX', 2, 0)]
    
    for plane, idx1, idx2 in projections:
        # Extract the two dimensions for the projection
        x = sampled_points[:, idx1]
        y = sampled_points[:, idx2]
        
        # Compute 2D histogram
        hist, xedges, yedges = np.histogram2d(x, y, bins=num_bins)
        
        # Construct arrays for plotting
        x_pos, y_pos = np.meshgrid(
            (xedges[:-1] + xedges[1:]) / 2,
            (yedges[:-1] + yedges[1:]) / 2,
            indexing='ij'
        )
        x_pos = x_pos.ravel()
        y_pos = y_pos.ravel()
        z_pos = np.zeros_like(x_pos)
        
        # The histogram values as heights
        dz = hist.ravel()
        
        # Filter out zero heights to reduce plotting time
        nonzero = dz > 0
        x_pos = x_pos[nonzero]
        y_pos = y_pos[nonzero]
        z_pos = z_pos[nonzero]
        dz = dz[nonzero]
        
        # Width and depth of the bars
        dx = (xedges[1] - xedges[0]) * np.ones_like(z_pos)
        dy = (yedges[1] - yedges[0]) * np.ones_like(z_pos)
        
        # Create 3D bar plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, shade=True)
        
        # Set labels and title
        ax.set_xlabel(f'{plane[0]} (m)')
        ax.set_ylabel(f'{plane[1]} (m)')
        ax.set_zlabel('Counts')
        ax.set_title(f'3D Histogram of {plane} Projection of Large Error Points')
        
        plt.show()
else:
    print('No large error points to plot or plotting disabled.')