import numpy as np
import matplotlib.pyplot as plt
#from mayavi import mlab
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from stl import mesh
from plyfile import PlyData
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial as spatial
from sklearn.neighbors import KDTree
import vg
import time
import ipywidgets as widgets
from IPython.display import display

print("starting functions")

def find_radius_3d(point1, point2, point3):
    """
    Returns the center and radius of the circumsphere of a triangle in 3D space.
    The circumsphere of a triangle is the sphere that passes through all three vertices of the triangle.
    The function uses the formula from this link: https://gamedev.stackexchange.com/questions/60630/how-do-i-find-the-circumcenter-of-a-triangle-in-3d
    """

    # Convert the input points to numpy arrays for easier manipulation
    p1 = np.array(point1)
    p2 = np.array(point2)
    p3 = np.array(point3)

    # Calculate the vectors from point1 to point2 and point1 to point3
    p12 = p2 - p1 # Vector from point1 to point2
    p13 = p3 - p1 # Vector from point1 to point3

    # Calculate the cross product of p12 and p13
    p12_X_p13 = np.cross(p12, p13) # Cross product of vectors p12 and p13

    # Calculate the vector from point1 to the circumsphere center
    toCircumsphereCenter = ((np.cross(p12_X_p13, p12) * np.dot(p13, p13)) + (np.cross(p13, p12_X_p13) * np.dot(p12, p12))) / (2 * np.dot(p12_X_p13, p12_X_p13))

    # Calculate the radius of the circumsphere
    circumsphereRadius = np.linalg.norm(toCircumsphereCenter) # The radius is the length of the vector to the circumsphere center

    # Calculate the coordinates of the circumsphere center
    ccs = point1  +  toCircumsphereCenter # The center is point1 plus the vector to the circumsphere center

    # Return the center and radius of the circumsphere
    return ccs, circumsphereRadius

def my_scatter(points, camera_view=(30, -60)):
    # Create a new figure
    fig = plt.figure()

    # Create a 3D plot
    ax = fig.add_subplot(111, projection='3d')

    # Plot the mean points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2]) # points

    # Set the labels with increased size
    ax.set_xlabel('X', labelpad=20, fontsize=14)
    ax.set_ylabel('Y', labelpad=20, fontsize=14)
    ax.set_zlabel('Z', labelpad=20, fontsize=14)

    # Increase the size of the tick labels
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Set the camera view
    ax.view_init(*camera_view)

    #print("Number of points: ", points.shape[0])
    plt.show()
    return None

def load_data_ply(file):
    plydata = PlyData.read(file)

    # Extract the vertex data
    vertex_data = plydata['vertex'].data

    # Extract x, y, z coordinates
    x = vertex_data['x']
    y = vertex_data['y']
    z = vertex_data['z']

    vertex_data = np.column_stack([x,y,z])
    return vertex_data


def load_data_npy(file):
    # Load the data from the .npy file
    vertex_data = np.load(file)
    return vertex_data


def mean_points_func(vertex_data, plot = False, N=4):
    # This is from ply data, we chose resolution 0, so its a square (4 points)

    # Initialize an empty list to store the mean points
    mean_points = []

    # Loop over the vertex_data array with a step of 4
    for i in range(0, vertex_data.shape[0] - 3, N):
        # Select the current 4 points
        current_points = vertex_data[i:i+N]
        
        # Compute the mean of the current 4 points
        mean_point = current_points.mean(axis=0)
        
        # Append the mean point to the list
        mean_points.append(mean_point)

    # Convert the list to a numpy array
    mean_points = np.array(mean_points)
    if plot == True:
        my_scatter(mean_points)
    return mean_points


def compute_equidistant_points(points, num_points, N=5):
    # Compute the cumulative distances along the points
    cumulative_distances = [0]
    for i in range(1, len(points)):
        distance = np.linalg.norm(points[i] - points[i - 1])
        cumulative_distances.append(cumulative_distances[-1] + distance)
    
    total_distance = cumulative_distances[-1]
    segment_length = total_distance / (num_points - 1)
    
    equidistant_points = [points[0]]
    current_distance = segment_length
    
    # Interpolate the points
    for i in range(1, len(points)):
        while current_distance <= cumulative_distances[i]:
            t = (current_distance - cumulative_distances[i - 1]) / (cumulative_distances[i] - cumulative_distances[i - 1])
            new_point = points[i - 1] * (1 - t) + points[i] * t
            equidistant_points.append(new_point)
            current_distance += segment_length
    
    # Ensure the last point is exactly the last point
    if len(equidistant_points) < num_points:
        equidistant_points.append(points[-1])
    
    equidistant_points = equidistant_points[::N]
    my_scatter(np.array(equidistant_points))
    return np.array(equidistant_points)


def compute_equidistant_points_V2(points, num_points):
    # Compute the cumulative distances along the points
    cumulative_distances = [0]
    for i in range(1, len(points)):
        distance = np.linalg.norm(points[i] - points[i - 1])
        cumulative_distances.append(cumulative_distances[-1] + distance)
    
    total_distance = cumulative_distances[-1]
    segment_length = total_distance / (num_points - 1)
    
    equidistant_points = [points[0]]
    current_distance = segment_length
    
    # Find the points that are approximately equidistant
    for i in range(1, len(points)):
        if cumulative_distances[i] >= current_distance:
            equidistant_points.append(points[i])
            current_distance += segment_length
    
    # Ensure the last point is exactly the last point
    if len(equidistant_points) < num_points:
        equidistant_points.append(points[-1])
    
    print("Number of equidistant points: ", len(equidistant_points))
    my_scatter(np.array(equidistant_points))
    return np.array(equidistant_points)

def my_sort_points(points, threshold):
    """
    This function sorts points based on their Euclidean distance. Starting from the point with the lowest x value,
    it finds the next closest point and adds it to the sorted list. This process continues until no points are left,
    or the distance to the next closest point is greater than a specified threshold.

    Parameters:
    points (numpy.ndarray): The points to be sorted. Each point is an array of coordinates.
    threshold (Int): The maximum allowed distance to the next point.

    Returns:
    numpy.ndarray: The sorted points.
    """

    # Find the point with the lowest x value
    start_point = min(points, key=lambda point: point[0])

    # Initialize the sorted points list with the start point
    sorted_points = [tuple(start_point)]

    # Create a set of the remaining points
    remaining_points = set(map(tuple, points))
    # Remove the start point from the remaining points
    remaining_points.remove(tuple(start_point))

    # Continue until no points are left
    while remaining_points:
        # Get the last point added to the sorted list
        current_point = sorted_points[-1]

        # Calculate the Euclidean distance from the current point to each remaining point
        distances = {point: np.linalg.norm(np.array(point) - np.array(current_point)) for point in remaining_points}

        # Find the point with the minimum distance to the current point
        next_point, min_distance = min(distances.items(), key=lambda item: item[1])

        # If the minimum distance is greater than the threshold, stop sorting
        if min_distance > threshold:
            break

        # Remove the next point from the remaining points
        remaining_points.remove(next_point)
        # Add the next point to the sorted list
        sorted_points.append(next_point)

    # Return the sorted points as a numpy array
    return np.array(sorted_points)

def my_sort_points_max_z(points, threshold):
    """
    This function sorts points based on their Euclidean distance. Starting from the point with the lowest x value,
    it finds the next closest point and adds it to the sorted list. This process continues until no points are left,
    or the distance to the next closest point is greater than a specified threshold.

    Parameters:
    points (numpy.ndarray): The points to be sorted. Each point is an array of coordinates.
    threshold (Int): The maximum allowed distance to the next point.

    Returns:
    numpy.ndarray: The sorted points.
    """

    # Find the point with the lowest x value
    start_point = max(points, key=lambda point: point[2])

    # Initialize the sorted points list with the start point
    sorted_points = [tuple(start_point)]

    # Create a set of the remaining points
    remaining_points = set(map(tuple, points))
    # Remove the start point from the remaining points
    remaining_points.remove(tuple(start_point))

    # Continue until no points are left
    while remaining_points:
        # Get the last point added to the sorted list
        current_point = sorted_points[-1]

        # Calculate the Euclidean distance from the current point to each remaining point
        distances = {point: np.linalg.norm(np.array(point) - np.array(current_point)) for point in remaining_points}

        # Find the point with the minimum distance to the current point
        next_point, min_distance = min(distances.items(), key=lambda item: item[1])

        # If the minimum distance is greater than the threshold, stop sorting
        if min_distance > threshold:
            break

        # Remove the next point from the remaining points
        remaining_points.remove(next_point)
        # Add the next point to the sorted list
        sorted_points.append(next_point)

    # Return the sorted points as a numpy array
    return np.array(sorted_points)

def my_sort_points_max_x(points, threshold):
    """
    This function sorts points based on their Euclidean distance. Starting from the point with the lowest x value,
    it finds the next closest point and adds it to the sorted list. This process continues until no points are left,
    or the distance to the next closest point is greater than a specified threshold.

    Parameters:
    points (numpy.ndarray): The points to be sorted. Each point is an array of coordinates.
    threshold (Int): The maximum allowed distance to the next point.

    Returns:
    numpy.ndarray: The sorted points.
    """

    # Find the point with the lowest x value
    start_point = max(points, key=lambda point: point[0])

    # Initialize the sorted points list with the start point
    sorted_points = [tuple(start_point)]

    # Create a set of the remaining points
    remaining_points = set(map(tuple, points))
    # Remove the start point from the remaining points
    remaining_points.remove(tuple(start_point))

    # Continue until no points are left
    while remaining_points:
        # Get the last point added to the sorted list
        current_point = sorted_points[-1]

        # Calculate the Euclidean distance from the current point to each remaining point
        distances = {point: np.linalg.norm(np.array(point) - np.array(current_point)) for point in remaining_points}

        # Find the point with the minimum distance to the current point
        next_point, min_distance = min(distances.items(), key=lambda item: item[1])

        # If the minimum distance is greater than the threshold, stop sorting
        if min_distance > threshold:
            break

        # Remove the next point from the remaining points
        remaining_points.remove(next_point)
        # Add the next point to the sorted list
        sorted_points.append(next_point)

    # Return the sorted points as a numpy array
    return np.array(sorted_points)


def interpolate_3d_points(points, num_interpolated_points = 3, skip_points=5, camera_view=(30, -60)):
    """
    This function interpolates additional points between each pair of consecutive points in a 3D space.
    It then skips a specified number of points to reduce the density of points.

    Parameters:
    points (numpy.ndarray): The original points in 3D space.
    num_interpolated_points (int): The number of points to interpolate between each pair of original points.
    skip_points (int): The number of points to skip in the final list of points.

    Returns:
    numpy.ndarray: The interpolated points.
    """

    # Create a list to store the new points
    interpolated_points = []
    
    # Iterate over each pair of consecutive points
    for i in range(len(points) - 1):
        # Get the start and end points of the current segment
        start_point = points[i]
        end_point = points[i + 1]
        
        # Append the start point of the current segment to the list
        interpolated_points.append(start_point)
        
        # Compute the interpolated points for the current segment
        for j in range(1, num_interpolated_points + 1):
            # Compute the parameter for the linear interpolation
            t = j / (num_interpolated_points + 1)
            # Compute the interpolated point
            interpolated_point = start_point * (1 - t) + end_point * t
            # Append the interpolated point to the list
            interpolated_points.append(interpolated_point)
    
    # Append the last point of the original array to the list
    interpolated_points.append(points[-1])

    # Skip points to reduce the density
    interpolated_points = interpolated_points[::skip_points]

    # Print the number of interpolated points
    print("Number of interpolated points: ", len(interpolated_points))

    # Visualize the interpolated points
    my_scatter(np.array(interpolated_points), camera_view=camera_view)
    # Return the interpolated points as a numpy array
    return np.array(interpolated_points)


def smoothing_signal(points, window_length, polyorder, camera_view=(30, -60)):
    """
    This function applies a Savitzky-Golay filter to smooth a signal represented by a set of points.
    It then visualizes the smoothed points using a scatter plot.

    Parameters:
    points (numpy.ndarray): The original points representing the signal.
    window_length (int): The length of the filter window (i.e., the number of points used for the polynomial regression).
    polyorder (int): The order of the polynomial used in the Savitzky-Golay filter.

    Returns:
    numpy.ndarray: The smoothed points.
    """

    # Apply the Savitzky-Golay filter to the points
    # The filter fits a polynomial of a specified order to a window of points and uses the polynomial to estimate the center point of the window
    # This process is repeated for each point in the signal, resulting in a smoothed version of the signal
    smooth_points = savgol_filter(points, window_length, polyorder, axis=0)

    # Visualize the points
    my_scatter(smooth_points, camera_view)

    # Print the number of smoothed points
    print("Number of smoothed points: ", len(smooth_points))

    # Return the smoothed points
    return smooth_points

def moving_average_signal(points, window_length, camera_view=(30, -60)):
    """
    This function applies a moving average filter to smooth a signal represented by a set of points.
    It then visualizes the smoothed points using a scatter plot.

    Parameters:
    points (numpy.ndarray): The original points representing the signal.
    window_length (int): The length of the filter window (i.e., the number of points to average).

    Returns:
    numpy.ndarray: The smoothed points.
    """

    # Create a one-dimensional moving average filter
    filter_weights = np.ones(window_length) / window_length

    # Initialize an empty list to store the smoothed points
    smooth_points = []

    # Apply the moving average filter to each dimension separately
    for i in range(points.shape[1]):
        # Apply the moving average filter to the points in the current dimension
        # The 'valid' mode means that the convolution product is only given for points where the signals overlap completely
        smooth_points_i = np.convolve(points[:, i], filter_weights, mode='valid')
        # Append the smoothed points in the current dimension to the list
        smooth_points.append(smooth_points_i)

    # Convert the list of smoothed points to a numpy array
    smooth_points = np.array(smooth_points).T

    # Visualize the points
    my_scatter(smooth_points, camera_view)

    # Print the number of smoothed points
    print("Number of smoothed points: ", len(smooth_points))

    # Return the smoothed points
    return smooth_points


def calculate_radii(points):
    """
    This function calculates the centers and radii of the circles passing through each triplet of consecutive points.
    It then appends a large radius at the start and end of the radii array to represent the end points of the line.

    Parameters:
    points (numpy.ndarray): The points through which the circles pass.

    Returns:
    tuple: A tuple containing two numpy arrays. The first array contains the centers of the circles, and the second array contains the radii of the circles.
    """

    # Initialize lists to store the centers and radii
    centers = []
    radii = []

    # Loop over the points
    for i in range(1, len(points) - 1):
        # Find the circle passing through the points points[i-1], points[i], points[i+1]
        center, radius = find_radius_3d(points[i-1], points[i], points[i+1])
        
        # Append the center and radius to the lists
        centers.append(center)
        radii.append(radius)

    # Convert the lists to numpy arrays
    centers = np.array(centers)
    radii = np.array(radii)

    # Append a large number at the end and start of the radii array to represent the end points of the line
    radii = np.append(radii, 10**5)
    radii = np.insert(radii, 0, 10**5)

    # Print the shapes of the radii and points arrays
    print(f"Radii shape: {radii.shape}\nequidistant_points shape: {points.shape}")
    # Print the minimum and maximum of the radii
    print(f"min of radii: {np.min(radii)}\nmax of radii: {np.max(radii)}")

    # Return the centers and radii
    return centers, radii


def plot_thresholded_points(smooth_points, radii, T=18, camera_view=(30, -60)):
    # Sort the radii
    radii_sort = np.sort(radii)

    threshold = radii_sort[T] # use the sorted list since the lower values should be the curves
    # Create a mask for the radii below the threshold
    mask = radii < threshold # use original array to get the correct index for points

    # Select the points corresponding to the radii below the threshold
    selected_points = smooth_points[mask]

    # Create a new figure
    fig = plt.figure()

    # Create a 3D plot
    ax = fig.add_subplot(111, projection='3d')

    # Plot the selected points
    ax.scatter(smooth_points[:, 0], smooth_points[:, 1], smooth_points[:, 2], color = "b",alpha=0.5) 
    ax.scatter(selected_points[:, 0], selected_points[:, 1], selected_points[:, 2], color = "r",alpha=1)

    # Set the labels with increased size
    ax.set_xlabel('X', labelpad=20, fontsize=14)
    ax.set_ylabel('Y', labelpad=20, fontsize=14)
    ax.set_zlabel('Z', labelpad=20, fontsize=14)

    # Increase the size of the tick labels
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Set the camera view
    ax.view_init(*camera_view)

    print(f"Total number of points below threshold: {len(selected_points)}\n")
    # Show the plot
    plt.show()
    # må lige lege med at finde ud af hvad man vil have af output
    return selected_points

def plot_turning_points(smooth_points, radii, T = 18, skip_indices = 4, camera_view=(30, -60)):

    radii_sort = np.sort(radii)
   # Define the threshold
    threshold = radii_sort[T]

    # Create a mask for the radii below the threshold
    # This will create a boolean array where each element is True 
    # if the corresponding radius is below the threshold, and False otherwise
    mask = radii < threshold

    # Select the points corresponding to the radii below the threshold
    # This will create a new array containing only the points that correspond to the radii below the threshold
    selected_points = smooth_points[mask]

    # Initialize an empty list to keep track of the indices of the points that have been plotted
    plotted_indices = []

    # Create a new figure
    fig = plt.figure()

    # Add a 3D subplot to the figure
    ax = fig.add_subplot(111, projection='3d')

    # Iterate over the selected points and their indices in the original points array
    for point, index in zip(selected_points, np.where(mask)[0]):
        # Check if the current index is within a range of 4 indices of any previously plotted point
        # If it is not, plot the point and add its index to the list of plotted indices
        if not any(abs(index - plotted_index) <= skip_indices for plotted_index in plotted_indices):
            ax.scatter(point[0], point[1], point[2], color = "r", alpha=1, s=100, marker='o', edgecolors='black')
            plotted_indices.append(index)

    # Plot the original points in blue
    ax.scatter(smooth_points[:, 0], smooth_points[:, 1], smooth_points[:, 2], color = "b", alpha=0.5)

    # Set the labels with increased size
    ax.set_xlabel('X', labelpad=20, fontsize=14)
    ax.set_ylabel('Y', labelpad=20, fontsize=14)
    ax.set_zlabel('Z', labelpad=20, fontsize=14)

    # Increase the size of the tick labels
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Set the camera view
    ax.view_init(*camera_view)

    plt.show()
    print(f"Number of turning points: {len(plotted_indices)}\n")
    return plotted_indices

def curve_characteristics(points):
    start_p, end_p = points[0],points[-1]
    distance = np.linalg.norm(start_p - end_p)
    print(f" The distance between the start and end point is: {distance} Voxels")
    # Initialize a variable to store the total length
    total_length = 0

    # Loop over the points
    for i in range(len(points) - 1):
        # Compute the Euclidean distance between the current point and the next point
        distance2 = np.linalg.norm(points[i] - points[i+1])
        
        # Add the distance to the total length
        total_length += distance2

    print(f" The total length of the line is: {total_length} Voxels")
    print(f" The tortuosity of the line is: {total_length/distance}")

        # Assuming smoothed_points is a numpy array of shape (n, 3)
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)

    # Calculate the lengths of the sides of the bounding box
    lengths = max_vals - min_vals

    # Calculate the volume of the bounding box
    volume = np.prod(lengths)

    print(" The volume of the bounding box is: ", volume, "Voxels")
    return distance, total_length, total_length/distance, volume

############## Interactive plots of different functions ####################

def animate_interp_points(mean_points):
    # Create sliders
    num_interpolated_points_slider = widgets.IntSlider(min=1, max=20, step=1, value=5, description='Interpolated Points:')
    skip_points_slider = widgets.IntSlider(min=1, max=40, step=1, value=20)
    elevation_slider = widgets.IntSlider(min=0, max=90, step=1, value=30, description='Elevation:')
    azimuth_slider = widgets.IntSlider(min=-180, max=180, step=1, value=-60, description='Azimuth:')


    # Function to update plot
    def update_plot(num_interpolated_points, skip_points, elevation, azimuth):
        interp_points = interpolate_3d_points(mean_points, num_interpolated_points, skip_points, camera_view=(elevation, azimuth))
        return interp_points

    # Interactive widget
    interactive_plot = widgets.interactive(update_plot, num_interpolated_points=num_interpolated_points_slider, skip_points=skip_points_slider, elevation=elevation_slider, azimuth=azimuth_slider)
    return interactive_plot

def animate_smooth_points(interp_points):
    # Create sliders
    window_length_slider = widgets.IntSlider(min=1, max=50, step=1, value=10, description='Window_length:')
    polyorder_slider = widgets.IntSlider(min=1, max=8, step=1, value=1, description='Polyorder:')
    elevation_slider = widgets.IntSlider(min=0, max=90, step=1, value=30, description='Elevation:')
    azimuth_slider = widgets.IntSlider(min=-180, max=180, step=1, value=-60, description='Azimuth:')

    def update_plot(window_length, polyorder, elevation, azimuth):
        if window_length <= polyorder:
            print("Error: window_length must be greater than polyorder.")
            return
        smoothed_points = smoothing_signal(interp_points, window_length, polyorder,camera_view=(elevation, azimuth))
        
        return smoothed_points
    
    # Interactive widget
    interactive_smooth = widgets.interactive(update_plot, window_length=window_length_slider, polyorder=polyorder_slider, elevation=elevation_slider, azimuth=azimuth_slider)
    return interactive_smooth

def animate_thresholed_points(smoothed_points, radii):
    # Create sliders
    T_slider = widgets.IntSlider(min=1, max=100, step=1, value=55, description='T:')
    elevation_slider = widgets.IntSlider(min=0, max=90, step=1, value=30, description='Elevation:')
    azimuth_slider = widgets.IntSlider(min=-180, max=180, step=1, value=-60, description='Azimuth:')

    # Function to update plot
    def update_plot(T, elevation, azimuth):
        selected_points = plot_thresholded_points(smoothed_points, radii, T,camera_view=(elevation, azimuth))
        return selected_points

    # Interactive widget
    interactive_selected_points = widgets.interactive(update_plot, T=T_slider, elevation=elevation_slider, azimuth=azimuth_slider)
    return interactive_selected_points

def animate_turning_points(smoothed_points, radii):
    # Create sliders
    T_slider = widgets.IntSlider(min=1, max=100, step=1, value=60, description='T:')
    skip_indices_slider = widgets.IntSlider(min=1, max=100, step=1, value=13, description='Skip Indices:')
    elevation_slider = widgets.IntSlider(min=0, max=90, step=1, value=30, description='Elevation:')
    azimuth_slider = widgets.IntSlider(min=-180, max=180, step=1, value=-60, description='Azimuth:')

    # Function to update plot
    def update_plot(T, skip_indices, elevation, azimuth):
        selected_turning_points = plot_turning_points(smoothed_points, radii, T, skip_indices, camera_view=(elevation, azimuth))
        return selected_turning_points

    # Interactive widget
    interactive_turning_points = widgets.interactive(update_plot, T=T_slider, skip_indices=skip_indices_slider, elevation=elevation_slider, azimuth=azimuth_slider)
    return interactive_turning_points

def animate_sorting_points(vertex_data):

    # Create sliders
    threshold_slider = widgets.FloatSlider(min=1, max=100, step=1, value=15, description='Threshold:')
    elevation_slider = widgets.IntSlider(min=0, max=90, step=1, value=30, description='Elevation:')
    azimuth_slider = widgets.IntSlider(min=-180, max=180, step=1, value=-60, description='Azimuth:')

    # Function to update plot
    def update_plot(threshold, elevation, azimuth):
        sorted_points = my_sort_points(vertex_data, threshold)
        my_scatter(sorted_points,camera_view=(elevation, azimuth))
        return sorted_points

    # Interactive widget
    interactive_sorted_points = widgets.interactive(update_plot, threshold=threshold_slider, elevation=elevation_slider, azimuth=azimuth_slider)
    return interactive_sorted_points

def animate_sorting_points_max_z(vertex_data):
    # Create sliders
    threshold_slider = widgets.FloatSlider(min=1, max=100, step=1, value=15, description='Threshold:')
    elevation_slider = widgets.IntSlider(min=0, max=90, step=1, value=30, description='Elevation:')
    azimuth_slider = widgets.IntSlider(min=-180, max=180, step=1, value=-60, description='Azimuth:')

    # Function to update plot
    def update_plot(threshold, elevation, azimuth):
        sorted_points = my_sort_points_max_z(vertex_data, threshold)
        my_scatter(sorted_points,camera_view=(elevation, azimuth))
        return sorted_points

    # Interactive widget
    interactive_sorted_points = widgets.interactive(update_plot, threshold=threshold_slider, elevation=elevation_slider, azimuth=azimuth_slider)
    return interactive_sorted_points

def animate_sorting_points_max_x(vertex_data):
    # Create sliders
    threshold_slider = widgets.FloatSlider(min=1, max=100, step=1, value=15, description='Threshold:')
    elevation_slider = widgets.IntSlider(min=0, max=90, step=1, value=30, description='Elevation:')
    azimuth_slider = widgets.IntSlider(min=-180, max=180, step=1, value=-60, description='Azimuth:')

    # Function to update plot
    def update_plot(threshold, elevation, azimuth):
        sorted_points = my_sort_points_max_x(vertex_data, threshold)
        my_scatter(sorted_points,camera_view=(elevation, azimuth))
        return sorted_points

    # Interactive widget
    interactive_sorted_points = widgets.interactive(update_plot, threshold=threshold_slider, elevation=elevation_slider, azimuth=azimuth_slider)
    return interactive_sorted_points



def animate_points(sorted_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter([], [], [])

    def update(num):
        ax.clear()
        ax.scatter(sorted_points[:num+1, 0], sorted_points[:num+1, 1], sorted_points[:num+1, 2])
        ax.set_xlim([sorted_points[:, 0].min(), sorted_points[:, 0].max()])
        ax.set_ylim([sorted_points[:, 1].min(), sorted_points[:, 1].max()])
        ax.set_zlim([sorted_points[:, 2].min(), sorted_points[:, 2].max()])

    ani = animation.FuncAnimation(fig, update, frames=len(sorted_points), interval=200)

    return HTML(ani.to_jshtml())

# Use the function like this:
# sorted_points = sort_points(points)
# HTML_display = animate_points(sorted_points)
# HTML_display


# == FUNCTIONS ========================================================================================================

# Takes points in [[x1, y1, z1], [x2, y2, z2]...] Numpy Array format
def thin_line(points, point_cloud_thickness=0.53, iterations=1,sample_points=0):
    total_start_time =  time.perf_counter()
    if sample_points != 0:
        points = points[:sample_points]
    
    # Sort points into KDTree for nearest neighbors computation later
    point_tree = spatial.cKDTree(points)

    # Empty array for transformed points
    new_points = []
    # Empty array for regression lines corresponding ^^ points
    regression_lines = []
    nn_time = 0
    rl_time = 0
    prj_time = 0
    for point in point_tree.data:
        # Get list of points within specified radius {point_cloud_thickness}
        start_time = time.perf_counter()
        points_in_radius = point_tree.data[point_tree.query_ball_point(point, point_cloud_thickness)]
        nn_time += time.perf_counter()- start_time

        # Get mean of points within radius
        start_time = time.perf_counter()
        data_mean = points_in_radius.mean(axis=0)

        # Calulate 3D regression line/principal component in point form with 2 coordinates
        uu, dd, vv = np.linalg.svd(points_in_radius - data_mean)
        linepts = vv[0] * np.mgrid[-1:1:2j][:, np.newaxis]
        linepts += data_mean
        regression_lines.append(list(linepts))
        rl_time += time.perf_counter() - start_time

        # Project original point onto 3D regression line
        start_time = time.perf_counter()
        ap = point - linepts[0]
        ab = linepts[1] - linepts[0]
        point_moved = linepts[0] + np.dot(ap,ab) / np.dot(ab,ab) * ab
        prj_time += time.perf_counter()- start_time

        new_points.append(list(point_moved))
    print("--- %s seconds to thin points ---" % (time.perf_counter() - total_start_time))
    print(f"Finding nearest neighbors for calculating regression lines: {nn_time}")
    print(f"Calculating regression lines: {rl_time}")
    print(f"Projecting original points on  regression lines: {prj_time}\n")
    return np.array(new_points), regression_lines

# Sorts points outputed from thin_points()
def sort_points(points, regression_lines, sorted_point_distance=0.2):
    sort_points_time = time.perf_counter()
    # Index of point to be sorted
    index = 0

    # sorted points array for left and right of intial point to be sorted
    sort_points_left = [points[index]]
    sort_points_right = []

    # Regression line of previously sorted point
    regression_line_prev = regression_lines[index][1] - regression_lines[index][0]

    # Sort points into KDTree for nearest neighbors computation later
    point_tree = spatial.cKDTree(points)


    # Iterative add points sequentially to the sort_points_left array
    while 1:
        # Calulate regression line vector; makes sure line vector is similar direction as previous regression line
        v = regression_lines[index][1] - regression_lines[index][0]
        if np.dot(regression_line_prev, v ) / (np.linalg.norm(regression_line_prev) * np.linalg.norm(v))  < 0:
            v = regression_lines[index][0] - regression_lines[index][1]
        regression_line_prev = v

        # Find point {distR_point} on regression line distance {sorted_point_distance} from original point 
        distR_point = points[index] + ((v / np.linalg.norm(v)) * sorted_point_distance)

        # Search nearest neighbors of distR_point within radius {sorted_point_distance / 3}
        points_in_radius = point_tree.data[point_tree.query_ball_point(distR_point, sorted_point_distance / 3)]
        if len(points_in_radius) < 1:
            break

        # Neighbor of distR_point with smallest angle to regression line vector is selected as next point in order
        # 
        # CAN BE OPTIMIZED
        # 
        nearest_point = points_in_radius[0]
        distR_point_vector = distR_point - points[index]
        nearest_point_vector = nearest_point - points[index]
        for x in points_in_radius: 
            x_vector = x - points[index]
            if vg.angle(distR_point_vector, x_vector) < vg.angle(distR_point_vector, nearest_point_vector):
                nearest_point_vector = nearest_point - points[index]
                nearest_point = x
        index = np.where(points == nearest_point)[0][0]

        # Add nearest point to 'sort_points_left' array
        sort_points_left.append(nearest_point)

    # Do it again but in the other direction of initial starting point 
    index = 0
    regression_line_prev = regression_lines[index][1] - regression_lines[index][0]
    while 1:
        # Calulate regression line vector; makes sure line vector is similar direction as previous regression line
        v = regression_lines[index][1] - regression_lines[index][0]
        if np.dot(regression_line_prev, v ) / (np.linalg.norm(regression_line_prev) * np.linalg.norm(v))  < 0:
            v = regression_lines[index][0] - regression_lines[index][1]
        regression_line_prev = v

        # Find point {distR_point} on regression line distance {sorted_point_distance} from original point 
        # 
        # Now vector is substracted from the point to go in other direction
        # 
        distR_point = points[index] - ((v / np.linalg.norm(v)) * sorted_point_distance)

        # Search nearest neighbors of distR_point within radius {sorted_point_distance / 3}
        points_in_radius = point_tree.data[point_tree.query_ball_point(distR_point, sorted_point_distance / 3)]
        if len(points_in_radius) < 1:
            break

        # Neighbor of distR_point with smallest angle to regression line vector is selected as next point in order
        # 
        # CAN BE OPTIMIZED
        # 
        nearest_point = points_in_radius[0]
        distR_point_vector = distR_point - points[index]
        nearest_point_vector = nearest_point - points[index]
        for x in points_in_radius: 
            x_vector = x - points[index]
            if vg.angle(distR_point_vector, x_vector) < vg.angle(distR_point_vector, nearest_point_vector):
                nearest_point_vector = nearest_point - points[index]
                nearest_point = x
        index = np.where(points == nearest_point)[0][0]

        # Add next point to 'sort_points_right' array
        sort_points_right.append(nearest_point)

    # Combine 'sort_points_right' and 'sort_points_left'
    sort_points_right.extend(sort_points_left[::-1])
    print("--- %s seconds to sort points ---" % (time.perf_counter() - sort_points_time))
    return np.array(sort_points_right)

print("Functions loaded")