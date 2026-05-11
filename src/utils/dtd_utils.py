import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.io as sio
from utils.dtd_math import get_rot_matrix, convert_3x3_to_1x6, generate_dtd_gauss_random_orientations, generate_fibers_WM_watson


def save_single_dtens_params(
    lambda_1, lambda_2, lambda_3, u1, u2, u3, filepath="single_dtens_params.json"
):
    """
    Save parameters for a single diffusion tensor to a JSON file.

    Parameters:
    lambda_1, lambda_2, lambda_3 (float): Eigenvalues.
    u1, u2, u3 (float): First eigenvector components.
    filepath (str): Path to the JSON file to save the parameters.
    """
    params = [
        {
            "lambda_1": lambda_1,
            "lambda_2": lambda_2,
            "lambda_3": lambda_3,
            "u1": u1,
            "u2": u2,
            "u3": u3,
        }
    ]

    with open(filepath, "w") as file:
        json.dump(params, file, indent=4)


def combine_dtens_from_file(file_paths, output_path):
    """
    Combine multiple JSON files containing diffusion tensors into a single file.

    Parameters:
    file_paths (list): List of file paths to JSON files containing diffusion tensors.
    output_path (str): Path to the output JSON file.
    """
    combined_dtens = []
    for file_path in file_paths:
        with open(file_path, "r") as file:
            data = json.load(file)
            combined_dtens.extend(data)

    with open(output_path, "w") as file:
        json.dump(combined_dtens, file, indent=4)


def load_dtens_from_file(filepath):
    with open(filepath, "r") as file:
        data = json.load(file)

    if isinstance(data, list) and all(isinstance(entry, dict) for entry in data):
        d_tensors = []
        for entry in data:
            lambda_1 = entry["lambda_1"]
            lambda_2 = entry["lambda_2"]
            lambda_3 = entry["lambda_3"]
            u1 = entry["u1"]
            u2 = entry["u2"]
            u3 = entry["u3"]

            d_tensor = calc_dtens(lambda_1, lambda_2, lambda_3, u1, u2, u3)
            d_tensors.append(d_tensor)

        return np.array(d_tensors, dtype=np.float64)
    else:
        raise ValueError("Invalid data format in JSON file")


def generate_dtens_params(
    n,
    lambda_1_range=None,
    lambda_2_range=None,
    lambda_3_range=None,
    u1_range=None,
    u2_range=None,
    u3_range=None,
    needle_count=0,
    lambda_1_range_needle=None,
    plate_count=0,
    lambda_1_range_plate=None,
    sphere_count=0,
    lambda_1_range_sphere=None,
    seed=None,
    method="random",
    filepath="dtens_params.json",
):
    """
    Generate parameters for diffusion tensors and save them to a JSON file.

    Parameters:
    lambda_1_range, lambda_2_range, lambda_3_range (tuple): Ranges for eigenvalues (min, max).
    u1_range, u2_range, u3_range (tuple): Ranges for first eigenvector components (min, max).
    n (int): Number of parameter sets to generate.
    method (str): 'random' or 'evenly'. Method to generate the parameters.
    filepath (str): Path to the JSON file to save the parameters.
    """
    if seed is not None:
        np.random.seed(seed)

    params = []

    # adjust methods as fixed or varying as needed, hard-code
    if method == "random":
        for _ in range(n):
            lambda_1 = np.random.uniform(*lambda_1_range)
            lambda_2 = np.random.uniform(*lambda_2_range)
            lambda_3 = np.random.uniform(*lambda_3_range)
            u1 = np.random.uniform(*u1_range)
            u2 = np.random.uniform(*u2_range)
            u3 = np.random.uniform(*u3_range)
            params.append(
                {
                    "lambda_1": lambda_1,
                    "lambda_2": lambda_2,
                    "lambda_3": lambda_3,
                    "u1": u1,
                    "u2": u2,
                    "u3": u3,
                }
            )
    elif method == "evenly":
        lambda_1_values = np.linspace(*lambda_1_range, n)
        lambda_2_values = np.linspace(*lambda_2_range, n)
        lambda_3_values = np.linspace(*lambda_3_range, n)
        u1_values = np.linspace(*u1_range, n)
        u2_values = np.linspace(*u2_range, n)
        u3_values = np.linspace(*u3_range, n)
        for i in range(n):
            params.append(
                {
                    "lambda_1": lambda_1_values[i],
                    "lambda_2": lambda_2_values[i],
                    "lambda_3": lambda_3_values[i],
                    "u1": u1_values[i],
                    "u2": u2_values[i],
                    "u3": u3_values[i],
                }
            )
    elif method == "random_xy_dir":
        lambda_1 = 1.2e-9
        lambda_2 = 0.12e-9
        lambda_3 = 0.12e-9
        u3 = 0.0
        for _ in range(n):
            u1 = np.random.uniform(*u1_range)
            u2 = np.random.uniform(*u2_range)
            params.append(
                {
                    "lambda_1": lambda_1,
                    "lambda_2": lambda_2,
                    "lambda_3": lambda_3,
                    "u1": u1,
                    "u2": u2,
                    "u3": u3,
                }
            )
    elif method == "random_xyz_dir":
        lambda_1 = 1.2e-9
        lambda_2 = 0.0e-9
        lambda_3 = 0.0e-9
        for _ in range(n):
            u1 = np.random.uniform(*u1_range)
            u2 = np.random.uniform(*u2_range)
            u3 = np.random.uniform(*u3_range)
            params.append(
                {
                    "lambda_1": lambda_1,
                    "lambda_2": lambda_2,
                    "lambda_3": lambda_3,
                    "u1": u1,
                    "u2": u2,
                    "u3": u3,
                }
            )
    elif method == "sticks_WM":
        lambda_1 = 0.52e-9
        lambda_2 = 0.0e-9
        lambda_3 = 0.0e-9
        for _ in range(n):
            u1 = np.random.uniform(*u1_range)
            u2 = np.random.uniform(*u2_range)
            u3 = np.random.uniform(*u3_range)
            params.append(
                {
                    "lambda_1": lambda_1,
                    "lambda_2": lambda_2,
                    "lambda_3": lambda_3,
                    "u1": u1,
                    "u2": u2,
                    "u3": u3,
                }
            )
    elif method == "sticks_GM":
        lambda_1 = 0.26e-9 # 0.08e-9
        lambda_2 = 0.0e-9
        lambda_3 = 0.0e-9
        for _ in range(n):
            u1 = np.random.uniform(*u1_range)
            u2 = np.random.uniform(*u2_range)
            u3 = np.random.uniform(*u3_range)
            params.append(
                {
                    "lambda_1": lambda_1,
                    "lambda_2": lambda_2,
                    "lambda_3": lambda_3,
                    "u1": u1,
                    "u2": u2,
                    "u3": u3,
                }
            )
    elif method == "fibers_WM":
        lambda_1 = 1.7e-09
        lambda_2 = 3.0e-10
        lambda_3 = 3.0e-10
        for _ in range(n):
            u1 = np.random.uniform(*u1_range)
            u2 = np.random.uniform(*u2_range)
            u3 = np.random.uniform(*u3_range)
            params.append(
                {
                    "lambda_1": lambda_1,
                    "lambda_2": lambda_2,
                    "lambda_3": lambda_3,
                    "u1": u1,
                    "u2": u2,
                    "u3": u3,
                }
            )
    elif method == "fibers_GM":
        lambda_1 = 1.33025e-09
        lambda_2 = 2.3475e-10
        lambda_3 = 2.3475e-10
        for _ in range(n):
            u1 = np.random.uniform(*u1_range)
            u2 = np.random.uniform(*u2_range)
            u3 = np.random.uniform(*u3_range)
            params.append(
                {
                    "lambda_1": lambda_1,
                    "lambda_2": lambda_2,
                    "lambda_3": lambda_3,
                    "u1": u1,
                    "u2": u2,
                    "u3": u3,
                }
            )
    elif method == "sticks_CSF":
        lambda_1 = 0.21e-9
        lambda_2 = 0.0e-9
        lambda_3 = 0.0e-9
        for _ in range(n):
            u1 = np.random.uniform(*u1_range)
            u2 = np.random.uniform(*u2_range)
            u3 = np.random.uniform(*u3_range)
            params.append(
                {
                    "lambda_1": lambda_1,
                    "lambda_2": lambda_2,
                    "lambda_3": lambda_3,
                    "u1": u1,
                    "u2": u2,
                    "u3": u3,
                }
            )
    elif method == "zeppelin_WM":
        lambda_1 = 2.07481576e-9
        lambda_2 = 0.66e-9
        lambda_3 = 0.66e-9
        for _ in range(n):
            u1 = np.random.uniform(*u1_range)
            u2 = np.random.uniform(*u2_range)
            u3 = np.random.uniform(*u3_range)
            params.append(
                {
                    "lambda_1": lambda_1,
                    "lambda_2": lambda_2,
                    "lambda_3": lambda_3,
                    "u1": u1,
                    "u2": u2,
                    "u3": u3,
                }
            )
    elif method == "zeppelin_GM":
        lambda_1 = 1.7e-9
        lambda_2 = 0.46651824e-9
        lambda_3 = 0.46651824e-9
        for _ in range(n):
            u1 = np.random.uniform(*u1_range)
            u2 = np.random.uniform(*u2_range)
            u3 = np.random.uniform(*u3_range)
            params.append(
                {
                    "lambda_1": lambda_1,
                    "lambda_2": lambda_2,
                    "lambda_3": lambda_3,
                    "u1": u1,
                    "u2": u2,
                    "u3": u3,
                }
            )
    elif method == "zeppelin_CSF":
        lambda_1 = 2.17e-9
        lambda_2 = 1.06e-9
        lambda_3 = 1.06e-9
        for _ in range(n):
            u1 = np.random.uniform(*u1_range)
            u2 = np.random.uniform(*u2_range)
            u3 = np.random.uniform(*u3_range)
            params.append(
                {
                    "lambda_1": lambda_1,
                    "lambda_2": lambda_2,
                    "lambda_3": lambda_3,
                    "u1": u1,
                    "u2": u2,
                    "u3": u3,
                }
            )
    elif method == "random_trace_spheres":
        u1 = 1.0
        u2 = 0.0
        u3 = 0.0
        for _ in range(n):
            lambda_1 = np.random.uniform(*lambda_1_range)
            lambda_2, lambda_3 = lambda_1, lambda_1
            params.append(
                {
                    "lambda_1": lambda_1,
                    "lambda_2": lambda_2,
                    "lambda_3": lambda_3,
                    "u1": u1,
                    "u2": u2,
                    "u3": u3,
                }
            )
    elif method == "random_trace_spheres":
        u1 = 1.0
        u2 = 0.0
        u3 = 0.0
        for _ in range(n):
            lambda_1 = np.random.uniform(*lambda_1_range)
            lambda_2, lambda_3 = lambda_1, lambda_1
            params.append(
                {
                    "lambda_1": lambda_1,
                    "lambda_2": lambda_2,
                    "lambda_3": lambda_3,
                    "u1": u1,
                    "u2": u2,
                    "u3": u3,
                }
            )
    elif method == "background":
        # Background method: generate a mix of needles, plates, and spheres.

        # Needle: one eigenvalue should be large,
        # while the other two the same, directions random
        for _ in range(needle_count):
            lambda_val = np.random.uniform(*lambda_1_range_needle)
            lambda_2 = 0.12e-9
            lambda_3 = 0.12e-9
            # Needles lie in xyz.
            u1 = np.random.uniform(*u1_range)
            u2 = np.random.uniform(*u2_range)
            u3 = np.random.uniform(*u3_range)
            params.append(
                {
                    "lambda_1": lambda_val,
                    "lambda_2": lambda_2,
                    "lambda_3": lambda_3,
                    "u1": u1,
                    "u2": u2,
                    "u3": u3,
                }
            )
        # Plate: two eigenvalue should be large and the same
        # while the third small, directions random
        for _ in range(plate_count):
            lambda_val = np.random.uniform(*lambda_1_range_plate)
            lambda_3 = 0.12e-9
            # Randomly sample the eigenvector components (all three directions)
            u1 = np.random.uniform(*u1_range)
            u2 = np.random.uniform(*u2_range)
            u3 = np.random.uniform(*u3_range)
            params.append(
                {
                    "lambda_1": lambda_val,
                    "lambda_2": lambda_val,
                    "lambda_3": lambda_3,
                    "u1": u1,
                    "u2": u2,
                    "u3": u3,
                }
            )
        # Sphere: all eigenvalues are equal and random, directions fixed
        for _ in range(sphere_count):
            # Sample one value and set all eigenvalues equal.
            val = np.random.uniform(*lambda_1_range_sphere)
            lambda_1 = lambda_2 = lambda_3 = val
            u1 = 1.0
            u2 = 0.0
            u3 = 0.0
            params.append(
                {
                    "lambda_1": lambda_1,
                    "lambda_2": lambda_2,
                    "lambda_3": lambda_3,
                    "u1": u1,
                    "u2": u2,
                    "u3": u3,
                }
            )
    elif method == "fibers_WM_watson":
        # Optional: load your 500-pt ER grid as udirs if you have it (shape [M,3]):
        udirs = np.loadtxt("utils/er_500_directions.txt")  # example
        udirs = None  # fallback to spherical Fibonacci if None
        new_params = generate_fibers_WM_watson(
            n=n, mu=(1,0,0), odi=0.23,
            base_d_par=1.7e-9, base_d_perp=3.0e-10,
            sigma_iso=0.15, sigma_delta=0.3,
            udirs=udirs, grid_size=2000
        )
        params.extend(new_params)

    elif method == "fibers_GM_distr":
        # No Watson; sample size/shape and assign random directions
        new_params = generate_dtd_gauss_random_orientations(
            n=n,
            base_d_par= 1.33025e-09, base_d_perp=2.3475e-10,
            sigma_iso=0.3, sigma_delta=0.3,
            orientation_mode="independent"   # or "shared"
        )
        params.extend(new_params)
   
    elif method == "GM_ball_distr":
        # No Watson; sample size/shape and assign random directions
        new_params = generate_dtd_gauss_random_orientations(
            n=n,
            base_d_par= 2e-09, base_d_perp=2e-09,
            sigma_iso=0.15, sigma_delta=0,
            orientation_mode="shared"   # or "shared"
        )
        params.extend(new_params)

    elif method == "CSF_distr":
        # No Watson; sample size/shape and assign random directions
        new_params = generate_dtd_gauss_random_orientations(
            n=n,
            base_d_par= 3.0e-09, base_d_perp=3.0e-09,
            sigma_iso=0.0, sigma_delta=0.0,
            orientation_mode="shared"   # or "shared"
        )
        params.extend(new_params)

    else:
        raise ValueError(f"Unknown method: {method}")
    with open(filepath, "w") as file:
        # print(f"✅ Saved generated dtens to {filepath}")
        json.dump(params, file, indent=4)


def calc_dtens(lambda_1, lambda_2, lambda_3, u1, u2, u3):
    """
    Create diffusion tensor from eigenvalues and first eigenvector.

    Parameters:
    lambda_1, lambda_2, lambda_3 (float): Eigenvalues.
    u1, u2, u3 (float): First eigenvector components.
    Tensor is created on coordinate axes and then rotated from x-axis onto first eigenvector.

    Returns:
    numpy.ndarray: 3x3 diffusion tensor.
    """
    d_eig = np.array([[lambda_1, 0, 0], [0, lambda_2, 0], [0, 0, lambda_3]])
    R = get_rot_matrix(np.array([u1, u2, u3]))
    # print(R)

    # ist das wirklich der gedrehte tensor/ellipsoid?
    # alternativ eine R für jede Achse ausrechnen und column-wise anwenden
    d_tensor = np.dot(np.dot(R, d_eig), R.T)
    return d_tensor


# not tested
def plot_single_dtens(d_tensor):
    # Create a grid of points
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Apply the tensor to the grid points
    for i in range(len(x)):
        for j in range(len(x)):
            x[i, j], y[i, j], z[i, j] = (
                d_tensor @ np.array([x[i, j], y[i, j], z[i, j]])
            ).reshape(3)

    # Plot the ellipsoid
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, color="b", alpha=0.6)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")

    # Set equal scaling
    max_range = (
        np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    )
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()


def plot_single_dtens2(d_tensor):
    # If d_tensor is batched (more than two dimensions), use the first tensor.
    if d_tensor.ndim > 2:
        d_tensor = d_tensor[0]

    # Create a grid of points using spherical coordinates
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    X0 = np.outer(np.cos(u), np.sin(v))
    Y0 = np.outer(np.sin(u), np.sin(v))
    Z0 = np.outer(np.ones(np.size(u)), np.cos(v))

    # Flatten the grid points for a vectorized transformation
    pts = np.array([X0.flatten(), Y0.flatten(), Z0.flatten()])

    # Apply the diffusion tensor
    transformed_pts = d_tensor @ pts  # expected shape: (3, 10000)

    # Reshape the transformed coordinates back to the grid shape
    X_trans = transformed_pts[0, :].reshape(X0.shape)
    Y_trans = transformed_pts[1, :].reshape(Y0.shape)
    Z_trans = transformed_pts[2, :].reshape(Z0.shape)

    # Compute bounds for equal aspect ratio
    x_min, x_max = X_trans.min(), X_trans.max()
    y_min, y_max = Y_trans.min(), Y_trans.max()
    z_min, z_max = Z_trans.min(), Z_trans.max()
    range_max = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
    mid_x = (x_max + x_min) / 2.0
    mid_y = (y_max + y_min) / 2.0
    mid_z = (z_max + z_min) / 2.0

    # Create a Plotly surface trace for the ellipsoid
    ellipsoid_surface = go.Surface(
        x=X_trans,
        y=Y_trans,
        z=Z_trans,
        showscale=False,
        opacity=0.9,
        hoverinfo="skip",
        colorscale="Viridis",
        lighting=dict(
            ambient=0.5,
            diffuse=0.7,
            specular=0.2,
            roughness=0.8,
            fresnel=0.2,
        ),
    )

    # Create and configure the Plotly figure for an interactive 3D view
    fig = go.Figure(data=[ellipsoid_surface])
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X-axis", range=[mid_x - range_max, mid_x + range_max]),
            yaxis=dict(title="Y-axis", range=[mid_y - range_max, mid_y + range_max]),
            zaxis=dict(title="Z-axis", range=[mid_z - range_max, mid_z + range_max]),
            aspectmode="cube",  # Ensures equal aspect ratio
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )
    fig.show()


# not tested
def plot_multiple_dtens(
    d_tensors, cube_size=8.7, ellipsoid_size=[0.2, 0.2, 1], margin=1
):
    num_chains = int(np.ceil(len(d_tensors) ** (1 / 3)))
    adjusted_cube_size = cube_size - 2 * margin

    # Create a grid of points for the ellipsoid
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_xlim([0, cube_size])
    ax.set_ylim([0, cube_size])
    ax.set_zlim([0, cube_size])
    ax.grid(True)

    # Plot the cube outline
    for sx in [0, cube_size]:
        for sy in [0, cube_size]:
            ax.plot([sx, sx], [sy, sy], [0, cube_size], "k", linewidth=2)
    for sx in [0, cube_size]:
        for sz in [0, cube_size]:
            ax.plot([sx, sx], [0, cube_size], [sz, sz], "k", linewidth=2)
    for sy in [0, cube_size]:
        for sz in [0, cube_size]:
            ax.plot([0, cube_size], [sy, sy], [sz, sz], "k", linewidth=2)

    # Plot the ellipsoids
    for nx in range(num_chains):
        for ny in range(num_chains):
            for nz in range(num_chains):
                cx = margin + (nx * adjusted_cube_size / (num_chains - 1))
                cy = margin + (ny * adjusted_cube_size / (num_chains - 1))
                cz = margin + (nz * adjusted_cube_size / (num_chains - 1))

                index = nx * num_chains * num_chains + ny * num_chains + nz
                if index < len(d_tensors):
                    d_tensor = d_tensors[index]
                    ellipsoid_points = np.array([x.flatten(), y.flatten(), z.flatten()])
                    transformed_points = (d_tensor @ ellipsoid_points).T

                    x_ellipsoid = transformed_points[:, 0].reshape(x.shape) + cx
                    y_ellipsoid = transformed_points[:, 1].reshape(x.shape) + cy
                    z_ellipsoid = transformed_points[:, 2].reshape(x.shape) + cz

                    ax.plot_surface(
                        x_ellipsoid, y_ellipsoid, z_ellipsoid, color="b", alpha=0.6
                    )

    plt.show()


def plot_multiple_dtens2(d_tensors, cube_size=8.7, ellipsoid_size=[1, 1, 1], margin=1):
    # Determine the grid dimensions based on the number of tensors.
    num_chains = int(np.ceil(len(d_tensors) ** (1 / 3)))
    adjusted_cube_size = cube_size - 2 * margin

    # Create the ellipsoid grid (a unit ellipsoid) and scale it by ellipsoid_size.
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    # Base ellipsoid before scaling:
    X0 = np.outer(np.cos(u), np.sin(v))
    Y0 = np.outer(np.sin(u), np.sin(v))
    Z0 = np.outer(np.ones(np.size(u)), np.cos(v))

    # Scale the ellipsoid points according to ellipsoid_size.
    X0 = ellipsoid_size[0] * X0
    Y0 = ellipsoid_size[1] * Y0
    Z0 = ellipsoid_size[2] * Z0

    # A list to store all plotly traces.
    traces = []

    # Loop over a grid and add ellipsoid surfaces (if a d_tensor is available for the current index).
    for nx in range(num_chains):
        for ny in range(num_chains):
            for nz in range(num_chains):
                # Compute center positions.
                if num_chains > 1:
                    cx = margin + (nx * adjusted_cube_size / (num_chains - 1))
                    cy = margin + (ny * adjusted_cube_size / (num_chains - 1))
                    cz = margin + (nz * adjusted_cube_size / (num_chains - 1))
                else:
                    # If only one ellipsoid, center it in the cube.
                    cx = cy = cz = cube_size / 2

                index = nx * num_chains**2 + ny * num_chains + nz
                if index < len(d_tensors):
                    d_tensor = d_tensors[index]
                    # Flatten the base ellipsoid arrays and combine into a 3 x N array.
                    pts = np.array([X0.flatten(), Y0.flatten(), Z0.flatten()])
                    # Apply the transformation.
                    transformed_pts = np.dot(d_tensor, pts)
                    # Reshape the transformed coordinates to the original grid shape and shift by the center position.
                    x_e = transformed_pts[0, :].reshape(X0.shape) + cx
                    y_e = transformed_pts[1, :].reshape(Y0.shape) + cy
                    z_e = transformed_pts[2, :].reshape(Z0.shape) + cz

                    # Add the ellipsoid as a surface trace with a uniform blue color.
                    ellipsoid_surface = go.Surface(
                        x=x_e,
                        y=y_e,
                        z=z_e,
                        # Use a constant "silver" color represented in RGB (192,192,192)
                        surfacecolor=np.zeros_like(x_e),
                        # colorscale=[[0, "rgb(192,192,192)"], [1, "rgb(192,192,192)"]],
                        colorscale="Viridis",
                        showscale=False,
                        opacity=1.0,
                        lighting=dict(
                            ambient=0.5,
                            diffuse=0.7,
                            specular=0.2,
                            roughness=0.8,
                            fresnel=0.2,
                        ),
                        # Light position helps enhance specular highlights
                        # lightposition=dict(x=100, y=200, z=0),
                        hoverinfo="skip",
                    )
                    traces.append(ellipsoid_surface)

    # Build the cube outline from its vertices.
    # Define the eight vertices of the cube.
    vertices = np.array(
        [
            [0, 0, 0],
            [cube_size, 0, 0],
            [cube_size, cube_size, 0],
            [0, cube_size, 0],
            [0, 0, cube_size],
            [cube_size, 0, cube_size],
            [cube_size, cube_size, cube_size],
            [0, cube_size, cube_size],
        ]
    )

    # Define the 12 edges by listing vertex pairs.
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),  # Bottom face
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),  # Top face
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # Vertical edges
    ]

    # Create lists that hold the x, y, z coordinates of all edges, inserting None between segments.
    cube_x, cube_y, cube_z = [], [], []
    for edge in edges:
        start, end = edge
        cube_x.extend([vertices[start, 0], vertices[end, 0], None])
        cube_y.extend([vertices[start, 1], vertices[end, 1], None])
        cube_z.extend([vertices[start, 2], vertices[end, 2], None])

    # Add the cube outline as a Scatter3d trace.
    cube_trace = go.Scatter3d(
        x=cube_x,
        y=cube_y,
        z=cube_z,
        mode="lines",
        line=dict(color="black", width=4),
        hoverinfo="none",
    )
    traces.append(cube_trace)

    # Create the figure and set up the layout.
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X-axis", range=[0, cube_size]),
            yaxis=dict(title="Y-axis", range=[0, cube_size]),
            zaxis=dict(title="Z-axis", range=[0, cube_size]),
            aspectmode="cube",  # Makes sure the aspect ratio is 1:1:1.
            camera=dict(eye=dict(x=1.25, y=1.25, z=1.25)),
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )

    # Display the interactive plot.
    fig.show()


def plot_voxel_dtens_cloud(
    d_tensors,
    voxel_size=1.0,
    ellipsoid_scale=0.07,
    max_tensors=180,
    seed=42,
    opacity=0.55,
    color_by="fa",
    show_voxel_box=True,
):
    """
    Plot a DTD as an ellipsoid cloud inside one voxel using eigen-decomposed tensors.

    The geometry is derived from tensor eigensystems (principal axes and lengths),
    which is more faithful than directly multiplying a sphere by the tensor matrix.
    """
    d_tensors = np.asarray(d_tensors, dtype=float)
    if d_tensors.ndim != 3 or d_tensors.shape[1:] != (3, 3):
        raise ValueError("d_tensors must have shape (N, 3, 3)")

    metric = str(color_by).lower().strip()
    if metric not in {"fa", "diso"}:
        raise ValueError("color_by must be either 'fa' or 'diso'")

    rng = np.random.default_rng(seed)
    n_total = d_tensors.shape[0]
    if n_total == 0:
        raise ValueError("d_tensors is empty")

    if n_total > max_tensors:
        idx = rng.choice(n_total, size=max_tensors, replace=False)
        d_use = d_tensors[idx]
    else:
        d_use = d_tensors
    n_use = d_use.shape[0]

    # Spherical template for all ellipsoids.
    u = np.linspace(0, 2 * np.pi, 24)
    v = np.linspace(0, np.pi, 18)
    x0 = np.outer(np.cos(u), np.sin(v))
    y0 = np.outer(np.sin(u), np.sin(v))
    z0 = np.outer(np.ones(np.size(u)), np.cos(v))
    pts = np.array([x0.ravel(), y0.ravel(), z0.ravel()])

    # Stratified random centers reduce severe overlap while keeping a cloud look.
    n_grid = int(np.ceil(n_use ** (1.0 / 3.0)))
    grid_vals = np.linspace(0.5 / n_grid, 1.0 - 0.5 / n_grid, n_grid)
    gx, gy, gz = np.meshgrid(grid_vals, grid_vals, grid_vals, indexing="ij")
    centers = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    rng.shuffle(centers)
    centers = centers[:n_use]
    jitter = (rng.random((n_use, 3)) - 0.5) * (0.45 / n_grid)
    centers = np.clip(centers + jitter, 0.03, 0.97) * voxel_size

    eigvals = np.zeros((n_use, 3), dtype=float)
    eigvecs = np.zeros((n_use, 3, 3), dtype=float)
    color_vals = np.zeros(n_use, dtype=float)

    for i, d in enumerate(d_use):
        d_sym = 0.5 * (d + d.T)
        lam, vec = np.linalg.eigh(d_sym)
        order = np.argsort(lam)[::-1]
        lam = np.maximum(lam[order], 0.0)
        vec = vec[:, order]

        eigvals[i] = lam
        eigvecs[i] = vec

        if metric == "diso":
            color_vals[i] = float(np.mean(lam))
        else:
            lam_mean = float(np.mean(lam))
            num = 1.5 * np.sum((lam - lam_mean) ** 2)
            den = np.sum(lam ** 2) + 1e-18
            color_vals[i] = float(np.sqrt(num / den))

    cmin = float(np.min(color_vals))
    cmax = float(np.max(color_vals))
    if np.isclose(cmin, cmax):
        cmax = cmin + 1e-12

    # Normalize ellipsoid sizes by robust reference of major eigenvalue.
    lam_ref = float(np.percentile(eigvals[:, 0], 95))
    lam_ref = max(lam_ref, 1e-18)

    traces = []
    for i in range(n_use):
        lam = eigvals[i]
        vec = eigvecs[i]

        # Semi-axis lengths scale with sqrt(lambda) for diffusion ellipsoid geometry.
        axes = ellipsoid_scale * np.sqrt(np.maximum(lam, 0.0) / lam_ref)
        transformed = vec @ (axes[:, None] * pts)

        cx, cy, cz = centers[i]
        xe = transformed[0, :].reshape(x0.shape) + cx
        ye = transformed[1, :].reshape(y0.shape) + cy
        ze = transformed[2, :].reshape(z0.shape) + cz

        traces.append(
            go.Surface(
                x=xe,
                y=ye,
                z=ze,
                surfacecolor=np.full_like(xe, color_vals[i]),
                colorscale="Turbo",
                cmin=cmin,
                cmax=cmax,
                showscale=(i == 0),
                colorbar=(
                    dict(title=("FA" if metric == "fa" else "D_iso"), len=0.72)
                    if i == 0
                    else None
                ),
                opacity=opacity,
                hoverinfo="skip",
                lighting=dict(ambient=0.55, diffuse=0.9, specular=0.12, roughness=0.95),
                lightposition=dict(x=500, y=350, z=800),
            )
        )

    if show_voxel_box:
        vertices = np.array(
            [
                [0, 0, 0],
                [voxel_size, 0, 0],
                [voxel_size, voxel_size, 0],
                [0, voxel_size, 0],
                [0, 0, voxel_size],
                [voxel_size, 0, voxel_size],
                [voxel_size, voxel_size, voxel_size],
                [0, voxel_size, voxel_size],
            ]
        )
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        cube_x, cube_y, cube_z = [], [], []
        for s, e in edges:
            cube_x.extend([vertices[s, 0], vertices[e, 0], None])
            cube_y.extend([vertices[s, 1], vertices[e, 1], None])
            cube_z.extend([vertices[s, 2], vertices[e, 2], None])

        traces.append(
            go.Scatter3d(
                x=cube_x,
                y=cube_y,
                z=cube_z,
                mode="lines",
                line=dict(color="black", width=3),
                hoverinfo="none",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", range=[0, voxel_size], showbackground=False),
            yaxis=dict(title="Y", range=[0, voxel_size], showbackground=False),
            zaxis=dict(title="Z", range=[0, voxel_size], showbackground=False),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.45, y=1.15, z=0.95)),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title=f"DTD voxel cloud (shown {n_use}/{n_total}, color={metric})",
    )
    fig.show()


def plot_multiple_dtens3(
    d_tensors,
    cube_size=8.7,
    ellipsoid_size=[1, 1, 1],
    margin=1,
    safe_margin=0.3,
    seed=228,
    show_boundaries=True,
):
    if seed is not None:
        np.random.seed(seed)
    N = len(d_tensors)

    # Create the base ellipsoid (unit sphere) grid using spherical coordinates
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    X0 = np.outer(np.cos(u), np.sin(v))
    Y0 = np.outer(np.sin(u), np.sin(v))
    Z0 = np.outer(np.ones(np.size(u)), np.cos(v))

    # Scale the base sphere by ellipsoid_size
    X0 = ellipsoid_size[0] * X0
    Y0 = ellipsoid_size[1] * Y0
    Z0 = ellipsoid_size[2] * Z0

    traces = []

    # Generate random centers for each diffusion tensor.
    # The centers are chosen uniformly from [margin, cube_size - margin] for each axis.
    centers = np.random.uniform(margin, cube_size - margin, size=(N, 3))

    # Initialize global bounds for later dynamic cube boundary computation.
    global_x_min, global_y_min, global_z_min = np.inf, np.inf, np.inf
    global_x_max, global_y_max, global_z_max = -np.inf, -np.inf, -np.inf

    # Loop over each tensor and its corresponding random center.
    for i, d_tensor in enumerate(d_tensors):
        cx, cy, cz = centers[i]
        # Flatten the ellipsoid grid to apply the transformation vectorized.
        pts = np.array([X0.flatten(), Y0.flatten(), Z0.flatten()])
        transformed_pts = d_tensor @ pts
        # Reshape transformed coordinates back into grid shape and shift by the random center.
        x_e = transformed_pts[0, :].reshape(X0.shape) + cx
        y_e = transformed_pts[1, :].reshape(Y0.shape) + cy
        z_e = transformed_pts[2, :].reshape(Z0.shape) + cz

        # Update the global bounds.
        global_x_min = min(global_x_min, x_e.min())
        global_y_min = min(global_y_min, y_e.min())
        global_z_min = min(global_z_min, z_e.min())
        global_x_max = max(global_x_max, x_e.max())
        global_y_max = max(global_y_max, y_e.max())
        global_z_max = max(global_z_max, z_e.max())

        # Add the ellipsoid surface trace with a uniform blue color.
        ellipsoid_surface = go.Surface(
            x=x_e,
            y=y_e,
            z=z_e,
            colorscale="gray",
            showscale=False,
            opacity=0.9,
            lighting=dict(
                ambient=0.8,
                diffuse=0.8,
                specular=0.2,
                roughness=0.9,
                fresnel=0.1,
            ),
            hoverinfo="skip",
        )
        traces.append(ellipsoid_surface)

    # Adjust overall global bounds by adding a safe margin.
    global_min = min(global_x_min, global_y_min, global_z_min) - safe_margin
    global_max = max(global_x_max, global_y_max, global_z_max) + safe_margin

    # Only add boundary cube trace if show_boundaries is True
    if show_boundaries:
        # Build the vertices of the outer cube from the dynamic bounds.
        vertices = np.array(
            [
                [global_min, global_min, global_min],
                [global_max, global_min, global_min],
                [global_max, global_max, global_min],
                [global_min, global_max, global_min],
                [global_min, global_min, global_max],
                [global_max, global_min, global_max],
                [global_max, global_max, global_max],
                [global_min, global_max, global_max],
            ]
        )

        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # Bottom face
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),  # Top face
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # Vertical edges
        ]

        cube_x, cube_y, cube_z = [], [], []
        for start, end in edges:
            cube_x.extend([vertices[start, 0], vertices[end, 0], None])
            cube_y.extend([vertices[start, 1], vertices[end, 1], None])
            cube_z.extend([vertices[start, 2], vertices[end, 2], None])

        cube_trace = go.Scatter3d(
            x=cube_x,
            y=cube_y,
            z=cube_z,
            mode="lines",
            line=dict(color="black", width=4),
            hoverinfo="none",
        )
        traces.append(cube_trace)

    # Create the interactive Plotly figure with axes ranges based on the global dynamic bounds.
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.25, y=1.25, z=1.25)),
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )
    fig.show()

def read_xps_mat(file_path):
    # Load .mat file
    container = sio.loadmat(file_path)

    # Extract "xps" dictionary
    xps_data = container["xps"][0, 0]  # Assuming the first element contains the data

    return pd.DataFrame(xps_data["bt"], dtype=np.float64).to_numpy()
'''
def read_xps_mat(file_path):
    DEPR, wir brauchen nur bt, Rest wird sowieso nicht genutzt.

    # Check if file is a NumPy array file
    if file_path.endswith(".npy"):
        # Simply load and return the NumPy array
        return np.load(file_path)

    # For .mat files, proceed with the original logic
    # Load .mat file
    container = sio.loadmat(file_path)

    # Extract "xps" dictionary
    xps_data = container["xps"][0, 0]  # Assuming the first element contains the data

    # Define field keys in order
    field_keys = ["n", "b", "b_delta", "b_eta", "bt", "u", "s_ind"]

    # Initialize dictionary to store processed data
    processed_data = {}

    for i, key in enumerate(field_keys):
        value = xps_data[i]  # Extract the field value

        if value.shape == (1, 1):  # Single scalar value
            processed_data[key] = value.item()  # Extract single value

        elif value.shape[1] == 1:  # Column vector (561,1) → Flatten to 1D
            processed_data[key] = value.flatten()

        else:  # Matrices (561,6) or (561,3) → Keep as 2D arrays
            processed_data[key] = value

    # Convert to Pandas DataFrame (excluding "n", "s_ind", "b_eta")
    df = pd.DataFrame(
        {
            "b": processed_data["b"],
            "b_delta": processed_data["b_delta"],
        }
    )

    return pd.DataFrame(processed_data["bt"], dtype=np.float64).to_numpy()
'''

def print_phantom_parameter_stats(
    phantom, parameter="md", scaling_factor=1.0, unit="m²/s"
):
    """
    Print statistics about a specific parameter for the phantom and its ROIs.

    Parameters:
    -----------
    phantom : QTI_Phantom
        The phantom object containing the parameter data
    parameter : str
        The parameter to display (e.g., "md", "fa", "ufa", "c_md", "c_c")
    scaling_factor : float
        Factor to scale the parameter values for display (e.g., 1e9 for nm²/s)
    unit : str
        Unit to display with the parameter value
    """
    # Check if parameter exists
    if parameter not in phantom.dps:
        print(f"Parameter '{parameter}' not found in phantom parameters.")
        return

    # Display parameter for phantom background
    param_value = phantom.dps[parameter][0] * scaling_factor
    print(f"The phantom background has {parameter}: {param_value:.3e} {unit}")

    # Display parameter for each ROI
    for i, roi in enumerate(phantom.rois):
        if parameter in roi.dps:
            roi_param_value = roi.dps[parameter][0] * scaling_factor
            print(
                f"The ROI (ID: {roi.roi_id}, POI: {roi.poi_id}) "
                f"has {parameter}: {roi_param_value:.3e} {unit}"
            )

    # Calculate average across all ROIs
    roi_values = [roi.dps[parameter][0] for roi in phantom.rois if parameter in roi.dps]
    if roi_values:
        avg_value = sum(roi_values) / len(roi_values) * scaling_factor
        print(f"\nAverage {parameter} across all ROIs: {avg_value:.3e} {unit}")

        # Calculate min and max
        min_value = min(roi_values) * scaling_factor
        max_value = max(roi_values) * scaling_factor
        print(f"Min {parameter}: {min_value:.3e} {unit}")
        print(f"Max {parameter}: {max_value:.3e} {unit}")

# Moritz added for btens
#--------------------------------------------------------------------------------------------------------------------------------------------------------
def _normalize_U(U):
    U = np.asarray(U, dtype=float)
    if U.ndim == 1: U = U[None, :]
    n = np.linalg.norm(U, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return U / n

def build_btens_shell(b, b_delta, U):
    if not (-0.5 - 1e-12 <= b_delta <= 1.0 + 1e-12):
        raise ValueError("b_delta must be in [-0.5, 1.0]")
    U = _normalize_U(U)
    return [{"b": float(b), "b_delta": float(b_delta),
             "u1": float(u[0]), "u2": float(u[1]), "u3": float(u[2])} for u in U]

def design_btens_json(shells, filepath=None):
    lists = [build_btens_shell(s["b"], s["b_delta"], s["U"]) for s in shells]
    combined = sum(lists, [])
    if filepath is not None:
        with open(filepath, "w") as f: json.dump(combined, f, indent=2)
    return combined

# ===== Build direction lists U_* per shell (unit vectors) =====
def diff_directions(mode, explicit=None, n=None, base=None, seed=None):
    if mode == "explicit" and explicit is not None:
        U = [tuple(np.array(u, float)/np.linalg.norm(u)) for u in explicit]
    elif mode == "repeat":
        u = np.array(base, float); u /= np.linalg.norm(u)
        U = [tuple(u)] * int(n)
    elif mode == "orthogonal_3":
        U = [(1,0,0),(0,1,0),(0,0,1)]
    elif mode == "platonic_6":
        # Vertices of an icosahedron (6 axes, 12 points, but 6 unique directions)
        U = [
            (0.525731, 0.850651, 0),
            (0.525731, -0.850651, 0),
            (0.850651, 0, 0.525731),
            (-0.850651, 0, 0.525731),
            (0, 0.525731, 0.850651),
            (0, 0.525731, -0.850651),
        ]

    elif mode == "platonic_10":
        # Vertices of a dodecahedron (10 unique directions)
        U = [
            (0.57735, 0.57735, 0.57735),
            (0, 0.934172, 0.356822),
            (0.356822, 0, 0.934172),
            (-0.356822, 0, 0.934172),
            (-0.57735, 0.57735, 0.57735),
            (0, 0.934172, -0.356822),
            (0.57735, 0.57735, -0.57735),
            (0.934172, 0.356822, 0),
            (0.57735, -0.57735, 0.57735),
            (0.934172, -0.356822, 0),
        ]

    else:
        raise ValueError(f"Unknown mode: {mode}")
    return U

def calc_btens_bbdelta(b, b_delta, u1, u2, u3):
    """
    Axisymmetric B-tensor from (b, b_delta, u).

    B = (b/3) I + b*b_delta * (u u^T - I/3)
    Eigenvalues: b_par=(b/3)(1+2 b_delta), b_perp=(b/3)(1-b_delta)
    """
    u = np.array([u1, u2, u3], dtype=float)
    n = np.linalg.norm(u)
    if n == 0:
        raise ValueError("u must be non-zero")
    u /= n

    I = np.eye(3)
    U = np.outer(u, u)
    B = (b / 3.0) * I + (b * b_delta) * (U - I / 3.0)
    return B

def load_btens_from_file(filepath):
    """
    Load a JSON list of B-tensors and return an array of 3×3 B-tensors.

    Supports two schemas per entry:
      A) {"b1","b2","b3","u1","u2","u3"}          # explicit eigenvalues
      B) {"b","b_delta","u1","u2","u3"}           # axisymmetric param
         (also accepts {"b","b_delta","u":[u1,u2,u3]})

    Returns
    -------
    np.ndarray, shape (N,3,3)
    """
    with open(filepath, "r") as file:
        data = json.load(file)

    if not (isinstance(data, list) and all(isinstance(e, dict) for e in data)):
        raise ValueError("Invalid data format in JSON file")

    b_tensors = []
    for e in data:
        # Schema A: three eigenvalues + direction
        if {"b1","b2","b3","u1","u2","u3"} <= e.keys():
            B = calc_dtens(e["b1"], e["b2"], e["b3"], e["u1"], e["u2"], e["u3"]) # brauchen nicht 2x gleiche funktion, ggf. namen general. tensor_from_eigvals oder so
            b_tensors.append(B)
            continue

        # Schema B: (b, b_delta, u)
        if {"b","b_delta","u1","u2","u3"} <= e.keys():
            B = calc_btens_bbdelta(e["b"], e["b_delta"], e["u1"], e["u2"], e["u3"])
            b_tensors.append(B)
            continue

        # Optional compact vector form for u
        if {"b","b_delta","u"} <= e.keys() and isinstance(e["u"], (list, tuple)) and len(e["u"]) == 3:
            u1, u2, u3 = e["u"]
            B = calc_btens_bbdelta(e["b"], e["b_delta"], u1, u2, u3)
            b_tensors.append(B)
            continue

        raise ValueError(
            "Each JSON entry must have either "
            "{b1,b2,b3,u1,u2,u3} or {b,b_delta,u1,u2,u3} (or 'u':[u1,u2,u3])."
        )

    return np.asarray(b_tensors)

def load_btens_any(source):
    """
    Accepts:
      • path to JSON list (either {b1,b2,b3,u1,u2,u3} or {b,b_delta,u1,u2,u3}),
      • path to .mat/.npy xps (expects field 'bt' -> (N,6)),
      • or a numpy array shaped (N,3,3) or (N,6).
    Returns:
      • (N,6) B-tensors in this project's Voigt convention.
    """
    if isinstance(source, str):
        if source.endswith(".json"):
            B33 = load_btens_from_file(source)      # -> (N,3,3)
            return convert_3x3_to_1x6(B33)          # -> (N,6)
        elif source.endswith((".mat", ".npy")):
            BT = read_xps_mat(source)               # -> (N,6) from xps['bt']
            return np.asarray(BT)
        else:
            raise ValueError("Unknown file type. Use .json, .mat, or .npy.")
    else:
        arr = np.asarray(source)
        if arr.ndim == 3 and arr.shape[-2:] == (3, 3):
            return convert_3x3_to_1x6(arr)
        if arr.ndim == 2 and arr.shape[-1] == 6:
            return arr
        raise ValueError("Array must be (N,3,3) or (N,6).")