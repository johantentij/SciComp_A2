import numpy as np
from scipy.spatial.distance import pdist

def correlation_dimension(cluster):
    """
    calculate correlation dimension (D_2) by counting pairs within distance r and fitting C(r) ~ r^{D_2}
    """
    i_coords, j_coords = np.where(cluster)
    points = np.column_stack((i_coords, j_coords))
    
    if len(points) < 10:
        return 0.0
        
    distances = pdist(points, metric='euclidean')
    
    # select fitting range to avoid boundary effects
    r_min = 2.0
    r_max = max(np.max(i_coords) - np.min(i_coords), np.max(j_coords) - np.min(j_coords)) / 3.0
    
    if r_max <= r_min:
        return 0.0
        
    r_vals = np.logspace(np.log10(r_min), np.log10(r_max), 15)
    C_r = np.zeros_like(r_vals)
    
    for idx, r in enumerate(r_vals):
        # count point pairs with distance less than r
        C_r[idx] = np.sum(distances < r)
        
    # log-log linear regression
    valid = C_r > 0
    if np.sum(valid) < 3:
        return 0.0
        
    log_r = np.log(r_vals[valid])
    log_C = np.log(C_r[valid])
    
    coeffs = np.polyfit(log_r, log_C, 1)
    return coeffs[0]

def radius_of_gyration(cluster):
    """
    Calculate the radius of gyration for the cluster, a robust measure of its size.
    """
    points = np.argwhere(cluster)
    if len(points) < 2:
        return 0.0
    center_of_mass = np.mean(points, axis=0)
    squared_distances = np.sum((points - center_of_mass)**2, axis=1)
    return np.sqrt(np.mean(squared_distances))

def box_counting_dimension(cluster):
    """
    Calculate the box-counting dimension (D_0).
    """
    points = np.argwhere(cluster)
    if len(points) < 10:
        return 0.0

    # Determine the range of box sizes
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    max_scale = (max_coords - min_coords).max()
    min_scale = 4.0

    if max_scale < min_scale:
        return 0.0

    # Generate box sizes on a logarithmic scale
    box_sizes = np.logspace(np.log10(min_scale), np.log10(max_scale), num=10, dtype=int)
    box_sizes = np.unique(box_sizes)

    counts = []
    for size in box_sizes:
        # Determine which boxes are occupied
        boxed_coords = points // size
        count = len(np.unique(boxed_coords, axis=0))
        counts.append(count)

    # Fit log(count) vs log(size) to find the dimension
    valid = np.array(counts) > 1
    if np.sum(valid) < 3:
        return 0.0

    coeffs = np.polyfit(np.log(box_sizes[valid]), np.log(np.array(counts)[valid]), 1)
    return -coeffs[0]

def calculate_metrics(cluster, seed_j=1):
    i_coords, j_coords = np.where(cluster)
    
    if len(i_coords) == 0:
        return {"Max Height": 0, "Horizontal Spread": 0, "Correlation Dimension": 0.0, "Radius of Gyration": 0.0, "Box-Counting Dimension": 0.0}
        
    max_height = np.max(j_coords) - seed_j
    spread_x = np.std(i_coords)
    d2 = correlation_dimension(cluster)
    rg = radius_of_gyration(cluster)
    d0 = box_counting_dimension(cluster)
    
    return {
        "Max Height": max_height,
        "Horizontal Spread": spread_x,
        "Correlation Dimension": d2,
        "Radius of Gyration": rg,
        "Box-Counting Dimension": d0
    }