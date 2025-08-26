import numpy as np

def compute_euclidean_distance(simulated_data: np.ndarray,
                               observed_data: np.ndarray)
                                -> np.ndarray:
    """
    Compute the Euclidean distance between simulated and observed data
    at each time step.

    Args:
        simulated_data: Array of simulated values with shape (T, n_features).
        observed_data: Array of observed values with the same shape.

    Returns:
        np.ndarray: 1D array of Euclidean distances per time step.
    """
    if simulated_data.shape != observed_data.shape:
        raise ValueError("Simulated and observed data must have the same shape.")

    diff = simulated_data - observed_data
    return np.linalg.norm(diff, axis=1)

