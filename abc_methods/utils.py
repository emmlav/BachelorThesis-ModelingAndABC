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


def apply_algorithm(u0,dx,dt,T,L,D, Nx,birth,carrying_capacity,lattice_threshold, time_points, algorithm):

    """ Uses a specific model to simulate given some parameters. The model can be Hybrid, Deterministic or Stochastic.
        It will return the density profiles for the input time_points.
    """
    if algorithm =="fk":
        try:
            return fk(u0,dx,dt,T,L,D,birth,carrying_capacity,time_points) [0]            
        except ValueError:
            raise ValueError
    if algorithm == "gillepsie":
        death_propensity_prob= r / carrying_capacity
        birth_propensity_prob = r 
        diffusionL_propensity_prob = D / dx**2
        diffusionR_propensity_prob = D / dx**2
        reaction_and_diffusions_coefficients = [
            death_propensity_prob,
            birth_propensity_prob,
            diffusionL_propensity_prob,
            diffusionR_propensity_prob,
        ]

        return gillepsie_algorithm(u0, T, reaction_and_diffusions_coefficients, Nx, time_points) [0]
    if algorithm == "hybrid":
        return hybrid_algorithm( D, T, L, dx, dt, birth,  time_points,carrying_capacity, lattice_threshold)[0]
