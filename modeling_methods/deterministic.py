from src.load_parameters import load_parameters 
import numpy as np 
import matplotlib.pyplot as plt 
import time

def deterministic_algorithm(u0, dx, dt, T, L, D, r, carrying_capacity, times_to_return=None):
    """
    Solves the Fisher-Kolmogorov equation using finite differences.
    Returns the solution array and time points, optionally subsampled at specified times.
    """
    # stability check 
    alpha = D * dt / dx ** 2
    if alpha > 1/2:
        print("Stability violated")
    
    Nt = int(T / dt)
    Nx = int(L / dx)
    u = np.zeros((Nt + 1, Nx))  # Nt+1 to include t=0 and t=T
    u[0, :] = u0

    for n in range(Nt):
        for i in range(1, Nx - 1):
            diffusion_term = D * (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / dx**2
            try:
                reaction_term = u[n, i] * (r - (r / carrying_capacity) * u[n, i])
            except RuntimeWarning:
                raise ValueError
            u[n+1, i] = u[n, i] + dt * (diffusion_term + reaction_term)

        # Boundary conditions
        u[n+1, 0] = carrying_capacity
        u[n+1, Nx - 1] = u[n+1, Nx - 2]

    # Full time points
    full_time_points = np.linspace(0, T, Nt + 1)

    if times_to_return is None:
        return u, full_time_points
    else:
        # Find nearest time indices
        indices_to_return = [int(round(t / dt)) for t in times_to_return]

        # Ensure indices are within bounds
        indices_to_return = [i for i in indices_to_return if 0 <= i <= Nt]

        return np.array(u[indices_to_return]), np.array(full_time_points[indices_to_return])

