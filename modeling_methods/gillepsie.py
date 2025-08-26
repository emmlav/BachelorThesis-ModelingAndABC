import numpy as np
from src.load_parameters import load_parameters 
import matplotlib.pyplot as plt

# State update function for the Gillespie algorithm
def state_update(current_state, state_number, reaction,Nx):
    """
    Updates the state vector according to the selected reaction and compartment.
    """
    if reaction == 0:  # Death
        current_state[state_number] -= 1
    elif reaction == 1:  # Birth
        current_state[state_number] += 1
    elif reaction == 2:  # Diffusion left
        if state_number > 0:
            current_state[state_number] -= 1
            current_state[state_number - 1] += 1
    elif reaction == 3:  # Diffusion right
        if state_number < Nx - 1:
            current_state[state_number] -= 1
            current_state[state_number + 1] += 1
    return current_state

def gillepsie_algorithm(u0, T, reaction_and_diffusions_coefficients, Nx, times_to_return=None):
    """
    Runs the Gillespie stochastic simulation algorithm for a reaction-diffusion system.
    Returns the state and time arrays, optionally subsampled at specified times.
    """
    # Initialization
    time_result = [0.0]
    state_result = [np.array(u0, dtype=np.int32)]
    N = np.array(u0, dtype=np.int32)
    t = 0

    # Initialize propensity matrix W
    W = np.zeros((Nx, 4))
    W[:, 0] = N * reaction_and_diffusions_coefficients[0] * (N - 1)  # Death
    W[:, 1] = N * reaction_and_diffusions_coefficients[1]            # Birth
    W[1:, 2] = N[1:] * reaction_and_diffusions_coefficients[2]       # Diffusion left
    W[0, 2] = 0  # No left diffusion in slot 0
    W[:-1, 3] = N[:-1] * reaction_and_diffusions_coefficients[3]     # Diffusion right
    W[-1, 3] = 0  # No right diffusion in slot -1

    cumulative_propensity_per_compartment = np.sum(W, axis=1)
    total_propensity = np.sum(cumulative_propensity_per_compartment)

    while t < T and total_propensity > 0:

        r1, r2 = np.random.random(2)
        tau = np.log(1.0 / r1) / total_propensity  # Time increment
        t += tau

        # Select compartment and reaction
        sum_prop = 0
        selected_compartment = None
        selected_reaction = None
        found = False

        for j in range(Nx):
            for i in range(4):
                sum_prop += W[j, i]
                if sum_prop >= r2 * total_propensity:
                    selected_compartment = j
                    selected_reaction = i
                    found = True
                    break
            if found:
                break

        # Update the state
        N = state_update(N, selected_compartment, selected_reaction, Nx)
        if N[selected_compartment] < 0:
            print("Negative population detected, stopping simulation.")
            break

        # Update propensities for affected compartments
        for i in [-1, 0, 1]:
            idx = selected_compartment + i
            if 0 <= idx < Nx:
                W[idx, 0] = N[idx] * reaction_and_diffusions_coefficients[0] * (N[idx] - 1)  # Death
                W[idx, 1] = N[idx] * reaction_and_diffusions_coefficients[1]                # Birth
                W[idx, 2] = N[idx] * reaction_and_diffusions_coefficients[2] if idx > 0 else 0  # Diffusion left
                W[idx, 3] = N[idx] * reaction_and_diffusions_coefficients[3] if idx < Nx - 1 else 0  # Diffusion right

        cumulative_propensity_per_compartment = np.sum(W, axis=1)
        total_propensity = np.sum(cumulative_propensity_per_compartment)

        # Store results
        time_result.append(t)
        state_result.append(N.copy())

    time_result = np.array(time_result)
    state_result = np.array(state_result)

    if times_to_return is None:
        return state_result, time_result
    else:
        times_to_return = np.array(times_to_return)
        # For each requested time, find the last simulated time <= requested time
        indices = np.searchsorted(time_result, times_to_return, side='right') - 1
        # Avoid negative indices
        indices[indices < 0] = 0
        return np.array(state_result[indices]), np.array(times_to_return)

def plot_gillespie(state_result, time_result, dx, plot_times=None):
    """
    Plots snapshots of the Gillespie simulation at specified time points.
    """
    Nx = state_result.shape[1]
    x = np.arange(Nx) * dx
    
    if plot_times is None:
        # Select 10 equally spaced time points across the time span
        plot_times = np.linspace(time_result[0], time_result[-1], 10)
    
    plt.figure(figsize=(10, 6))
    
    for t_plot in plot_times:
        # Find index closest to requested time
        idx = np.searchsorted(time_result, t_plot, side='left')
        if idx >= len(time_result):
            idx = len(time_result) - 1
        
        plt.plot(x, state_result[idx], label=f't={time_result[idx]:.2f}')
    
    plt.xlabel('Position')
    plt.ylabel('Population')
    plt.title('Gillespie simulation snapshots')
    plt.legend()
    plt.grid(True)
    plt.show()
