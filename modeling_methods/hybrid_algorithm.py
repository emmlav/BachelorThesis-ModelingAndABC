from gillepsie import state_update
import numpy as np

def probabilistic_round(value):
    """Rounds a value probabilistically to conserve mass."""
    try:
        integer_part = int(value)
    except OverflowError:
        return -1
    fractional_part = value - integer_part
    if np.random.rand() < fractional_part:
        return integer_part + 1  # Round up with probability equal to the fractional part
    else:
        return integer_part  # Round down

def numericalSolution(D, kI,dx, dt, birth_coefficent, death_coefficent, u0,carrying_capacity_hybrid ):
    """
    Applies a finite difference scheme for the deterministic region of the hybrid model.
    Handles boundary and interface conditions, and probabilistic rounding at the interface.
    """
    N = u0.copy()
    N1 = np.zeros(len(u0))  # Maintain same size as u0
    N[kI] = N[kI]/dx
    # Apply finite difference scheme for the deterministic region
    for i in range(1, kI):  # Exclude stochastic domain
        if N[i]> 1e10:
            raise ValueError
        diffusion_term = D * (N[i+1] - 2*N[i] + N[i-1]) / dx**2
        reaction_term = N[i] * (birth_coefficent - death_coefficent * N[i])
        N1[i] = N[i] + dt * (diffusion_term + reaction_term)

    # Dirichlet boundary condition at x = 0
    N1[0] = carrying_capacity_hybrid

    # Neumann boundary condition at the last deterministic point (kI)
    N1[kI+1] = N1[kI-1]  

    # Apply interface update equation at kI
    N1[kI] = N[kI] + dt * (D / dx**2) * (-N[kI] + N[kI-1])

    # Keep stochastic region unchanged
    N1[kI+1:] = u0[kI+1:]

    # Probabilistic rounding at kI (if needed)
    N1[kI] = probabilistic_round(N1[kI]*dx)
    
    return N1

def hybrid_algorithm( D, T, L, dx, dt, birth_prob,  observed_times,carrying_capacity, lattice_threshold):
    """
    Simulates the hybrid reaction-diffusion process combining deterministic and stochastic models.
    Handles interface movement and stores results at specified observation times.
    """
    Nx = int(L / dx)
    diffusion_prob = D / dx**2

    result_array = []
    time_result = []
    death_prob = birth_prob / carrying_capacity

    # Initial condition
    N = np.zeros(Nx)
    N[:10] = carrying_capacity
    
    interface_idx = np.argmax(N < lattice_threshold)
    t = 0
    
    while t < T+2:
        # Compute stochastic propensities for the stochastic region
        W = np.zeros((Nx - interface_idx, 4))
        for i in range(interface_idx, Nx):
            if N[i] > 0:
                if N[i]> 1e10: #check for overflow
                    return  result_array, time_result
                W[i - interface_idx, 0] = N[i] * death_prob * (N[i] - 1)  # Death
                W[i - interface_idx, 1] = N[i] * birth_prob  # Birth
        
        W[:, 2] = N[interface_idx:] * diffusion_prob  # Diffusion left
        W[:-1, 3] = N[interface_idx:-1] * diffusion_prob  # Diffusion right
        
        cumulative_propensity = np.sum(W, axis=1)
        total_propensity = np.sum(cumulative_propensity)
        
        r1, r2 = np.random.random(2)
        tau = np.log(1.0 / r1) / total_propensity if total_propensity > 0 else dt
        tau = min(tau,0.1) # dont allow big tau
        t += tau

        # Select compartment and reaction
        sum_prob = 0
        selected_compartment, selected_reaction = None, None
        for j in range(0,Nx - interface_idx):
            if total_propensity == 0:
                break 
            for i in range(4):
                sum_prob += W[j, i]
                if sum_prob >= (r2 * total_propensity):
                    selected_compartment, selected_reaction = j + interface_idx , i
                    break
            if selected_compartment is not None:
                break
        
        if selected_compartment == interface_idx and selected_reaction == 2: ## flux to the deterministic part:
            N[interface_idx]= N[interface_idx]-1
            N[interface_idx-1] = N[interface_idx-1] +  1/dx 
        else:
        # Apply state update
            N = state_update(N, selected_compartment, selected_reaction,Nx)
        
        try:
            N =  numericalSolution(D, interface_idx,  dx,tau,birth_prob,death_prob,N,carrying_capacity)
        except ValueError:
            return np.array(result_array), time_result

        # Update interface dynamically
        new_interface_idx = np.argmax(N < lattice_threshold)
        if new_interface_idx > Nx-3:
            return result_array, time_result
        if new_interface_idx > interface_idx : # moves right
            N[interface_idx] = probabilistic_round(N[interface_idx]*dx)  
        elif  new_interface_idx < interface_idx: # moves left
             N[interface_idx] =  N[interface_idx]/dx
        
        interface_idx = new_interface_idx
        
        # Store results
        if t - observed_times[0] >= 0:
            time_result.append(t)
            result_array.append(N.copy())
            observed_times = observed_times[1:]
        
        if  len(observed_times)==0:
            break 

    return np.array(result_array), np.array(time_result)