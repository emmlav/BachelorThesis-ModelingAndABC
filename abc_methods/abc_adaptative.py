import time
import heapq
import numpy as np
from typing import List, Tuple, Optional
from utils import  apply_algorithm, compute_euclidean_distance
Params = Tuple[float, float]  # (birth_rate, death_rate)


def abc_adaptive_prior(
    u0, dx, dt, T, L, D, Nx, lattice_threshold,
    observed_data, time_points, tolerance, N,
    warmup: int = 30,
    algorithm: str = "FK",
    timeout: float = 18_000.0,
    start_time: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    prior_birth: Tuple[float, float] = (0.0, 0.5),
    prior_death: Tuple[float, float] = (0.0, 1e-4),
    min_death: float = 1e-6,
) -> List[Params]:
    """
    Adaptive Prior Approximate Bayesian Computation (ABC) method.
    
    Phase 1: Warm-up from uniform priors, keep N best.
    Phase 2: Proposals drawn from Gaussian fitted to current best.
    
    Args:
        u0, dx, dt, T, L, D, Nx: Model parameters
        lattice_threshold: Hybrid model threshold
        observed_data: Observed trajectories
        time_points: Observation times
        tolerance: Distance tolerance
        N: Number of best parameters to keep
        warmup: Number of warm-up prior samples
        algorithm: Simulation algorithm ("FK", "hybrid", etc.)
        timeout: Maximum run time (seconds)
        start_time: Optional reference start time
        rng: Optional numpy Generator for reproducibility
        prior_birth, prior_death: Prior ranges for (σ, ν)
        min_death: Smallest allowed ν value
    Returns:
        List of best parameter tuples [(σ, ν), ...] ordered best→worst.
    """
            
    def push_candidate(score: float, params: Params):
            """Keep only N best in heap."""
            heapq.heappush(heap, (score, params))
            if len(heap) > N:
                heap[:] = heapq.nsmallest(N, heap)
                heapq.heapify(heap)

    def evaluate_candidate(birth: float, death: float) -> Optional[float]:
        """Run simulation and return score (worst distance), or None if failed."""
        if death < min_death:
            death = min_death
        K = birth / death
        try:
            simulated = apply_algorithm(
                u0, dx, dt, T, L, D, Nx,
                birth, K, lattice_threshold, time_points, algorithm
            )
            distances = compute_euclidean_distance(simulated, observed_data)
            return float(max(distances))
        except (ValueError, np.linalg.LinAlgError):
            return None

    if rng is None:
        rng = np.random.default_rng()
    if start_time is None:
        start_time = time.time()

    # Heap for top-N candidates (score, params)
    heap: List[Tuple[float, Params]] = []

    # -------------------
    # Phase 1: Warm-up
    # -------------------
    while len(heap) < warmup and (time.time() - start_time) < timeout:
        birth = rng.uniform(*prior_birth)
        death = rng.uniform(*prior_death)
        score = evaluate_candidate(birth, death)
        if score is not None:
            push_candidate(score, (birth, death))

    # -------------------
    # Phase 2: Adaptive proposals
    # -------------------
    while (time.time() - start_time) < timeout:
        # Fit Gaussian to best candidates
        accepted = np.array([p for _, p in heap])
        mean = np.mean(accepted, axis=0)
        cov = np.cov(accepted.T) + np.diag([1e-6, 1e-10])  # regularization

        try:
            proposal = rng.multivariate_normal(mean, cov)
        except np.linalg.LinAlgError:
            proposal = rng.uniform([prior_birth[0], prior_death[0]],
                                   [prior_birth[1], prior_death[1]])

        # Clip to valid ranges
        birth = np.clip(proposal[0], *prior_birth)
        death = np.clip(proposal[1], *prior_death)

        # Evaluate and update heap if improved
        score = evaluate_candidate(birth, death)
        if score is not None:
            worst_score = heapq.nlargest(1, heap)[0][0]
            if len(heap) < N or score < worst_score:
                push_candidate(score, (birth, death))

    # Sort best→worst
    best_sorted = sorted(heap, key=lambda z: z[0])
    print("Timeout reached. Returning the best parameters found so far.")
    return [params for _, params in best_sorted]
