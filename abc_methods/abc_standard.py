import math
import time
import heapq
from typing import List, Tuple, Optional
import numpy as np
from utils import push_candidate, evaluate_candidate

Params = Tuple[float, float]  # (birth_rate, death_rate)

def abc_method(
    u0: np.ndarray,
    dx: float,
    dt: float,
    T: float,
    L: float,
    D: float,
    Nx: int,
    lattice_threshold: float,
    observed_data: np.ndarray,
    time_points: List[float],
    tolerance: float,
    number_posterior_samples: int,
    algorithm: str = "FK",
    timeout: float = 18_000.0,
    start_time: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    prior_birth: Tuple[float, float] = (0.0, 0.5),
    prior_death: Tuple[float, float] = (0.0, 1e-4),
    min_death: float = 1e-6,
) -> List[Params]:
    """
    ABC rejection method with top-N selection.
    Keeps the best parameter sets (σ, ν) based on worst-case Euclidean distance.

    Returns:
        A list of parameter tuples [(σ, ν), ...], ordered best → worst.
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

    # Min-heap for storing (score, params)
    heap: List[Tuple[float, Params]] = []

    # Main loop
    while (time.time() - start_time) < timeout:
        birth = rng.uniform(*prior_birth)
        death = rng.uniform(*prior_death)
        score = evaluate_candidate(birth,death)

        # Keep only top-N
        if score is not None:
            if (len(heap) < number_posterior_samples) or (
                score < heapq.nlargest(1, heap)[0][0]
            ):
                push_candidate(score, (birth, death))

    best_sorted = sorted(heap, key=lambda z: z[0])
    print("Timeout reached. Returning the best parameters found so far.")
    return [params for _, params in best_sorted]
