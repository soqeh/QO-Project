# ==============================================================================
# Parallelized Standard Model Hypercharge Anomaly Search
#
# Concept & Core Logic: by ChatGPT o3 and Seong-Dong Kim
# Parallelization & Refinements: by Gemini
# Date: June 22, 2025
#
# Description:
# This script performs an exhaustive search for integer vectors (k) that
# satisfy the Standard Model's gauge anomaly cancellation conditions,
# the gcd=1 criterion, and the Gell-Mann-Nishijima relation.
# It uses Python's multiprocessing library to significantly speed up the search
# across all available CPU cores.
# ==============================================================================

import itertools
import json
import math
import multiprocessing

# --- Physical Constants and Field Definitions ---
# These constants will be pre-loaded into each worker process using an initializer
# to ensure robust parallelization and avoid pickling errors.

# Field order: νL, eL, eR, uL, dL, uR, dR
_COLOR = (1, 1, 1, 3, 3, 3, 3)
_IS_QUARK = (0, 0, 0, 1, 1, 1, 1)
_IS_DOUB = (1, 1, 0, 1, 1, 0, 0)
_CHIRAL = (1, 1, -1, 1, 1, -1, -1)  # Left-handed (1), Right-handed (-1, treated as LH conjugate)

# Known physical values for verification
_T3 = (0.5, -0.5, 0, 0.5, -0.5, 0, 0)
_Q = (0, -1, -1, 2/3, -1/3, 2/3, -1/3)


def init_worker(color, is_quark, is_doub, chiral, t3, q):
    """
    Initializer function called once per worker process.
    This pre-loads all necessary constants into the worker's global scope.
    """
    global COLOR, IS_QUARK, IS_DOUB, CHIRAL, T3, Q
    COLOR, IS_QUARK, IS_DOUB, CHIRAL, T3, Q = color, is_quark, is_doub, chiral, t3, q

def anomaly_ok(k):
    """Checks if the given k-vector satisfies all gauge anomaly cancellation conditions."""
    # Calculate effective hypercharges (Y_eff) for left-handed Weyl spinors
    Y_eff = [s * ki / 6.0 for s, ki in zip(CHIRAL, k)]

    # (1) Gravitational Anomaly: Sum[Y_eff] = 0
    if abs(sum(c * y for c, y in zip(COLOR, Y_eff))) > 1e-9: return False
    # (2) U(1)^3 Anomaly: Sum[Y_eff^3] = 0
    if abs(sum(c * (y**3) for c, y in zip(COLOR, Y_eff))) > 1e-9: return False
    # (3) SU(2)^2 * U(1) Anomaly: Sum over doublets [Y_eff] = 0
    if abs(sum(c * y for c, y, d in zip(COLOR, Y_eff, IS_DOUB) if d)) > 1e-9: return False
    # (4) SU(3)^2 * U(1) Anomaly: Sum over quarks [Y_eff] = 0
    if abs(sum(y for y, q in zip(Y_eff, IS_QUARK) if q)) > 1e-9: return False
    
    return True

def is_physically_viable(k):
    """Checks if the k-vector produces the known physical electric charges (Q)."""
    # Calculate standard hypercharges Y = k/6
    Y = [ki / 6.0 for ki in k]
    for i in range(7):
        # Verify using the Gell-Mann-Nishijima relation: Q = T3 + Y
        if abs(Q[i] - (T3[i] + Y[i])) > 1e-9:
            return False
    return True

def check_k_vector(k_vector):
    """
    The main worker function that checks a single k-vector against all conditions.
    Returns the vector if it passes, otherwise returns None.
    """
    # Condition 1: Exclude the trivial solution (all zeros)
    if all(v == 0 for v in k_vector):
        return None
    # Condition 2: Check the Greatest Common Divisor (gcd)
    if math.gcd(*k_vector) != 1:
        return None
    # Condition 3: Check for anomaly cancellation
    if not anomaly_ok(k_vector):
        return None
    # Condition 4: Check for physical viability (correct electric charge)
    if not is_physically_viable(k_vector):
        return None
    
    # If all conditions pass, return the solution
    return list(k_vector)


# --- Main Execution Block ---
if __name__ == "__main__":
    # Define the search space for each integer k_i
    k_range = range(-8, 9)
    
    # Create an iterator for all possible combinations
    possible_k_iterator = itertools.product(k_range, repeat=7)
    total_combinations = len(k_range)**7

    # Prepare arguments for the worker initializer
    init_args = (_COLOR, _IS_QUARK, _IS_DOUB, _CHIRAL, _T3, _Q)

    # Use all available CPU cores
    try:
        num_processes = multiprocessing.cpu_count()
    except NotImplementedError:
        num_processes = 4  # Set a default if detection fails

    print(f"Starting parallel processing on {num_processes} CPU cores...")
    print(f"Search range: k_i in {list(k_range)}")
    print(f"Total combinations to test: {total_combinations:,}")
    print("(This may take a significant amount of time)")

    passed_solutions = []
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_processes, initializer=init_worker, initargs=init_args) as pool:
        # Use imap_unordered for efficient, parallel execution
        results_iterator = pool.imap_unordered(check_k_vector, possible_k_iterator, chunksize=10000)
        
        # Collect results as they are completed
        for result in results_iterator:
            if result is not None:
                passed_solutions.append(result)

    # --- Final Results ---
    output_data = {
        "comment": "Anomaly-free AND physically viable (correct Q) integer solutions with gcd=1",
        "tested_range": f"k in {list(k_range)}",
        "total_combinations_tested": total_combinations,
        "passed_count": len(passed_solutions),
        "solutions": passed_solutions
    }

    print("\n--- Search Complete ---")
    print(json.dumps(output_data, indent=2))
