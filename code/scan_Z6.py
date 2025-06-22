#!/usr/bin/env python3
# scan_Z6.py  --  Exhaustive search for anomaly–free AND physically viable
#                 hyper-charge vectors  (Y_i = k_i / 6,  first generation).

import itertools, json, math, multiprocessing

# --- Static descriptors for the seven SM Weyl fields ---
#   0: ν_L , 1: e_L , 2: e_R ,
#   3: u_L , 4: d_L , 5: u_R , 6: d_R
_COLOR      = (1, 1, 1, 3, 3, 3, 3)        # number of colour copies
_IS_QUARK   = (0, 0, 0, 1, 1, 1, 1)        # quark flag (SU(3) triplet)
_IS_DOUBLET = (1, 1, 0, 1, 1, 0, 0)        # SU(2)_L doublet flag
_CHIRAL     = (1, 1, -1, 1, 1, -1, -1)     # +1 (LH) , –1 (RH → LH†)
_T3         = ( +0.5, -0.5,  0.0,  +0.5, -0.5, 0.0, 0.0 )
_Q          = (  0.0,  -1.0, -1.0, +2/3, -1/3, +2/3, -1/3 )

# ------------------------------------------------------------------------
# Pool initialiser : copies constants into subprocess namespace
def _init_worker(color, is_q, is_d, chiral, t3, q):
    global COLOR, IS_QUARK, IS_DOUB, CHIRAL, T3, Q
    COLOR, IS_QUARK, IS_DOUB  = color, is_q,   is_d
    CHIRAL, T3, Q             = chiral, t3,    q

# ---------- Anomaly cancellation tests ----------------------------------
def _anomaly_free(k):
    """Return True iff the integer vector k_i cancels all gauge anomalies."""
    Y_eff = [s * ki / 6.0 for s, ki in zip(CHIRAL, k)]     # RH → LH† sign flip

    # (Gravity)^2 × U(1)_Y  (linear)
    if abs(sum(c * y               for c, y in zip(COLOR, Y_eff)))       > 1e-9:
        return False
    # U(1)_Y^3  (cubic)
    if abs(sum(c * y**3            for c, y in zip(COLOR, Y_eff)))       > 1e-9:
        return False
    # SU(2)_L^2 × U(1)_Y  (sum over doublets)
    if abs(sum(c * y               for c, y, d in zip(COLOR, Y_eff, IS_DOUBLET) if d)) > 1e-9:
        return False
    # SU(3)_C^2 × U(1)_Y  (sum over colour triplets, one colour factor already in COLOR)
    if abs(sum(       y            for y, q in zip(Y_eff, IS_QUARK) if q)) > 1e-9:
        return False
    return True

# ---------- Physical-charge filter --------------------------------------
def _matches_SM_charge(k):
    """Return True iff Q_i = T3_i + Y_i reproduces the observed electric charges."""
    Y = [ki / 6.0 for ki in k]
    return all(abs(Q[i] - (T3[i] + Y[i])) < 1e-9 for i in range(7))

# ---------- Worker ------------------------------------------------------
def _check_vector(k):
    """Return k (list) if it passes all filters, else None."""
    if all(v == 0 for v in k):                   # trivial vector
        return None
    if math.gcd(*k) != 1:                        # gcd criterion (Lemma)
        return None
    if not _anomaly_free(k):
        return None
    if not _matches_SM_charge(k):
        return None
    return list(k)

# ------------------------------------------------------------------------
if __name__ == "__main__":
    # Search domain :  k_i ∈ {-8, …, +8}
    k_range = range(-8, 9)
    total   = len(k_range) ** 7

    # Spawn worker pool
    n_proc  = multiprocessing.cpu_count()
    print(f"Using {n_proc} CPU cores …")
    print(f"Scanning k_i in {list(k_range)}  (total {total:,} vectors)")

    pool = multiprocessing.Pool(
        processes=n_proc,
        initializer=_init_worker,
        initargs=(_COLOR, _IS_QUARK, _IS_DOUBLET, _CHIRAL, _T3, _Q)
    )

    passed = []
    try:
        for res in pool.imap_unordered(_check_vector,
                                       itertools.product(k_range, repeat=7),
                                       chunksize=10000):
            if res is not None:
                passed.append(res)
    finally:
        pool.close(); pool.join()

    # ---------- Summary JSON --------------------------------------------
    out = {
        "comment": "Anomaly-free AND physically viable solutions with gcd=1",
        "tested_range": f"k_i ∈ [{min(k_range)}, …, {max(k_range)}]",
        "total_vectors": total,
        "passed_count": len(passed),
        "solutions": passed,
    }
    print(json.dumps(out, indent=2))
