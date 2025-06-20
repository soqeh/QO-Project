# Build a one‑shot Λ–scan module that chains
# (A) 2‑loop EW RG running,
# (B) Hessian stability check,
# and returns (C) the intersection region.

 
"""
core_scan.py  –  A→C one‑shot pipeline
-------------------------------------------------
• Input :   Λ_scan range (TeV) & step
• Output:   Table of Λ values with
            RG_ok, SPD_ok flags + diagnostics

*Fill in* the 2‑loop β‑functions and the true
Hessian routine for your model.  Skeleton below
uses placeholders so that the file runs.
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import List
import pickle, json

# ------------------------------------------------
# User‑set sweep parameters
Λ_MIN, Λ_MAX, N_SCAN = 2.0, 10.0, 25            # TeV
SIN2_W_TARGET   = 0.23122                       # PDG @ M_Z
SIN2_W_START    = 0.25                          # at Λ
RG_TOL          = 5e-4                          # RG pass window
HESSIAN_N_RAD   = 80                            # lattice pts
SAVE_PATH       = "scan_records.json"
# ------------------------------------------------

@dataclass
class ScanRow:
    Λ_TeV: float
    sin2_MZ: float
    RG_ok: bool
    λ_min: float
    SPD_ok: bool

# ========= (A) 2‑loop RG running ================
def beta_g1(g1, g2, g3, yt, lam):
    # <<<  *implement 2‑loop MS β*  >>>
    b1 = 41*g1**3/(96*np.pi**2)      # 1‑loop only placeholder
    return b1

def run_RG(sin2_start: float, Λ_start_GeV: float, MZ=91.1876, steps=600):
    """
    Very compressed 1‑loop running just for demo.
    Replace with full 2‑loop system!
    """
    g2 = np.sqrt(4*np.pi*0.0338)   # α2(MZ) placeholder
    g1 = g2*np.sqrt( (1-sin2_start)/sin2_start ) * np.sqrt(5/3)
    μs = np.geomspace(Λ_start_GeV, MZ, steps)
    for μ in μs[1:]:
        t = np.log(μs[0]/μ)
        g1 += beta_g1(g1,g2,0,0,0)*t   # absurdly rough
    sin2 = 1/(1+ (g2**2)/( (3/5)*g1**2 ))
    return sin2

# ========= (B) Hessian SPD check ================
def profile_F(r, Λ):
    # Example double‑wall profile; tune as needed
    return 2*np.arctan( np.exp(-Λ*r) )

def hessian_min_eig(Λ):
    """
    Dummy finite‑difference Hessian; returns positive number
    if Λ in 3.5–4.2 TeV just for illustration.
    """
    if 3.4 <= Λ <= 4.2:
        return 0.05      # stable
    return -0.02         # unstable

# ========= (C) Λ‑scan loop ======================
records: List[ScanRow] = []
Λ_vals = np.linspace(Λ_MIN, Λ_MAX, N_SCAN)

for Λ in Λ_vals:
    sin2 = run_RG(0.25, Λ*1e3)
    RG_ok = abs(sin2 - SIN2_W_TARGET) < RG_TOL
    λ_min = hessian_min_eig(Λ)
    SPD_ok = λ_min > 0
    row = ScanRow(Λ_TeV=float(Λ), sin2_MZ=float(f"{sin2:.5f}"),
                  RG_ok=RG_ok, λ_min=float(f"{λ_min:.3e}"), SPD_ok=SPD_ok)
    records.append(row)
    print(f"Λ={Λ:.2f} TeV | sin²={sin2:.5f} | RG_ok={RG_ok} | "
          f"λ_min={λ_min:.3e} | SPD_ok={SPD_ok}")

# Save
with open(SAVE_PATH,"w") as f:
    json.dump([asdict(r) for r in records], f, indent=2)
print("\n=== scan complete ===  results ↗", SAVE_PATH)
 
 
