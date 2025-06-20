"""skyrme_bi.py
Profile solver + Hessian for Skyrme+BI Q=1 hedgehog.
"""
import numpy as np, math
from scipy.integrate import solve_bvp

# --- physical constants (can be parameterised)
v = 246.0
lambda_BI = 4.0e3  # GeV (placeholder)

def shooting_profile(R_F, points=400):
    """Return F(r) numeric profile via ansatz fallback."""
    r = np.linspace(0,10/R_F,points)
    F = 2*np.arctan(np.exp(-r/R_F))
    return r,F

def hessian_min(R_F):
    r,F = shooting_profile(R_F)
    dr = r[1]-r[0]
    dF = np.gradient(F,dr)
    # crude λ_min ~ integral of (dF^2) just placeholder
    lam_min = np.trapz(dF**2,r)
    return lam_min

def hessian_min_pair(R_F,R_phi):
    # treat symmetric for now
    return hessian_min(R_F)