
"""
skyrmion.py  — Minimal SU(2) Skyrmion + Born–Infeld core utilities.

All lengths are GeV⁻¹.  BI ansatz follows paper eq.(P‑10, p.32).

Public API
----------
energy(R_f, R_phi, *, Lambda_GeV, v_EW=246.22)
hessian_min(R_f, R_phi, *, Lambda_GeV, eps_ratio=1e‑2)
"""
import numpy as np

def _profiles(r, R_f, R_phi, v):
    F = 2.0 * np.arctan(np.exp(-r / R_f))
    dF = np.gradient(F, r)
    phi = v * np.tanh(r / R_phi)
    dphi = np.gradient(phi, r)
    return F, dF, phi, dphi

def energy(R_f, R_phi, *, Lambda_GeV, v_EW=246.22, n_r=1201):
    r_max = 30.0 / Lambda_GeV
    r = np.linspace(1e-5, r_max, n_r)
    dr = r[1] - r[0]

    F, dF, phi, dphi = _profiles(r, R_f, R_phi, v_EW)

    dens = 0.5 * dphi**2 + phi**2 * (0.5 * dF**2 + np.sin(F)**2 / r**2)
    dens += np.sin(F)**4 / (2 * r**4)  # Skyrme 4‑deriv term
    BI = phi**2 * 0.5 * dF**2
    dens += Lambda_GeV**4 * (np.sqrt(1 + BI / Lambda_GeV**4) - 1)

    return 4.0 * np.pi * np.trapz(dens * r**2, r)

def hessian_min(R_f, R_phi, *, Lambda_GeV, eps_ratio=1e-2):
    """Finite‑difference proxy for minimal Hessian eigenvalue (radial, 2‑var)."""
    def E(R): return energy(R, R, Lambda_GeV=Lambda_GeV)
    h = eps_ratio * R_f
    return (E(R_f + h) - 2 * E(R_f) + E(R_f - h)) / h**2
