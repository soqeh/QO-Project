import numpy as np
from run_RG_two_loop import evolve

# === 전역 상수 ===
V_EW   = 246.22        # GeV
MU_HI  = 4000.0        # GeV  (4 TeV)
MU_LO  = 91.1876       # GeV  (M_Z)
ALPHA_INV = 123.0      # α⁻¹(Λ)
ALPHA_S   = 0.099      # α_s(Λ)

# ------------------------------------------------------------------
# 1) run_RG  : exactly 3 positional args, returns (tuple, status, sol)
# ------------------------------------------------------------------

def run_RG(sin2_in, mt_dec, mh_dec):
    """2‑loop RG → (sin2_MZ, g1, g2, g3, yt, lam)"""
    alpha = 1.0/ALPHA_INV
    e     = (4*np.pi*alpha)**0.5
    g2_0  =  e / sin2_in**0.5
    g1_0  = (5/3)**0.5 * e / (1-sin2_in)**0.5
    g3_0  = (4*np.pi*ALPHA_S)**0.5
    yt_0  = (2**0.5) * 173.0 / V_EW
    lam_0 = 125.0**2 / (2*V_EW**2)
    y0    = [g1_0, g2_0, g3_0, yt_0, lam_0]

    sol = evolve(MU_HI, MU_LO, y0)  # evolve(mu_hi, mu_lo, y0)
    if (not sol.success) or (not np.all(np.isfinite(sol.y))):
        return None, 'RG_OVERFLOW', None

    g1, g2, g3, yt, lam = sol.y[:, -1]
    sin2_MZ = (3/5)*g1**2 / ((3/5)*g1**2 + g2**2)
    return (sin2_MZ, g1, g2, g3, yt, lam), 'OK', sol

# ------------------------------------------------------------------
# 2) skyrmion_energy  : (R_f, R_phi) in GeV⁻¹
# ------------------------------------------------------------------

def skyrmion_energy(R_f, R_phi):
    Λ = MU_HI
    r = np.linspace(1e-5, 30.0/Λ, 1201)
    dr = r[1] - r[0]

    F    = 2*np.arctan(np.exp(-r/R_f))
    dF   = np.gradient(F, dr)
    phi  = V_EW*np.tanh(r/R_phi)
    dphi = np.gradient(phi, dr)

    dens  = 0.5*dphi**2 + phi**2*(0.5*dF**2 + np.sin(F)**2/r**2)
    dens += (np.sin(F)**4)/(2*r**4)
    BI    = phi**2*0.5*dF**2
    dens += Λ**4*(np.sqrt(1+BI/Λ**4)-1)

    return 4*np.pi*np.trapz(dens*r**2, r)

# ------------------------------------------------------------------
# 3) hessian_energy  : single‑arg wrapper (Lambda_GeV) → λ_min
# ------------------------------------------------------------------

def hessian_energy(Lambda_GeV, eps_ratio=1e-2):
    R = 1.0 / Lambda_GeV           # GeV⁻¹   (대칭 ansatz)
    h = eps_ratio * R
    # 1D second derivative as proxy for λ_min
    E  = lambda Rv: skyrmion_energy(Rv, Rv)
    lam_min = (E(R+h) - 2*E(R) + E(R-h)) / h**2
    return lam_min

# ------------------------------------------------------------------
# 4) check_hessian  : returns (bool, λ_min)  (호환용)
# ------------------------------------------------------------------

def check_hessian():
    lam_min = hessian_energy(MU_HI)
    return (lam_min > 0.0), lam_min