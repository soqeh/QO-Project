"""
skyrmion_v2.py — Skyrmion + BI + collapse barrier (λ_Sk, κ_BI, λ_col)
Units:
    * length r  : GeV⁻¹
    * energy    : GeV
    * Λ_TeV     : input scale in TeV (used to set core size R = 1/Λ_TeV in TeV⁻¹)
    * Natural units ℏ = c = 1
"""
import numpy as np

def _profiles(r, R_f, R_phi, v):
    F = 2.0 * np.arctan(np.exp(-r / R_f))
    dF = np.gradient(F, r)
    phi = v * np.tanh(r / R_phi)
    dphi = np.gradient(phi, r)
    return F, dF, phi, dphi

def energy(Lambda_TeV, *, v_EW=246.22, 
           lambda_Sk=1.8, kappa_BI=2.0, lambda_col=0.05,
           n_r=2001):
    """Return total core energy (GeV) for given parameters."""
    R_core = 1.0 / Lambda_TeV   # TeV^{-1}
    R_core *= 1e-3              # convert to GeV^{-1}
    R_f = R_phi = R_core
    r_max = 30.0 * R_core
    r = np.linspace(1e-6, r_max, n_r)
    F, dF, phi, dphi = _profiles(r, R_f, R_phi, v_EW)
    
    dens = 0.5 * dphi**2 + phi**2 * (0.5 * dF**2 + np.sin(F)**2 / r**2)
    # Skyrme 4-derivative with adjustable lambda_Sk
    dens += lambda_Sk * (np.sin(F)**4) / (2 * r**4)
    # BI term with kappa_BI
    BI = phi**2 * 0.5 * dF**2
    dens += kappa_BI * Lambda_TeV**4 * 1e12 * (np.sqrt(1 + BI / ( (Lambda_TeV*1e3)**4)) - 1)
    # collapse barrier term (approx) using F profile
    collapse = lambda_col * (np.sin(F/2.0)**2)
    dens += collapse
    return 4.0 * np.pi * np.trapz(dens * r**2, r)
def hessian_min(Lambda_TeV, *, eps_ratio=1e-2, **kw):
    dLam = eps_ratio          # 작은 ΔΛ(TeV)로 바로 씁니다
    E0 = energy(Lambda_TeV, **kw)               # 중앙값
    Ep = energy(Lambda_TeV + dLam, **kw)        # +Δ
    Em = energy(Lambda_TeV - dLam, **kw)        # –Δ
    return (Ep - 2*E0 + Em) / dLam**2           # 2차 유한차분
# def hessian_min(Lambda_TeV, *, eps_ratio=1e-2, **kw):
#     R = 1.0 / Lambda_TeV
#     R *= 1e-3
#     h = eps_ratio * R
#     def E(Rvar):
#         return energy(Lambda_TeV, **kw, lambda_Sk=kw.get('lambda_Sk',1.8), 
#                       kappa_BI=kw.get('kappa_BI',2.0), lambda_col=kw.get('lambda_col',0.05))
#     E0 = E(Lambda_TeV)
#     Ep = energy(Lambda_TeV + eps_ratio, **kw)
#     Em = energy(Lambda_TeV - eps_ratio, **kw)
#     return (Ep - 2*E0 + Em) / (eps_ratio**2)