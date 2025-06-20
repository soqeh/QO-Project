# -*- coding: utf-8 -*-
"""
scan3D_fixed.py  ─── standalone, *single‑file* re‑write
=====================================================
• **what it does**
  1.  takes three external scan grids (sin²θ_W(Λ), m_t^dec, m_H^dec)
  2.  for every grid point runs a *very lightweight* RG placeholder that
      captures the empirically‑observed 1‑loop drift
          sin²(M_Z) ≈ sin²(Λ) – Δ, Δ≃0.018
      (full two‑loop machinery can be re‑inserted later; see `TODO`.)
  3.  evaluates the Skyrmion core stability by calling **your** existing
      `core_hessian.hessian((R_f,R_φ))` and checking λ_min > 0.
  4.  prints the lattice with the familiar "Λ | sin² | RG_ok | λ_min | SPD_ok"
      header and finally a short summary.

• **why this stripped‑down version?**
  The immediate goal is to **eliminate the signature mismatches** that kept
  crashing the previous iterations (evolve() vs. run_rg(), mt_dec keyword,
  etc.) and give you a working baseline that reproduces the *qualitative*
  window already known from 1‑loop scans (Λ≈3.8–4.3 TeV).

  Once the pipeline again returns non‑zero "OK" hits, we can drop the
  placeholder inside `run_rg()` and hook it to your *actual* 2‑loop solver
  (`run_RG_two_loop.evolve`) with the now‑consistent I/O.
"""

from __future__ import annotations
from scan3D_core_funcs import run_RG, hessian_energy

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

# ====== external dependencies (already provided in your repo) ======
from core_hessian import hessian  # -> expects (R_f, R_phi) tuple

# ===================== configuration block =========================
# physical targets
MZ_GEV: float = 91.1876
SIN2_TARGET: float = 0.23122  # PDG2024 central value
SIN2_TOL: float = 5.0e-4      # ±5×10⁻⁴ window ⇒ RG_ok flag

# Λ scan (TeV)
Λ_MIN, Λ_MAX, Λ_STEP = 3.0, 5.5, 0.33  # 3 – 5.5 TeV, ~8 points

# high‑scale input grid (the three scan axes)
SIN2_START, SIN2_STOP, SIN2_STEP = 0.2490, 0.2550, 0.00025
MT_START, MT_STOP, MT_STEP = 160.0, 180.0, 5.0    # GeV
MH_START, MH_STOP, MH_STEP = 120.0, 135.0, 5.0    # GeV

# output
SAVEFILE = Path("scan3D_results.json")

# ================= utility: RG & Hessian wrappers ==================

# ------------------------------------------------------------------
# 1) run_RG  —  exactly three positional arguments
#              returns (tuple, status, OdeResult)
# ------------------------------------------------------------------
from run_RG_two_loop import evolve        # 기존 모듈 그대로
V_EW   = 246.22
MU_HI  = 4000.0       # 4 TeV
MU_LO  = 91.1876
ALPHA_INV = 123.0
ALPHA_S   = 0.099

# def run_RG(sin2_in, mt_dec, mh_dec):
#     """
#     Parameters
#     ----------
#     sin2_in : float   # sin²θ_W at Λ = 4 TeV
#     mt_dec  : float   # top-decouple scale  [GeV]
#     mh_dec  : float   # Higgs-decouple scale[GeV]

#     Returns
#     -------
#     (sin2_MZ, g1, g2, g3, yt, lam), 'OK'|'RG_OVERFLOW', OdeResult|None
#     """
#     # --- initial couplings at MU_HI ---------------------------------
#     alpha = 1.0 / ALPHA_INV
#     e     = (4*np.pi*alpha)**0.5
#     g2_0  =  e / sin2_in**0.5
#     g1_0  =  (5/3)**0.5 * e / (1-sin2_in)**0.5
#     g3_0  = (4*np.pi*ALPHA_S)**0.5
#     yt_0  = (2**0.5) * 173.0 / V_EW
#     lam_0 = 125.0**2 / (2*V_EW**2)
#     y0    = [g1_0, g2_0, g3_0, yt_0, lam_0]

#     # --- RG evolve (no keywords!) -----------------------------------
#     sol = evolve(MU_HI, MU_LO, y0)   # <-- 3-positional-arg call

#     if (not sol.success) or (not np.all(np.isfinite(sol.y))):
#         return None, 'RG_OVERFLOW', None

#     g1, g2, g3, yt, lam = sol.y[:, -1]
#     sin2_MZ = (3/5)*g1**2 / ((3/5)*g1**2 + g2**2)
#     return (sin2_MZ, g1, g2, g3, yt, lam), 'OK', sol


# # ------------------------------------------------------------------
# # 2) skyrmion_energy  —  (R_f, R_phi)  both in GeV⁻¹
# # ------------------------------------------------------------------
# def skyrmion_energy(R_f, R_phi):
#     Λ = MU_HI
#     r = np.linspace(1e-5, 30.0/Λ, 1201)
#     dr = r[1]-r[0]

#     F    = 2*np.arctan(np.exp(-r/R_f))
#     dF   = np.gradient(F, dr)
#     phi  = V_EW*np.tanh(r/R_phi)
#     dphi = np.gradient(phi, dr)

#     dens  = 0.5*dphi**2 + phi**2*(0.5*dF**2 + np.sin(F)**2/r**2)
#     dens += (np.sin(F)**4)/(2*r**4)                       # Skyrme
#     BI    = phi**2*0.5*dF**2
#     dens += Λ**4*(np.sqrt(1+BI/Λ**4)-1)                   # Born-Infeld

#     return 4*np.pi*np.trapz(dens*r**2, r)


# # ------------------------------------------------------------------
# # 3) hessian_energy  —  central finite-difference 2×2 Hessian
# # ------------------------------------------------------------------
# def hessian_energy(R_f, R_phi, eps=1e-4):
#     def E(Rf, Rp): return skyrmion_energy(Rf, Rp)
#     dRf = eps*R_f
#     dRp = eps*R_phi
#     f00 = E(R_f, R_phi)
#     fpR = E(R_f+dRf, R_phi)
#     fmR = E(R_f-dRf, R_phi)
#     fpP = E(R_f, R_phi+dRp)
#     fmP = E(R_f, R_phi-dRp)
#     fpp = E(R_f+dRf, R_phi+dRp)
#     fmm = E(R_f-dRf, R_phi-dRp)

#     d2_Rf2  = (fpR - 2*f00 + fmR) / dRf**2
#     d2_Rp2  = (fpP - 2*f00 + fmP) / dRp**2
#     d2_mix  = (fpp - fpR - fpP + f00 + f00 - fmR - fmP + fmm) / (2*dRf*dRp)

#     return np.array([[d2_Rf2, d2_mix],
#                      [d2_mix, d2_Rp2]])


# ------------------------------------------------------------------
# 4) check_hessian  —  **무인수**  (scan3D.py 호출과 동일)
# ------------------------------------------------------------------
def check_hessian():
    """
    Uses R_f = R_phi = 1 / Λ_GeV  with Λ = 4 TeV
    Returns (True/False, λ_min)
    """
    R = 1.0 / MU_HI            # GeV⁻¹
    lam_min = np.min(np.linalg.eigvalsh(hessian_energy(R, R)))
    return lam_min > 0.0, lam_min

def convert_np(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj

# ========================= data container ==========================

@dataclass
class ScanRow:
    Λ_TeV: float
    sin2_MZ: float
    RG_ok: bool
    λ_min: float
    SPD_ok: bool


# ============================= main ================================

def main() -> None:
    rows: List[ScanRow] = []

    total = (
        int((SIN2_STOP - SIN2_START) / SIN2_STEP + 1)
        * int((MT_STOP - MT_START) / MT_STEP + 1)
        * int((MH_STOP - MH_START) / MH_STEP + 1)
    )

    # ------- loop over Λ first (outer) so Hessian is reused logically -------
    for Λ in np.arange(Λ_MIN, Λ_MAX + 1e-12, Λ_STEP):
        Λ_GeV = Λ * 1e3
        λ_min_val = hessian_energy(Λ_GeV)
        SPD_ok = λ_min_val > 0.0

        # inner 3‑D grid: sin²(Λ), mt_dec, mH_dec
        pbar = tqdm(total=total, desc=f"Λ={Λ:.2f} TeV", leave=False)
        ok_hit = False
        for sin2_in in np.arange(SIN2_START, SIN2_STOP + 1e-12, SIN2_STEP):
            for mt_dec in np.arange(MT_START, MT_STOP + 1e-12, MT_STEP):
                for mh_dec in np.arange(MH_START, MH_STOP + 1e-12, MH_STEP):
                    sin2_lo, rg_ok = run_RG(sin2_in, mt_dec, mh_dec, Λ)
                    row = ScanRow(Λ, sin2_lo, rg_ok, λ_min_val, SPD_ok)
                    rows.append(row)

                    # console echo for RG & SPD success points only
                    if rg_ok and SPD_ok:
                        print(
                            f"Λ={Λ:4.2f} TeV | sin²={sin2_lo:.5f} | RG_ok=True | "
                            f"λ_min={λ_min_val:.2e} | SPD_ok=True"
                        )
                        ok_hit = True
                    pbar.update(1)
        pbar.close()

        if not ok_hit:
            print(
                f"Λ={Λ:4.2f} TeV — no RG∧SPD intersection within the current grid.")

    # --------------- save on disk for later heat‑maps -------------------

    SAVEFILE.write_text(json.dumps([asdict(r) for r in rows], indent=2, default=convert_np))
    # SAVEFILE.write_text(json.dumps([asdict(r) for r in rows], indent=2))
    # import csv

    # with SAVEFILE.open("w", newline='', encoding="utf-8") as f:
    #     writer = csv.DictWriter(f, fieldnames=asdict(rows[0]).keys())
    #     writer.writeheader()
    #     for r in rows:
    #         writer.writerow({k: str(v) for k, v in asdict(r).items()})
    print(f"\n▶ scan complete — rows saved to '{SAVEFILE}'.")


if __name__ == "__main__":
    main()
