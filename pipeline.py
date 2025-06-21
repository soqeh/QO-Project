
"""
pipeline.py  — RG ↔ Skyrmion 교차 스캐너
"""
import numpy as np, csv, datetime, os
from rg_core import evolve
from skyrmion_v2 import hessian_min

_HEADER = ["Lambda_TeV", "sin2W_MZ", "SPD_ok", "lambda_min"]

def scan_lambda(lam_grid_TeV, *, y0, mu_lo=91.1876, decouple=True, outfile=None):
    rows = []
    for Lam in lam_grid_TeV:
        mu_hi = Lam * 1e3  # TeV→GeV
        couplings = evolve(mu_hi, mu_lo, y0, decouple=decouple)
        sin2W = couplings['sin2W']
        # lam_min = hessian_min(1.0/mu_hi, 1.0/mu_hi, Lambda_GeV=mu_hi)
        lam_min = hessian_min(Lam,
                             lambda_Sk=1.8,      # 필요 시 params.yaml 로 빼도 OK
                             kappa_BI=2.0,
                             lambda_col=0.05,
                             eps_ratio=1e-2)
        SPD_ok = lam_min > 0.0
        rows.append([Lam, sin2W, SPD_ok, lam_min])

    if outfile is None:
        outfile = f"scan_{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.csv"
    with open(outfile, "w", newline="") as f:
        w = csv.writer(f); w.writerow(_HEADER); w.writerows(rows)
    return outfile, rows
