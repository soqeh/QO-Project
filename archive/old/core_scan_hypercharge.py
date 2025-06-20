# =========================================================
#  core_scan_hypercharge.py   ← 새 파일로 저장하거나 기존에 덮어쓰기
# =========================================================

import numpy as np, json, yaml
from dataclasses import dataclass, asdict
from typing import List
from scipy.optimize import minimize_scalar

param = {"alpha_inv": 123, "sin2thetaW": 0.2475, "alpha_s": 0.099, "mt": 173.0}
with open("Document/Hypercharge/param.yaml", "w") as f:
    yaml.dump(param, f)

# ---------- 사용자 스윕 설정 ----------
Λ_MIN, Λ_MAX, N_SCAN = 2.0, 10.0, 25      # TeV
SIN2_W_TARGET = 0.23122
RG_TOL        = 5e-4
SAVE_PATH     = "Document/Hypercharge/scan_records.json"
# -------------------------------------

# ===== 2-loop RG (run_RG_two_loop.py) =====
from run_RG_two_loop import evolve       # 이미 업로드돼 있음
import run_RG_two_loop as rg

# ★ param.yaml 경로를 /mnt/data 로 강제 지정 ★
def _load_my_param(fp="Document/Hypercharge/param.yaml"):
    import yaml, pathlib
    with open(pathlib.Path(fp), encoding="utf-8") as f:
        return yaml.safe_load(f)
rg.load = _load_my_param                 # 원래 load() 교체

def run_RG(Λ_start_GeV: float):
    """2-loop EW RG → sin²θ_W(M_Z)  (threshold decouple ON)"""
    out = evolve(Λ_start_GeV, 91.1876, steps=4000, decouple=True)
    return out["sin2W"]

# ===== Hessian SPD (core_hessian.py) =====
from core_hessian import hessian

def hessian_min_eig(Λ):
    """R_F = R_phi = R 를 0.2/Λ–1.0/Λ 범위 1-D 최적화"""
    def λ_min(R):
        return np.min(np.linalg.eigvalsh(hessian((R, R))))
    res = minimize_scalar(λ_min,
                          bounds=(0.2/Λ, 1.0/Λ),
                          method="bounded")
    return λ_min(res.x)

# ---------- 데이터 구조 ----------
@dataclass
class ScanRow:
    Λ_TeV: float; sin2_MZ: float
    RG_ok: bool; λ_min: float; SPD_ok: bool

# ============= Λ-스캔 =============
records: List[ScanRow] = []
for Λ in np.linspace(Λ_MIN, Λ_MAX, N_SCAN):
    sin2 = run_RG(Λ*1e3)
    λ_min = hessian_min_eig(Λ)
    row = ScanRow(Λ_TeV=Λ,
                  sin2_MZ=sin2,
                  RG_ok=abs(sin2-SIN2_W_TARGET) < RG_TOL,
                  λ_min=λ_min,
                  SPD_ok=λ_min > 0)
    records.append(row)
    print(f"Λ={Λ:4.2f} TeV | sin²={sin2:.5f} | RG_ok={row.RG_ok}"
          f" | λ_min={λ_min:.2e} | SPD_ok={row.SPD_ok}")

# ---------- 저장 ----------
with open(SAVE_PATH, "w") as f:
    json.dump([asdict(r) for r in records], f, indent=2)
print("\n▶ scan complete — 결과:", SAVE_PATH)
