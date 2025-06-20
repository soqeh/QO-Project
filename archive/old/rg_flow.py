"""
rg_flow.py
2-loop electroweak RG evolution + top/Higgs decoupling
– 독립 래퍼 evolve_decouple() 제공
"""

import math, numpy as np
from run_RG_two_loop import beta_g1, beta_g2, beta_g3, beta_yt, load

def evolve_decouple(mu_hi, mu_lo, *,
                    steps        = 4000,
                    alpha_inv    = 123.0,
                    alpha_s      = 0.099,
                    sin2thetaW   = 0.2465,
                    mt           = 173.0,
                    mt_decouple  = 173.0,
                    mH_decouple  = 125.0):
    """
    Return dict {sin2W, g1, g2, g3}  + status 'OK' / 'RG_ERR'
    No dependency on run_RG_two_loop.evolve signature.
    """
    try:
        P = load()                     # other SM inputs
        P.update(alpha_inv = alpha_inv,
                 alpha_s   = alpha_s,
                 sin2thetaW= sin2thetaW,
                 mt        = mt)

        # ---  initial couplings  (MS-bar, GUT-normalised g1)  ---
        alpha = 1.0 / P["alpha_inv"]
        g1 = math.sqrt(4*math.pi*alpha/(1.0 - P["sin2thetaW"]))
        g2 = math.sqrt(4*math.pi*alpha/        P["sin2thetaW"] )
        g3 = math.sqrt(4*math.pi*P["alpha_s"])
        yt = math.sqrt(2)*P["mt"]/246.0

        logs = np.linspace(math.log(mu_hi), math.log(mu_lo), steps)
        for i in range(1, len(logs)):
            dlog  = logs[i] - logs[i-1]
            scale = math.exp(logs[i])

            yt_eff = yt if scale >= mt_decouple else 0.0   # top decouple

            # --- 2-loop β (from run_RG_two_loop)  -------------
            g1 += -beta_g1(g1, g2, g3, yt_eff) * dlog
            g2 += -beta_g2(g1, g2, g3, yt_eff) * dlog
            g3 += -beta_g3(g1, g2, g3, yt_eff) * dlog
            yt += -beta_yt(yt_eff, g1, g2, g3)  * dlog

            # 방어적 NaN / inf 체크
            if not all(map(math.isfinite, (g1, g2, g3))):
                return None, "RG_OVERFLOW"

        sin2W = g1**2 / (g1**2 + g2**2)
        return dict(sin2W=sin2W, g1=g1, g2=g2, g3=g3), "OK"

    except Exception as e:
        return None, f"RG_ERR:{e}"
