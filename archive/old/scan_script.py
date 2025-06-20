import numpy as np
import math, csv
from core_hessian import hessian
from run_RG_two_loop import load, beta_g1, beta_g2, beta_g3, beta_yt

def evolve_decouple(mu_hi, mu_lo, steps=5000,
                    alpha_inv=123, alpha_s=0.099,
                    sin2thetaW=0.2465, mt=173.0,
                    mt_decouple=173.0, mH_decouple=125.0):
    P = load()
    P["alpha_inv"], P["alpha_s"], P["sin2thetaW"], P["mt"] = \
        alpha_inv, alpha_s, sin2thetaW, mt
    # init couplings
    alpha = 1/P["alpha_inv"]
    g1 = math.sqrt(4*math.pi*alpha/(1-P["sin2thetaW"]))
    g2 = math.sqrt(4*math.pi*alpha/(P["sin2thetaW"]))
    g3 = math.sqrt(4*math.pi*P["alpha_s"])
    yt = math.sqrt(2)*P["mt"]/246.0
    logs = np.linspace(math.log(mu_hi), math.log(mu_lo), steps)
    for i in range(1,len(logs)):
        dlog = logs[i]-logs[i-1]
        scale = math.exp(logs[i])
        yt_eff = yt if scale>=mt_decouple else 0.0
        # optional Higgs decouple:
        # lam_eff = lam if scale>=mH_decouple else 0.0
        g1 += -beta_g1(g1,g2,g3,yt_eff)*dlog
        g2 += -beta_g2(g1,g2,g3,yt_eff)*dlog
        g3 += -beta_g3(g1,g2,g3,yt_eff)*dlog
        yt += -beta_yt(yt,  g1,g2,g3)*dlog
    sin2W = g1**2/(g1**2+g2**2)
    return {"sin2W": sin2W}

def full_scan(output_csv="Document/Hypercharge/degenerative_scan.csv"):
    Λ_TeV, μ_lo, STEPS = 4.0, 91.1876, 4000
    RG_TOL, SIN2_TARGET = 5e-4, 0.23122
    sin2_inputs = np.arange(0.2450,0.2481,0.00005)
    mt_range      = np.arange(160,191,30)
    mH_range      = np.arange(115,136,20)
    with open(output_csv,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["sin2_in","mt_dcp","mH_dcp","sin2_MZ","RG_ok","SPD_ok","λ_min"])
        for sin2_in in sin2_inputs:
            for mt_dcp in mt_range:
                for mH_dcp in mH_range:
                    try:
                        out=evolve_decouple(Λ_TeV*1e3,μ_lo,STEPS,
                                            sin2thetaW=sin2_in,
                                            mt_decouple=mt_dcp,
                                            mH_decouple=mH_dcp)
                        s2=out["sin2W"]
                        RG_ok=abs(s2-SIN2_TARGET)<RG_TOL
                        R=0.5/Λ_TeV
                        λ_min=np.min(np.linalg.eigvalsh(hessian((R,R))))
                        SPD_ok=(λ_min>0)
                        w.writerow([sin2_in,mt_dcp,mH_dcp,
                                    f"{s2:.6f}",RG_ok,SPD_ok,
                                    f"{λ_min:.6e}"])
                        
                        print(f"Λ={Λ_TeV:4.2f} TeV | sin²={s2:.5f} | RG_ok={RG_ok} | λ_min={λ_min:.2e} | SPD_ok={SPD_ok}")
                    except Exception as e:
                        w.writerow([sin2_in,mt_dcp,mH_dcp,"ERROR",e,"",""])
    print("Scan complete:", output_csv)

if __name__=="__main__":
    full_scan()
