#!/usr/bin/env python3
import numpy as np, yaml, math, pprint

# ---------- β-function coefficients (Machacek–Vaughn) ----------
b1_1, b2_1, b3_1 = 41/10, -19/6, -7
# 2-loop gauge parts
b11, b12, b13 = 199/50, 27/10, 44/5
b21, b22, b23 = 9/10, 35/6, 12
b31, b32, b33 = 11/10, 9/2, -26
# Yukawa piece (top only)
c1, c2, c3 = 17/10, 3/2, 2

def load(fp="Document/Hypercharge/param.yaml"):
    with open(fp, encoding='utf-8') as f: return yaml.safe_load(f)

def beta_g1(g1,g2,g3,yt):
    t1=b1_1*g1**3; t2=g1**3*(b11*g1**2+b12*g2**2+b13*g3**2-c1*yt**2)
    return (t1/(16*math.pi**2) + t2/(16*math.pi**2)**2)

def beta_g2(g1,g2,g3,yt):
    t1=b2_1*g2**3; t2=g2**3*(b21*g1**2+b22*g2**2+b23*g3**2-c2*yt**2)
    return (t1/(16*math.pi**2) + t2/(16*math.pi**2)**2)

def beta_g3(g1,g2,g3,yt):
    t1=b3_1*g3**3; t2=g3**3*(b31*g1**2+b32*g2**2+b33*g3**2-c3*yt**2)
    return (t1/(16*math.pi**2) + t2/(16*math.pi**2)**2)

def beta_yt(yt,g1,g2,g3):
    return yt/(16*math.pi**2)*(4.5*yt**2 - 0.85*g1**2 - 2.25*g2**2 - 8*g3**2)

def evolve(mu_hi, mu_lo, steps=5000, decouple=False):
    P=load()
    alpha  = 1/P["alpha_inv"]
    g1 = math.sqrt(4*math.pi*alpha/(1-P["sin2thetaW"]))
    g2 = math.sqrt(4*math.pi*alpha/P["sin2thetaW"])
    g3 = math.sqrt(4*math.pi*P["alpha_s"])
    yt = math.sqrt(2)*P["mt"]/246.0
    logs = np.linspace(math.log(mu_hi), math.log(mu_lo), steps)
    for i in range(1,len(logs)):
        dlog = logs[i]-logs[i-1]
        yt_eff = yt if math.exp(logs[i])>P["mt"] and not decouple else 0.0
        g1 += beta_g1(g1,g2,g3,yt_eff)*dlog
        g2 += beta_g2(g1,g2,g3,yt_eff)*dlog
        g3 += beta_g3(g1,g2,g3,yt_eff)*dlog
        yt += beta_yt(yt,g1,g2,g3)*dlog
    sin2W = g1**2/(g1**2+g2**2)
    return dict(g1=g1, g2=g2, g3=g3, yt=yt, sin2W=sin2W)

if __name__=="__main__":
    out = evolve(5e3, 91.1876)
    print(">>> Couplings at mZ after 2-loop RG:")
    pprint.pprint({k:f"{v:.6f}" for k,v in out.items()})
