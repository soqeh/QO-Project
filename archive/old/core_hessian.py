
"""core_hessian.py  –  minimal Skyrme+BI Hessian evaluator

Edit constants or profile forms as needed.  Run directly:
    python core_hessian.py
to see eigenvalues for the nominal parameters at bottom.

"""

import numpy as np
from math import pi
# constants
v=246.0; f_pi=246.0; e_gauge=0.65; lam_H=0.13; Lambda_BI=3400.0
R_max=10.0; N=4000
r = np.linspace(1e-6, R_max, N)
dr = r[1]-r[0]

def F_profile(r, R_F):      return 2*np.arctan( np.exp(-r/R_F) )
def phi_profile(r, R_phi):  return v*np.tanh( r/R_phi )

def derivatives(a):
    d = np.empty_like(a); d[1:-1]=(a[2:]-a[:-2])/(2*dr)
    d[0]=(a[1]-a[0])/dr; d[-1]=(a[-1]-a[-2])/dr; return d

def energy(R_F, R_phi):
    F=F_profile(r,R_F); phi=phi_profile(r,R_phi)
    Fp=derivatives(F); phip=derivatives(phi)
    sin2=np.sin(F)**2; r2=r**2
    eps=0.5*phip**2+0.5*phi**2*Fp**2
    eps+=(f_pi**2/8)*(Fp**2+2*sin2/r2)
    eps+=(1/(2*e_gauge**2))*sin2/r2*(Fp**2+sin2/(2*r2))
    eps+=lam_H/4*(phi**2-v**2)**2
    eps+=-Lambda_BI**4*(np.sqrt(1+F**2/(2*Lambda_BI**4))-1)
    return 4*pi*np.trapz(eps*r2,r)

def hessian(params,delta=0.02):
    H=np.zeros((2,2)); E0=energy(*params)
    for i in range(2):
        dp=list(params); dp[i]+=delta; Ep=energy(*dp)
        dm=list(params); dm[i]-=delta; Em=energy(*dm)
        H[i,i]=(Ep-2*E0+Em)/delta**2
    # off‑diag
    pp=[params[0]+delta, params[1]+delta]; Epp=energy(*pp)
    pm=[params[0]+delta, params[1]-delta]; Epm=energy(*pm)
    mp=[params[0]-delta, params[1]+delta]; Emp=energy(*mp)
    mm=[params[0]-delta, params[1]-delta]; Emm=energy(*mm)
    off=(Epp-Epm-Emp+Emm)/(4*delta**2); H[0,1]=H[1,0]=off
    return H

if __name__=='__main__':
    nominal=[0.5,0.5]
    H=hessian(nominal); vals=np.linalg.eigvalsh(H)
    print("Eigenvalues:",vals); print("SPD?", np.all(vals>0))
