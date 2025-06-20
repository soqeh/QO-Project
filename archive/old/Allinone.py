###############################################################################
#  ε₀–Topology “ALL‑IN‑ONE”  :  Global Fit  +  Leave‑One‑Out Cross‑Validation #
#                                                                           #
#  • Monte‑Carlo 1 000× random initialisations (Nelder–Mead local search)   #
#  • Real‑time logging every 50 trials, checkpoint every 200                #
#  • Final best‑fit stored in  best_fit.pkl                                  #
#  • Optional leave‑one‑out CV: set DO_LOO = True                           #
#                                                                           #
#  How to run in Colab / local:                                             #
#    !python all_in_one_epsilon_topology.py                                  #
###############################################################################
import numpy as np, pickle, time, sys
from pathlib import Path
from scipy.optimize import minimize

# ====== 1. TARGET DATA ======
CKM_T = np.array([[0.97446,0.22452,0.00365],
                  [0.22438,0.97359,0.04214],
                  [0.00896,0.04133,0.999105]])

PMNS_T = np.array([[0.821,0.549,0.150],
                   [0.358,0.707,0.612],
                   [0.450,0.446,0.773]])

KOIDE_T = 2/3
LEP_M   = np.sqrt(np.array([0.000511, 0.10566, 1.77686]))   # GeV^½
QUA_M   = np.sqrt(np.array([0.0022,1.27,173, 0.0047,0.096,4.18]))

# λ weights (Fine‑λ best)
LAMB_K, LAMB_M = 0.3, 0.5

# ====== 2. HELPER FUNCTIONS ======
def rot(a,b,c):
    Rz1=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    Ry =np.array([[np.cos(b),0,np.sin(b)],[0,1,0],[-np.sin(b),0,np.cos(b)]])
    Rz2=np.array([[np.cos(c),-np.sin(c),0],[np.sin(c),np.cos(c),0],[0,0,1]])
    return Rz1@Ry@Rz2

def diagphase(p1,p2):
    return np.diag([np.exp(1j*p1), np.exp(1j*p2), 1])

def koide(rho):  # rho = √m vector(3)
    return (rho**2).sum() / (rho.sum()**2)

# -------- observables → 19‑vector  
def observable_vector(params):
    (a,b,c, pd1,pd2, pe1,pe2, dR) = params[:8]
    rhoL = params[8:11]
    rhoQ = params[11:17]

    R   = rot(a,b,c)
    V   = R @ diagphase(pd1,pd2) @ R.T
    Rℓ  = rot(0,dR,0) @ R
    U   = Rℓ @ diagphase(pe1,pe2) @ Rℓ.T

    obs = [
        abs(V[0,0]), abs(V[0,1]), abs(V[0,2]),
        abs(V[1,0]), abs(V[1,1]), abs(V[1,2]),
        abs(V[2,0]), abs(V[2,1]), abs(V[2,2]),
        abs(U[0,0]), abs(U[0,1]), abs(U[0,2]),
        abs(U[1,0]), abs(U[1,1]), abs(U[1,2]),
        abs(U[2,0]), abs(U[2,1]), abs(U[2,2]),
        koide(rhoL)
    ] + list(rhoL**2) + list(rhoQ**2)
    return np.array(obs)

# -------- target vector (= flatten CKM & PMNS |.| + Koide + masses²) 
OBS_TARGET = np.concatenate([
    CKM_T.flatten(), PMNS_T.flatten(), np.array([KOIDE_T]), LEP_M**2, QUA_M**2
])

def error_vector(params):
    return observable_vector(params) - OBS_TARGET

# -------- loss (with optional mask for leave‑one‑out) 
def loss(params, mask=None):
    diff = error_vector(params)
    if mask is not None:
        diff = diff[mask]
    # split contributions
    ckm_err  = np.linalg.norm(diff[0:9])
    pmns_err = np.linalg.norm(diff[9:18])
    koide_err= diff[18]
    mass_err = np.linalg.norm(diff[19:])
    return ckm_err + pmns_err + LAMB_K*koide_err**2 + LAMB_M*mass_err

# ====== 3. FIT ROUTINES ======
def random_init():
    ang  = np.random.uniform(-np.pi, np.pi, 3)
    phs  = np.random.uniform(-np.pi, np.pi, 4)
    dR   = np.random.uniform(-0.12, 0.12)
    rhoL = LEP_M * np.random.uniform(0.8,1.3,3)
    rhoQ = QUA_M * np.random.uniform(0.8,1.3,6)
    return np.concatenate([ang, phs, [dR], rhoL, rhoQ])

def local_fit(init, mask=None):
    res = minimize(loss, init, args=(mask,), method='Nelder-Mead',
                   options={'maxiter':1200, 'fatol':1e-7, 'xatol':1e-6})
    return res.x, res.fun

# ====== 4. MONTE‑CARLO GLOBAL SEARCH ======
N_TRIALS = 10000
LOG_EVERY= 100
CKPT_EVERY= 500
SAVE_DIR = Path("./")
best_loss, best_par = 1e9, None
t0 = time.time()

for i in range(1, N_TRIALS+1):
    p0 = random_init()
    par, loss_val = local_fit(p0)
    if loss_val < best_loss:
        best_loss, best_par = loss_val, par.copy()

    if i % LOG_EVERY == 0:
        ck, pm = np.linalg.norm(error_vector(best_par)[:9]), np.linalg.norm(error_vector(best_par)[9:18])
        print(f"[{i}/{N_TRIALS}] best L={best_loss:.4f}  CKMerr={ck:.3f}  PMNSerr={pm:.3f}  Δt={time.time()-t0:.1f}s")
        sys.stdout.flush()

    if i % CKPT_EVERY == 0:
        # UNCOMMENT ↓ TO SAVE CHECKPOINTS
        with (SAVE_DIR / f"checkpoint_{i}.pkl").open("wb") as f:
            pickle.dump({'loss':best_loss, 'param':best_par}, f)
        pass

with (SAVE_DIR / "best_fit.pkl").open("wb") as f:
    pickle.dump({'loss':best_loss, 'param':best_par}, f)
print("\n=== Monte‑Carlo completed ===  best loss =", best_loss)

# ====== 5. LEAVE‑ONE‑OUT CROSS‑VALIDATION ======
DO_LOO = True   # <- 필요 시 True 로
if DO_LOO:
    print("\n--- Leave‑One‑Out Cross‑Validation ---")
    TOTAL_OBS = len(OBS_TARGET)
    errs=[]
    for k in range(TOTAL_OBS):
        mask = np.ones(TOTAL_OBS, dtype=bool); mask[k]=False
        p_fit, _ = local_fit(random_init(), mask)
        pred_err = abs(error_vector(p_fit)[k])
        errs.append(pred_err)
        print(f"  removed idx {k:2d}  pred.abs.err = {pred_err:.4e}")
    print("RMS(pred err) =", (np.mean(np.square(errs)))**0.5)



###############################
# End of ALL‑IN‑ONE script    #
###############################