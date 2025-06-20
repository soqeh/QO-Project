import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean

# Load best parameter vector
with open("best_fit.pkl", "rb") as f:
    data = pickle.load(f)

best_par = data['param']

# Parameter decoding
angles = best_par[0:3]
phases_d = best_par[3:5]
phases_e = best_par[5:7]
dR = best_par[7]
rhoL = best_par[8:11]
rhoQ = best_par[11:17]

# CKM and PMNS target matrices
CKM_T = np.array([[0.97446, 0.22452, 0.00365],
                  [0.22438, 0.97359, 0.04214],
                  [0.00896, 0.04133, 0.999105]])

PMNS_T = np.array([[0.821, 0.549, 0.150],
                   [0.358, 0.707, 0.612],
                   [0.450, 0.446, 0.773]])

# Helper functions
def rot(a, b, c):
    Rz1 = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz2 = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return Rz1 @ Ry @ Rz2

def diagphase(p1, p2):
    return np.diag([np.exp(1j * p1), np.exp(1j * p2), 1])

# Construct CKM and PMNS matrices from parameters
R = rot(*angles)
V = R @ diagphase(*phases_d) @ R.T
Rℓ = rot(0, dR, 0) @ R
U = Rℓ @ diagphase(*phases_e) @ Rℓ.T

# Extract magnitudes
CKM_fit = np.abs(V)
PMNS_fit = np.abs(U)

# Compute error matrices
CKM_err = np.abs(CKM_fit - CKM_T)
PMNS_err = np.abs(PMNS_fit - PMNS_T)

# Plotting
fig, axs = plt.subplots(2, 3, figsize=(16, 8))
sns.heatmap(CKM_fit, annot=True, fmt=".5f", ax=axs[0,0], cbar=False, cmap="Blues")
axs[0,0].set_title("Fitted CKM Matrix")
sns.heatmap(CKM_T, annot=True, fmt=".5f", ax=axs[0,1], cbar=False, cmap="Greens")
axs[0,1].set_title("Target CKM Matrix")
sns.heatmap(CKM_err, annot=True, fmt=".2e", ax=axs[0,2], cbar=False, cmap="Reds")
axs[0,2].set_title("CKM Absolute Error")

sns.heatmap(PMNS_fit, annot=True, fmt=".5f", ax=axs[1,0], cbar=False, cmap="Blues")
axs[1,0].set_title("Fitted PMNS Matrix")
sns.heatmap(PMNS_T, annot=True, fmt=".5f", ax=axs[1,1], cbar=False, cmap="Greens")
axs[1,1].set_title("Target PMNS Matrix")
sns.heatmap(PMNS_err, annot=True, fmt=".2e", ax=axs[1,2], cbar=False, cmap="Reds")
axs[1,2].set_title("PMNS Absolute Error")

plt.tight_layout()

# Sensitivity heatmap: parameter vs. observable shift magnitude (finite difference approximation)
def observable_vector(params):
    a, b, c, pd1, pd2, pe1, pe2, dR = params[:8]
    rhoL = params[8:11]
    rhoQ = params[11:17]
    R = rot(a, b, c)
    V = R @ diagphase(pd1, pd2) @ R.T
    Rℓ = rot(0, dR, 0) @ R
    U = Rℓ @ diagphase(pe1, pe2) @ Rℓ.T
    obs = list(np.abs(V.flatten())) + list(np.abs(U.flatten()))
    return np.array(obs)

eps = 1e-4
base_obs = observable_vector(best_par)
sensitivity = []

for i in range(len(best_par)):
    perturbed = best_par.copy()
    perturbed[i] += eps
    pert_obs = observable_vector(perturbed)
    delta = np.linalg.norm(pert_obs - base_obs)
    sensitivity.append(delta)

# Visualize sensitivity
plt.figure(figsize=(12, 2))
sns.heatmap([sensitivity], annot=False, cmap="magma", cbar=True)
plt.yticks([0], ["Sensitivity"])
plt.xticks(np.arange(len(best_par)) + 0.5, labels=[
    "a", "b", "c", "pd1", "pd2", "pe1", "pe2", "dR",
    "ρL1", "ρL2", "ρL3",
    "ρQ1", "ρQ2", "ρQ3", "ρQ4", "ρQ5", "ρQ6"
], rotation=45, ha="right")
plt.title("Parameter Sensitivity Heatmap")
plt.tight_layout()

plt.show()