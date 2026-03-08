import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from DLA.dla_methods import dla_simulation
from dla_mc import monte_carlo_dla
from cluster_metrics import radius_of_gyration, box_counting_dimension, correlation_dimension

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

def evaluate_pde(eta):
    _, cluster_frames = dla_simulation(eta=eta, logging=False, frame_interval=1)
    cluster = cluster_frames[-1]

    return np.array([
        radius_of_gyration(cluster),
        box_counting_dimension(cluster),
        correlation_dimension(cluster)
    ])

def evaluate_mc(p_s):
    cluster = monte_carlo_dla(p_s=p_s)
    
    return np.array([
        radius_of_gyration(cluster),
        box_counting_dimension(cluster),
        correlation_dimension(cluster)
    ])

N_p = 10
sticking_values = np.linspace(0.1, 1, N_p)
repeats_mc = 128

gyration_mc = np.empty((N_p, 2))
boxDim_mc = np.empty((N_p, 2))
corrDim_mc = np.empty((N_p, 2))
for i, p_s in enumerate(sticking_values):
    print("p_s value %d out of %d" % (i, N_p))

    results = Parallel(n_jobs=-1)(delayed(evaluate_mc)(p_s) for _ in range(repeats_mc))

    gyrations, boxDims, corrDims = np.array(results).T

    gyration_mc[i, :] = [np.mean(gyrations), np.std(gyrations)]
    boxDim_mc[i, :] = [np.mean(boxDims), np.std(boxDims)]
    corrDim_mc[i, :] = [np.mean(corrDims), np.std(corrDims)]

etas = np.linspace(0, 2, N_p)
repeats_pde = 16

gyration_pde = np.empty((N_p, 2))
boxDim_pde = np.empty((N_p, 2))
corrDim_pde = np.empty((N_p, 2))
for i, eta in enumerate(etas):
    print("eta value %d out of %d" % (i, N_p))

    results = Parallel(n_jobs=-1)(delayed(evaluate_pde)(eta) for _ in range(repeats_pde))

    gyrations, boxDims, corrDims = np.array(results).T

    gyration_pde[i, :] = [np.mean(gyrations), np.std(gyrations)]
    boxDim_pde[i, :] = [np.mean(boxDims), np.std(boxDims)]
    corrDim_pde[i, :] = [np.mean(corrDims), np.std(corrDims)]

# fig, axes = plt.subplots(3, 2)

# # radius of gyration
# ax1, ax2 = axes[0, :]

# ax1.errorbar(etas, gyration_pde[:, 0], yerr=gyration_pde[:, 1])
# ax1.set_title("PDE")
# ax1.set_ylabel("Radius of gyration")

# ax2.errorbar(sticking_values, gyration_mc[:, 0], yerr=gyration_mc[:, 1])
# ax2.set_title("MC")

# # box-dimension
# ax1, ax2 = axes[1, :]

# ax1.errorbar(etas, boxDim_pde[:, 0], yerr=boxDim_pde[:, 1])
# ax1.set_ylabel("Box-dimension")

# ax2.errorbar(sticking_values, boxDim_mc[:, 0], yerr=boxDim_mc[:, 1])

# # correlation dimension
# ax1, ax2 = axes[2, :]

# ax1.errorbar(etas, corrDim_pde[:, 0], yerr=corrDim_pde[:, 1])
# ax1.set_xlabel("$\\eta$")
# ax1.set_ylabel("Correlation dimension")

# ax2.errorbar(sticking_values, corrDim_mc[:, 0], yerr=corrDim_mc[:, 1])
# ax2.set_xlabel("$p_s$")

# plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.errorbar(
    gyration_pde[:, 0], boxDim_pde[:, 0], 
    xerr=gyration_pde[:, 1], yerr=boxDim_pde[:, 1],
    fmt="none", color="grey", alpha=0.5
)
sc_pde = ax1.scatter(
    gyration_pde[:, 0], boxDim_pde[:, 0],
    c=etas, marker='o', s=80, label='PDE',
    zorder=2
)

cbar_pde = plt.colorbar(sc_pde, ax=ax1, pad=0.01)
cbar_pde.set_label("PDE: $\\eta$")

ax1.errorbar(
    gyration_mc[:, 0], boxDim_mc[:, 0], 
    xerr=gyration_mc[:, 1], yerr=boxDim_mc[:, 1],
    fmt="none", color="grey", alpha=0.5
)
sc_mc = ax1.scatter(
    gyration_mc[:, 0], boxDim_mc[:, 0],
    c=sticking_values, marker='X', s=80, label='MC',
    zorder=2
)
cbar_mc = plt.colorbar(sc_mc, ax=ax1, pad=0.05)
cbar_mc.set_label("MC: $p_s$")

ax1.set_xlabel("Radius of Gyration")
ax1.set_ylabel("Box-counting Dimension")
ax1.legend()

ax2.errorbar(
    gyration_pde[:, 0], corrDim_pde[:, 0], 
    xerr=gyration_pde[:, 1], yerr=corrDim_pde[:, 1],
    fmt="none", color="grey", alpha=0.5
)
sc_pde = ax2.scatter(
    gyration_pde[:, 0], corrDim_pde[:, 0],
    c=etas, marker='o', s=80, label='PDE',
    zorder=2
)
cbar_pde = plt.colorbar(sc_pde, ax=ax2, pad=0.01)
cbar_pde.set_label("PDE: $\\eta$")

ax2.errorbar(
    gyration_mc[:, 0], corrDim_mc[:, 0], 
    xerr=gyration_mc[:, 1], yerr=corrDim_mc[:, 1],
    fmt="none", color="grey", alpha=0.5
)
sc_mc = ax2.scatter(
    gyration_mc[:, 0], corrDim_mc[:, 0],
    c=sticking_values, marker='X', s=80, label='MC',
    zorder=2
)
cbar_mc = plt.colorbar(sc_mc, ax=ax2, pad=0.05)
cbar_mc.set_label("MC: $p_s$")

ax2.set_xlabel("Radius of Gyration")
ax2.set_ylabel("Correlation Dimension")
ax2.legend()
plt.show()

    