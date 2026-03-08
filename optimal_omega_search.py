import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from joblib import Parallel, delayed
from DLA.dla_methods import *

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

N = 100

def evaluate(w, eta):
    _, _, iters = dla_simulation(N=N, omega=w, eta=eta, outputSORiters=True, frame_interval=1, logging=False)

    return np.mean(iters)

N_omegas = 20
N_etas = 8

omegas = np.linspace(1.7, 1.95, N_omegas)
repeats = 64

etas = np.linspace(0, 2, N_etas)
optimal_omega_range = np.empty((N_etas, 2))
optimal_omegas = np.empty(N_etas)

costs = np.empty((N_etas, N_omegas, repeats))
for k, eta in enumerate(etas):
    print("eta value %d out of %d" % (k, N_etas))

    for i, w in enumerate(omegas):
        print("omega value %d out of %d" % (i, N_omegas))

        results = Parallel(n_jobs=-1)(delayed(evaluate)(w, eta) for _ in range(repeats))

        costs[k, i, :] = np.array(results)

    best_w_index = np.argmin(np.mean(costs[k, :, :], axis=1))
    p_vals = np.ones(N_omegas)
    for i, w in enumerate(omegas):
        if i != best_w_index:
            result = stats.ttest_ind(costs[k, best_w_index, :], costs[k, i, :], equal_var=False, alternative="less")

            p_vals[i] = result.pvalue

    insignificant = omegas[p_vals > .05]

    optimal_omegas[k] = omegas[best_w_index]
    optimal_omega_range[k, :] = [np.min(insignificant), np.max(insignificant)]

plt.fill_between(etas, optimal_omega_range[:, 0], optimal_omega_range[:, 1], alpha=0.5)
plt.plot(etas, optimal_omegas)
plt.grid(True)
plt.yticks(omegas)
labels = [f"{val:.3f}" if i % 2 == 0 else "" for i, val in enumerate(omegas)]
plt.gca().set_yticklabels(labels)
plt.xticks(etas)
plt.xlabel("$\\eta$")
plt.ylabel("Optimal $\\omega$")

plt.show()