import numpy as np
import matplotlib.pyplot as plt

from DLA.dla_methods import *

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

etas = [0, 0.5, 1, 2]
fig, axes = plt.subplots(2, 2)

N = 100
x = np.arange(N)

for i, ax in enumerate(axes.flatten()):
    field_frames, cluster_frames = dla_simulation(N=N, omega=1.88, eta=etas[i])

    final = field_frames[-1]
    final[cluster_frames[-1]] = np.nan

    im = ax.pcolor(x, x, final.T)
    ax.set_aspect(1)
    ax.set_title(f"$\\eta = {etas[i]}$")

axes[0, 0].set_ylabel("$y$")
axes[1, 0].set_xlabel("$x$")
axes[1, 0].set_ylabel("$y$")
axes[1, 1].set_xlabel("$x$")

cbar = plt.colorbar(im, ax=axes)
cbar.set_label("Concentration")

plt.show()