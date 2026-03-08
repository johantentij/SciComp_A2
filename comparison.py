import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import time

from DLA.dla_methods import dla_simulation
from dla_mc import monte_carlo_dla
from cluster_metrics import calculate_metrics

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

def plot_clusters(ax, N=100, num_particles=800, p_s=1):
    cluster = monte_carlo_dla(N=N, num_particles=num_particles, p_s=p_s)
    
    x = np.arange(N)
    ax.pcolor(x, x, cluster.T)
    ax.set_title(f"$p_s={p_s}$")

def compare_metrics(N=80, particle_list=[200, 400, 600], trials=10):
    metrics_keys = ["Max Height", "Horizontal Spread", "Radius of Gyration", "Correlation Dimension", "Box-Counting Dimension"]
    
    results = {"MC": {k: [] for k in metrics_keys}, "PDE": {k: [] for k in metrics_keys}}
    
    for p_num in particle_list:
        print(f"Computing for {p_num} particles...")
        mc_temp = {k: [] for k in metrics_keys}
        pde_temp = {k: [] for k in metrics_keys}
        
        for t in range(trials):
            # mc simulation and metric extraction
            mc_cluster = monte_carlo_dla(N=N, num_particles=p_num, p_s=1.0)
            mc_res = calculate_metrics(mc_cluster)
            
            # pde simulation and metric extraction with unique seed
            seed_val = np.random.randint(1, 100000)
            _, pde_frames = dla_simulation(N=N, growth_steps=p_num, eta=1.0, parallel=True, prng_seed=seed_val, logging=False)
            pde_res = calculate_metrics(pde_frames[-1])
            
            for k in metrics_keys:
                mc_temp[k].append(mc_res[k])
                pde_temp[k].append(pde_res[k])
                
        # record statistical distributions
        for k in metrics_keys:
            results["MC"][k].append(mc_temp[k])
            results["PDE"][k].append(pde_temp[k])
            
    return results

def plot_metrics(particle_list, results):
    metrics_keys = list(results["PDE"].keys())
    fig, axes = plt.subplots(1, 2, figsize=(20, 9), dpi=120)
    
    # plot 1: fractal dimensions
    ax1 = axes[0]
    dim_metrics = ["Correlation Dimension", "Box-Counting Dimension"]
    model_linestyles = {'MC': '-', 'PDE': '--'}
    metric_colors_dim = {'Correlation Dimension': '#2ca02c', 'Box-Counting Dimension': '#ff7f0e'}

    for metric in dim_metrics:
        for model in ['MC', 'PDE']:
            means = [np.mean(trials) for trials in results[model][metric]]
            stds = [np.std(trials) for trials in results[model][metric]]
            ax1.errorbar(particle_list, means, yerr=stds, 
                         label=f'{model} - {metric.replace(" Dimension", "")}', 
                         linestyle=model_linestyles[model],
                         marker='o' if model == 'MC' else 's',
                         color=metric_colors_dim[metric],
                         capsize=5, alpha=0.9)

    ax1.set_title("Fractal Dimensions Comparison")
    ax1.set_xlabel("Number of Particles ($N$)")
    ax1.set_ylabel("Dimension")
    ax1.legend(loc='best')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_ylim(1, 2)

    # plot 2: size metrics
    ax2 = axes[1]
    size_metrics = ["Max Height", "Horizontal Spread", "Radius of Gyration"]
    metric_colors_size = {'Max Height': '#1f77b4', 'Horizontal Spread': '#d62728', 'Radius of Gyration': '#9467bd'}

    for metric in size_metrics:
        for model in ['MC', 'PDE']:
            means = [np.mean(trials) for trials in results[model][metric]]
            stds = [np.std(trials) for trials in results[model][metric]]
            ax2.errorbar(particle_list, means, yerr=stds, 
                         label=f'{model} - {metric}',
                         linestyle=model_linestyles[model],
                         marker='o' if model == 'MC' else 's',
                         color=metric_colors_size[metric],
                         capsize=5, alpha=0.9)

    ax2.set_title("Cluster Size Metrics Comparison")
    ax2.set_xlabel("Number of Particles ($N$)")
    ax2.set_ylabel("Size (pixels)")
    ax2.legend(loc='best')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # coral overlapping plot
    N_grid = 100
    # particle_counts = [200, 400, 600, 800]
    p_s_values = [.25, .5, .75, 1]
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 6), dpi=120)
    
    axesFlat = axes.flatten()
    for i, p_s in enumerate(p_s_values):
        plot_clusters(axesFlat[i], N=N_grid, num_particles=800, p_s=p_s)
    

    axes[0, 0].set_ylabel("$y$")
    axes[1, 0].set_xlabel("$x$")
    axes[1, 0].set_ylabel("$y$")
    axes[1, 1].set_xlabel("$x$")

    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    # stats comparison plot
    N_particles = [100, 200, 300, 400, 500, 600, 700, 800]
    stats = compare_metrics(N=N_grid, particle_list=N_particles, trials=50)
    plot_metrics(N_particles, stats)