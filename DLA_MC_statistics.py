import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import time
from cluster_metrics import calculate_metrics

@njit(fastmath=True)
def monte_carlo_dla(N=100, num_particles=800, p_s=1.0, seed_i=None, seed_j=1):
    if seed_i is None:
        seed_i = N // 2
        
    cluster = np.zeros((N, N), dtype=np.bool_)
    cluster[seed_i, seed_j] = True
    
    # directions: right, left, down, up
    directions = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
    
    particles_added = 0
    while particles_added < num_particles:
        # random walkers from upper bound
        i = np.random.randint(0, N)
        j = N - 1 
        
        # prevent spawning inside the cluster if it reaches the ceiling
        if cluster[i, j]:
            break
            
        alive = True
        while alive:
            dir_idx = np.random.randint(0, 4)
            di, dj = directions[dir_idx]
            
            ni = (i + di) % N  # x axis periodic boundary
            nj = j + dj        # y axis
            
            # remove if goes out of the boundary (cleaner than alive = False + continue)
            if nj < 0 or nj >= N:
                break
                
            # bounce off the cluster
            if cluster[ni, nj]:
                continue 
                
            i, j = ni, nj
            
            # check the neighbors
            for d in range(4):
                nni = (i + directions[d, 0]) % N
                nnj = j + directions[d, 1]
                
                if 0 <= nnj < N and cluster[nni, nnj]:
                    if np.random.rand() < p_s:
                        cluster[i, j] = True
                        particles_added += 1
                        alive = False
                    # stop checking other neighbors if we touched the cluster but didn't stick
                    break
                    
    return cluster


def run_ps_statistics(N=80, num_particles=400, trials=5, ps_values=None):
    metrics_keys = ["Max Height", "Horizontal Spread", "Radius of Gyration", "Correlation Dimension", "Box-Counting Dimension"]
    
    # initialize dictionary to store results
    results = {k: {"means": [], "stds": []} for k in metrics_keys}
    
    print(f"starting p_s parameter sweep. N={N}, particles={num_particles}, trials={trials}")
    
    for ps in ps_values:
        print(f"--> testing p_s = {ps}...")
        temp_data = {k: [] for k in metrics_keys}
        
        for t in range(trials):
            start_t = time.time()
            # generate cluster
            cluster = monte_carlo_dla(N=N, num_particles=num_particles, p_s=ps)
            
            # calculate metrics
            metrics = calculate_metrics(cluster)
            for k in metrics_keys:
                temp_data[k].append(metrics[k])
                
            print(f"    trial {t+1}/{trials} done in {time.time()-start_t:.2f}s")
            
        # calculate mean and std for this p_s
        for k in metrics_keys:
            results[k]["means"].append(np.mean(temp_data[k]))
            results[k]["stds"].append(np.std(temp_data[k]))
            
    return ps_values, results

def plot_ps(ps_values, results):
    metrics_keys = list(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=120)
    
    # plot 1: fractal dimensions
    ax1 = axes[0]
    dim_metrics = ["Correlation Dimension", "Box-Counting Dimension"]
    dim_colors = {'Correlation Dimension': '#2ca02c', 'Box-Counting Dimension': '#ff7f0e'}
    dim_markers = {'Correlation Dimension': 'o', 'Box-Counting Dimension': 's'}

    for metric in dim_metrics:
        means = results[metric]["means"]
        stds = results[metric]["stds"]
        ax1.errorbar(ps_values, means, yerr=stds, label=metric,
                     fmt=f'-{dim_markers[metric]}', capsize=5, color=dim_colors[metric], 
                     markerfacecolor='white', markeredgewidth=2)

    ax1.set_title("Fractal Dimensions vs. $p_s$")
    ax1.set_xlabel("Sticking Probability ($p_s$)")
    ax1.set_ylabel("Dimension")
    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(1, 2)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # plot 2: size metrics
    ax2 = axes[1]
    size_metrics = ["Max Height", "Horizontal Spread", "Radius of Gyration"]
    size_colors = {'Max Height': '#1f77b4', 'Horizontal Spread': '#d62728', 'Radius of Gyration': '#9467bd'}
    size_markers = {'Max Height': 'o', 'Horizontal Spread': 's', 'Radius of Gyration': '^'}

    for metric in size_metrics:
        means = results[metric]["means"]
        stds = results[metric]["stds"]
        ax2.errorbar(ps_values, means, yerr=stds, label=metric,
                     fmt=f'-{size_markers[metric]}', capsize=5, color=size_colors[metric])

    ax2.set_title("Cluster Size Metrics vs. $p_s$")
    ax2.set_xlabel("Sticking Probability ($p_s$)")
    ax2.set_ylabel("Size (pixels)")
    ax2.set_xlim(0, 1.05)
    ax2.set_ylim(bottom=0)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ps_list = [0.01, 0.2, 0.4, 0.6, 0.8, 1]
    ps_vals, stats = run_ps_statistics(N=100, num_particles=700, trials=50, ps_values=ps_list)
    plot_ps(ps_vals, stats)