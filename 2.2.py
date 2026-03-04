import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# params
grid_size = 100
num_particles = 800
p_s = 0.1 # sticking probability

# init grid and seed
grid = np.zeros((grid_size, grid_size), dtype=bool)
grid[grid_size - 1, grid_size // 2] = True

# directions: right, down, left, up
directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

def random_walk():
    # start from random position on top boundary
    y = np.random.randint(0, grid_size)
    x = 0
    
    while True:
        dx, dy = directions[np.random.randint(0, 4)]
        new_x = x + dx
        new_y = (y + dy) % grid_size  # periodic horizontal boundary

        # vertical boundary: remove if walking out
        if new_x < 0 or new_x >= grid_size:
            return None 

        # prevent walking into the cluster
        if grid[new_x, new_y]:
            continue

        x, y = new_x, new_y

        # check neighbors for attachment
        for n_dx, n_dy in directions:
            nx = x + n_dx
            ny = (y + n_dy) % grid_size
            if 0 <= nx < grid_size and grid[nx, ny]:
                # attach based on probability
                if np.random.rand() < p_s:
                    grid[x, y] = True
                    return x, y
                break  # bounce off but do not attach yet
            
def generate_particles():
    attached = 0
    # loop until target number of particles stick
    while attached < num_particles:
        pos = random_walk()
        if pos:
            attached += 1
            yield grid, attached

# visualization
fig, ax = plt.subplots()
img = ax.imshow(grid, cmap='magma', interpolation='nearest')
plt.axis('off')

time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, color='white')

def update(frame):
    current_grid, time_step = frame
    img.set_data(current_grid)
    time_text.set_text(f'particles: {time_step}/{num_particles} (sticking probability={p_s})')
    return [img, time_text]

ani = FuncAnimation(fig, update, frames=generate_particles, interval=1, blit=True, cache_frame_data=False, repeat=False)
plt.show()