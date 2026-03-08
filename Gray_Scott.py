import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import scipy.sparse as sp

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

N = 200

D_u = .16
D_v = .08

A = [.04, .01]
B = [.062, .03]
C = [.06, .045]
# f = .035
# k = .06

k, f = C

dx = 1
dt = 1

x = np.arange(N) * dx

epsilon = 0

# D2 = -2 * np.identity(N)
# D2 += np.roll(np.identity(N), 1, axis=0)
# D2 += np.roll(np.identity(N), -1, axis=0)
# D2 = sp.csr_matrix(D2)
D2 = sp.diags((np.ones(N - 1), -2 * np.ones(N), np.ones(N)), (-1, 0, 1))

I = sp.eye(N, format="csr")

D2_2 = sp.kron(I, D2) + sp.kron(D2, I)

def dState_dt(state):
    U, V = state

    dU_dt = D_u * D2_2.dot(U) - U * V * V + f * (1 - U)
    dV_dt = D_v * D2_2.dot(V) + U * V * V - (f + k) * V

    return np.array([dU_dt, dV_dt])

def RK4_step(state):
    k1 = dState_dt(state)
    k2 = dState_dt(state + .5 * k1 * dt)
    k3 = dState_dt(state + .5 * k2 * dt)
    k4 = dState_dt(state + k3 * dt)

    return state + (k1 + 2 * (k2 + k3) + k4) * dt / 6

def Euler_step(state):
    k1 = dState_dt(state)

    return state + k1 * dt

N_steps = 20

U_init = .5 * np.ones((N, N))
V_init = np.zeros((N, N))
squareRadius = 4

xc = N // 2
V_init[xc - squareRadius : xc + squareRadius, xc - squareRadius : xc + squareRadius] = .25

xc = N // 2 - 50
V_init[xc - squareRadius : xc + squareRadius, xc - squareRadius : xc + squareRadius] = .25

U_init = U_init.flatten(order='F') + epsilon * np.random.random(N ** 2)
V_init = V_init.flatten(order='F') + epsilon * np.random.random(N ** 2)
# U_init = .5 * np.ones(N ** 2) + epsilon * np.random.random(N ** 2)
# V_init = .25 * np.ones(N ** 2) + epsilon * np.random.random(N ** 2)

state = np.array([U_init, V_init])

U, V = state
U = np.reshape(U, (N, N), order='F')
V = np.reshape(V, (N, N), order='F')



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

im1 = ax1.imshow(
    U,
    origin='lower',
    extent=[0, np.max(x), 0, np.max(x)],
    cmap='viridis',
    animated=True,
    vmin=0,
    vmax=1
)
ax1.set_title("$U(x, y)$")
ax1.set_xlabel("$x$")
ax1.set_ylabel("$y$")

im2 = ax2.imshow(
    V,
    origin='lower',
    extent=[0, np.max(x), 0, np.max(x)],
    cmap='viridis',
    animated=True,
    vmin=0,
    vmax=1
)
ax2.set_title("$V(x, y)$")
ax2.set_xlabel("$x$")
ax2.set_ylabel("$y$")

cbar = fig.colorbar(im1, ax=[ax1, ax2])
cbar.set_label("Concentration")

frame_step = 50

def update(frame):
    global state

    for _ in range(frame_step):
        # state = RK4_step(state)
        state = Euler_step(state)

    U, V = state
    U = np.reshape(U, (N, N), order='F')
    V = np.reshape(V, (N, N), order='F')

    im1.set_data(U)
    im2.set_data(V)

    fig.suptitle(f"t = {frame * frame_step * dt:.2f}")
    
    return [im1, im2]
    
ani = animation.FuncAnimation(
    fig, update, interval=10, cache_frame_data=False
) 

plt.show()