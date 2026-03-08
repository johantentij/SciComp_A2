import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import scipy.sparse as sp

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 25,
    "axes.labelsize": 25,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
    "legend.fontsize": 12,
})

N = 2000

D_u = .16
D_v = .08
# f = .035
# k = .06

f = np.linspace(0, .07, N)
k = np.linspace(0, .07, N)

F = np.outer(f, np.ones(N))
F = F.flatten(order='F')

K = np.outer(np.ones(N), k)
K = K.flatten(order='F')

dx = 1
dt = 1

x = np.arange(N) * dx

epsilon = .01

# D2 = -2 * np.identity(N)
# D2 += np.roll(np.identity(N), 1, axis=0)
# D2 += np.roll(np.identity(N), -1, axis=0)
# D2 = sp.csr_matrix(D2)
D2 = sp.diags((np.ones(N - 1), -2 * np.ones(N), np.ones(N)), (-1, 0, 1))

I = sp.eye(N, format="csr")

D2_2 = sp.kron(I, D2) + sp.kron(D2, I)

def dState_dt(state):
    U, V = state

    dU_dt = D_u * D2_2.dot(U) - U * V * V + F * (1 - U)
    dV_dt = D_v * D2_2.dot(V) + U * V * V - (F + K) * V

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

initDivs = 150
noiseAmplitude = 5
for i in range(initDivs):
    xc = i * N // initDivs     
    for j in range(initDivs):
        yc = j * N // initDivs

        xo = int((2 * np.random.random() - 1) * noiseAmplitude)
        yo = int((2 * np.random.random() -1 ) * noiseAmplitude)
        V_init[xc + xo - squareRadius : xc + xo + squareRadius, yc + yo - squareRadius : yc + yo + squareRadius] = .25


U_init = U_init.flatten(order='F') + epsilon * np.random.random(N ** 2)
V_init = V_init.flatten(order='F') + epsilon * np.random.random(N ** 2)
# U_init = .5 * np.ones(N ** 2) + epsilon * np.random.random(N ** 2)
# V_init = .25 * np.ones(N ** 2) + epsilon * np.random.random(N ** 2)

state = np.array([U_init, V_init])

U, V = state
U = np.reshape(U, (N, N), order='F')
V = np.reshape(V, (N, N), order='F')



fig, ax = plt.subplots(1, figsize=(12, 12))

im = ax.imshow(
    U,
    origin='lower',
    extent=[np.min(f), np.max(f), np.min(k), np.max(k)],
    cmap='viridis',
    animated=True,
    vmin=0,
    vmax=1
)
ax.set_title("$U(x, y)$")
ax.set_xlabel("$k$")
ax.set_ylabel("$f$")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Concentration")

frame_step = 1

def update(frame):
    global state

    for _ in range(frame_step):
        # state = RK4_step(state)
        state = Euler_step(state)

    U, V = state
    U = np.reshape(U, (N, N), order='F')
    V = np.reshape(V, (N, N), order='F')

    im.set_data(U)

    fig.suptitle(f"t = {frame * frame_step * dt:.2f}")
    
    return [im]
    
ani = animation.FuncAnimation(
    fig, update, interval=10, cache_frame_data=False
) 

plt.show()