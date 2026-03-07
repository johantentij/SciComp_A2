import numpy as np
from numba import njit, prange
import warnings
import time

@njit
def seed_numba(seed_value):
    np.random.seed(seed_value)

@njit(parallel=True)
def sor_step(c, cluster, omega, color, N):
    """
        Performs one color-pass (Red or Black) of the Successive Over-Relaxation (SOR)
        method to solve the Laplace equation on a 2D grid.

        This implementation is decorated with @njit(parallel=True) and utilizes
        Red-Black ordering to allow for race-condition-free parallel updates.

        Parameters:
        -----------
        c : numpy.ndarray (2D, float64)
            The grid containing the values (e.g., potential or temperature).
            Updated in-place.
        cluster : numpy.ndarray (2D, bool)
            A boolean mask where True indicates a fixed boundary or internal
            constraint (e.g., a conductor) that should not be updated.
        omega : float
            The relaxation parameter.
            - 1.0 results in Gauss-Seidel iteration.
            - 1.0 < omega < 2.0 provides over-relaxation for faster convergence.
        color : int (0 or 1)
            Specifies which set of the checkerboard to update:
            - 0: Red cells (where (i + j) % 2 == 0)
            - 1: Black cells (where (i + j) % 2 == 1)
        N : int
            The size of the grid (assumed square, including boundary layers).

        Returns:
        --------
        max_diff : float
            The maximum absolute change (L-infinity norm) recorded across all
            updated cells during this pass. Used for convergence monitoring.
    """

    # We only parallelize the outer loop to keep overhead low
    # We use a 1D array to store max diffs to avoid concurrency issues
    row_diffs = np.zeros(N, dtype=np.float64)
    inv_4 = 0.25 * omega
    one_minus_omega = 1.0 - omega

    for i in prange(N):
        ip = (i + 1) % N
        im = (i - 1) % N
        row_max = 0.0

        # Determine if the first j in this row is Red or Black
        # if (i + 1) % 2 == color, start at 1, else start at 2
        start_j = 1 if (i + 1) % 2 == color else 2

        # Use a step of 2
        for j in range(start_j, N - 1, 2):
            if not cluster[i, j]:
                old = c[i, j]
                # Inlined SOR math
                res = (c[ip, j] + c[im, j] + c[i, j + 1] + c[i, j - 1])
                c[i, j] = one_minus_omega * old + inv_4 * res

                diff = abs(c[i, j] - old)
                if diff > row_max:
                    row_max = diff

        row_diffs[i] = row_max

    return np.max(row_diffs)

@njit(fastmath=True)
def solve_laplace_red_black(c, cluster, omega, max_iter, tol, N):
    """
        Solves the Laplace equation on a 2D grid using the Successive Over-Relaxation
        (SOR) method with Red-Black checkerboard ordering.

        This function orchestrates the iterative solver by alternating between
        Red and Black update passes until the solution converges or the
        iteration limit is reached.

        Parameters:
        -----------
        c : numpy.ndarray (2D, float64)
            The grid of values (potential, temperature, etc.). Updated in-place.
        cluster : numpy.ndarray (2D, bool)
            A mask where True indicates fixed nodes (boundary conditions or
            internal conductors) that should not be updated.
        omega : float
            The relaxation parameter.
            - omega = 1: Gauss-Seidel.
            - 1 < omega < 2: Successive Over-Relaxation (accelerates convergence).
        max_iter : int
            The maximum number of full iterations (one Red + one Black pass each).
        tol : float
            The convergence tolerance for the maximum absolute change in cell values.
        N : int
            The size of the grid.

        Returns:
        --------
        c : numpy.ndarray
            The updated 2D grid after convergence or reaching max_iter.
        it : int
            The total number of iterations completed.
        """

    for it in range(1,max_iter+1):
        # Update Red sites then Black sites
        diff_red = sor_step(c, cluster, omega, 0, N)
        diff_black = sor_step(c, cluster, omega, 1, N)

        if max(diff_red, diff_black) < tol:
            break
    return c, it


@njit(fastmath=True)
def solve_laplace_sor(c, cluster, omega, max_iter, tol, N):
    """
        Standard Successive Over-Relaxation (SOR) solver for the Laplace equation.

        This version uses a "warm start" approach, preserving the values in 'c'
        from previous calls to accelerate convergence in iterative growth models.

        Parameters:
        -----------
        c : numpy.ndarray (2D, float64)
            The grid of values, updated in-place.
        cluster : numpy.ndarray (2D, bool)
            Mask where True indicates fixed boundary/obstacle nodes.
        omega : float
            Relaxation parameter (1.0 for Gauss-Seidel, >1.0 for SOR).
        max_iter : int
            Maximum number of iterations to perform.
        tol : float
            The convergence threshold for the maximum absolute difference.
        N : int
            The grid dimension.

        Returns:
        --------
        c : numpy.ndarray
            The solved 2D grid.
        it : int
            The number of iterations performed until convergence.
    """
    for it in range(1,max_iter+1):
        max_diff = 0.0
        for i in range(N):
            ip, im = (i + 1) % N, (i - 1) % N
            for j in range(1, N - 1):
                if cluster[i, j]: continue

                old = c[i, j]
                new = 0.25 * (c[ip, j] + c[im, j] + c[i, j + 1] + c[i, j - 1])
                c[i, j] = (1 - omega) * old + omega * new

                diff = abs(c[i, j] - old)
                if diff > max_diff: max_diff = diff
        if max_diff < tol:
            break
    return c, it

@njit(fastmath=True)
def get_candidates_and_choose(is_candidate, c, eta):
    """
        Selects a growth site from potential candidates based on a probability
        distribution derived from the field 'c'.

        Parameters:
        -----------
        is_candidate : numpy.ndarray (2D, bool)
            A mask identifying potential growth sites (the "skin" of the cluster).
        c : numpy.ndarray (2D, float64)
            The field values (e.g., probability amplitude or potential).
        eta : float
            The power-law exponent. Controls the fractal dimension of growth:
            - eta = 0: Random growth (Eden model).
            - eta = 1: Dielectric Breakdown Model (DBM) / Diffusion Limited Aggregation (DLA).

        Returns:
        --------
        (i, j) : tuple of int
            The coordinates of the chosen site. Returns (-1, -1) if no
            candidates exist.
    """

    idx_i, idx_j = np.where(is_candidate)
    if len(idx_i) == 0:
        return -1, -1

    weights = np.empty(len(idx_i), dtype=np.float64)
    total_w = 0.0
    for k in range(len(idx_i)):
        # ensure non-negative concentrations
        val = max(0, c[idx_i[k], idx_j[k]])

        w = val ** eta
        weights[k] = w
        total_w += w

    if total_w == 0.0:
        print("Hit")
        return -2, -2

    r = np.random.rand() * total_w

    cum = 0.0
    for k in range(len(idx_i)):
        cum += weights[k]
        if r <= cum:
            return idx_i[k], idx_j[k]
    return idx_i[0], idx_j[0]

@njit(fastmath=True)
def update_candidate_mask(i_new, j_new, cluster, is_candidate, N, c):
    """
        Efficiently updates the list of growth candidates after a new site
        has been added to the cluster.

        Parameters:
        -----------
        i_new, j_new : int
            Coordinates of the newly added cluster site.
        cluster : numpy.ndarray (2D, bool)
            The current state of the grown structure.
        is_candidate : numpy.ndarray (2D, bool)
            The mask of potential sites to be updated.
        N : int
            Grid dimension.
        c : numpy.ndarray (2D, float64)
            The field array, used here to verify if a neighbor is a valid
            new candidate.
    """

    is_candidate[i_new, j_new] = False
    for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        ni, nj = (i_new + di) % N, j_new + dj
        if 0 < nj < N - 1 and (not cluster[ni, nj]):
            is_candidate[ni, nj] = True


def dla_simulation(N=100, seed_i=None, seed_j=1, omega=1.94,
                   max_sor_iter=5000, tolerance=1e-5, growth_steps=800,
                   eta=1.0, frame_interval=10, parallel=False, prng_seed=None, 
                   outputSORiters=False, logging=True):
    """
        Simulates fractal growth using the Dielectric Breakdown Model (DBM) /
        Diffusion Limited Aggregation (DLA) framework via iterative Laplace solving.

        This function manages the high-level simulation loop, alternating between
        solving the potential field and stochastically attaching new particles
        to the growing cluster.

        Parameters:
        -----------
        N : int
            Grid resolution (N x N).
        seed_i, seed_j : int, optional
            Initial coordinates for the cluster seed. Defaults to (N//2, 1).
        omega : float
            Successive Over-Relaxation parameter for the Laplace solver.
        max_sor_iter : int
            Limit on iterations for the SOR solver per growth step.
        tolerance : float
            Convergence threshold for the field solver.
        growth_steps : int
            Total number of particles to add to the cluster.
        eta : float
            The growth exponent (η).
            - η=1: Standard DLA.
            - η > 1: More branched/needle-like growth.
            - η < 1: More compact/globular growth.
        frame_interval : int
            Frequency of saving grid states for animation/visualization.
        parallel : bool
            If True, uses the Red-Black parallel SOR solver.
            If False, uses the standard serial SOR solver with a warm start.
        prng_seed : int, optional
            Seed for the Numba-compatible random number generator.
        outputSORiters : bool, optional
            Enables return of the amount of SOR iterations used 
        logging : bool, optional
            Sets whether to log the progress

        Returns:
        --------
        frames_field : list of numpy.ndarray
            Snapshot history of the potential field 'c'.
        frames_cluster : list of numpy.ndarray
            Snapshot history of the boolean 'cluster' mask.
    """

    if prng_seed:
        seed_numba(prng_seed)

    if seed_i is None:
        seed_i = N//2

    cluster = np.zeros((N, N), dtype=np.bool_)
    is_candidate = np.zeros((N, N), dtype=np.bool_)
    c = np.zeros((N, N))

    def system_init():

        # Linear concentration gradient for empty system
        for j in range(N):
            c[:, j] = j / (N - 1)

        cluster[seed_i, seed_j] = True
        c[seed_i, seed_j] = 0.0
        update_candidate_mask(seed_i, seed_j, cluster, is_candidate, N, c)

    system_init()

    # Initial solve
    c, iters = solve_laplace_sor(c, cluster, omega, max_sor_iter, tolerance, N)

    # ============================================================
    # DLA LOOP
    # ============================================================
    frames_field = []
    frames_cluster = []
    frames_field.append(c.copy())
    frames_cluster.append(cluster.copy())

    SOR_iters = [iters]

    start = time.perf_counter()

    if logging:
        print("Simulating growth...")
    for step in range(1, growth_steps + 1):
        # Choose new site from the boundary only
        i_new, j_new = get_candidates_and_choose(is_candidate, c, eta)
        if i_new == -2:
            warnings.warn(f" Step {step} | All candidate propensities are 0, impossible to continue DLA.")
            break
        elif i_new == -1:
            warnings.warn(f"Step {step} | No candidates left. Exiting...")

        # Add to cluster and update boundary mask
        cluster[i_new, j_new] = True
        c[i_new, j_new] = 0.0

        if (j_new == N - 2):
            frames_field.append(c.copy())
            frames_cluster.append(cluster.copy())
            SOR_iters.append(iters)
            if logging:
                print(f"Top has been reached at step {step}")

            break

        update_candidate_mask(i_new, j_new, cluster, is_candidate, N, c)

        if parallel:
            # Parallel SOR solve
            c, iters = solve_laplace_red_black(c, cluster, omega, max_sor_iter, tolerance, N)

        else:
            # Solve starting from previous field (Warm Start)
            c, iters = solve_laplace_sor(c, cluster, omega, max_sor_iter, tolerance, N)

        if step % frame_interval == 0:
            frames_field.append(c.copy())
            frames_cluster.append(cluster.copy())
            SOR_iters.append(iters)
            if step % 200 == 0 and logging:
                print(f"Step {step} | SOR Iters: {iters}")

    t_python = time.perf_counter() - start
    if logging:
        print(f"Python time: {t_python:.6f} seconds")

    if outputSORiters:
        return frames_field, frames_cluster, SOR_iters
    else:
        return frames_field, frames_cluster

# dla_simulation(eta=1,prng_seed=42)