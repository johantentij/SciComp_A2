import shlex
import timeit
from numba import njit
import numpy as np
import scipy.stats as st

probs = np.random.rand(10000)
probs /= probs.sum()

def sample_discrete_scipy(probs):
    """Randomly sample an index with probability given by probs."""
    return st.rv_discrete(values=(range(len(probs)), probs)).rvs()


def sample_discrete_numpy(probs):
    """Randomly sample an index with probability given by probs."""
    probs = np.asarray(probs, dtype=float)
    return np.random.choice(len(probs), p=probs)


@njit
def sample_discrete(probs):
    """Randomly sample an index with probability given by probs."""
    # Generate random number
    q = np.random.rand()

    # Find index
    i = 0
    p_sum = 0.0
    while p_sum < q:
        p_sum += probs[i]
        i += 1
    return i - 1

print("Scipy")
timeit.main(args=shlex.split("""-s'from __main__ import sample_discrete_scipy, probs'  
'sample_discrete_scipy(probs)'"""))
print("Numpy")
timeit.main(args=shlex.split("""-s'from __main__ import sample_discrete_numpy, probs'  
'sample_discrete_numpy(probs)'"""))
print("By hand")
timeit.main(args=shlex.split("""-s'from __main__ import sample_discrete, probs'  
'sample_discrete(probs)'"""))