import os, sys, time, logging
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import lobpcg

def fiedler_on_gcc(A, N):
    """Returns Fiedler value on GCC, plus whether graph was disconnected."""
    n_comp, labels = connected_components(A, directed=False, return_labels=True)
    
    disconnected = n_comp > 1
    if disconnected:
        # Slice out the GCC
        gcc_label = np.bincount(labels).argmax()
        mask = labels == gcc_label
        A = A.tocsr()[mask][:, mask]
        N = mask.sum()

    L = laplacian(A, normed=False).asfptype().tocsr()
    diag = L.diagonal().copy()
    diag[diag == 0] = 1.0
    M = diags(1.0 / diag)

    rng = np.random.default_rng(42)
    X = rng.standard_normal((N, 2))

    try:
        evals, _ = lobpcg(L, X, M=M, largest=False, tol=1e-5, maxiter=300)
        evals.sort()
        fiedler = max(0.0, float(evals[1]))
    except Exception:
        from scipy.sparse.linalg import eigsh
        evals, _ = eigsh(L, k=2, sigma=1e-6, which='LM')
        fiedler = max(0.0, float(sorted(evals)[1]))

    return fiedler, disconnected