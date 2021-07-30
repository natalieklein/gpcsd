"""
Utility functions for CSD methods.
"""

import autograd.numpy as np

def normalize(x):
    return x/np.max(np.abs(x), axis=(0, 1))

def sort_grid(x):
    xsrt = x[x[:, 1].argsort()]
    xsrt = xsrt[xsrt[:, 0].argsort(kind='mergesort')]
    return xsrt

def expand_grid(x1,x2):
    """
    Creates (len(x1)*len(x2), 2) array of points from two vectors.
    :param x1: vector 1, (nx1, 1)
    :param x2: vector 2, (nx2, 1)
    :return: (nx1*nx2, 2) points
    """
    lc = [(a, b) for a in x1 for b in x2]
    return np.squeeze(np.array(lc))

def reduce_grid(x):
    """
    Undoes expand_grid to take (nx, 2) array to two vectors containing unique values of each col.
    :param x: (nx, 2) points
    :return: x1, x2 each vectors
    """
    x1 = np.sort(np.unique(x[:,0]))
    x2 = np.sort(np.unique(x[:,1]))
    return x1, x2

def mykron(A, B):
    """
    Efficient Kronecker product.
    """
    a1, a2 = A.shape
    b1, b2 = B.shape
    C = np.reshape(np.expand_dims(A, (1, 3)) * np.expand_dims(B, (0, 2)), (a1*b1, a2*b2))
    return C

def comp_eig_D(Ks, Kt, sig2n):
    """
    Computes eigvecs and diagonal D for inversion of kron(Ks, Kt) + sig2n * I
    :param Ks: spatial covariance
    :param Kt: temporal covariance
    :param sig2n: noise variance
    :return: eigvec(Ks), eigvec(Kt), Dvec
    """
    nx = Ks.shape[0]
    nt = Kt.shape[0]
    if np.isscalar(sig2n):
        sig2n_vec = sig2n*np.ones(nx*nt)
    else:
        sig2n_vec = np.repeat(sig2n, nt) #sig2n can be nx dimension
    evals_t, evec_t = np.linalg.eigh(Kt)
    evals_s, evec_s = np.linalg.eigh(Ks)
    #import scipy.linalg
    #evals_t, evec_t = scipy.linalg.eigh(Kt)
    #evals_s, evec_s = scipy.linalg.eigh(Ks)
    Dvec = np.repeat(evals_s, nt) * np.tile(evals_t, nx) + sig2n_vec
    return evec_s, evec_t, Dvec