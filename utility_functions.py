"""
Utility functions for CSD methods.
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt


def plot_im(arr, v1, v2):
    p = plt.imshow(arr, vmin=-np.nanmax(np.abs(arr)), vmax=np.nanmax(np.abs(arr)), cmap='bwr', aspect='auto',
                   extent=[np.min(v1), np.max(v1), np.max(v2), np.min(v2)])
    return p


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
    x1 = np.unique(x[:,0])
    x2 = np.unique(x[:,1])
    return x1, x2


def mykron(A, B):
    """
    Efficient Kronecker product.
    """
    a1, a2 = A.shape
    b1, b2 = B.shape
    C = np.reshape(A[:, np.newaxis, :, np.newaxis] * B[np.newaxis, :, np.newaxis, :], (a1*b1, a2*b2))
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
    evals_t, evec_t = scipy.linalg.eigh(Kt)
    evals_s, evec_s = scipy.linalg.eigh(Ks)
    Dvec = np.repeat(evals_s, nt) * np.tile(evals_t, nx) + sig2n*np.ones(nx*nt)
    return evec_s, evec_t, Dvec