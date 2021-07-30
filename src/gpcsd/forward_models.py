"""
Forward models for both 1D and 2D CSDs.
"""

import autograd.numpy as np
import scipy.integrate


def b_fwd_1d(r, R):
    """
    Computes forward model weight function b.
    :param r: vector of differences between spatial coords
    :param R: cylinder radius parameter R
    :return: values of b_fwd for elements in r (same shape as r)
    """
    b = np.sqrt(np.square(np.divide(r, R)) + 1) - np.sqrt(np.square(np.divide(r, R)))
    return b


def fwd_model_1d(arr, x, z, R, varsigma=1):
    """
    Apply fwd model to arbitrary space-time array arr; predicts at same time points as input.
    (Note: want dense input within integral bounds to get good results!)
    :param arr: matrix (nz, nt)
    :param x: x vector of observations
    :param z: z vector for predictions (can differ from x)
    :param R: forward model parameter
    :param varsigma: scalar conductivity
    :return: matrix (nx, nt)
    """
    nx, nt = arr.shape
    nz = z.shape[0]
    res = np.zeros((nz, nt))
    for t in range(nt):
        arr_tmp = np.squeeze(arr[:, t])[:,None]
        for i in range(nz):
            A = b_fwd_1d(z[i]-x, R)
            res[i, t] = scipy.integrate.trapz(np.squeeze(A * arr_tmp), x=np.squeeze(x), axis=0)
    return R/(2*varsigma) * res


def b_fwd_2d(delta1, delta2, R, eps, w=None):
    """
    Computes b weight function from forward model for parameters R, epsilon (to handle singularity).
    :param delta1: differences between spatial coords dimension 1 (vector)
    :param delta2: differences between spatial coords dimension 2 (vector)
    :param R: parameter R
    :param eps: distance of zero charge between array and CSD to handle singularity
    :return: values of b_fwd for elements in delta1, delta2
    """
    if w is None:
        w = np.array(np.sqrt( np.square(delta1) + np.square(delta2) ))
    wt = np.log(R+eps+np.sqrt((R+eps)**2 + w**2)) - np.log(eps+np.sqrt(eps**2 + w**2))
    return wt


def fwd_model_2d(arr, x1, x2, z, R, eps, varsigma=1):
    """
    Apply fwd model to space-time array arr (must be on grid of spatial locs); predicts at same time points as input.
    (Note: want dense input within integral bounds to get good results!) Output can be requested on non-grid.
    :param arr: matrix (nx1, nx2, nt)
    :param x1: x1 (nx1, 1) vector of observation spatial locations
    :param x2: x2 (nx2, 1) vector of observation spatial locations
    :param z: z (nz, 2) matrix of predicted spatial locations
    :param R: forward model parameter
    :param eps: forward model singularity parameter
    :param varsigma: scalar conductivity
    :return: matrix (nz, nt)
    """
    nt = arr.shape[2]
    nz = z.shape[0]
    res = np.zeros((nz, nt))
    for t in range(nt):
        arr_tmp = np.squeeze(arr[:, :, t]) # (nx1, nx2)
        for i in range(nz):
            deltax1 = z[i,0] - x1   # (nx1, 1)
            deltax2 = z[i,1] - x2.T # (1, nx2)
            wt = b_fwd_2d(deltax1, deltax2, R, eps)
            toint = wt * arr_tmp # (nx1, nx2)
            res[i, t] = scipy.integrate.trapz(scipy.integrate.trapz(toint, x=np.squeeze(x1), axis=0), x=np.squeeze(x2), axis=0)
    return res #/(4*np.pi*varsigma)