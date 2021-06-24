import numpy as np
import quadpy

from utility_functions import expand_grid
from forward_models import b_fwd_1d, b_fwd_2d


"""
Covariance computations
"""


def compKt(t, tprime, sig2M, ellM, sig2SE, ellSE):
    """
    Compute Ks (temporal cov, Exp/Matern(1/2) plus SE)
    :param t: vector (nt, 1) times
    :param tprime: vector (ntprime, 1) times
    :param sig2M: Matern variance
    :param ellM: Matern ell
    :param sig2SE: SE variance
    :param ellSE: SE ell
    :return: tuple of cov mats, (Matern, SE)
    """
    dist = t-tprime.T
    se = sig2SE*np.exp(-0.5*np.square(dist)/np.square(ellSE))
    m = sig2M*np.exp(-np.abs(dist)/ellM)
    return (m, se)


def compKs_1d(x, ellSE):
    """
    Compute Ks (CSD spatial correlation), squared exponential unit variance.
    :param x: x vector (nx, 1)
    :param ellSE: GP parameter ell
    :return: cov mat
    """
    return np.exp(-0.5*np.square(x-x.T)/np.square(ellSE))


def compKs_2d(x, ellSE1, ellSE2):
    """
    Compute Ks (CSD spatial correlation), squared exponential unit variance.
    :param x: x vector (nx, 2) of spatial locations
    :param ellSE1: GP parameter ell for 1st spatial dim
    :param ellSE2: GP parameter for 2nd spatial dim
    :return: cov mat (nx, nx)
    """
    x1 = x[:,0][:,None]
    x2 = x[:,1][:,None]
    return np.exp(-0.5*np.square((x1-x1.T)/ellSE1))*np.exp(-0.5*np.square((x2-x2.T)/ellSE2))


def compKphig_1d(x, z, R, ellSE, a, b, ngl=50):
    """
    Compute spatial cross-cov between CSD and LFP (fwd model applied to the x part)
    :param x: vector (nx, 1) of LFP locations
    :param z: vector (nz, 1) of CSD locations
    :param R: fwd model param
    :param ellSE: spatial GP lengthscale
    :param a, b: integration limits
    :param ngl: optional number of gauss-legendre points
    :return: covariance matrix
    """
    def fun_quad(u, x):
        xudiff = x - u # (nx, nu)
        A = np.exp(-0.5*np.square(z-u)/np.square(ellSE)) # (nz, nu)
        B = b_fwd_1d(xudiff,R) # (nx, nu)
        res = np.transpose((np.expand_dims(A.T,2)*np.expand_dims(B.T,1)),[1,2,0]) # (nu, nz, nx) -> (nz, nx, nu)
        return res
    # works with quadpy==0.12.0
    #res = quadpy.line_segment.integrate(lambda z: fun_quad(z, x), [a, b], quadpy.line_segment.GaussLegendre(ngl))
    scheme = quadpy.c1.gauss_legendre(ngl)
    res = scheme.integrate(lambda z: fun_quad(z, x), [a, b])
    return res # (nz, nx)


def compKphig_2d(x, z, R, ellSE1, ellSE2, a1, b1, a2, b2, eps, ngl1=20, ngl2=50):
    """
    Compute spatial cross-cov between CSD and LFP for 2d spatial data.
    :param x: (nx, 2) LFP locations
    :param z: (nz, 2) CSD locations
    :param R: fwd model parameter
    :param ellSE1: spatial lengthscale dim 1
    :param ellSE2: spatial lengthscale dim 2
    :param a1: integral bound 1 dim 1
    :param b1: integral bound 1 dim 2
    :param a2: integral bound 2 dim 1
    :param b2: integral bound 2 dim 2
    :param ngl1: number integral points dim 1
    :param ngl2: number integral points dim 2
    :return: covariance matrix
    """
    gl_x1 = np.linspace(a1, b1, ngl1)
    gl_x2 = np.linspace(a2, b2, ngl2)
    gl_x = expand_grid(gl_x1, gl_x2)
    Ks = np.exp(-0.5*np.square((gl_x[:, 0][:, None]-z[:, 0][None, :])/ellSE1))*np.exp(-0.5*np.square((gl_x[:, 1][:, None]-z[:, 1][None, :])/ellSE2))
    delta1 = gl_x[:, 0][None, :] - x[:, 0][:, None]
    delta2 = gl_x[:, 1][None, :] - x[:, 1][:, None]
    fwd_wts = b_fwd_2d(delta1, delta2, R, eps)  # (nx, ngl)
    A = (gl_x1[1] - gl_x1[0]) * (gl_x2[1] - gl_x2[0]) * fwd_wts
    Kphig = np.dot(A, Ks)
    return Kphig


def compKphi_1d(x, xp, R, ellSE, a, b, ngl=50):
    """
    Compute spatial LFP-LFP cov (fwd model applied to both x, xp)
    :param x: vector (nx, 1) of LFP locations
    :param xp: vector (nz, 1) of LFP locations
    :param R: parameter R
    :param ellSE: GP parameter ell
    :param a: lower bound of integral
    :param b: upper bound of integral
    :param ngl: optional number of gauss-legendre points
    :return: covariance matrix
    """
    def fun_quad(u, v, x, xp):
        xudiff = x - u # (nx, nu)
        xpvdiff = xp - v # (nxp, nu)
        A = np.exp(-0.5*np.square(u-v)/np.square(ellSE)) # (nu, )
        B = b_fwd_1d(xudiff,R) # (nx, nu)
        C = b_fwd_1d(xpvdiff,R) # (nxp, nu)
        res = A*np.transpose((np.expand_dims(B.T,2)*np.expand_dims(C.T,1)),[1,2,0]) # this works for quadrilateral: (nu, nx, nxp)
        return res
    #res = quadpy.quadrilateral.integrate(lambda z: fun_quad(z[0], z[1], x, xp), [[[a, a], [b, a]], [[a, b], [b, b]]],
    #                                     quadpy.quadrilateral.Product(quadpy.line_segment.GaussLegendre(ngl)))
    scheme = quadpy.c2.product(quadpy.c1.gauss_legendre(ngl))
    res = scheme.integrate(lambda z: fun_quad(z[0], z[1], x, xp), [[[a, a], [b, a]], [[a, b], [b, b]]])
    return res # (nx, nxp)


def compKphi_2d(x, R, ellSE1, ellSE2, a1, b1, a2, b2, eps, ngl1=20, ngl2=50):
    """
    Compute spatial LFP-LFP cov for 2d spatial
    :param x: (nx, 2) spatial locations
    :param R: fwd model parameter
    :param ellSE1: spatial lengthscale dim 1
    :param ellSE2: spatial lengthscale dim 2
    :param a1: integral bound 1 dim 1
    :param b1: integral bound 1 dim 2
    :param a2: integral bound 2 dim 1
    :param b2: integral bound 2 dim 2
    :param eps: fwd model singularity param
    :param ngl1: number integral points dim 1
    :param ngl2: number integral points dim 2
    :return: covariance matrix
    """
    gl_x1 = np.linspace(a1, b1, ngl1)
    gl_x2 = np.linspace(a2, b2, ngl2)
    gl_x = expand_grid(gl_x1, gl_x2)
    delta1 = gl_x[:, 0][None, :] - x[:, 0][:, None]
    delta2 = gl_x[:, 1][None, :] - x[:, 1][:, None]
    fwd_wts = b_fwd_2d(delta1, delta2, R, eps)  # (nx, ngl)
    A = (gl_x1[1] - gl_x1[0]) * (gl_x2[1] - gl_x2[0]) * fwd_wts
    Ks = compKs_2d(gl_x, ellSE1, ellSE2)
    Kphi = np.linalg.multi_dot([A, Ks, A.T])
    return Kphi