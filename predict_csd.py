import numpy as np

from utility_functions import mykron, comp_eig_D
from gp_cov import compKphi_1d, compKphi_2d, compKphig_1d, compKphig_2d, compKt


"""
Predictions
"""


def predictcsd_trad_1d(lfp):
    """
    Does traditional CSD estimator with no smoothing (may want to smooth LFP first)
    :param lfp: (nx, nt, ntrial)
    :return: (nx, nt, ntrial) values of CSD
    """
    nx = lfp.shape[0]
    nt = lfp.shape[1]
    ntrials = lfp.shape[2]
    csd = np.zeros((nx, nt, ntrials))
    for x in range(1,nx-1):
        csd[x, :, :] = lfp[x+1, :, :] + lfp[x-1, :, :] - 2*lfp[x, :, :]
    return -csd


def predictcsd_trad_2d(lfp):
    """
    Does traditional CSD estimator with no smoothing, columnwise, assuming data already on
    grid (interpolate to grid if needed). May also want to pre-smooth data.
    :param lfp: (nx1, nx2, nt, ntrial)
    :return: (nx1, nx2, nt, ntrial) values of CSD
    """
    nx1, nx2, nt, ntrials = lfp.shape
    csd = np.nan*np.ones((nx1, nx2, nt, ntrials))
    for row in range(nx1):
        lfprow = lfp[row, :, :, :]
        for col in range(1, nx2-1):
            csd[row, col, :, :] = lfprow[col+1, :, :] + lfprow[col-1, :, :] - 2*lfprow[col, :, :]
    return -csd


def predictcsd_1d(R, ellSE, sig2tM, elltM, sig2tSE, elltSE, sig2n, zstar, tstar, x, t, y, a, b, ngl=50):
    """
    Prediction of mean-zero CSD g from LFP residuals (mean subtracted), returns slow and Matern components.
    :param R: fwd model param
    :param ellSE: spatial lengthscale
    :param sig2tM: Matern temporal cov variance
    :param elltM: Matern temporal cov lengthscale
    :param sig2tSE: SE temporal cov variance
    :param elltSE: SE temporal cov lengthscale
    :param sig2n: noise variance
    :param zstar: vector of z values where to predict CSD (nzstar, 1)
    :param tstar: vector of t values where to predict CSD (ntstar, 1)
    :param x: data x vector (nx, 1)
    :param t: data t vector (nt, 1)
    :param y: data y matrix (nx, nt, ntrials)
    :param a: lower bound for integral
    :param b: upper bound for integral
    :return: tuple (maternCSD, slowCSD) arrays, each (nzstar, ntstar, ntrials)
    """
    nx = x.shape[0]
    nt = t.shape[0]
    ntrials = y.shape[2]
    nzstar = zstar.shape[0]
    ntstar = tstar.shape[0]

    yvec = np.reshape(y, (nx * nt, ntrials))

    # Compute inverse matrix
    Ks = compKphi_1d(x, x, R, ellSE, a, b, ngl=ngl)
    Kt_res = compKt(t, t, sig2tM, elltM, sig2tSE, elltSE)
    Kt = Kt_res[0] + Kt_res[1]
    Qs, Qt, Dvec = comp_eig_D(Ks, Kt, sig2n)
    ktmp = mykron(Qs, Qt)
    invmat = np.linalg.multi_dot([ktmp, np.diag(1. / Dvec), ktmp.T])

    # Compute cross cov
    Ktstar_res = compKt(tstar, tstar, sig2tM, elltM, sig2tSE, elltSE)
    Kphig = compKphig_1d(x, zstar, R, ellSE, a, b, ngl=ngl)

    gstar_m = np.reshape(np.linalg.multi_dot([mykron(Kphig,Ktstar_res[0]), invmat,yvec]),(nzstar,ntstar,ntrials))
    gstar_se = np.reshape(np.linalg.multi_dot([mykron(Kphig, Ktstar_res[1]), invmat, yvec]), (nzstar, ntstar, ntrials))

    return (gstar_m, gstar_se)


def predictcsd_2d(R, ellSE1, ellSE2, sig2tM, elltM, sig2tSE, elltSE, sig2n, zstar, tstar, x, t, y, a1, b1, a2, b2, eps, ngl1=20, ngl2=50):
    """
    Prediction of mean-zero CSD g from LFP, returns slow and Matern components.
    :param R: fwd model parameter
    :param ellSE1: spatial lengthscale dim 1
    :param ellSE2: spatial lengthscale dim 2
    :param sig2tM: Matern temporal cov variance
    :param elltM: Matern tmeporal cov lengthscale
    :param sig2tSE: SE temporal cov variance
    :param elltSE: SE temporal cov lengthscale
    :param sig2n: noise variance
    :param zstar: vector of z values where to predict CSD (nzstar, 2)
    :param tstar: vector of t values where to predict CSD (ntstar, 1)
    :param x: data x vector (nx, 2)
    :param t: data t vector (nt, 1)
    :param y: data y matrix (nx, nt, ntrials)
    :param a1: lower bound for integral
    :param b1: upper bound for integral
    :param a2: lower bound for integral
    :param b2: upper bound for integral
    :param eps: fwd model singularity param
    :return: (maternCSD, slowCSD) arrays, each (nzstar, ntstar, ntrials)
    """
    nx = x.shape[0]
    nt = t.shape[0]
    ntrials = y.shape[2]
    nzstar = zstar.shape[0]
    ntstar = tstar.shape[0]

    # vectorize space/time so trials is the second dimension
    yvec = np.reshape(y, (nx * nt, ntrials))

    # Compute inverse matrix
    Ks = compKphi_2d(x, R, ellSE1, ellSE2, a1, b1, a2, b2, eps, ngl1, ngl2)
    Kt_res = compKt(t, t, sig2tM, elltM, sig2tSE, elltSE)
    Kt = Kt_res[0] + Kt_res[1]
    Qs, Qt, Dvec = comp_eig_D(Ks, Kt, sig2n)
    invDvec = 1./Dvec
    ktmp = mykron(Qs, Qt)

    # Compute cross cov
    Ktstar_res = compKt(tstar, tstar, sig2tM, elltM, sig2tSE, elltSE)
    Kphig = compKphig_2d(x, zstar, R, ellSE1, ellSE2, a1, b1, a2, b2, eps, ngl1, ngl2).T

    gstar_m = np.reshape(np.linalg.multi_dot([mykron(Kphig, Ktstar_res[0]), ktmp, np.diag(invDvec), ktmp.T, yvec]), (nzstar,ntstar,ntrials))
    gstar_se = np.reshape(np.linalg.multi_dot([mykron(Kphig, Ktstar_res[1]), ktmp, np.diag(invDvec), ktmp.T, yvec]), (nzstar, ntstar, ntrials))

    return (gstar_m, gstar_se)
