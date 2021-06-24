import numpy as np

from utility_functions import comp_eig_D
from gp_cov import compKphi_1d, compKphi_2d, compKt #, compute_lfp_mean_quad_vec

# For numerical stability in squared exponential covariance
JITTER = 1e-10

def marg_lik_cov_1d(R, ellSE, sig2tM, elltM, sig2tSE, elltSE, sig2n, x, t, yvec, a, b, ngl=50):
    """
    Define marginal likelihood function for covariance hyperparameters with zero mean.
    :param R: fwd model parameter
    :param ellSE: spatial lengthscale
    :param sig2tM: Matern temporal cov variance
    :param elltM: Matern temporal cov lengthscale
    :param sig2tSE: SE temporal cov variance
    :param elltSE: SE temporal cov lengthscale
    :param sig2n: noise variance
    :param x: spatial locations (nx,1)
    :param t: temporal locations (nt,1)
    :param yvec: vectorized y data (data shape: (nx*nt,ntrials) )
    :param a, b: (limits for integrals)
    :param ngl: number quadrature points
    :return: value of log likelihood, up to normalization constants
    """

    ntrials = yvec.shape[1]
    nx = x.shape[0]
    nt = t.shape[0]

    Ks = compKphi_1d(x, x, R, ellSE, a, b, ngl=ngl) + JITTER
    Kt_res = compKt(t, t, sig2tM, elltM, sig2tSE, elltSE)
    Kt = Kt_res[0] + Kt_res[1] 
    Qs, Qt, Dvec = comp_eig_D(Ks, Kt, sig2n)
    logdet = -0.5*ntrials*np.sum(np.log(Dvec))

    quad = 0
    for trial in range(ntrials):
        alpha = np.reshape(np.linalg.multi_dot([Qs.T,np.reshape(yvec[:,trial],(nx,nt)),Qt]),(nx*nt))
        quad += np.sum(alpha**2/Dvec)
    quad *= -0.5

    return -np.squeeze(logdet + quad)


def marg_lik_cov_2d(R, ellSE1, ellSE2, sig2tM, elltM, sig2tSE, elltSE, sig2n, x, t, yvec, a1, b1, a2, b2, eps, ngl1=20, ngl2=50):
    """
    Marginal likelihood function for covariance parameters, assuming zero mean. Returns negative log likelihood
    :param R: forward model parameter
    :param ellSE1: spatial lengthscale dimension 1
    :param ellSE2: spatial lengthscale dimension 2
    :param sig2tM: Matern temporal cov variance
    :param elltM: Matern temporal cov lengthscale
    :param sig2tSE: SE temporal cov variance
    :param elltSE: SE temporal cov lengthscale
    :param sig2n: noise variance
    :param x: spatial locations (nx, 2)
    :param t: temporal locations (nt, 1)
    :param yvec: vectorized y values
    :param a1: integral boundary for fwd model
    :param b1: integral boundary for fwd model
    :param a2: integral boundary for fwd model
    :param b2: integral boundary for fwd model
    :param eps: fwd model singularity param
    :param ngl1: number of integration points dim 1
    :param ngl2: number of integration points dim 2
    :return: negative log likelihood value
    """

    ntrials = yvec.shape[1]
    nx = x.shape[0]
    nt = t.shape[0]

    Ks = compKphi_2d(x, R, ellSE1, ellSE2, a1, b1, a2, b2, eps, ngl1, ngl2)

    Kt_res = compKt(t, t, sig2tM, elltM, sig2tSE, elltSE)
    Kt = Kt_res[0] + Kt_res[1]
    Qs, Qt, Dvec = comp_eig_D(Ks, Kt, sig2n)
    logdet = -0.5*ntrials*np.sum(np.log(Dvec))

    quad = 0
    for trial in range(ntrials):
        alpha = np.reshape(np.linalg.multi_dot([Qs.T,np.reshape(yvec[:,trial],(nx,nt)),Qt]),(nx*nt))
        quad += np.sum(alpha**2/Dvec)
    quad *= -0.5

    nll = -1*np.squeeze(logdet + quad)

    return nll