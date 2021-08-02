"""
Spatial and temporal covariance functions for GPCSD.

"""

import autograd.numpy as np

from gpcsd.priors import *
from gpcsd.forward_models import *
from gpcsd.utility_functions import expand_grid, reduce_grid

class GPCSD1DSpatialCov:
    def __init__(self, x, a, b, ngl):
        self.x = x
        if a is None:
            a = np.min(x)
        if b is None:
            b = np.max(x)
        self.a = a
        self.b = b
        self.ngl = ngl
        gl_x, gl_w = scipy.special.roots_legendre(ngl)
        # Transform Gauss-Legendre weights to interval
        gl_t = 0.5*(gl_x + 1)*(b - a) + a
        gl_w = 0.5 * (b - a) * gl_w
        self.gl_x = gl_t
        self.gl_w = gl_w

class GPCSD1DSpatialCovSE(GPCSD1DSpatialCov):
    def __init__(self, x, ell_prior=None, a=None, b=None, ngl=100):
        """
        GPCSD1D Spatial covariance (Squared exponential).
        :param x: spatial locations of observed LFP
        :param ell_prior: GPCSDPrior instance or None
        :param a: lower boundary for integration
        :param b: upper boundary for integration
        :param ngl: number of points to use in integration
        """
        GPCSD1DSpatialCov.__init__(self, x, a, b, ngl)
        if ell_prior is None:
            ell_prior = GPCSDInvGammaPrior()
            lb = 1.2 * np.min(np.diff(self.x.squeeze()))
            ub = 0.8 * (np.max(self.x.squeeze()) - np.min(self.x.squeeze()))
            ell_prior.set_params(lb, ub)
        ell = ell_prior.sample()
        ell_min = 0.5 * np.min(np.diff(self.x.squeeze()))
        ell_max = np.max(self.x.squeeze()) - np.min(self.x.squeeze())
        self.params = {'ell':{'value': ell, 'prior':ell_prior, 'min':ell_min, 'max':ell_max}}

    def compute_Ks(self):
        """
        Compute Ks (CSD spatial correlation).
        :return: cov mat
        """
        ell = self.params['ell']['value']
        return np.exp(-0.5*np.square(self.x-self.x.T)/np.square(ell))

    def compKphig_1d(self, z, R):
        """
        Compute spatial cross-cov between CSD and LFP (fwd model applied to the x part).
        :param z: vector (nz, 1) of CSD locations
        :param R: fwd model param value
        :return: cross-covariance matrix
        """
        ell = self.params['ell']['value']
        gl_x_e = np.expand_dims(self.gl_x, 0)
        Ks = np.exp(-0.5*np.square((gl_x_e-z)/ell)).T
        delta = gl_x_e - self.x
        fwd_wts = b_fwd_1d(delta, R) 
        A = np.expand_dims(self.gl_w, 0) * fwd_wts
        res = np.dot(A, Ks)
        return res # (nx, nz)

    def compKphi_1d(self, R, xp=None):
        """
        Compute spatial LFP-LFP cov (fwd model applied to both x, xp)
        :param xp: vector (nz, 1) of LFP locations (if None, use self.x)
        :param R: parameter R
        :return: covariance matrix
        """
        ell = self.params['ell']['value']
        if xp is None:
            xp = self.x
        gl_x_e = np.expand_dims(self.gl_x, 0)
        # x
        delta = gl_x_e - self.x # (nx, ngl)
        fwd_wts = b_fwd_1d(delta, R)  # (nx, ngl)
        A = np.expand_dims(self.gl_w, 0) * fwd_wts
        Ks = np.exp(-0.5*np.square((gl_x_e.T-gl_x_e)/ell)) # (ngl, ngl)
        res_x = np.dot(A, Ks)
        # xp
        delta = gl_x_e - xp # (nx, ngl)
        fwd_wts = b_fwd_1d(delta, R)  # (nx, ngl)
        A = np.expand_dims(self.gl_w, 0) * fwd_wts
        res = np.dot(res_x, A.T)
        return res # (nx, nxp)


class GPCSD2DSpatialCov:
    def __init__(self, x, a1, b1, a2, b2, ngl1, ngl2):
        self.x = x
        self.a1 = a1
        self.b1 = b1
        self.a2 = a2
        self.b2 = b2
        self.ngl1 = ngl1
        self.ngl2 = ngl2
        # Trying non Gauss-Legendre in case...
        #self.gl_x1 = np.linspace(a1, b1, ngl1)
        #self.gl_x2 = np.linspace(a2, b2, ngl2)
        #self.gl_w1 = np.repeat(self.gl_x1[1] - self.gl_x1[0], ngl1)
        #self.gl_w2 = np.repeat(self.gl_x2[1] - self.gl_x2[0], ngl2)
        # Gauss-Legendre
        gl_x1, gl_w1 = scipy.special.roots_legendre(ngl1)
        gl_x2, gl_w2 = scipy.special.roots_legendre(ngl2)
        # Transform Gauss-Legendre weights to interval
        gl_t1 = 0.5*(gl_x1 + 1)*(b1 - a1) + a1
        gl_w1 = 0.5 * (b1 - a1) * gl_w1
        self.gl_x1 = gl_t1
        self.gl_w1 = gl_w1
        gl_t2 = 0.5*(gl_x2 + 1)*(b2 - a2) + a2
        gl_w2 = 0.5 * (b2 - a2) * gl_w2
        self.gl_x2 = gl_t2
        self.gl_w2 = gl_w2
        self.gl_x_grid = expand_grid(self.gl_x1, self.gl_x2)
        self.gl_w_prod = np.prod(expand_grid(self.gl_w1, self.gl_w2), axis=1, keepdims=True)
        self.delta1 = self.gl_x_grid[:, 0][None, :] - self.x[:, 0][:, None] # (nx1*nx2, ngl1*ngl2)
        self.delta2 = self.gl_x_grid[:, 1][None, :] - self.x[:, 1][:, None] # (nx1*nx2, ngl1*ngl2)
        self.gl_x1_sqdist = np.square(np.expand_dims(self.gl_x_grid[:, 0], 1)-np.expand_dims(self.gl_x_grid[:, 0], 1).T)
        self.gl_x2_sqdist = np.square(np.expand_dims(self.gl_x_grid[:, 1], 1)-np.expand_dims(self.gl_x_grid[:, 1], 1).T)
        self.delta_w = np.sqrt(np.square(self.delta1) + np.square(self.delta2))

    def reset_x(self, x_new):
        self.x = x_new
        self.delta1 = self.gl_x_grid[:, 0][None, :] - self.x[:, 0][:, None] # (nx1*nx2, ngl1*ngl2)
        self.delta2 = self.gl_x_grid[:, 1][None, :] - self.x[:, 1][:, None] # (nx1*nx2, ngl1*ngl2)
        self.delta_w = np.sqrt(np.square(self.delta1) + np.square(self.delta2))


class GPCSD2DSpatialCovSE(GPCSD2DSpatialCov):
    def __init__(self, x, ell_prior1=None, ell_prior2=None, a1=None, b1=None, a2=None, b2=None, ngl1=100, ngl2=100):
        """
        GPCSD1D Spatial covariance (Squared exponential).
        :param x: spatial locations of observed LFP in 2D (n_spatial_lfp, 2)
        :param ell_prior1: GPCSDPrior instance or None, prior for lengthscale in first spatial dim
        :param ell_prior2: GPCSDPrior instance or None, prior for lengthscale in second spatial dim
        :param a1: lower boundary for integration in first spatial dim
        :param b1: upper boundary for integration in first spatial dim
        :param a2: lower boundary for integration in second spatial dim
        :param b2: upper boundary for integration in second spatial dim
        :param ngl1: number of points to use in integration in first spatial dim
        :param ngl2: number of points to use in integration in second spatial dim
        """
        GPCSD2DSpatialCov.__init__(self, x, a1, b1, a2, b2, ngl1, ngl2)
        x1, x2 = reduce_grid(x)
        if ell_prior1 is None:
            ell_prior1 = GPCSDInvGammaPrior()
            lb = 2.0 * np.min(np.diff(np.sort(x1).squeeze()))
            ub = 2.0 * (np.max(np.sort(x1).squeeze()) - np.min(np.sort(x1).squeeze()))
            ell_prior1.set_params(lb, ub)
        if ell_prior2 is None:
            ell_prior2 = GPCSDInvGammaPrior()
            lb = 2.0 * np.min(np.diff(np.sort(x2).squeeze()))
            ub = (np.max(np.sort(x2).squeeze()) - np.min(np.sort(x2).squeeze()))
            ell_prior2.set_params(lb, ub)
        # setup ell1 param
        ell1 = ell_prior1.sample()
        ell_min1 = np.min(np.diff(np.sort(x1).squeeze()))
        ell_max1 = 5.0 * np.max(np.sort(x1).squeeze()) - np.min(np.sort(x1).squeeze())
        # setup ell2 param
        ell2 = ell_prior2.sample()
        ell_min2 = np.min(np.diff(np.sort(x2).squeeze()))
        ell_max2 = np.max(np.sort(x2).squeeze()) - np.min(np.sort(x2).squeeze())
        self.params = {'ell1':{'value': ell1, 'prior':ell_prior1, 'min':ell_min1, 'max':ell_max1},
                       'ell2':{'value': ell2, 'prior':ell_prior2, 'min':ell_min2, 'max':ell_max2}}

    def compute_Ks(self):
        """
        Compute Ks (CSD spatial correlation).
        :return: cov mat
        """
        x1 = self.x[:,0][:,None]
        x2 = self.x[:,1][:,None]
        ell1 = self.params['ell1']['value']
        ell2 = self.params['ell2']['value']
        return np.exp(-0.5*np.square((x1-x1.T)/ell1))*np.exp(-0.5*np.square((x2-x2.T)/ell2))

    def compKphig_2d(self, z, R, eps):
        """
        Compute spatial cross-cov between CSD and LFP (fwd model applied to the x part).
        :param z: vector (nz, 2) of 2D CSD locations
        :param R: fwd model param value
        :param eps: spacing in front of array to assume zero charge
        :return: cross-covariance matrix
        """
        ell1 = self.params['ell1']['value']
        ell2 = self.params['ell2']['value']
        Ks = np.exp(-0.5*np.square((self.gl_x_grid[:, 0][:, None]-z[:, 0][None, :])/ell1))*np.exp(-0.5*np.square((self.gl_x_grid[:, 1][:, None]-z[:, 1][None, :])/ell2))
        fwd_wts = b_fwd_2d(None, None, R, eps, self.delta_w) # (nx1*nx2, ngl1*ngl2)
        A = self.gl_w_prod.T * fwd_wts 
        res = np.dot(A, Ks)
        return res

    def compKphi_2d(self, R, eps, xp=None):
        """
        Compute spatial LFP-LFP cov (fwd model applied to both x, xp)
        :param xp: vector (nz, 2) of LFP locations for cross-cov with self.x (if None, use self.x)
        :param R: parameter R
        :param eps: spacing in front of array to assume zero charge
        :return: covariance matrix
        """ 
        ell1 = self.params['ell1']['value']
        ell2 = self.params['ell2']['value']
        #gl_x1_grid = self.gl_x_grid[:, 0][:, None]
        #gl_x2_grid = self.gl_x_grid[:, 1][:, None]
        Ks = np.exp(-0.5*self.gl_x1_sqdist/(ell1**2))*np.exp(-0.5*self.gl_x2_sqdist/(ell2**2)) # (ngl1*ngl2, ngl1*ngl2)
        # x
        #delta1 = gl_x[:, 0][None, :] - self.x[:, 0][:, None] # (nx1*nx2, ngl1*ngl2)
        #delta2 = gl_x[:, 1][None, :] - self.x[:, 1][:, None] # (nx1*nx2, ngl1*ngl2)
        fwd_wts = b_fwd_2d(None, None, R, eps, w=self.delta_w) # (nx1*nx2, ngl1*ngl2)
        A = self.gl_w_prod.T * fwd_wts
        #res_x = np.dot(A, Ks)
        res_x = np.matmul(A, Ks)
        if xp is not None:
            # xp
            delta1 = self.gl_x_grid[:, 0][None, :] - xp[:, 0][:, None] # (ngl1*ngl2, nx1*nx2)
            delta2 = self.gl_x_grid[:, 1][None, :] - xp[:, 1][:, None] # (ngl1*ngl2, nx1*nx2)
            fwd_wts = b_fwd_2d(delta1, delta2, R, eps) # (nx1*nx2, ngl1*ngl2)
            A = self.gl_w_prod.T * fwd_wts
        #res = np.dot(res_x, A.T)
        res = np.matmul(res_x, A.T)
        return res


class GPCSDTemporalCov:
    def __init__(self, t):
        self.t = t


class GPCSDTemporalCovSE(GPCSDTemporalCov):
    def __init__(self, t, ell_prior=None, sigma2_prior=None):
        GPCSDTemporalCov.__init__(self, t)
        if ell_prior is None:
            ell_prior = GPCSDInvGammaPrior()
            lb = 1.2 * np.min(np.diff(self.t.flatten()))
            ub = 0.8 * (np.max(self.t.flatten()) - np.min(self.t.flatten()))
            ell_prior.set_params(lb, ub)
        if sigma2_prior is None:
            sigma2_prior = GPCSDHalfNormalPrior(1.0)
        ell_min = 0.5 * np.min(np.diff(self.t.flatten()))
        ell_max = (np.max(self.t.flatten()) - np.min(self.t.flatten()))
        ell = ell_prior.sample()
        sigma2 = sigma2_prior.sample()
        self.params = {'ell':{'value': ell, 'prior':ell_prior, 'min':ell_min, 'max':ell_max},
                       'sigma2':{'value': sigma2, 'prior':sigma2_prior, 'min':1e-8, 'max':np.inf}}

    def compute_Kt(self, t=None, tprime=None):
        """
        Compute Kt (temporal cov)
        :param tprime: vector (ntprime, 1) times (if None, use t)
        :return: temporal cov mat
        """
        ell = self.params['ell']['value']
        sigma2 = self.params['sigma2']['value']
        if t is None:
            t = self.t
        if tprime is None:
            tprime = self.t
        dist = t-tprime.T
        cov = sigma2 * np.exp(-0.5*np.square(dist)/np.square(ell))
        return cov


class GPCSDTemporalCovMatern(GPCSDTemporalCov):
    def __init__(self, t, ell_prior=None, sigma2_prior=None):
        GPCSDTemporalCov.__init__(self, t)
        if ell_prior is None:
            ell_prior = GPCSDInvGammaPrior()
            lb = 1.2 * np.min(np.diff(self.t.flatten()))
            ub = 0.8 * (np.max(self.t.flatten()) - np.min(self.t.flatten()))
            ell_prior.set_params(lb, ub)
        if sigma2_prior is None:
            sigma2_prior = GPCSDHalfNormalPrior(1.0)
        ell_min = 0.5 * np.min(np.diff(self.t.flatten()))
        ell_max = (np.max(self.t.flatten()) - np.min(self.t.flatten()))
        ell = ell_prior.sample()
        sigma2 = sigma2_prior.sample()
        self.params = {'ell':{'value': ell, 'prior':ell_prior, 'min':ell_min, 'max':ell_max},
                       'sigma2':{'value': sigma2, 'prior':sigma2_prior, 'min':0, 'max':np.inf}}

    def compute_Kt(self, t=None, tprime=None):
        """
        Compute Kt (temporal cov)
        :param tprime: vector (ntprime, 1) times (if None, use t)
        :return: temporal cov mat
        """
        ell = self.params['ell']['value']
        sigma2 = self.params['sigma2']['value']
        if t is None:
            t = self.t
        if tprime is None:
            tprime = self.t
        dist = t-tprime.T
        cov = sigma2 * np.exp(-np.sqrt(np.square(dist))/ell)
        return cov