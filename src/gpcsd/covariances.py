"""
Spatial and temporal covariance functions for GPCSD.

"""

import quadpy
import autograd.numpy as np

from gpcsd.priors import *
from gpcsd.forward_models import *

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