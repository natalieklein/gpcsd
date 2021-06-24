"""
Class for 1D GPCSD model, fitting, and prediction.

"""

import numpy as np
import scipy
from scipy.stats import invgamma, halfnorm
from tqdm import tqdm

from predict_csd import *
from forward_models import *
from gp_lik import *
from utility_functions import *

# TODO some stuff hard coded, missing parameters (was doing case with fixed Matern, don't like how priors are handled)

class GPCSD_1D:

    def __init__(self, lfp, x, t, a=None, b=None, ngl=100):
        self.lfp = lfp
        self.x = x
        self.t = t
        if a is None:
            a = np.min(x)
        if b is None:
            b = np.max(x)
        self.a = a
        self.b = b
        self.ngl = ngl

    def fit(self, priors, n_restarts=5, method='L-BFGS-B', verbose=False,
            options={'disp': True, 'gtol':1e-5, 'ftol':1e7 * np.finfo(float).eps}):
        nll_values = []
        params = []
        aR, bR, aSE, bSE, atSE, btSE, atM, btM = priors # ugly hardcoded for now

        def obj_fun(tparams, *args):
            """
            Objective function (likelihood with priors)
            :param tparams: (R, ellSE, sig2tSE, elltSE, sig2n)
            :param args: lfpdata
            :return: value of negative log likelihood
            """

            lfp = args[0]
            nx = len(self.x)
            nt = len(self.t)

            R = np.exp(tparams[0])
            ellSE = np.exp(tparams[1])
            sig2tSE = np.exp(tparams[2])
            elltSE = np.exp(tparams[3])
            elltM = np.exp(tparams[4])
            sig2tM = np.exp(tparams[5])
            sig2n = np.exp(tparams[6])

            llik = marg_lik_cov_1d(R, ellSE, sig2tM, elltM, sig2tSE, elltSE, sig2n, self.x, self.t, lfp.reshape((nx*nt, -1)), self.a, self.b, self.ngl)
            Rprior = inv_gamma_lpdf(R, aR, bR)
            ellSEprior = inv_gamma_lpdf(ellSE, aSE, bSE)
            elltMprior = inv_gamma_lpdf(elltM, atM, btM)
            elltSEprior = inv_gamma_lpdf(elltSE, atSE, btSE)
            sig2tMprior = half_normal_lpdf(sig2tM, 2.)
            sig2tSEprior = half_normal_lpdf(sig2tSE, 2.)
            sig2nprior = half_normal_lpdf(sig2n, 0.5)

            nll = - (llik + Rprior + ellSEprior + elltMprior + elltSEprior + sig2tMprior + sig2tSEprior + sig2nprior)
            return nll

        for i in tqdm(range(n_restarts), desc="Restarts"):
            R0 = np.log(invgamma.rvs(aR, scale=bR))
            ellSE0 = np.log(invgamma.rvs(aSE, scale=bSE))
            sig2tSE0 = np.log(halfnorm.rvs(scale=2))
            elltSE0 = np.log(invgamma.rvs(atSE, scale=btSE))
            elltM0 = np.log(invgamma.rvs(atM, scale=btM))
            sig2tM0 = np.log(halfnorm.rvs(scale=2))
            sig2n0 = np.log(halfnorm.rvs(scale=0.5))

            tpcov0 = np.array([R0, ellSE0, sig2tSE0, elltSE0, elltM0, sig2tM0, sig2n0])
            if verbose:
                tqdm.write('\nStarting parameter values')
                tqdm.write('R = %0.2g, ellSE = %0.2g, sig2tSE = %0.2g, elltM = %0.2g, elltSE = %0.2g, sig2tM = %0.2g, sig2n = %0.2g' % \
                    (np.exp(R0), np.exp(ellSE0), np.exp(sig2tSE0), np.exp(elltM0), np.exp(elltSE0), np.exp(sig2tM0), np.exp(sig2n0)))

            try:
                optrescov = scipy.optimize.minimize(obj_fun, tpcov0, args=(self.lfp), method=method, options=options)

                tpcovstar = optrescov.x
                nllcov = optrescov.fun

                R = np.exp(tpcovstar[0])
                ellSE = np.exp(tpcovstar[1])
                sig2tSE = np.exp(tpcovstar[2])
                elltSE = np.exp(tpcovstar[3])
                elltM = np.exp(tpcovstar[4])
                sig2tM = np.exp(tpcovstar[5])
                sig2n = np.exp(tpcovstar[6])

                if verbose:
                    tqdm.write('\nEstimated parameters')
                    tqdm.write('R = %0.2g, ellSE = %0.2g, sig2tSE = %0.2g, elltM = %0.2g, elltSE = %0.2g, sig2tM = %0.2g, sig2n = %0.2g' % \
                        (R, ellSE, sig2tSE, elltM, elltSE, sig2tM, sig2n))                    
                    tqdm.write('Attained nll %0.5g'%nllcov)

                nll_values.append(nllcov)
                params.append(tpcovstar)
            except ValueError as e:
                print(e)

        nll_values = np.array(nll_values)
        if len(nll_values) < 1:
            print('problem with optimization!')
            return
        best_ind = np.argmin(nll_values[np.isfinite(nll_values)])
        params = [params[i] for i in range(len(nll_values)) if np.isfinite(nll_values[i])]

        self.R = np.exp(params[best_ind][0])
        self.ellSE = np.exp(params[best_ind][1])
        self.sig2tSE = np.exp(params[best_ind][2])
        self.elltSE = np.exp(params[best_ind][3])
        self.elltM = np.exp(params[best_ind][4])
        self.sig2tM = np.exp(params[best_ind][5])
        self.sig2n = np.exp(params[best_ind][6])

    def predict(self, z, t): # TODO save each time scale
        def normalize(x):
            return x/np.std(x)
        gpcsd_pred = predictcsd_1d(self.R, self.ellSE, self.sig2tM, self.elltM, self.sig2tSE, self.elltSE, self.sig2n, z, t, self.x, self.t, np.atleast_3d(self.lfp), self.a, self.b, self.ngl)
        gpcsd_pred = gpcsd_pred[0] + gpcsd_pred[1]
        gpcsd_pred = normalize(np.squeeze(gpcsd_pred))
        self.csd = gpcsd_pred