"""
Class for 2D GPCSD model, fitting, and prediction.

"""

import autograd.numpy as np
np.seterr(all='ignore')
from autograd import grad
import scipy
from tqdm import tqdm

from gpcsd.priors import GPCSDHalfNormalPrior, GPCSDInvGammaPrior
from gpcsd.covariances import GPCSD2DSpatialCovSE, GPCSDTemporalCovSE, GPCSDTemporalCovMatern
from gpcsd.utility_functions import reduce_grid, comp_eig_D, mykron

JITTER = 1e-7

class GPCSD2D:

    def __init__(self, lfp, x, t, a1=None, b1=None, a2=None, b2=None, ngl1=20, ngl2=60, 
                 spatial_cov=None, temporal_cov_list=None, R_prior=None, sig2n_prior=None, eps=None):
        """
        :param lfp: LFP array, shape (n_spatial_lfp, n_time, n_trials); recommend rescaling to approximately std dev = 1
        :param x: LFP observed spatial locations, shape (n_spatial_lfp, 2), in microns
        :param t: LFP observed time points, shape (n_time, 1), in milliseconds
        :param a1: Edge of range for integration in first spatial direction (defaults to np.min(x[:, 0]))
        :param b1: Edge of range for integration in first spatial direction (defaults to np.max(x[:, 0]))
        :param a2: Edge of range for integration in second spatial direction (defaults to np.min(x[:, 1]))
        :param b2: Edge of range for integration in second spatial direction (defaults to np.max(x[:, 1]))
        :param ngl1: order of Gauss-Legendre integration in first spatial direction (defaults to 100)
        :param ngl2: order of Gauss-Legendre integration in second spatial direction (defaults to 100)
        :param spatial_cov: Instance of GPCSD2DSpatialCovSE
        :param temporal_cov_list: list of instances of temporal covariance objects (GPCSDTemporalCovSE or GPCSDTemporalCovMatern)
        :param R_prior: Instance of a prior for R (defaults to GPCSDInvGammaPrior)
        :param sig2n_prior: Instance of a prior for noise variance (defaults to GPCSDHalfNormalPrior)
        """
        lfp = np.atleast_3d(lfp)
        self.lfp = lfp
        self.x = x
        self.t = t
        if a1 is None:
            a1 = np.min(x[:, 0])
        if b1 is None:
            b1 = np.max(x[:, 0])
        self.a1 = a1
        self.b1 = b1
        if a2 is None:
            a2 = np.min(x[:, 1])
        if b2 is None:
            b2 = np.max(x[:, 1])
        self.a2 = a2
        self.b2 = b2
        self.ngl1 = ngl1
        self.ngl2 = ngl2
        if spatial_cov is None:
            spatial_cov = GPCSD2DSpatialCovSE(self.x, a1=a1, b1=b1, a2=a2, b2=b2, ngl1=ngl1, ngl2=ngl2)
        self.spatial_cov = spatial_cov
        if temporal_cov_list is None:
            temporal_cov_list = [GPCSDTemporalCovSE(t), GPCSDTemporalCovMatern(t)]
        self.temporal_cov_list = temporal_cov_list
        x1, x2 = reduce_grid(x)
        min_delta_x = np.min([np.min(np.diff(x1.squeeze())), np.min(np.diff(x2.squeeze()))])
        max_delta_x = np.max([b1-a1, b2-a2])
        if R_prior is None:
            R_prior = GPCSDInvGammaPrior()
            R_prior.set_params(min_delta_x, 0.5 * max_delta_x)
        self.R = {'value': R_prior.sample(), 'prior':R_prior, 
                  'min':0.5 * min_delta_x, 'max':0.8 * max_delta_x}
        if eps is None:
            eps = 5 * min_delta_x
        self.eps = eps
        if sig2n_prior is None: 
            sig2n_prior = GPCSDHalfNormalPrior(1.0)
            self.sig2n = {'value': sig2n_prior.sample(), 'prior': sig2n_prior, 'min': 1e-8, 'max': 10.0}
        elif isinstance(sig2n_prior, list):
            self.sig2n = {'value': np.array([sp.sample() for sp in sig2n_prior]), 'prior': sig2n_prior, 
                          'min': [1e-8] * len(sig2n_prior), 'max': [10.0] * len(sig2n_prior)}
        else:
            self.sig2n = {'value': sig2n_prior.sample(), 'prior': sig2n_prior, 'min': 1e-8, 'max': 10.0}

    def __str__(self):
        s = "GPCSD1D object\n"
        s += "LFP shape: (%d, %d, %d)\n" % (self.lfp.shape[0], self.lfp.shape[1], self.lfp.shape[2])
        s += "Integration bounds: (%d, %d), (%d, %d)\n" % (self.a1, self.b1, self.a2, self.b2)
        s += "Integration number points: %d, %d\n" % (self.ngl1, self.ngl2)
        s += "R parameter prior: %s\n" % str(self.R['prior'])
        s += "R parameter value %0.4g\n" % self.R['value']
        # TODO handle list if exists
        #s += "sig2n parameter prior: %s\n" % str(self.sig2n['prior'])
        #s += "sig2n parameter value %0.4g\n" % self.sig2n['value']
        s += "Spatial covariance ell prior (dim 1): %s\n" % str(self.spatial_cov.params['ell1']['prior'])
        s += "Spatial covariance ell value (dim 1) %0.4g\n" % self.spatial_cov.params['ell1']['value']
        s += "Spatial covariance ell prior (dim 2): %s\n" % str(self.spatial_cov.params['ell2']['prior'])
        s += "Spatial covariance ell value (dim 2) %0.4g\n" % self.spatial_cov.params['ell2']['value']
        for i in range(len(self.temporal_cov_list)):
            s += "Temporal covariance %d class name: %s\n" % (i+1, type(self.temporal_cov_list[i]).__name__)
            s += "Temporal covariance %d ell prior: %s\n" % (i+1, str(self.temporal_cov_list[i].params['ell']['prior']))
            s += "Temporal covariance %d ell value %0.4g\n" % (i+1, self.temporal_cov_list[i].params['ell']['value'])
            s += "Temporal covariance %d sigma2 prior: %s\n" % (i+1, str(self.temporal_cov_list[i].params['sigma2']['prior']))
            s += "Temporal covariance %d sigma2 value %0.4g\n" % (i+1, self.temporal_cov_list[i].params['sigma2']['value'])
        return s

    def extract_model_params(self):
        params = {}
        params['R'] = self.R['value']
        params['eps'] = self.eps
        params['sig2n'] = self.sig2n['value'] # will be list possibly
        params['spatial_ell1'] = self.spatial_cov.params['ell1']['value']
        params['spatial_ell2'] = self.spatial_cov.params['ell2']['value']
        params['temporal_ell_list'] = [tc.params['ell']['value'] for tc in self.temporal_cov_list]
        params['temporal_sigma2_list'] = [tc.params['sigma2']['value'] for tc in self.temporal_cov_list]
        return params

    def restore_model_params(self, params):
        self.R['value'] = params['R']
        self.eps = params['eps']
        self.sig2n['value'] = params['sig2n'] # will be list possibly
        self.spatial_cov.params['ell1']['value'] = params['spatial_ell1']
        self.spatial_cov.params['ell2']['value'] = params['spatial_ell2']
        if len(self.temporal_cov_list) != len(params['temporal_ell_list']):
            print('different number of temporal covariance functions! stopping.')
            return
        for i, tc in enumerate(self.temporal_cov_list):
            tc.params['ell']['value'] = params['temporal_ell_list'][i]
            tc.params['sigma2']['value'] = params['temporal_sigma2_list'][i]

    def update_lfp(self, new_lfp, t, x=None):
        if x is not None:
            self.x = x
            self.spatial_cov.reset_x(x)
        self.t = t
        for tcov in self.temporal_cov_list:
            tcov.t = t
        self.lfp = np.atleast_3d(new_lfp)

    def loglik(self):
        nx = len(self.x)
        nt = len(self.t)
        ntrials = self.lfp.shape[2]
        Ks = self.spatial_cov.compKphi_2d(R=self.R['value'], eps=self.eps) + JITTER * np.eye(nx)
        Kt = np.zeros((nt, nt))
        for i in range(len(self.temporal_cov_list)):
            Kt = Kt + self.temporal_cov_list[i].compute_Kt()
        Qs, Qt, Dvec = comp_eig_D(Ks, Kt, self.sig2n['value'])
        logdet = -0.5*ntrials*np.sum(np.log(Dvec))
        quad = 0
        for trial in range(ntrials):
            alpha = np.reshape(np.matmul(np.matmul(Qs.T,self.lfp[:, :, trial]),Qt),(nx*nt))
            quad = quad + np.sum(np.square(alpha)/Dvec)
        quad = -0.5 * quad
        return np.squeeze(logdet + quad)

    def fit(self, n_restarts=10, method='L-BFGS-B', fix_R=False, verbose=False, profile=False,
            options={'maxiter':500, 'disp': False, 'gtol':1e-5, 'ftol':1e7 * np.finfo(float).eps}):
        # Store nll values and params over restarts
        nll_values = []
        params = []
        term_msg = []
        # Get bounds from objects
        bounds = []
        bounds.append((np.log(self.R['min']/100), np.log(self.R['max']/100))) # Bounds on R
        bounds.append((np.log(self.spatial_cov.params['ell1']['min']/100), np.log(self.spatial_cov.params['ell1']['max']/100)))
        bounds.append((np.log(self.spatial_cov.params['ell2']['min']/100), np.log(self.spatial_cov.params['ell2']['max']/100)))
        for i in range(len(self.temporal_cov_list)):
            ell_min = self.temporal_cov_list[i].params['ell']['min']
            ell_max = self.temporal_cov_list[i].params['ell']['max']
            sig2_min = self.temporal_cov_list[i].params['sigma2']['min']
            sig2_max = self.temporal_cov_list[i].params['sigma2']['max']
            bounds.append((np.log(ell_min), np.log(ell_max)))
            bounds.append((np.log(sig2_min), np.log(sig2_max)))
        if np.isscalar(self.sig2n['value']):
            bounds.append((np.log(self.sig2n['min']), np.log(self.sig2n['max']))) # bounds on log(sig2n)
        else:
            for i in range(len(self.sig2n['value'])):
                bounds.append((np.log(self.sig2n['min'][i]), np.log(self.sig2n['max'][i])))

        def obj_fun(tparams):
            """
            Objective function (likelihood with priors)
            :param tparams: list of log-transformed parameters
            :return: value of negative log likelihood
            """

            # Get parameters
            if not fix_R:
                self.R['value'] = np.exp(tparams[0]) * 100.0
            self.spatial_cov.params['ell1']['value'] = np.exp(tparams[1]) * 100.0
            self.spatial_cov.params['ell2']['value'] = np.exp(tparams[2]) * 100.0
            n_temp_cov = len(self.temporal_cov_list)
            pind = 3
            for i in range(n_temp_cov):
                self.temporal_cov_list[i].params['ell']['value'] = np.exp(tparams[pind])
                pind = pind + 1
                self.temporal_cov_list[i].params['sigma2']['value'] = np.exp(tparams[pind])
                pind = pind + 1
            if np.isscalar(self.sig2n['value']):
                self.sig2n['value'] = np.exp(tparams[pind])
            else:
                self.sig2n['value'] = np.exp(tparams[pind:])

            # compute log priors
            prior_lpdf = self.R['prior'].lpdf(self.R['value'])
            prior_lpdf = prior_lpdf + self.spatial_cov.params['ell1']['prior'].lpdf(self.spatial_cov.params['ell1']['value'])
            prior_lpdf = prior_lpdf + self.spatial_cov.params['ell2']['prior'].lpdf(self.spatial_cov.params['ell2']['value'])
            for i in range(n_temp_cov):
                prior_lpdf = prior_lpdf + self.temporal_cov_list[i].params['ell']['prior'].lpdf(self.temporal_cov_list[i].params['ell']['value'])
                prior_lpdf = prior_lpdf + self.temporal_cov_list[i].params['sigma2']['prior'].lpdf(self.temporal_cov_list[i].params['sigma2']['value'])
            if np.isscalar(self.sig2n['value']):
                prior_lpdf = prior_lpdf + self.sig2n['prior'].lpdf(self.sig2n['value'])
            else:
                for i in range(len(self.sig2n['prior'])):
                    prior_lpdf = prior_lpdf + self.sig2n['prior'][i].lpdf(self.sig2n['value'][i])

            # Compute likelihood
            try:
                llik = self.loglik()
            except np.linalg.LinAlgError as e:
                llik = -np.inf
                #print(e)
            nll = -1.0 * (llik + prior_lpdf) #* 1e-7
            return nll

        for _ in tqdm(range(n_restarts), desc="Restarts"):
            tparams0 = []
            if fix_R:
                tparams0.append(np.log(self.R['value']) - np.log(100))
            else:
                tparams0.append(np.log(self.R['prior'].sample()) - np.log(100)) # starting R
            tparams0.append(np.log(self.spatial_cov.params['ell1']['prior'].sample()) - np.log(100)) # starting spatial lengthscale
            tparams0.append(np.log(self.spatial_cov.params['ell2']['prior'].sample()) - np.log(100)) # starting spatial lengthscale
            for i in range(len(self.temporal_cov_list)):
                tparams0.append(np.log(self.temporal_cov_list[i].params['ell']['prior'].sample()))
                tparams0.append(np.log(self.temporal_cov_list[i].params['sigma2']['prior'].sample()))
            if np.isscalar(self.sig2n['value']):
                tparams0.append(np.log(self.sig2n['prior'].sample())) # starting sig2n
            else:
                for i in range(len(self.sig2n['value'])):
                    tparams0.append(np.log(self.sig2n['prior'][i].sample())) # starting sig2n
            tparams0 =  np.array(tparams0)

            # Profiling
            if profile and _ == 0:
                import cProfile
                cProfile.runctx('obj_fun(tparams0)', None, locals(), filename='objfunstats')
                g_obj_fun = grad(obj_fun)
                cProfile.runctx('g_obj_fun(tparams0)', None, locals(), filename='gradobjfunstats')
                return

            try:
                optrescov = scipy.optimize.minimize(obj_fun, tparams0, method=method, options=options, bounds=bounds, jac=grad(obj_fun))

                tparams_fit = optrescov.x
                nllcov = optrescov.fun

                nll_values.append(nllcov)
                params.append(tparams_fit)
                term_msg.append(optrescov.message)
            except (ValueError, np.linalg.LinAlgError) as e:
                print(e)
                print('\nrestarting optimization...')

        nll_values = np.array(nll_values)
        if len(nll_values) < 1:
            print('problem with optimization!')
            return
        best_ind = np.argmin(nll_values[np.isfinite(nll_values)])
        params = [params[i] for i in range(len(nll_values)) if np.isfinite(nll_values[i])]
        if verbose:
            print('\nNeg log lik values across different initializations:')
            print(nll_values)
            print('Best index termination message')
            print(term_msg[best_ind])

        if not fix_R:
            self.R['value'] = np.exp(params[best_ind][0]) * 100
        self.spatial_cov.params['ell1']['value'] = np.exp(params[best_ind][1]) * 100
        self.spatial_cov.params['ell2']['value'] = np.exp(params[best_ind][2]) * 100
        pind = 3
        for i in range(len(self.temporal_cov_list)):
            self.temporal_cov_list[i].params['ell']['value'] = np.exp(params[best_ind][pind])
            pind += 1
            self.temporal_cov_list[i].params['sigma2']['value'] = np.exp(params[best_ind][pind])
            pind += 1
        if np.isscalar(self.sig2n['value']):
            self.sig2n['value'] = np.exp(params[best_ind][pind])
        else:
            self.sig2n['value'] = np.exp(params[best_ind][pind:])

    def predict(self, z, t, type="csd"): 
        nx = self.x.shape[0]
        nt = self.t.shape[0]
        ntrials = self.lfp.shape[2]
        nzstar = z.shape[0]
        ntstar = t.shape[0]

        yvec = np.reshape(self.lfp, (nx * nt, ntrials))

        # Compute inverse matrix
        Ks = self.spatial_cov.compKphi_2d(R=self.R['value'], eps=self.eps)
        Kt = np.zeros((nt, nt))
        for i in range(len(self.temporal_cov_list)):
            Kt += self.temporal_cov_list[i].compute_Kt()
        Qs, Qt, Dvec = comp_eig_D(Ks, Kt, self.sig2n['value'])
        ktmp = mykron(Qs, Qt)
        invmat = np.linalg.multi_dot([ktmp, np.diag(1. / Dvec), ktmp.T])
        invy = np.dot(invmat,yvec)

        # Compute cross cov
        csd_list = []
        csd = np.zeros((nzstar, ntstar, ntrials))
        lfp_list = []
        lfp = np.zeros((nzstar, ntstar, ntrials))
        if type == "both" or type == "csd":
            Kphig = self.spatial_cov.compKphig_2d(z=z, R=self.R['value'], eps=self.eps)
        if type == "both" or type == "lfp":
            Kphi = self.spatial_cov.compKphi_2d(R=self.R['value'], eps=self.eps, xp=z)
        for i in range(len(self.temporal_cov_list)):
            Ktstar = self.temporal_cov_list[i].compute_Kt(t)
            if type == "both" or type == "csd":
                csd_tmp = np.reshape(np.dot(mykron(Kphig,Ktstar).T, invy),(nzstar,ntstar,ntrials))
                csd_list.append(csd_tmp)
                csd += csd_tmp
            if type == "both" or type == "lfp":
                lfp_tmp = np.reshape(np.dot(mykron(Kphi,Ktstar).T, invy),(nzstar,ntstar,ntrials))
                lfp_list.append(lfp_tmp)
                lfp += lfp_tmp
        if type == "both" or type == "csd":
            self.csd_pred_list = csd_list
            self.csd_pred = csd
        if type == "both" or type == "lfp":
            self.lfp_pred_list = lfp_list
            self.lfp_pred = lfp
        self.t_pred = t
        self.x_pred = z

    def sample_prior(self, ntrials, type="csd", seed=1):
        # Note: sometimes get LinAlg errors if trying to sample dense LFP; sampling LFP is experimental. 
        np.random.seed(seed)
        nt = self.t.shape[0]
        nx = self.x.shape[0]
        if type == "csd" or type == "both":
            Ks_csd = self.spatial_cov.compute_Ks()
            Ls_csd = np.linalg.cholesky(Ks_csd + JITTER * np.eye(nx))
        if type == "lfp" or type == "both":
            Ks_lfp = self.spatial_cov.compKphi_2d(R=self.R['value'], eps=self.eps)
            Ls_lfp = np.linalg.cholesky(Ks_lfp + JITTER * np.eye(nx))
        Kt = np.zeros((nt, nt))
        for i in range(len(self.temporal_cov_list)):
            Kt += self.temporal_cov_list[i].compute_Kt()
        Lt = np.linalg.cholesky(Kt)
        csd = np.nan * np.zeros((nx, nt, ntrials))
        lfp = np.nan * np.zeros((nx, nt, ntrials))
        rand_samp = np.random.normal(0, 1, (nx, nt, ntrials))
        if type == "csd" or type == "both":
            for trial in range(ntrials):
                csd[:, :, trial] = np.dot(np.dot(Ls_csd, rand_samp[:, :, trial]), Lt.T)
        if type == "lfp" or type == "both":
            for trial in range(ntrials):
                lfp[:, :, trial] = np.dot(np.dot(Ls_lfp, rand_samp[:, :, trial]), Lt.T)
        return csd, lfp

    