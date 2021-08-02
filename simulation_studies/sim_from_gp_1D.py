"""
Simulate some data from GP to test methods and evaluate errors.
In this case, the fitted GPCSD model matches the generating model in form.
Reproduces results from Section "Quantifying performance on repeated trials" in the paper.

"""

# %% Imports
import matplotlib.pyplot as plt
import autograd.numpy as np
import scipy.interpolate
import scipy.stats

from kcsd import KCSD1D # version https://github.com/Neuroinflab/kCSD-python/releases/tag/v2.0

from gpcsd.gpcsd1d import GPCSD1D
from gpcsd.covariances import *
from gpcsd.utility_functions import normalize
from gpcsd.predict_csd import predictcsd_trad_1d

np.random.seed(1)

 # %%
# Setup
ntrials = 50
fit_gpcsd = False # Fit GPCSD or use true generating parameters?

a = 0
b = 2300
nt = 60
t = np.linspace(0, nt, nt)[:, None]

nx = 24
x = np.linspace(a, b, nx)[:, None]
deltax = x[1] - x[0]
xshort = x[1:-1]
nz = 100
z = np.linspace(a, b, nz)[:, None]
deltaz = z[1] - z[0]

R_true = 100
ellSE_true = 200
sig2tM_true = 0.7
elltM_true = 5.0
sig2tSE_true = 0.5
elltSE_true = 20.0
sig2n_true = 0.0001

gpcsd_gen = GPCSD1D(np.zeros((nz, nt)), z, t, temporal_cov_list=[GPCSDTemporalCovSE(t), GPCSDTemporalCovMatern(t)])
gpcsd_gen.R['value'] = R_true
gpcsd_gen.sig2n['value'] = sig2n_true
gpcsd_gen.spatial_cov.params['ell']['value'] = ellSE_true
gpcsd_gen.temporal_cov_list[0].params['ell']['value'] = elltSE_true
gpcsd_gen.temporal_cov_list[0].params['sigma2']['value'] = sig2tSE_true
gpcsd_gen.temporal_cov_list[1].params['ell']['value'] = elltM_true
gpcsd_gen.temporal_cov_list[1].params['sigma2']['value'] = sig2tM_true

# %% Generate CSD and sample at interior electrode positions for comparing to tCSD
csd = gpcsd_gen.sample_prior(2*ntrials)
csd_interior_electrodes = np.zeros((nx-2, nt, 2*ntrials))
for trial in range(2*ntrials):
    csdinterp = scipy.interpolate.RectBivariateSpline(z, t, csd[:, :, trial])
    csd_interior_electrodes[:, :, trial] = csdinterp(xshort, t)

# %% Pass through forward model, add white noise
lfp = np.zeros((nx, nt, 2*ntrials))
for trial in range(2*ntrials):
    lfp[:, :, trial] = fwd_model_1d(csd[:, :, trial], z, x, R_true) 
lfp = lfp + np.random.normal(0, np.sqrt(sig2n_true), size=(nx, nt, 2*ntrials))
lfp = normalize(lfp)

# %% Visualize one trial
plt.figure()
plt.subplot(121)
plt.imshow(csd[:, :, 0], vmin=-1, vmax=1, cmap='bwr', aspect='auto')
plt.title('CSD')
plt.xlabel('Time')
plt.ylabel('depth')
plt.colorbar()
plt.subplot(122)
plt.imshow(lfp[:, :, 0], cmap='bwr', aspect='auto')
plt.title('LFP')
plt.xlabel('Time')
plt.colorbar()
plt.tight_layout()
plt.show()

# %% trad CSD on test data
tcsd_pred = predictcsd_trad_1d(lfp[:, :, 50:])[1:-1, :, :]

# %% Fit GPCSD
if fit_gpcsd:
    matern_cov = GPCSDTemporalCovMatern(t, ell_prior=GPCSDInvGammaPrior(), sigma2_prior=GPCSDHalfNormalPrior(sd=5))
    matern_cov.params['ell']['prior'].set_params(1, 20)
    SE_cov = GPCSDTemporalCovSE(t, ell_prior=GPCSDInvGammaPrior(), sigma2_prior=GPCSDHalfNormalPrior(sd=5))
    SE_cov.params['ell']['prior'].set_params(20, 100)
    gpcsd_model = GPCSD1D(lfp[:, :, :50], x, t, a=np.min(z),b=np.max(z))#, temporal_cov_list=[SE_cov, matern_cov])
    gpcsd_model.fit(n_restarts=10)
    gpcsd_model.update_lfp(lfp[:, :, 50:], t)
else:
    gpcsd_model = GPCSD1D(lfp[:, :, 50:], x, t)
    gpcsd_model.R['value'] = R_true
    gpcsd_model.sig2n['value'] = sig2n_true
    gpcsd_model.spatial_cov.params['ell']['value'] = ellSE_true
    gpcsd_model.temporal_cov_list[0].params['ell']['value'] = elltSE_true
    gpcsd_model.temporal_cov_list[0].params['sigma2']['value'] = sig2tSE_true
    gpcsd_model.temporal_cov_list[1].params['ell']['value'] = elltM_true
    gpcsd_model.temporal_cov_list[1].params['sigma2']['value'] = sig2tM_true

print(gpcsd_model)
gpcsd_model.predict(xshort, t)

# %% kCSD estimation
# use first five trials concatenated for estimating parameters (for computational reasons)
kcsd_model = KCSD1D(x, lfp[:, :, :5].reshape((nx, -1)), gdx=deltax/20, h=R_true)
kcsd_model.cross_validate(Rs=np.linspace(100, 1000, 15))

kcsd_R = kcsd_model.R
kcsd_lambda = kcsd_model.lambd
kcsd_values = []
# Predict on test set
for i in range(ntrials):
    kcsd_model_tmp = KCSD1D(x, lfp[:, :, 50+i].squeeze(), gdx=deltax/20, h=R_true,
                            R_init=kcsd_R, lambd=kcsd_lambda)
    kcsd_values_tmp = kcsd_model_tmp.values()
    kcsd_values_interp = scipy.interpolate.RectBivariateSpline(kcsd_model_tmp.estm_x, t, kcsd_values_tmp)
    kcsd_values.append(kcsd_values_interp(x, t))
kcsd_values = np.array(kcsd_values).transpose([1, 2, 0])

# %% Visualize one trial
plt.figure(figsize=(12, 5))
plt.subplot(141)
plt.imshow(normalize(csd_interior_electrodes[:, :, 50]), vmin=-1, vmax=1, cmap='bwr', aspect='auto')
plt.title('Ground truth CSD')
plt.xlabel('Time')
plt.ylabel('Depth')
plt.subplot(142)
plt.imshow(normalize(gpcsd_model.csd_pred[:, :, 0]), vmin=-1, vmax=1, cmap='bwr', aspect='auto')
plt.title('GPCSD')
plt.xlabel('Time')
plt.subplot(143)
plt.imshow(normalize(kcsd_values[1:-1, :, 0]), vmin=-1, vmax=1, cmap='bwr', aspect='auto')
plt.title('kCSD')
plt.xlabel('Time')
plt.subplot(144)
plt.imshow(lfp[:, :, 50], vmin=-1, vmax=1, cmap='bwr', aspect='auto')
plt.title('LFP')
plt.xlabel('Time')
plt.tight_layout()
plt.show()

# %% Compute MSE -- mean squared error across space/time
tcsd_meansqerr = np.nanmean(np.square(normalize(tcsd_pred[1:-1, :, :]) - normalize(csd_interior_electrodes[1:-1, :, 50:])), axis=(0, 1)) 
gpcsd_meansqerr = np.nanmean(np.square(normalize(gpcsd_model.csd_pred[1:-1, :, :]) - normalize(csd_interior_electrodes[1:-1, :, 50:])), axis=(0, 1)) 
kcsd_meansqerr = np.nanmean(np.square(normalize(kcsd_values[2:-2, :, :]) - normalize(csd_interior_electrodes[1:-1, :, 50:])), axis=(0, 1)) 

tcsd_rsq = 1 - np.sum(np.square(normalize(tcsd_pred[1:-1, :, :]) - normalize(csd_interior_electrodes[1:-1, :, 50:])), axis=(0, 1))/np.sum(np.square(normalize(csd_interior_electrodes[1:-1, :, 50:])), axis=(0, 1))
gpcsd_rsq = 1 - np.sum(np.square(normalize(gpcsd_model.csd_pred[1:-1, :, :]) - normalize(csd_interior_electrodes[1:-1, :, 50:])), axis=(0, 1))/np.sum(np.square(normalize(csd_interior_electrodes[1:-1, :, 50:])), axis=(0, 1))
kcsd_rsq = 1 - np.sum(np.square(normalize(kcsd_values[2:-2, :, :]) - normalize(csd_interior_electrodes[1:-1, :, 50:])), axis=(0, 1))/np.sum(np.square(normalize(csd_interior_electrodes[1:-1, :, 50:])), axis=(0, 1))

plt.figure()
plt.boxplot([tcsd_meansqerr, gpcsd_meansqerr, kcsd_meansqerr], labels=['tCSD', 'GPCSD', 'kCSD'])
plt.ylabel('MSE')
plt.show()

plt.figure()
plt.boxplot([tcsd_rsq, gpcsd_rsq, kcsd_rsq], labels=['tCSD', 'GPCSD', 'kCSD'])
plt.ylabel('R^2')
plt.show()

print('tCSD average MSE across trials: %0.3g' % np.mean(tcsd_meansqerr))
print('kCSD average MSE across trials: %0.3g' % np.mean(kcsd_meansqerr))
print('GPCSD average MSE across trials: %0.3g' % np.mean(gpcsd_meansqerr))

print('tCSD median MSE across trials: %0.3g' % np.median(tcsd_meansqerr))
print('kCSD median MSE across trials: %0.3g' % np.median(kcsd_meansqerr))
print('GPCSD median MSE across trials: %0.3g' % np.median(gpcsd_meansqerr))

print('tCSD min MSE across trials: %0.3g' % np.min(tcsd_meansqerr))
print('kCSD min MSE across trials: %0.3g' % np.min(kcsd_meansqerr))
print('GPCSD min MSE across trials: %0.3g' % np.min(gpcsd_meansqerr))

print('tCSD average R^2 across trials: %0.3g' % np.mean(tcsd_rsq))
print('kCSD average R^2 across trials: %0.3g' % np.mean(kcsd_rsq))
print('GPCSD average R^2 across trials: %0.3g' % np.mean(gpcsd_rsq))

# %%
tcsd_spacetime_rmse = np.sqrt(np.nanmean(np.square(normalize(tcsd_pred) - normalize(csd_interior_electrodes[:, :, 50:])), axis=2))
gpcsd_spacetime_rmse = np.sqrt(np.nanmean(np.square(normalize(gpcsd_model.csd_pred) - normalize(csd_interior_electrodes[:, :, 50:])), axis=2))
kcsd_spacetime_rmse = np.sqrt(np.nanmean(np.square(normalize(kcsd_values[1:-1, :, :]) - normalize(csd_interior_electrodes[:, :, 50:])), axis=2))

plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(8, 5))
plt.plot(np.arange(2, 24)*100, np.mean(kcsd_spacetime_rmse, 1), 'r.-', label="kCSD")
plt.plot(np.arange(2, 24)*100, np.mean(gpcsd_spacetime_rmse, 1), 'g.-', label="GPCSD")
plt.legend()
plt.xlabel('Electrode position (depth, microns)')
plt.ylabel('RMSE (averaged across trials/time)')
plt.show()

#%%
from scipy.stats import ttest_rel

tstat1, pval1 = ttest_rel(tcsd_meansqerr, gpcsd_meansqerr)
print('tcsd - gpcsd diff means t %0.2g, pval %0.5g' % (tstat1, pval1))

tstat2, pval2 = ttest_rel(kcsd_meansqerr, gpcsd_meansqerr)
print('kcsd - gpcsd diff means t %0.2g, pval %0.5g' % (tstat2, pval2))

# %%
