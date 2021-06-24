# Simulate some data from GP to test methods and evaluate error on left out set

# %%
import matplotlib.pyplot as plt
import pickle
import numpy as np
import scipy.ndimage.filters as filters
import scipy.interpolate
import scipy.stats
from scipy.stats import invgamma, halfnorm

# TODO use package/module style imports
from forward_models import fwd_model_1d
from predict_csd import predictcsd_trad_1d
from utility_functions import inv_gamma_lim
from gp_cov import compKs_1d, compKt
import gpcsd_1d

np.random.seed(42)

# Normalizes so that SD over whole array is 1, returns rescaled
def normalize(x):
    return x/np.std(x)

# Setup
ntrain = 50
ntest = 50
a = 0
b = 24
nt = 50
t = np.linspace(0, 50, nt)[:, None]
nx = 24
#x = np.linspace(a, b, nx)[:, None]
x = np.arange(b)[:, None]
deltax = x[1] - x[0]
xshort = x[1:-1]
nz = 100
z = np.linspace(a-2, b+2, nz)[:, None]
deltaz = z[1] - z[0]

R_true = 0.5
ellSE_true = 2.0
sig2tM_true = 0.7
elltM_true = 5.0
sig2tSE_true = 0.5
elltSE_true = 20.0
sig2n_true = 0.0001

# %%
# Calculate cov funcs
Ks_csd = compKs_1d(z, ellSE_true)
Kt_res = compKt(t, t, sig2tM_true, elltM_true, sig2tSE_true, elltSE_true)
Kt = Kt_res[0] + Kt_res[1]

# Generate data
print('generating CSD...')
Lt = np.linalg.cholesky(Kt)
Ls = np.linalg.cholesky(Ks_csd + 1e-7*np.eye(nz))

csd_train = np.zeros((nz, nt, ntrain))
csd_test = np.zeros((nz, nt, ntest))
for trial in range(ntrain):
    csd = np.dot(np.dot(Ls, np.random.normal(0, 1, (nz, nt))), Lt.T)
    csd_train[:, :, trial] = np.reshape(csd, (nz, nt))
for trial in range(ntest):
    csd = np.dot(np.dot(Ls, np.random.normal(0, 1, (nz, nt))), Lt.T)
    csd_test[:, :, trial] = np.reshape(csd, (nz, nt))

# Put on original x locations for tCSD error computation
csd_test_lowdim = np.zeros((nx-2, nt, ntest))
for trial in range(ntest):
    csdinterp = scipy.interpolate.RectBivariateSpline(z, t, csd_test[:, :, trial])
    csd_test_lowdim[:, :, trial] = csdinterp(xshort, t)

# Pass through forward model, add white noise
lfp_train = np.zeros((nx, nt, ntrain))
lfp_test = np.zeros((nx, nt, ntest))
for trial in range(ntrain):
    lfp_train[:, :, trial] = fwd_model_1d(csd_train[:, :, trial], z, x, R_true) + np.random.normal(0, np.sqrt(sig2n_true), size=(nx, nt))
for trial in range(ntest):
    lfp_test[:, :, trial] = fwd_model_1d(csd_test[:, :, trial], z, x, R_true) + np.random.normal(0, np.sqrt(sig2n_true), size=(nx, nt))

# Normalize for comparison
lfp_test_norm = normalize(lfp_test)
csd_test_norm = normalize(csd_test)
csd_test_lowdim_norm = normalize(csd_test_lowdim)

# %% trad CSD on test data
tcsd_pred_test = predictcsd_trad_1d(lfp_test)[1:-1, :, :]
#for trial in range(ntest):
#    tcsd_pred_test[:, :, trial] = normalize(tcsd_pred_test[:, :, trial])

tcsd_pred_test_norm = normalize(tcsd_pred_test)
tcsd_meansqerr = np.nanmean(np.square(tcsd_pred_test_norm - csd_test_lowdim_norm), axis=(0, 1)) # mean squared error across space/time


vmlfp = np.amax(np.abs(lfp_test_norm))
vmcsd = np.amax(np.abs(csd_test_norm))
#vmtcsd = np.amax(np.abs(tcsd_pred_test))

f = plt.figure()
plt.subplot(321)
plt.imshow(lfp_test_norm[:, :, 0], vmin=-vmlfp, vmax=vmlfp, aspect='auto', cmap='bwr')
plt.title('LFP true')
plt.colorbar()
plt.subplot(322)
plt.imshow(lfp_test_norm[:, :, 1], vmin=-vmlfp, vmax=vmlfp, aspect='auto', cmap='bwr')
plt.title('LFP true')
plt.colorbar()
plt.subplot(323)
plt.imshow(csd_test_norm[:, :, 0], vmin=-vmcsd, vmax=vmcsd, aspect='auto', cmap='bwr')
plt.title('CSD true')
plt.colorbar()
plt.subplot(324)
plt.imshow(csd_test_norm[:, :, 1], vmin=-vmcsd, vmax=vmcsd, aspect='auto', cmap='bwr')
plt.title('CSD true')
plt.colorbar()
plt.subplot(325)
plt.imshow(tcsd_pred_test_norm[:, :, 0], aspect='auto', vmin=-vmcsd, vmax=vmcsd, cmap='bwr')
plt.title('tCSD pred')
plt.colorbar()
plt.subplot(326)
plt.imshow(tcsd_pred_test_norm[:, :, 1], aspect='auto', vmin=-vmcsd, vmax=vmcsd, cmap='bwr')
plt.title('tCSD pred')
plt.colorbar()
f.set_size_inches(6,15)
plt.show()


# %% GPCSD: train on training, evaluate on test
# Set up priors and objective function
# True values
# R_true = 0.5
# ellSE_true = 2.0
# sig2tM_true = 0.7
# elltM_true = 5.0
# sig2tSE_true = 0.5
# elltSE_true = 20.0
# sig2n_true = 0.0001

aR, bR = inv_gamma_lim(0.1, 10.0)
aSE, bSE = inv_gamma_lim(1., 20.)
atM, btM = inv_gamma_lim(1., 20.)
atSE, btSE = inv_gamma_lim(10., nt)

gpcsd = gpcsd_1d.GPCSD_1D(lfp_test, x, t)
gpcsd.fit([aR, bR, aSE, bSE, atSE, btSE, atM, btM], verbose=True)
gpcsd.predict(xshort, t)

#gpcsd_meansqerr = np.mean(np.square(normalize(gpcsd_pred_test) - normalize(csd_test)), axis=(0, 1)) # mean squared error across space/time
gpcsd_meansqerr = np.mean(np.square(gpcsd.csd - csd_test_lowdim_norm), axis=(0, 1))

print('tCSD MSE mean across trials: %0.3f' % np.mean(tcsd_meansqerr))
print('GPCSD MSE mean across trials: %0.3f' % np.mean(gpcsd_meansqerr))

f = plt.figure()
plt.subplot(321)
plt.imshow(lfp_test_norm[:, :, 0], vmin=-vmlfp, vmax=vmlfp, aspect='auto', cmap='bwr')
plt.title('LFP true')
plt.colorbar()
plt.subplot(322)
plt.imshow(lfp_test_norm[:, :, 1], vmin=-vmlfp, vmax=vmlfp, aspect='auto', cmap='bwr')
plt.title('LFP true')
plt.colorbar()
plt.subplot(323)
plt.imshow(csd_test_norm[:, :, 0], vmin=-vmcsd, vmax=vmcsd, aspect='auto', cmap='bwr')
plt.title('CSD true')
plt.colorbar()
plt.subplot(324)
plt.imshow(csd_test_norm[:, :, 1], vmin=-vmcsd, vmax=vmcsd, aspect='auto', cmap='bwr')
plt.title('CSD true')
plt.colorbar()
plt.subplot(325)
plt.imshow(gpcsd.csd[:, :, 0], vmin=-vmcsd, vmax=vmcsd, aspect='auto', cmap='bwr')
plt.title('gpCSD pred')
plt.colorbar()
plt.subplot(326)
plt.imshow(gpcsd.csd[:, :, 1], vmin=-vmcsd, vmax=vmcsd, aspect='auto', cmap='bwr')
plt.title('gpCSD pred')
plt.colorbar()
f.set_size_inches(6, 15)
plt.show()


# f = plt.figure()
# plt.boxplot([tcsd_meansqerr, kcsd_meansqerr, gpcsd_meansqerr], labels=['tCSD', 'kCSD', 'GPCSD'])
# plt.ylabel('Per-trial MSE')
# plt.savefig('%s/plots/sim_1D_GP_leaveoneouterror_MSEboxplot.png'%savepath)
# plt.close()

# f = plt.figure(figsize=(4, 6))
# plt.boxplot([tcsd_meansqerr-gpcsd_meansqerr, kcsd_meansqerr-gpcsd_meansqerr], labels=['tCSD - GPCSD', 'kCSD - GPCSD'])
# plt.ylabel('Difference in per-trial MSE')
# plt.tight_layout()
# plt.savefig('%s/plots/sim_1D_GP_leaveoneouterror_MSEboxplot.png'%savepath)
# plt.close()

# tstat1, pval1 = scipy.stats.ttest_rel(tcsd_meansqerr, gpcsd_meansqerr)
# print('tcsd - gpcsd diff means pval %0.5g'%pval1)

# tstat2, pval2 = scipy.stats.ttest_rel(kcsd_meansqerr, gpcsd_meansqerr)
# print('kcsd - gpcsd diff means pval %0.5g'%pval2)








# %%
