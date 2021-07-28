# %%
import matplotlib.pyplot as plt
import pickle
import numpy as np
import scipy.ndimage.filters as filters
import scipy.interpolate


# TODO use package/module style imports
from predict_csd import *
from forward_models import *
from gp_lik import *
from gp_cov import *
from utility_functions import *

import os
savepath = os.path.abspath('..')
# TODO put data in standard place, or link to data from repo...
dat_path = '/Users/neklein/Documents/Code/gpcsd/neuropixel/'

np.random.seed(42)

# Normalizes so that SD over whole array is 1, returns rescaled
def normalize(x):
    return x/np.std(x)

# %% Setup
ntrain = 50
ntest = 50

# Load data to use as reference for xlim, t, etc
m_id = '405751'
with open('%s/neuropixel_viz_probeA_m%s.pkl'%(dat_path, m_id),'rb') as f:
    d = pickle.load(f)

x_data = d['x'][0:50, :] # subset because reference channel or something at end
t = d['t'][200:220][:,None]

nx = x_data.shape[0]
nt = t.shape[0]

# Integration bounds
x1_data, x2_data = reduce_grid(x_data) # note it's not a grid, so x1/x2 not really meaningful, just using to get min/max/range
print('obs x1 (%0.2g,%0.2g):'%(np.min(x1_data),np.max(x1_data)))
print('delta x1 %0.2g'%(x1_data[1]-x1_data[0]))
print('obs x2 (%0.2g,%0.2g):'%(np.min(x2_data),np.max(x2_data)))
print('delta x2 %0.2g'%(x2_data[1]-x2_data[0]))

x1range = np.max(x1_data) - np.min(x1_data)
x2range = np.max(x2_data) - np.min(x2_data)
a1 = np.amin(x1_data) - 0.1*x1range
b1 = np.amax(x1_data) + 0.1*x1range
a2 = np.amin(x2_data) - 0.1*x2range
b2 = np.amax(x2_data) + 0.1*x2range

# Uniform grid z values for simulated CSD
nz1 = 15
nz2 = 100
z1 = np.linspace(np.min(a1),np.max(b1),nz1)[:,None]
z2 = np.linspace(np.min(a2),np.max(b2),nz2)[:,None]
z = expand_grid(z1, z2)
nz = z.shape[0]

# Grid for LFP instead of real locations
nx1 = 4
nx2 = 20
x1 = np.linspace(np.min(x1_data),np.max(x1_data),nx1)[:,None]
x2 = np.linspace(np.min(x2_data),np.max(x2_data),nx2)[:,None]
x = expand_grid(x1, x2)
nx = x.shape[0]
print('gen x1 (%0.2g,%0.2g):'%(np.min(x1),np.max(x1)))
print('delta x1 %0.2g'%(x1[1]-x1[0]))
print('gen x2 (%0.2g,%0.2g):'%(np.min(x2),np.max(x2)))
print('delta x2 %0.2g'%(x2[1]-x2[0]))

# True GP parameters
R = 75
ellSE1 = 25
ellSE2 = 100
sig2tM = 2**2
elltM = 1.0
sig2tSE = 2**2
elltSE = 15
sig2n = 0.0001


# Calculate cov funcs
Ks_csd = compKs_2d(z, ellSE1, ellSE2)
Kt_res = compKt(t, t, sig2tM, elltM, sig2tSE, elltSE)
Kt = Kt_res[0] + Kt_res[1]

# %% Generate data
print('generating CSD...')
Lt = np.linalg.cholesky(Kt)
Ls = np.linalg.cholesky(Ks_csd + 1e-7*np.eye(nz))
csd_mat_train = np.zeros((nz1,nz2,nt,ntrain))
csd_mat_test = np.zeros((nz1,nz2,nt,ntrain))
for trial in range(ntrain):
    csd = np.dot(np.dot(Ls,np.random.normal(0,1,(nz,nt))),Lt.T) # using identity, much better
    csd_mat_train[:,:,:,trial] = np.reshape(csd, (nz1,nz2,nt))
for trial in range(ntest):
    csd = np.dot(np.dot(Ls,np.random.normal(0,1,(nz,nt))),Lt.T) # using identity, much better
    csd_mat_test[:,:,:,trial] = np.reshape(csd, (nz1,nz2,nt))

# pass through forward model to get LFP at same spatial loc as data
print('starting forward model...')
lfp_train = np.zeros((nx,nt,ntrain))
lfp_test = np.zeros((nx,nt,ntest))
for trial in range(ntrain):
    lfp_train[:,:,trial] = fwd_model_2d(csd_mat_train[:,:,:,trial], z1, z2, x, R, 0.1) + np.sqrt(sig2n)*np.random.normal(0,1,(nx,nt))
lfp_mat_train = np.reshape(lfp_train,(nx1,nx2,nt,ntrain))
for trial in range(ntest):
    lfp_test[:,:,trial] = fwd_model_2d(csd_mat_test[:,:,:,trial], z1, z2, x, R, 0.1) + np.sqrt(sig2n)*np.random.normal(0,1,(nx,nt))
lfp_mat_test = np.reshape(lfp_test,(nx1,nx2,nt,ntest))


# Normalize for comparison
lfp_test_norm = normalize(lfp_mat_test)
csd_test_norm = normalize(csd_mat_test)


# %% GPCSD: train on training, evaluate on test
# Set up priors and objective function
# True values
# R = 75
# ellSE1 = 25
# ellSE2 = 100
# sig2tM = 2**2
# elltM = 1.0
# sig2tSE = 2**2
# elltSE = 15
# sig2n = 0.0001

aR, bR = inv_gamma_lim(50.0, 100.0)
aSE1, bSE1 = inv_gamma_lim(16., 50.)
aSE2, bSE2 = inv_gamma_lim(20., 200.)
atM, btM = inv_gamma_lim(0.5, 10.)
atSE, btSE = inv_gamma_lim(10., 20.)
bounds = [(50., 100.), (10, 60), (10, 250), (1e-3, 5), (0.1, 15), (1e-3, 5), (10, 20), (1e-6, 0.5)]

print('R prior mean %0.2f, mode %0.2f' % (bR / (aR - 1), bR / (aR + 1)))
print('ellSE1 prior mean %0.2f, mode %0.2f' % (bSE1 / (aSE1 - 1), bSE1 / (aSE1 + 1)))
print('ellSE2 prior mean %0.2f, mode %0.2f' % (bSE2 / (aSE2 - 1), bSE2 / (aSE2 + 1)))
print('elltSE prior mean %0.2f, mode %0.2f' % (btSE / (atSE - 1), btSE / (atSE + 1)))
print('elltM prior mean %0.2f, mode %0.2f' % (btM / (atM - 1), btM / (atM + 1)))

# Let's try starting at prior mean...
R0 = bR / (aR - 1)
ellSE10 = bSE1 / (aSE1 - 1)
ellSE20 = bSE2 / (aSE2 - 1)
sig2tM0 = 0.5
elltM0 = btM / (atM - 1)
sig2tSE0 = 0.5
elltSE0 = btSE / (atSE - 1)
sig2n0 = 1e-2  # noise variance
tpcov0 = [R0, ellSE10, ellSE20, sig2tM0, elltM0, sig2tSE0, elltSE0, sig2n0]

y_vec = np.reshape(lfp_test, (nx * nt, ntest))

def obj_fun(tparams):
    """
    Objective function (likelihood with priors)
    :param tparams: (R, ellSE1, ellSE2, sig2tM, elltM, sig2tSE, elltSE, sig2n)
    :return: value of negative log likelihood
    """

    R = tparams[0]
    ellSE1 = tparams[1]
    ellSE2 = tparams[2]
    sig2tM = tparams[3]
    elltM = tparams[4]
    sig2tSE = tparams[5]
    elltSE = tparams[6]
    sig2n = tparams[7]

    if np.any(np.array([R, ellSE1, ellSE2, sig2tM, elltM, sig2tSE, elltSE, sig2n]) <= 1e-8):
        return np.inf

    llik = marg_lik_cov_2d(R, ellSE1, ellSE2, sig2tM, elltM, sig2tSE, elltSE, sig2n, x, t, y_vec, a1, b1, a2, b2,
                        ngl=40)
    Rprior = inv_gamma_lpdf(R, aR, bR)
    ellSE1prior = inv_gamma_lpdf(ellSE1, aSE1, bSE1)
    ellSE2prior = inv_gamma_lpdf(ellSE2, aSE2, bSE2)
    elltMprior = inv_gamma_lpdf(elltM, atM, btM)
    elltSEprior = inv_gamma_lpdf(elltSE, atSE, btSE)
    sig2tMprior = half_normal_lpdf(sig2tM, 2.)
    sig2tSEprior = half_normal_lpdf(sig2tSE, 2.)
    sig2nprior = half_normal_lpdf(sig2n, 0.5)

    nll = - (llik + Rprior + ellSE1prior + ellSE2prior + elltMprior + elltSEprior + sig2tMprior + sig2tSEprior + sig2nprior)

    return nll


# Try gp opt or forest opt

# res = skopt.gp_minimize(obj_fun, dimensions=bounds, x0=tpcov0, verbose=True, n_random_starts=20, random_state=42, noise=1e-8)
# res = skopt.forest_minimize(obj_fun, dimensions=bounds, x0=tpcov0, verbose=True, n_random_starts=20, random_state=42)
#res = skopt.gbrt_minimize(obj_fun, dimensions=bounds, x0=tpcov0, verbose=True,
#                          n_random_starts=20, random_state=42, n_jobs=-1, n_calls=50)

# %%
#tpcovstar = res.x
#nllcov = res.fun

# TODO trying something to see...
tpcovstar = tpcov0
nllcov = 0

R = tpcovstar[0]
ellSE1 = tpcovstar[1]
ellSE2 = tpcovstar[2]
sig2tM = tpcovstar[3]
elltM = tpcovstar[4]
sig2tSE = tpcovstar[5]
elltSE = tpcovstar[6]
sig2n = tpcovstar[7]

print('Start value')
print(
    'R = %0.2g, ellSE1 = %0.2g, ellSE2 = %0.2g, sig2tM = %0.2g, elltM = %0.2g, sig2tSE = %0.2g, elltSE = %0.2g, sig2n = %0.2g' % (
        R0, ellSE10, ellSE20, sig2tM0, elltM0, sig2tSE0, elltSE0, sig2n0))

print('Fitted GP')
print(
    'R = %0.2g, ellSE1 = %0.2g, ellSE2 = %0.2g, sig2tM = %0.2g, elltM = %0.2g, sig2tSE = %0.2g, elltSE = %0.2g, sig2n = %0.2g' % (
        R, ellSE1, ellSE2, sig2tM, elltM, sig2tSE, elltSE, sig2n))

print('Attained nll %0.3g' % nllcov)

pred = predictcsd_2d(R, ellSE1, ellSE2, sig2tM, elltM, sig2tSE, elltSE, sig2n, x, t, x, t, lfp_test, a1, b1, a2, b2, 0.1)

gpcsd_pred_test_norm = normalize(pred[0] + pred[1])

csd_test_lowdim_gpcsd = np.zeros((nx1, nx2, nt, ntest))
for trial in range(ntest):
    for ti in range(nt):
        csdinterp = scipy.interpolate.RectBivariateSpline(z1, z2, csd_mat_test[:, :, ti, trial])
        csd_test_lowdim_gpcsd[:, :, ti, trial] = csdinterp(x1, x2)
csd_test_lowdim_gpcsd_norm = np.reshape(normalize(csd_test_lowdim_gpcsd), (nx, nt, ntest))

gpcsd_meansqerr = np.mean(np.square(gpcsd_pred_test_norm - csd_test_lowdim_gpcsd_norm), axis=(0,1)) # mean squared error across space/time

# %%
f = plt.figure()
trplot = 0
plot_ind = 1
for tind in range(0,nt,5):
    plt.subplot(2,4,plot_ind)
    plt.imshow(np.squeeze(csd_mat_test[:,:,tind,trplot]).T,aspect='auto',
               extent=[np.min(z1),np.max(z1),np.max(z2),np.min(z2)])
    plt.colorbar()
    plt.title('CSD gen trial %d t=%0.2f'%(trplot,t[tind]))
    plot_ind += 1
for tind in range(0,nt,5):
    plt.subplot(2,4,plot_ind)
    plt.imshow(np.reshape(gpcsd_pred_test_norm[:,tind,trplot], (nx1, nx2)).T,
               aspect='auto',extent=[np.min(x1),np.max(x1),np.max(x2),np.min(x2)])
    plt.colorbar()
    plt.title('GPCSD trial %d t=%0.2f'%(trplot,t[tind]))
    plot_ind += 1
f.set_size_inches(20,15)
plt.show()

# %%
f = plt.figure()
plt.boxplot([kcsd_meansqerr, gpcsd_meansqerr], labels=['kCSD', 'GPCSD'])
plt.ylabel('Per-trial MSE')
plt.show()

print('kCSD MSE mean across trials: %0.3f' % np.mean(kcsd_meansqerr))
print('GPCSD MSE mean across trials: %0.3f' % np.mean(gpcsd_meansqerr))

mse_dict = {'gpcsd_mse': gpcsd_meansqerr,  'kcsd_mse':kcsd_meansqerr,
             'kcsd_pred_test':kcsd_pred_test}
with open('%s/pickles/sim_2D_GP_leaveoneouterror_mse.pkl'%savepath, 'wb') as f:
    pickle.dump(mse_dict, f)
# %%
