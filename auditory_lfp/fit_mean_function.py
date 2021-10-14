"""
Estimate evoked CSD, segment into independent components, and estimate per-trial time shifts.
Produces Figures 4 and 5 from the paper.

"""

# %% Imports
import numpy as np
np.seterr(all='ignore')
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, ConnectionStyle
import pickle
import scipy.ndimage.filters as filters
from scipy.ndimage import gaussian_filter
from skimage import segmentation
from skimage import morphology
import scipy.interpolate
import copy
import os
import os.path
import scipy.io
from joblib import Parallel, delayed
import multiprocessing

from gpcsd.gpcsd1d import GPCSD1D
from gpcsd.covariances import *
from gpcsd.utility_functions import normalize, comp_eig_D
from gpcsd.forward_models import fwd_model_1d

from kcsd import KCSD1D # https://github.com/Neuroinflab/kCSD-python/releases/tag/v2.0

# %% Setup
root_path = '/'.join(os.path.abspath(__file__).split('/')[:-1])

np.random.seed(0)

# Probe limits (in microns)
a = 0.0
b = 2300.0
nx = 24
fs = 1000. # sampling rate Hz

# Probe spacing (in 1000 micron units)
x = np.linspace(a, b, nx)[:, None]
# Dense prediction grid
z = np.arange(a, b)[:, None]

# Layer boundaries, defined by authors
layerbounds = 100*(np.array([11, 16])-1)
layerlabs = ['Superficial', 'Medium', 'Deep']

# %% Get GPCSD and KCSD evoked response estimates
time = np.loadtxt('%s/data/time.txt' % root_path) * 1000. # Time in seconds, convert to ms
trial_pred_time_idx = np.logical_and(time >= 0, time <= 150)
trial_pred_t = time[trial_pred_time_idx][:, None] 

if os.path.isfile('%s/results/gpcsd_lfp_trials.pkl' % root_path):
    with open('%s/results/gpcsd_lfp_trials.pkl' % root_path, 'rb') as f:
        lfp_trials = pickle.load(f)
else:
    lfp_trials = {}
evoked = {}
cov = {}
for probe_name in ['medial', 'lateral']:
    if os.path.isfile('%s/results/gpcsd_evoked_%s.pkl' % (root_path, probe_name)):
        with open('%s/results/gpcsd_evoked_%s.pkl' % (root_path, probe_name), 'rb') as f:
            evoked[probe_name] = pickle.load(f)
        with open('%s/results/gpcsd_cov_%s.pkl' % (root_path, probe_name), 'rb') as f:
            cov[probe_name] = pickle.load(f)
    else:
        evoked_probe = {}
        cov_probe = {}
        # Load LFP data
        lfp = []
        for i in range(24):
            lfp.append(np.loadtxt('%s/data/%s_electrode%d.txt' % (root_path, probe_name, i+1)))
        lfp = np.array(lfp) # (n_elec, time, trials)
        lfp /= 100.0 # Rescale 
        lfp_trial_pred = lfp[:, trial_pred_time_idx, :]
        lfp_trial_pred_evoked = np.mean(lfp_trial_pred, 2)
        ntrials = lfp.shape[2]
        evoked_probe['lfp'] = lfp_trial_pred_evoked
        lfp_trials[probe_name] = lfp_trial_pred

        # Load GPCSD model
        spatial_cov = GPCSD1DSpatialCovSE(x, a=-200.0, b=2600.0)
        matern_cov = GPCSDTemporalCovMatern(trial_pred_t)
        matern_cov.params['ell']['prior'].set_params(1., 20.)
        SE_cov = GPCSDTemporalCovSE(trial_pred_t)
        SE_cov.params['ell']['prior'].set_params(30., 100.)
        sig2n_prior = [GPCSDHalfNormalPrior(0.1) for i in range(nx)]
        gpcsd_model = GPCSD1D(lfp_trial_pred, x, trial_pred_t, 
                            sig2n_prior=sig2n_prior,
                            spatial_cov=spatial_cov, 
                            temporal_cov_list=[SE_cov, matern_cov], a=-200.0, b=2600.0)
        with open('%s/results/gpcsd_model_%s.pkl' % (root_path, probe_name), 'rb') as f:
            params = pickle.load(f)
        gpcsd_model.restore_model_params(params)
        # Store covariance information for later
        Kt = gpcsd_model.temporal_cov_list[0].compute_Kt() + gpcsd_model.temporal_cov_list[1].compute_Kt()
        Ks = gpcsd_model.spatial_cov.compKphi_1d(gpcsd_model.R['value']) + 1e-8 * np.eye(nx)
        Qs, Qt, Dvec = comp_eig_D(Ks, Kt, gpcsd_model.sig2n['value'])
        cov_probe['Qs'] = Qs
        cov_probe['Qt'] = Qt
        cov_probe['Dvec'] = Dvec

        # Compute empirical mean of CSD estimated by GPCSD as evoked response
        gpcsd_model.predict(z, trial_pred_t)
        evoked_probe['gpcsd'] = np.mean(gpcsd_model.csd_pred, 2)

        # kCSD estimation of evoked response for comparison
        kcsd_evoked_model = KCSD1D(x, lfp_trial_pred_evoked, gdx=1., h=gpcsd_model.R['value'])
        kcsd_evoked_model.cross_validate(Rs=np.linspace(100, 800, 15), lambdas=np.logspace(1,-15,25,base=10.))
        evoked_probe['kcsd'] = kcsd_evoked_model.values()

        with open('%s/results/gpcsd_evoked_%s.pkl' % (root_path, probe_name), 'wb') as f:
            pickle.dump(evoked_probe, f)
        evoked[probe_name] = evoked_probe

        with open('%s/results/gpcsd_cov_%s.pkl' % (root_path, probe_name), 'wb') as f:
            pickle.dump(cov_probe, f)
        cov[probe_name] = cov_probe

    # Supplemental figure: comparing kCSD on evoked and GPCSD evoked
    plt.figure(figsize=(14, 6))
    plt.subplot(131)
    plt.imshow(normalize(evoked[probe_name]['lfp']), vmin=-1, vmax=1, aspect='auto',cmap='bwr')
    plt.title('%s evoked LFP' % probe_name)
    plt.subplot(132)
    plt.imshow(normalize(evoked[probe_name]['gpcsd']), vmin=-1, vmax=1, aspect='auto',cmap='bwr')
    plt.title('GPCSD evoked')
    plt.subplot(133)
    plt.imshow(normalize(evoked[probe_name]['kcsd']), cmap='bwr', vmin=-1, vmax=1, aspect='auto')
    plt.title('kCSD on evoked')

with open('%s/results/gpcsd_lfp_trials.pkl' % root_path, 'wb') as f:
    pickle.dump(lfp_trials, f)

# %% Load evoked MUA, compute trial values relative to baseline
mua_lateral_evoked = np.loadtxt('%s/data/lateral_evoked_mua.txt' % root_path)
mua_base = np.mean(mua_lateral_evoked[:, :100], 1, keepdims=True)
mua_lateral_evoked = mua_lateral_evoked - mua_base
mua_lateral_evoked = mua_lateral_evoked[:, 100:250]/np.max(mua_lateral_evoked[:, 100:250])

mua_medial_evoked = np.loadtxt('%s/data/medial_evoked_mua.txt' % root_path)
mua_base = np.mean(mua_medial_evoked[:, :100], 1, keepdims=True)
mua_medial_evoked = mua_medial_evoked - mua_base
mua_medial_evoked = mua_medial_evoked[:, 100:250]/np.max(mua_medial_evoked[:, 100:250])

# %% Image segmentation on GPCSD evoked response
csd_segments = {}
mu_lfp = {}
for probe_name in ['medial', 'lateral']:
    # Apply a small amount of smoothing to avoid spurious local maxima
    filt_evoked = gaussian_filter(normalize(evoked[probe_name]['gpcsd']), (1, 1))
    abs_csd = np.abs(filt_evoked)
    # Find evoked local maxima
    local_max = (abs_csd == filters.maximum_filter(abs_csd, size=(10, 10)))
    # remove ones before/after time period of interest (10ms to 140ms)
    local_max[:, :10] = False
    local_max[:, 140:] = False
    local_max_ind = np.where(local_max)
    # Remove very small (absolute value) extrema
    markers = morphology.label(local_max)
    markers_pos = markers.copy()
    markers_pos[filt_evoked < 1e-4] = -1
    markers_neg = markers.copy()
    markers_neg[filt_evoked > -1e-4] = -1
    # Use watershed algorithm to segment positive and negative evoked components
    cparam = 1e-3
    labels_pos_ws = segmentation.watershed(-filt_evoked, markers_pos, compactness=cparam)
    labels_neg_ws = segmentation.watershed(filt_evoked, markers_neg, compactness=cparam)
    # Merge positive and negative into one set of labels
    labels_all = np.zeros_like(labels_pos_ws)
    labelnames_pos = np.unique(labels_pos_ws)
    labelnames_neg = np.unique(labels_neg_ws)
    nlabels_pos = labelnames_pos.shape[0]
    nlabels_neg = labelnames_neg.shape[0]
    counter = 1
    for i in range(1, nlabels_pos):
        labels_all[labels_pos_ws == labelnames_pos[i]] = counter
        counter += 1
    for i in range(1, nlabels_neg):
        labels_all[labels_neg_ws == labelnames_neg[i]] = counter
        counter += 1
    labelnames_all = np.unique(labels_all)

    csd_segments[probe_name] = labels_all

    print('%s probe, found %d regions'% (probe_name, len(labelnames_all)-1))

    # Load GPCSD model params to get R value
    with open('%s/results/gpcsd_model_%s.pkl' % (root_path, probe_name), 'rb') as f:
        params = pickle.load(f)

    # Compute fwd model on each cluster
    mu_lfp_tmp = np.zeros((x.shape[0], trial_pred_t.shape[0], len(labelnames_all)))
    for ci, cname in enumerate(labelnames_all):
        csd_clust_tmp = np.copy(evoked[probe_name]['gpcsd'])
        csd_clust_tmp[labels_all != cname] = 0
        mu_lfp_tmp[:, :, ci] = fwd_model_1d(csd_clust_tmp, z, x, params['R']) * 2/params['R']

    mu_lfp[probe_name] = mu_lfp_tmp

    plt.figure()
    plt.subplot(121)
    plt.imshow(np.sum(mu_lfp[probe_name], 2), aspect='auto', cmap='bwr')
    plt.title('%s fwd modeled segments' % probe_name)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(evoked[probe_name]['lfp'], aspect='auto', cmap='bwr')
    plt.colorbar()
    plt.title('LFP evoked')

# %% Figure 4 - show segments and evoked along with MUA

plt.rcParams.update({'font.size': 16})
cmap = copy.copy(matplotlib.cm.get_cmap("gist_ncar"))
cmap.set_bad(color='white')

plt.figure(figsize=(12, 10))
plt.subplot(231) # MUA 
for ci in range(24):
    plt.fill_between(np.arange(150), -mua_lateral_evoked[ci, :] * 75 + (ci+1)*100, (ci+1)*100, color='black')
plt.title('Lateral evoked MUA')
plt.ylabel('Electrode depth (microns)')
plt.xlabel('Time (ms)')
plt.gca().invert_yaxis()
ax=plt.subplot(232)
plt.imshow(normalize(evoked['lateral']['gpcsd']), aspect='auto', vmin=-1, vmax=1, cmap='bwr')
plt.xlabel('Time (ms)')
plt.title('Lateral evoked CSD')
plt.subplot(233,sharey=ax)
segments_plot = np.copy(csd_segments['lateral']).astype(float)
segments_plot[segments_plot < 1] = np.nan
segment_boundaries = np.where(segmentation.find_boundaries(csd_segments['lateral']))
plt.imshow(segments_plot, aspect='auto', cmap=cmap, alpha=0.7)
plt.plot(segment_boundaries[1], segment_boundaries[0], 'k.', markersize=1.0)
plt.xlabel('Time (ms)')
plt.title('Lateral components')
plt.subplot(234) # MUA medial
for ci in range(24):
    plt.fill_between(np.arange(150), -mua_medial_evoked[ci, :] * 75 + (ci+1)*100, (ci+1)*100, color='black')
plt.title('Medial evoked MUA')
plt.ylabel('Electrode depth (microns)')
plt.xlabel('Time (ms)')
plt.gca().invert_yaxis()
ax=plt.subplot(235)
plt.imshow(normalize(evoked['medial']['gpcsd']), aspect='auto', vmin=-1, vmax=1, cmap='bwr')
plt.xlabel('Time (ms)')
plt.title('Medial evoked CSD')
plt.subplot(236,sharey=ax)
segments_plot = np.copy(csd_segments['medial']).astype(float)
segments_plot[segments_plot < 1] = np.nan
segment_boundaries = np.where(segmentation.find_boundaries(csd_segments['medial']))
plt.imshow(segments_plot, aspect='auto', cmap=cmap, alpha=0.7)
plt.plot(segment_boundaries[1], segment_boundaries[0], 'k.', markersize=1.0)
plt.xlabel('Time (ms)')
plt.title('Medial components')
plt.tight_layout()

# %% Test time shifting and visualize
probe_name = 'lateral'
n_seg = np.max(csd_segments[probe_name])
tau = -10 * np.ones(n_seg)
mu_f = {}
for i in range(1, n_seg+1):
    mu_f[i] = scipy.interpolate.interp1d(trial_pred_t.squeeze(), mu_lfp[probe_name][:, :, i], axis=1, fill_value="extrapolate")

mu_new = np.copy(mu_lfp[probe_name][:, :, 0]) # background
for i in range(1, n_seg+1):
    mu_new += mu_f[i](trial_pred_t.squeeze() + tau[i-1])

plt.figure(figsize=(20, 6))
plt.subplot(151)
plt.imshow(np.sum(mu_lfp[probe_name], 2), aspect='auto', cmap='bwr')
plt.title('sum mu_lfp_scl fwd')
plt.colorbar()
plt.subplot(152)
plt.imshow(np.mean(lfp_trials[probe_name], 2), aspect='auto', cmap='bwr')
plt.title('average evoked LFP')
plt.colorbar()
plt.subplot(153)
plt.imshow(mu_new, aspect='auto', cmap='bwr')
plt.title('with time shifts')
plt.colorbar()
plt.subplot(154)
plt.imshow(lfp_trials[probe_name][:, :, 0], aspect='auto', cmap='bwr')
plt.colorbar()
plt.title('single trial lfp')
plt.subplot(155)
plt.imshow(lfp_trials[probe_name][:, :, 0] - np.sum(mu_lfp[probe_name], 2), aspect='auto', cmap='bwr')
plt.title('lfp resid')
plt.colorbar()

# %% Per-component shifts -- may take a while, depending on number of processors (optimization done in parallel for each trial)
for probe_name in ['lateral', 'medial']:
    if not os.path.isfile('%s/results/per_trial_shifts_%s.txt' % (root_path, probe_name)):
        n_seg = np.max(csd_segments[probe_name])
        tau0 = np.zeros(n_seg)
        # prior/regularizer on shifts
        mutau = 0.0
        sigtau = 10.0

        mu_f = {}
        for i in range(1, n_seg+1):
            mu_f[i] = scipy.interpolate.interp1d(trial_pred_t.squeeze(), mu_lfp[probe_name][:, :, i], axis=1, fill_value="extrapolate")

        # Optimization
        def obj_fun(tau, *args):
            lfp_trial = args[0]
            mu_new = np.copy(mu_lfp[probe_name][:, :, 0]) # background
            for i in range(1, n_seg+1):
                mu_new += mu_f[i](trial_pred_t.squeeze() + tau[i-1])
            resid = lfp_trial - mu_new
            alpha = np.reshape(np.linalg.multi_dot([cov[probe_name]['Qs'].T,resid,cov[probe_name]['Qt']]), (nx*len(trial_pred_t)))
            quad = -0.5*np.sum(alpha**2/cov[probe_name]['Dvec'])
            nll = -1 * np.squeeze(quad)
            nll += -np.sum(-0.5*np.square((tau - mutau)/sigtau))
            return nll
            
        def minfunc(ti):
            optres = scipy.optimize.minimize(obj_fun, tau0, method='l-bfgs-b', args=(lfp_trials[probe_name][:, :, ti]))
            return optres.x, optres.success, optres.message

        num_cores = multiprocessing.cpu_count()
        optres = Parallel(n_jobs=num_cores, verbose=50)(delayed(minfunc)(i) for i in range(ntrials))

        tau_hat, suc, msg = zip(*optres)
        if not np.all(suc):
            print('warning: not all optimizations succeeded')
            print('succeeded percent: %0.2f' % np.mean(suc))
        
        tau_hat = np.array(tau_hat) # (ntrials, n_seg)
        np.savetxt('%s/results/per_trial_shifts_%s.txt' % (root_path, probe_name), tau_hat)

# %% Load precomputed shifts for further analysis/visualization
tau = {}
tau['lateral'] = np.loadtxt('%s/results/per_trial_shifts_lateral.txt' % root_path)
tau['medial'] = np.loadtxt('%s/results/per_trial_shifts_medial.txt' % root_path)

print('lateral time shift m=%0.2g, sd=%0.2g' % (np.mean(tau['lateral']), np.std(tau['lateral'])))
print('medial time shift m=%0.2g, sd=%0.2g' % (np.mean(tau['medial']), np.std(tau['medial'])))

# %% Get center of mass of each segment, (spatial, temporal) points
cmcoords = {}
shifted_times = {}
for probe_name in ['lateral', 'medial']:
    cmcoords_tmp = scipy.ndimage.center_of_mass(normalize(evoked[probe_name]['gpcsd']), 
                                                csd_segments[probe_name], 
                                                index=range(1,np.max(csd_segments[probe_name]+1)))
    cmcoords[probe_name] = np.array(cmcoords_tmp)
    cmcoords_times = cmcoords[probe_name][:, 1]
    shifted_times[probe_name] = cmcoords_times[None, :] - tau[probe_name]

    # quick visual check
    plt.figure()
    plt.imshow(csd_segments[probe_name], aspect='auto', cmap=cmap)
    plt.plot(cmcoords[probe_name][:, 1], cmcoords[probe_name][:, 0], 'k.')

# %% Compute KDE of estimated times for each component
# Compute kde for times
tnew = np.linspace(0, 150, 1000)
times_kde = {probe_name: np.zeros((tnew.shape[0], tau[probe_name].shape[1])) for probe_name in ['lateral', 'medial']}
for probe_name in ['lateral', 'medial']:
    for i in range(tau[probe_name].shape[1]):
        k = scipy.stats.gaussian_kde(shifted_times[probe_name][:, i])
        res = k(tnew)
        res = res/np.max(res)
        res[res < 1e-2] = np.nan
        times_kde[probe_name][:, i] = res

# %% Compute correlation between per-trial time shifts per component
tau1 = shifted_times['lateral'] - np.mean(shifted_times['lateral'], 0)
tau1 /= np.std(tau1, 0)
tau2 = shifted_times['medial'] - np.mean(shifted_times['medial'], 0)
tau2 /= np.std(tau2, 0)
tau_all = np.hstack([tau1, tau2])

ntrials = tau_all.shape[0]
taucorr = np.corrcoef(tau_all, rowvar=False)
# use Fisher's z to get pvalues
taucorr_z = np.arctanh(taucorr)
se = 1/np.sqrt(ntrials-3)
pval = 2*(1-scipy.stats.norm.cdf(np.abs(taucorr_z), loc=0, scale=se))

ntests = (taucorr.shape[0])*(taucorr.shape[0]-1)/2
alpha_uncorr = 0.0001
alpha = alpha_uncorr/ntests # Bonferroni correction
taucorr_sig = taucorr.copy()
taucorr_sig[pval > alpha] = np.nan

# Get indices of segments with significant connections
n_lat = tau['lateral'].shape[1]
n_med = tau['medial'].shape[1]
within_inds = {}
within_inds['lateral'] = np.logical_not(np.isnan(taucorr_sig[:n_lat, :n_lat]))
within_inds['medial'] = np.logical_not(np.isnan(taucorr_sig[n_lat:, n_lat:]))
between_inds = np.logical_not(np.isnan(taucorr_sig[n_lat:, :n_lat]))

# %% Graph (Figure 5a)
probe_names = ['Lateral', 'Medial']
ms1 = dict(color='white', marker='o', markersize=10, linewidth=0)
ms2 = dict(color='black', marker='o', markersize=7, linewidth=0)
f = plt.figure(figsize=(12, 5))
ax_list = []

for i, probe_name in enumerate(['lateral', 'medial']):
    labels_tmp = csd_segments[probe_name].copy().astype(float)
    labels_tmp[labels_tmp == 0] = np.nan

    ax = plt.subplot(1,3,i+1)
    ax_list += [ax]
    plt.imshow(labels_tmp, aspect='auto', cmap=cmap, alpha=0.5)
    segment_boundaries = np.where(segmentation.find_boundaries(csd_segments[probe_name]))
    plt.plot(segment_boundaries[1], segment_boundaries[0], 'k.', markersize=0.5)
    if i == 0:
        plt.ylabel('Depth (microns)')
    if i == 1:
        labels = [item.get_text() for item in ax.get_yticklabels()]
        empty_string_labels = ['        '] * len(labels)
        ax.set_yticklabels(empty_string_labels)
    plt.plot(cmcoords[probe_name][:, 1], cmcoords[probe_name][:, 0], **ms1)
    plt.plot(cmcoords[probe_name][:, 1], cmcoords[probe_name][:, 0], **ms2)
    plt.xlabel('Time (ms)')
    plt.title(probe_names[i])
    # within probe connections
    for ci in range(len(cmcoords[probe_name])):
        for cj in range(ci, len(cmcoords[probe_name])):
            if within_inds[probe_name][ci, cj]:
                xi_tmp = cmcoords[probe_name][ci, 1]
                xj_tmp = cmcoords[probe_name][cj, 1]
                yi_tmp = cmcoords[probe_name][ci, 0]
                yj_tmp = cmcoords[probe_name][cj, 0]
                alpha_tmp = np.min([1.0, 5*np.abs(taucorr_sig[ci, cj])])
                lw_tmp = np.min([3.0, 30*np.abs(taucorr_sig[ci, cj])])
                plt.plot([xi_tmp, xj_tmp], [yi_tmp, yj_tmp], color='black', linewidth=lw_tmp, alpha=alpha_tmp)
# between probe connections
for ci in range(n_lat):
    for cj in range(n_med):
        if between_inds[ci, cj]:
            c1 = (cmcoords['lateral'][ci, 1], cmcoords['lateral'][ci, 0])
            c2 = (cmcoords['medial'][cj, 1], cmcoords['medial'][cj, 0])
            alpha_tmp = np.min([1.0, 5*np.abs(taucorr_sig[ci, cj+n_lat])])
            lw_tmp = np.min([3.0, 30*np.abs(taucorr_sig[ci, cj+n_lat])])
            con = ConnectionPatch(xyA=c2, xyB=c1, coordsA="data", coordsB="data",
                                  axesA=ax_list[1], axesB=ax_list[0], color="black", linewidth=lw_tmp, 
                                  alpha=alpha_tmp, connectionstyle=ConnectionStyle.Arc3(rad=0.08))
            ax_list[1].add_artist(con)
ax = plt.subplot(133)
for i, c in enumerate(cmcoords['lateral']):
    if i == 0:
        plt.plot(tnew, -100 * times_kde['lateral'][:, i] + c[0], 'b', label='L')
    else:
        plt.plot(tnew, -100 * times_kde['lateral'][:, i] + c[0], 'b')
for i, c in enumerate(cmcoords['medial']):
    if i == 0:
        plt.plot(tnew, -100 * times_kde['medial'][:, i] + c[0], 'r', label='M')
    else:
        plt.plot(tnew, -100 * times_kde['medial'][:, i] + c[0], 'r')
plt.axhline(layerbounds[0], linestyle='dashed', color='grey')
plt.axhline(layerbounds[1], linestyle='dashed', color='grey')
plt.xlabel('Time (ms)')
plt.gca().invert_yaxis()
labels = [item.get_text() for item in ax.get_yticklabels()]
empty_string_labels = ['        '] * len(labels)
ax.set_yticklabels(empty_string_labels)
plt.legend(loc='lower right', handlelength=0.2)
f.text(0.05, 0.9, 'A', fontsize=22)
f.text(0.65, 0.9, 'B', fontsize=22)



# %%
