"""
Fit zero-mean GPCSD to baseline period of auditory LFP recordings.
Investigate power and extract phases for coupling analysis in Matlab.
Creates Figure 2 in the paper.

"""

# %% Imports
import autograd.numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy.io
import pickle
import os.path
root_path = os.path.abspath(__file__)

from gpcsd.gpcsd1d import GPCSD1D
from gpcsd.covariances import *
from gpcsd.predict_csd import predictcsd_trad_1d
from gpcsd.utility_functions import normalize

# %% Setup
np.random.seed(0)

n_restarts = 10 # how many random initializations for GP fitting
ntrials_fit = None # how many trials to use in fitting; None uses all
probe_name = "lateral" # choose "medial" or "lateral"
reload_model = True

# Probe limits (in 1000 micron units)
a = 0
b = 2300
nx = 24
fs = 1000. # sampling rate Hz

# Probe spacing (in 1000 micron units)
x = np.linspace(a, b, nx)[:, None]

# Layer boundaries, defined by authors
layerbounds = 100*(np.array([11, 16])-1)
layerlabs = ['Superficial', 'Medium', 'Deep']

# %% Functions
# PLV, assuming axis 0 is trials
def plv(x, y):
    complex_phase_diff = np.exp(complex(0, 1) * (x - y))
    plv = np.abs(np.sum(complex_phase_diff, axis=0)) / x.shape[0]
    return plv

# Construct butterworth bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

# %% Load LFP data
lfp = []
for i in range(24):
    lfp.append(np.loadtxt('%s/data/%s_electrode%d.txt' % (root_path, probe_name, i+1)))
lfp = np.array(lfp) # (n_elec, time, trials)
lfp /= 100.0 # Rescale for numerical reasons
lfp -= np.mean(lfp, 2, keepdims=True)

# %% Subset to baseline period
time = np.loadtxt('%s/data/time.txt' % root_path) * 1000. # Time in seconds, convert to ms
time_idx = time < 0
t = time[time_idx][:, None] 
lfp_baseline = lfp[:, time_idx, :]

# %% Set up GPCSD and fit
# Note: could speed up with lower n_restarts, or don't use all trials
nx, nt, ntrials = lfp_baseline.shape
if ntrials_fit is None:
    trials_sel = np.arange(ntrials)
else:
    trials_sel = np.random.choice(ntrials, ntrials_fit, replace=False)

spatial_cov = GPCSD1DSpatialCovSE(x, a=-200.0, b=2600.0)
matern_cov = GPCSDTemporalCovMatern(t)
matern_cov.params['ell']['prior'].set_params(1., 20.)
SE_cov = GPCSDTemporalCovSE(t)
SE_cov.params['ell']['prior'].set_params(30., 100.)
sig2n_prior = [GPCSDHalfNormalPrior(0.1) for i in range(nx)]
gpcsd_model = GPCSD1D(lfp_baseline[:, :, trials_sel], x, t, 
                      sig2n_prior=sig2n_prior,
                      spatial_cov=spatial_cov, 
                      temporal_cov_list=[SE_cov, matern_cov], a=-200.0, b=2600.0)

if reload_model and os.path.isfile('%s/results/gpcsd_model_%s.pkl' % (root_path, probe_name)):
    with open('%s/results/gpcsd_model_%s.pkl' % (root_path, probe_name), 'rb') as f:
        params = pickle.load(f)
    gpcsd_model.restore_model_params(params)
else:
    gpcsd_model.fit(n_restarts=n_restarts, verbose=True)

params = gpcsd_model.extract_model_params()
with open('%s/results/gpcsd_model_%s.pkl' % (root_path, probe_name), 'wb') as f:
    pickle.dump(params, f)

# %% Predict during baseline period
gpcsd_model.update_lfp(lfp_baseline, t)
print(gpcsd_model)
gpcsd_model.predict(x, t)
tcsd = predictcsd_trad_1d(lfp_baseline)

# %% Single trial image plots
trial = 0
plt.figure(figsize=(14, 5))
plt.subplot(151)
plt.imshow(normalize(gpcsd_model.csd_pred_list[0][:, :, trial]), aspect='auto', cmap='bwr', vmin=-1, vmax=1)
plt.title('Slow CSD')
plt.subplot(152)
plt.imshow(normalize(gpcsd_model.csd_pred_list[1][:, :, trial]), aspect='auto', cmap='bwr', vmin=-1, vmax=1)
plt.title('Fast CSD')
plt.subplot(153)
plt.imshow(normalize(gpcsd_model.csd_pred[:, :, trial]), aspect='auto', cmap='bwr', vmin=-1, vmax=1)
plt.title('GPCSD')
plt.subplot(154)
plt.imshow(normalize(tcsd[:, :, trial]), aspect='auto', cmap='bwr', vmin=-1, vmax=1)
plt.title('tCSD')
plt.subplot(155)
plt.imshow(normalize(lfp_baseline[:, :, trial]), aspect='auto', cmap='bwr', vmin=-1, vmax=1)
plt.title('LFP')
plt.tight_layout()
plt.show()

# %% Time series plots for medium depth electrode
plt.figure(figsize=(12, 4))
plt.subplot(221)
plt.plot(t, lfp_baseline[11, :, ::100])
plt.plot(t, np.mean(lfp_baseline[11, :, :], 1), 'k', linewidth=3)
plt.title('LFP')
plt.xlabel('Time (ms)')
plt.subplot(222)
plt.plot(t, gpcsd_model.csd_pred[11, :, ::100])
plt.plot(t, np.mean(gpcsd_model.csd_pred[11, :, :], 1), 'k', linewidth=3)
plt.title('GPCSD')
plt.xlabel('Time (ms)')
plt.show()

# %% Prediction during trial - both CSD and LFP at the two timescales
t_ind = np.logical_and(time >= 0, time < 500)
lfp_trial = lfp[:, t_ind, :]
#lfp_trial = lfp_trial - np.mean(lfp_trial, 2, keepdims=True)
gpcsd_model.update_lfp(lfp_trial, time[t_ind][:, None])
gpcsd_model.predict(x, time[t_ind][:, None], type="both")
tcsd = predictcsd_trad_1d(lfp_trial)

# %%
trial_plot = 0
vmlfp = 0.9 * np.max(np.abs(lfp_trial[:, :, trial_plot]))
vmgpcsd = 0.9 * np.max(np.abs(gpcsd_model.csd_pred[:, :, trial_plot]))
vmtcsd = 0.9 * np.max(np.abs(tcsd[:, :, trial_plot]))

plt.figure(figsize=(12, 5))
plt.subplot(231)
plt.imshow(gpcsd_model.csd_pred_list[0][1:-1, :, trial_plot], aspect='auto', cmap='bwr',
           vmin=-vmgpcsd, vmax=vmgpcsd)
plt.title('GPCSD (slow)')
plt.colorbar()
plt.subplot(232)
plt.imshow(gpcsd_model.csd_pred_list[1][1:-1, :, trial_plot], aspect='auto', cmap='bwr',
           vmin=-vmgpcsd, vmax=vmgpcsd)
plt.title('GPCSD (fast)')
plt.colorbar()
plt.subplot(233)
plt.imshow(tcsd[1:-1, :, trial_plot], aspect='auto', cmap='bwr',
           vmin=-vmtcsd, vmax=vmtcsd)
plt.title('tCSD')
plt.colorbar()
plt.subplot(234)
plt.imshow(gpcsd_model.lfp_pred_list[0][1:-1, :, trial_plot], aspect='auto', cmap='bwr',
           vmin=-vmlfp, vmax=vmlfp)
plt.title('LFP (slow)')
plt.colorbar()
plt.subplot(235)
plt.imshow(gpcsd_model.lfp_pred_list[1][1:-1, :, trial_plot], aspect='auto', cmap='bwr',
           vmin=-vmlfp, vmax=vmlfp)
plt.title('LFP (fast)')
plt.colorbar()
plt.subplot(236)
plt.imshow(lfp_trial[1:-1, :, trial_plot], aspect='auto', cmap='bwr',
           vmin=-vmlfp, vmax=vmlfp)
plt.title('LFP')
plt.colorbar()
plt.tight_layout()
plt.show()

# %% Compute spectral power of each component

f_slow, spec0 = signal.welch(gpcsd_model.csd_pred_list[0], fs=fs, nfft=512, detrend=False, nperseg=512, axis=1)
f_fast, spec1 = signal.welch(gpcsd_model.csd_pred_list[1], fs=fs, nfft=512, detrend=False, nperseg=512, axis=1)
f_all, spec_csd = signal.welch(gpcsd_model.csd_pred, nfft=512, fs=fs, detrend=False, nperseg=512, axis=1)
f_slow, spec0_lfp = signal.welch(gpcsd_model.lfp_pred_list[0], fs=fs, nfft=512, detrend=False, nperseg=512, axis=1)
f_fast, spec1_lfp = signal.welch(gpcsd_model.lfp_pred_list[1], fs=fs, nfft=512, detrend=False, nperseg=512, axis=1)
f_all, spec_lfp = signal.welch(gpcsd_model.lfp_pred, fs=fs, nfft=512, detrend=False, nperseg=512, axis=1)
f_all, spec_y = signal.welch(lfp_trial, fs=fs, nfft=512, detrend=False, nperseg=512, axis=1)

maxfind = 20

maxcsdpower = np.max(np.mean(spec_csd[1:,:],2))
maxlfppower = np.max(np.mean(spec_y[1:,:],2))

spec0_lfp_mean = np.mean(spec0_lfp, 2).T
spec0_csd_mean = np.mean(spec0, 2).T
spec1_lfp_mean = np.mean(spec1_lfp, 2).T
spec1_csd_mean = np.mean(spec1, 2).T
spec_lfp_mean = np.mean(spec_lfp, 2).T
spec_csd_mean = np.mean(spec_csd, 2).T

# %% Figure 2 from paper
plt.rcParams.update({'font.size': 16})
scl = 700
e_ind_list = [3, 6, 13, 16, 19, 22]
xdepth = np.linspace(0, 2300, 24)
maxcsdpower = np.max(np.mean(spec_csd, 2)[:, 2:])
maxlfppower = np.max(np.mean(spec_y, 2)[:, 2:])

f, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [2, 2, 1.5]}, figsize=(10, 7))
# CSD
for ci, cix in enumerate(e_ind_list):
    ax[0].hlines(xdepth[cix], 0, f_all[maxfind], color='darkgrey')
    ax[0].plot(f_all[1:maxfind], xdepth[cix] - scl * spec0_csd_mean[1:maxfind, cix]/maxcsdpower, 'k--')
    ax[0].plot(f_all[1:maxfind], xdepth[cix] - scl * spec1_csd_mean[1:maxfind, cix] / maxcsdpower, 'k')
ax[0].set_ylim([0, 2300])
ax[0].invert_yaxis()
ax[0].set_ylabel('Depth (microns)')
ax[0].set_xlabel('Freq (Hz)')
ax[0].set_title('CSD')
ax[0].hlines(layerbounds, 0, f_all[maxfind], linestyles='dotted', color='grey', linewidth=2)
ax[0].legend(['Slow', 'Fast'])
ax[0].text(-13, -1.6, 'A', fontsize=20)
# LFP
for ci, cix in enumerate(e_ind_list):
    ax[1].hlines(xdepth[cix], 0, f_all[maxfind], color='darkgrey')
    ax[1].plot(f_all[1:maxfind], xdepth[cix] - scl * spec0_lfp_mean[1:maxfind, cix]/maxlfppower, 'k--')
    ax[1].plot(f_all[1:maxfind], xdepth[cix] - scl * spec1_lfp_mean[1:maxfind, cix] / maxlfppower, 'k')
ax[1].set_ylim([0, 2300])
ax[1].invert_yaxis()
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_title('LFP')
ax[1].hlines(layerbounds, 0, f_all[maxfind], linestyles='dotted', color='grey', linewidth=2)
# Relative 10Hz power
f_ix = 5 # index of ~10Hz power
ax[2].plot(spec1_csd_mean[f_ix, 1:-1] / np.max(spec1_csd_mean[f_ix, 1:-1]), xdepth[1:-1], 'r-o')
ax[2].plot(spec1_lfp_mean[f_ix, 1:-1] / np.max(spec1_lfp_mean[f_ix, 1:-1]), xdepth[1:-1], 'b-o')
ax[2].hlines(layerbounds, 0, 1.1, linestyles='dotted', color='grey', linewidth=2)
ax[2].set_xlabel('Relative 10 Hz power')
ax[2].set_ylim([0, 2300])
ax[2].invert_yaxis()
ax[2].legend(['CSD', 'LFP'])
ax[2].text(-0.5, -1.6, 'B', fontsize=20)
plt.tight_layout()
plt.show()

plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(gpcsd_model.t_pred, gpcsd_model.csd_pred_list[0][13, :, 0], 'r', label='Slow')
plt.plot(gpcsd_model.t_pred, gpcsd_model.csd_pred_list[1][13, :, 0], 'g', label='Fast')
plt.plot(gpcsd_model.t_pred, gpcsd_model.csd_pred[13, :, 0], 'k', label='Total')
plt.xlabel('Time (ms)')
plt.ylabel('CSD')
plt.text(-180.0, 0.01, 'C', fontsize=20)
plt.subplot(122)
plt.plot(gpcsd_model.t_pred, gpcsd_model.lfp_pred_list[0][13, :, 0], 'r', label='Slow')
plt.plot(gpcsd_model.t_pred, gpcsd_model.lfp_pred_list[1][13, :, 0], 'g', label='Fast')
plt.plot(gpcsd_model.t_pred, gpcsd_model.lfp_pred[13, :, 0], 'k', label='Total')
plt.xlabel('Time (ms)')
plt.ylabel('LFP')
plt.legend()
plt.tight_layout()
plt.show()

# %% Timecourses from electrode 13
plt.figure(figsize=(12, 16))
ax_csd = plt.subplot(321)
plt.plot(gpcsd_model.t_pred, gpcsd_model.csd_pred[13, :, ::100])
plt.title('CSD')
ax_lfp = plt.subplot(322)
plt.plot(gpcsd_model.t_pred, gpcsd_model.lfp_pred[13, :, ::100])
plt.title('LFP')
plt.subplot(323, sharey=ax_csd)
plt.plot(gpcsd_model.t_pred, gpcsd_model.csd_pred_list[0][13, :, ::100])
plt.title('CSD slow')
plt.subplot(324, sharey=ax_lfp)
plt.plot(gpcsd_model.t_pred, gpcsd_model.lfp_pred_list[0][13, :, ::100])
plt.title('LFP slow')
plt.subplot(325, sharey=ax_csd)
plt.plot(gpcsd_model.t_pred, gpcsd_model.csd_pred_list[1][13, :, ::100])
plt.title('CSD fast')
plt.subplot(326, sharey=ax_lfp)
plt.plot(gpcsd_model.t_pred, gpcsd_model.lfp_pred_list[1][13, :, ::100])
plt.title('LFP fast')
plt.show()


# %% Filter to 10Hz to extract phase angles
plt.figure()
for order in [1, 2, 3, 4, 5, 6, 9]:
    b, a = butter_bandpass(8., 12., fs, order=order)
    w, h = signal.freqz(b, a, worN=2000)
    plt.plot((fs * 0.5 / np.pi) * w[:200], abs(h[:200]), label="order = %d" % order)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.legend(loc='best')

b, a = butter_bandpass(8., 12., fs=1000., order=3)

csd_fast_filt = signal.filtfilt(b, a, gpcsd_model.csd_pred_list[1], axis=1)
csd_fast_phase = np.angle(signal.hilbert(csd_fast_filt, axis=1), deg=False)
lfp_fast_filt = signal.filtfilt(b, a, gpcsd_model.lfp_pred_list[1], axis=1)
lfp_fast_phase = np.angle(signal.hilbert(lfp_fast_filt, axis=1), deg=False)

# %% compute within-probe PLV matrix over time
plv_csd = np.nan * np.zeros((24, 24, gpcsd_model.t_pred.shape[0]))
plv_lfp = np.nan * np.zeros((24, 24, gpcsd_model.t_pred.shape[0]))
for ci in range(24):
    for cj in range(ci, 24):
        if ci == cj:
            continue
        res_csd = plv(np.squeeze(csd_fast_phase[ci, :, :]).T, np.squeeze(csd_fast_phase[cj, :, :]).T)
        plv_csd[ci, cj, :] = res_csd
        plv_csd[cj, ci, :] = res_csd
        res_lfp = plv(np.squeeze(lfp_fast_phase[ci, :, :]).T, np.squeeze(lfp_fast_phase[cj, :, :]).T)
        plv_lfp[ci, cj, :] = res_lfp
        plv_lfp[cj, ci, :] = res_lfp

# %% Visualize PLV briefly
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(gpcsd_model.t_pred.squeeze(), plv_csd.reshape((24*24, -1)).T)
plt.plot(gpcsd_model.t_pred.squeeze(), np.nanmean(plv_csd.reshape((24*24, -1)), 0), 'k', linewidth=3)
plt.subplot(122)
plt.plot(gpcsd_model.t_pred.squeeze(), plv_lfp.reshape((24*24, -1)).T)
plt.plot(gpcsd_model.t_pred.squeeze(), np.nanmean(plv_lfp.reshape((24*24, -1)), 0), 'k', linewidth=3)
plt.show()

# %% Save phases at time point index 350 for Matlab analysis
scipy.io.savemat('%s/results/csd_lfp_filt_phases_%s.mat' % (root_path, probe_name), {'csd':csd_fast_phase[:, 350, :], 'lfp':lfp_fast_phase[:, 350, :]})


