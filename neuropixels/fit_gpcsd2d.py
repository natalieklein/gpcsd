"""
Fit GPCSD2D to visual area Neuropixels data.
Run after extract_data.py (which gets visual area data from raw data).

"""
# %% imports
import autograd.numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from scipy import signal
import scipy.io
import os.path
root_path = os.path.abspath(__file__)

from gpcsd.gpcsd2d import GPCSD2D
from gpcsd.covariances import GPCSD2DSpatialCov, GPCSD2DSpatialCovSE, GPCSDTemporalCovMatern, GPCSDTemporalCovSE
from gpcsd.priors import GPCSDInvGammaPrior, GPCSDHalfNormalPrior

# %%
reload_model = True # reload model if pickle file already exists?
plot_ol = False # plot outlier trials? (plotting is a little slow)

# %% Load data
csd_loc = {'probeC': np.array([2260., 2450., 2650., 2785.]), 'probeD': np.array([2215., 2410., 2590., 2720.])}

lfp = {}
x = {}
z = {}
for probe in ['probeC', 'probeD']:

    # Load data (saved by extract_data.py)
    with open('%s/results/neuropixel_viz_%s_m405751.pkl' % (root_path, probe), 'rb') as f:
        d = pickle.load(f)

    x[probe] = d['x']   # (69, 2) spatial locations, microns
    t = d['t']   # (2500, 1) time points in seconds (in 0.4ms increments)
    t *= 1000.   # convert to ms
    t_ind = np.logical_and(t >= -40.0, t <= 110.0)
    t = np.expand_dims(t[t_ind], 1)
    lfp_tmp = d['y'][:, t_ind, :] # (69, 2500, 150) which is (spatial, time, trials)
    lfp_tmp /= 100. # Scaling
    # remove evoked LFP
    lfp_tmp = lfp_tmp - np.mean(lfp_tmp, 2, keepdims=True)
    lfp[probe] = lfp_tmp

    # Desired CSD prediction locations
    z[probe] = np.stack([24. * np.ones(len(csd_loc[probe])), csd_loc[probe]]).T

# %% Visualize data, check for outlier trials
ol_bool = {}
for probe in ['probeC', 'probeD']:
    trial_sd = np.std(lfp[probe], axis=2, keepdims=True)
    ol = np.any(np.abs(lfp[probe]) > 5 * trial_sd, axis=(0, 1))
    ol_bool[probe] = ol

ol = np.logical_or(ol_bool['probeC'], ol_bool['probeD'])
print('outlier trials: %d' % np.sum(ol))
if plot_ol:
    for probe in ['probeC', 'probeD']:
        x1 = np.unique(x[probe][:, 0])
        for j in x1:
            plt.figure(figsize=(6, 16))
            for i, xi in enumerate(x[probe][x[probe][:, 0] == j]):
                plt.plot(t, xi[1] + 3*lfp[probe][i, :, np.logical_not(ol)].T, 'k')
                plt.plot(t, xi[1] + 3*lfp[probe][i, :, ol].T, 'r')
            plt.title('%s x1 = %0.2f microns' % (probe,j))
            plt.show()

for probe in ['probeC', 'probeD']:
    lfp[probe] = lfp[probe][:, :, np.logical_not(ol)]

# %%
csdSE = {}
csdMatern = {}
for probe in ['probeC', 'probeD']:

    # Create GPCSD model
    R_prior = GPCSDInvGammaPrior()
    R_prior.set_params(50, 300)
    ellSEprior = GPCSDInvGammaPrior()
    ellSEprior.set_params(20, 200)
    temporal_cov_SE = GPCSDTemporalCovSE(t, ell_prior=ellSEprior)
    ellMprior = GPCSDInvGammaPrior()
    ellMprior.set_params(1, 20)
    temporal_cov_M = GPCSDTemporalCovMatern(t, ell_prior=ellMprior)
    gpcsd_model = GPCSD2D(lfp[probe], x[probe], t, R_prior=R_prior, 
                          temporal_cov_list=[temporal_cov_SE, temporal_cov_M], 
                          eps=1, ngl1=30, ngl2=120,
                          a1=np.min(x[probe][:, 0])-16, b1=np.max(x[probe][:, 0])+16, 
                          a2=np.min(x[probe][:, 1])-100, b2=np.max(x[probe][:, 1])+100)
    print(gpcsd_model)

    if reload_model and os.path.isfile('%s/results/%s_model_csd_pred.pkl' % (root_path, probe)):
        with open('%s/results/%s_model_csd_pred.pkl' % (root_path, probe), 'rb') as f:
            results = pickle.load(f)
        gpcsd_model.restore_model_params(results['params'])
        csdSE[probe] = results['csd0']
        csdMatern[probe] = results['csd1']
    else:
        # Fit GPCSD model
        gpcsd_model.fit(n_restarts=20, verbose=True)

        # Selected parameters
        print(gpcsd_model)
    
        # Predictions
        gpcsd_model.predict(z[probe], t)

        csd_pred0 = gpcsd_model.csd_pred_list[0]
        csd_pred1 = gpcsd_model.csd_pred_list[1]

        # Saving
        params = gpcsd_model.extract_model_params()
        with open('%s/results/%s_model_csd_pred.pkl' % (root_path, probe), 'wb') as f:
            pickle.dump({'params':params, 'csd0':csd_pred0, 'csd1':csd_pred1, 'x':z[probe], 't':t}, f)

# %% Power spectra
layer_labs = ['L6', 'L5', 'L4', 'L 2/3']
for probe, probe_name in zip(['probeC', 'probeD'], ['V1', 'LM']):

    f, res = signal.welch(csdMatern[probe], fs=2500., axis=1, noverlap=256, nfft=5000, nperseg=512)
    pgramMatern = np.mean(res, 2)

    f, res = signal.welch(csdSE[probe], fs=2500., axis=1, noverlap=256, nfft=5000, nperseg=512)
    pgramSE = np.mean(res, 2)

    colors = ['k', 'r', 'g', 'b']
    fig = plt.figure(figsize=(6, 5))
    for i in range(4):
        plt.plot(f[1:100], pgramMatern[i, 1:100].T, label=layer_labs[i], color=colors[i])
        plt.plot(f[1:100], pgramSE[i, 1:100].T, label=layer_labs[i], color=colors[i])
    plt.xlabel('Frequency (Hz)')
    plt.title("%s CSD periodogram" % probe_name)
    plt.legend()
    plt.show()

# %% Filtering
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

fs = 2500
bands = {'theta': [3., 7.], 'beta':[20., 24.]}
for probe, probe_name in zip(['probeC', 'probeD'], ['V1', 'LM']):
    phases = {}
    for bk in bands.keys():
        ord, wn = signal.buttord([bands[bk][0], bands[bk][1]], [bands[bk][0]-2, bands[bk][1]+2], 10, 20, fs=fs)
        sos = signal.butter(1, [bands[bk][0], bands[bk][1]], btype='bandpass', fs=fs, output='sos')
        res = signal.sosfiltfilt(sos, csdMatern[probe]+csdSE[probe], axis=1)
        phase_tmp = np.angle(signal.hilbert(res, axis=1), deg=False)
        # TODO get time index of interest
        ti = np.array([np.argmin(np.abs(t.squeeze() - 0.0)), np.argmin(np.abs(t.squeeze() - 70.0))])
        phases[bk] = phase_tmp[:, ti, :]
    scipy.io.savemat('%s/results/neuropixel_csd_%s_phases.mat' % (root_path, probe_name), phases)
# %%
