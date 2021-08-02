"""
Extracts visual area LFPs from Neuropixels nwb.lfp files, does some plotting, 
and saves out per-region LFPs as pickle files for further analysis.
Creates Figure 6A from the paper.
"""

# %% imports
import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import os.path
root_path = '/'.join(os.path.abspath(__file__).split('/')[:-1])

plt.rcParams.update({'font.size': 16})

# %% function definition (written by Josh Siegle)
def get_channel_location(channel):
    """
    Returns physical location (in microns) of a Neuropixels channel,
    relative to the probe tip.
    Parameters:
    -----------
    channel - int
        channel number (0-383)
    Returns:
    --------
    location - tuple
        (xpos, ypos) in microns
    isReference - bool
        True if channel is a reference, False otherwise

    """
    xlocations = [16, 48, 0, 32]
    try:
        [36, 75, 112, 151, 188, 227, 264, 303, 340, 379].index(channel)
        isReference = True
    except ValueError:
        isReference = False
    return (xlocations[channel % 4], np.floor(channel / 2) * 20), isReference

# %% Settings
viz_roi_labels = {'probeA':'VISam', 'probeB':'VISpm', 'probeC':'V1', 'probeD':'LM', 'probeE':'VISal', 'probeF':'VISrl'}

m_id = '405751' # mouse ID
probe_list = ['probeC', 'probeD'] # Which probes we are interested in

roi_codes = {'V':1, 'C':2, 'D':3, 'T':4, 'S':5}
layer_boundaries = {'probeC':[273, 258, 233], 'probeD':[264, 255, 229]} # only have for ProbeC probeD; channel numbers
csd_loc = {'probeC': [2260., 2450., 2650., 2785.], 'probeD': [2215., 2410., 2590., 2720.]}

# %% Files for LFP and spiking data
# Data available at https://doi.org/10.5281/zenodo.5150708
lfp_nwb_file = '%s/data/mouse405751.lfp.nwb' % root_path
nwb_lfp = h5.File(lfp_nwb_file)

spikes_nwb_file = '%s/data/mouse405751.spikes.nwb' % root_path
nwb = h5.File(spikes_nwb_file)

# %% Loop over probes to get data
for pi, probe in enumerate(probe_list): 

    print('starting %s'%probe)

    if not probe in nwb['processing'].keys():
        continue

    # Get channel indices for LFPs
    electrodes = nwb_lfp['acquisition']['timeseries'][probe]['electrode_idx'][()]

    # Get channel indices for spikes
    spike_units = nwb['processing'][probe]['unit_list'][()]
    spike_electrodes = []
    for unit_idx, unit in enumerate(spike_units):
        spike_electrodes += [nwb['processing'][probe]['UnitTimes'][str(unit)]['channel'][()]]

    spike_electrodes = np.array(spike_electrodes)
    unique_spike_electrodes = np.sort(np.unique(spike_electrodes))

    # Get putative area for each spiking unit
    # Visual areas: VISam (probe A), VISpm (probe B), VISp (probe C), VISl (probe D),
    #               VISal (probe E), VISrl (probeF)
    # Hippocampus: CA (CA1/CA3), DG (dentate gyrus)
    # Other: TH (thalamus), SC (superior colliculus)
    ch_labels = np.zeros(384)

    units = nwb['processing'][probe]['unit_list']

    for unit_idx, unit in enumerate(units):

        try:
            ccf_structure = nwb['processing'][probe]['UnitTimes'][str(unit)]['ccf_structure'][()].decode("utf-8")
        except AttributeError:
            ccf_structure = None

        if ccf_structure is not None:
            # Visual areas: VISam (probe A), VISpm (probe B), VISp (probe C), VISl (probe D),
            #               VISal (probe E), VISrl (probeF)
            # Hippocampus: CA (CA1/CA3), DG (dentate gyrus)
            # Other: TH (thalamus), SC (superior colliculus)
            channel = nwb['processing'][probe]['UnitTimes'][str(unit)]['channel'][()]
            if ccf_structure[0] == 'V': # visual cortex
                ch_labels[channel] = roi_codes['V']
            elif ccf_structure[0] == 'C':
                ch_labels[channel] = roi_codes['C']
            elif ccf_structure[0] == 'D':
                ch_labels[channel] = roi_codes['D']
            elif ccf_structure[0] == 'T':
                ch_labels[channel] = roi_codes['T']
            elif ccf_structure[0] == 'S':
                ch_labels[channel] = roi_codes['S']

    print('Viz elec: %d'%sum(ch_labels == roi_codes['V']))
    print('CA1 elec: %d'%sum(ch_labels == roi_codes['C']))
    print('DG elec: %d' % sum(ch_labels == roi_codes['D']))
    print('Thal elec: %d'%sum(ch_labels == roi_codes['T']))
    print('SC elec: %d' % sum(ch_labels == roi_codes['S']))

    # Plot channels, marking references in yellow and recorded indices in red
    viz_loc = np.zeros((sum(ch_labels == roi_codes['V']), 2))
    viz_ind = 0
    ca_loc = np.zeros((sum(ch_labels == roi_codes['C']), 2))
    ca_ind = 0
    dg_loc = np.zeros((sum(ch_labels == roi_codes['D']), 2))
    dg_ind = 0
    th_loc = np.zeros((sum(ch_labels == roi_codes['T']), 2))
    th_ind = 0
    sc_loc = np.zeros((sum(ch_labels == roi_codes['S']), 2))
    sc_ind = 0

    for i in range(0, 384):
        loc, isRef = get_channel_location(i)
        if i in electrodes:
            if ch_labels[i] == roi_codes['V']:
                viz_loc[viz_ind, :] = loc
                viz_ind += 1
            elif ch_labels[i] == roi_codes['C']:
                ca_loc[ca_ind, :] = loc
                ca_ind += 1
            elif ch_labels[i] == roi_codes['D']:
                dg_loc[dg_ind, :] = loc
                dg_ind += 1
            elif ch_labels[i] == roi_codes['T']:
                th_loc[th_ind, :] = loc
                th_ind += 1
            elif ch_labels[i] == roi_codes['S']:
                sc_loc[sc_ind, :] = loc
                sc_ind += 1
            else:
                continue

    layer_locs = [get_channel_location(li)[0][1] for li in layer_boundaries[probe]]
    print('%s layer locs microns' % probe)
    print(layer_locs)

    f = plt.figure(figsize=(5, 7))
    for i in range(0, 384):
        loc, isRef = get_channel_location(i) # TODO should exclude ref by relabeling so they aren't V?
        if isRef:
            dotStyle = '.w'
        else:
            if i in electrodes:
                if ch_labels[i] == roi_codes['V']:
                    dotStyle = '.r'
                elif ch_labels[i] == roi_codes['C']:
                    dotStyle = '.b'
                elif ch_labels[i] == roi_codes['D']:
                    dotStyle = '.g'
                elif ch_labels[i] == roi_codes['T']:
                    dotStyle = '.m'
                elif ch_labels[i] == roi_codes['S']:
                    dotStyle = '.y'
                else:
                    dotStyle = '.c'
            else:
                dotStyle = '.k'
        plt.plot(loc[0], loc[1], dotStyle)
    for csd_y in csd_loc[probe]:
        plt.plot(24, csd_y, 'kD', ms=5)
        plt.plot(24, csd_y, 'yD', ms=2)
    plt.xlim([-20, 70])
    plt.xlabel('X pos (microns)')
    plt.ylabel('Y pos (microns)')
    plt.title('LFP channels %s' % viz_roi_labels[probe])
    plt.hlines(layer_locs, xmin=-10, xmax=55)
    plt.hlines([np.min(csd_loc[probe]), np.max(csd_loc[probe])], xmin=-10, xmax=55, linestyles="dashed")
    plt.text(55, layer_locs[0] + 40, "2/3", fontsize=12)
    plt.text(56, layer_locs[0] - 100, "4", fontsize=12)
    plt.text(56, layer_locs[1] - 160, "5", fontsize=12)
    plt.text(56, layer_locs[2] - 100, "6", fontsize=12)
    red_patch = mpatches.Patch(color='red', label='VIS')
    blue_patch = mpatches.Patch(color='blue', label='CA')
    green_patch = mpatches.Patch(color='green', label='DG')
    mag_patch = mpatches.Patch(color='magenta', label='TH')
    plt.legend(handles=[red_patch, blue_patch, green_patch, mag_patch], bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.title(viz_roi_labels[probe])
    plt.subplots_adjust(right=0.7)
    plt.tight_layout()
    plt.show()

    # Try to get epochs (extract 1 second centered at stim)
    lfp_data = nwb_lfp['acquisition']['timeseries'][probe]['data']

    print('got %d LFP elecs, %d unit elecs (%d unique)'%(len(electrodes), spike_electrodes.shape[0], unique_spike_electrodes.shape[0]))
    print('LFP data shape %d'%lfp_data.shape[1])
    # steps by 0.04ms
    lfp_timestamps = nwb_lfp['acquisition']['timeseries'][probe]['timestamps']

    stim = 'flash_250ms_1'
    trial_times = np.squeeze(nwb['stimulus']['presentation'][stim]['timestamps'][()])[:, 0]
    ntrials = trial_times.shape[0]
    electrodes = nwb_lfp['acquisition']['timeseries'][probe]['electrode_idx'][()]
    nx = electrodes.shape[0]

    lfp_sample_rate = 2500

    lfp_mat = np.zeros((nx, lfp_sample_rate, ntrials))

    lfp_ch_labels = np.zeros(nx)

    for trial in range(ntrials):

        start_index = np.argmin(np.abs(lfp_timestamps - trial_times[trial])) - int(lfp_sample_rate * 0.5)

        for idx, ch in enumerate(electrodes):
            channel_data = lfp_data[start_index:start_index + lfp_sample_rate, idx] * 0.195  # convert to microvolts
            lfp_mat[idx, :, trial] = channel_data
            lfp_ch_labels[idx] = ch_labels[ch]

    t = np.linspace(-0.5, 0.5, len(channel_data))

    lfp_evoked = np.mean(lfp_mat,2)

    # Plot evokeds for channels identified to each area
    f = plt.figure(figsize=(6, 20))
    ax = plt.subplot(511)
    if sum(ch_labels == roi_codes['V']) > 0:
        plt.plot(t,lfp_evoked[lfp_ch_labels == roi_codes['V'], :].T)
    plt.axvline(0.0, color='black', linestyle='--')
    plt.axhline(0.0, color='black')
    plt.xlabel('Time (s)')
    plt.ylabel('Microvolts')
    plt.xlim([-0.5, 0.5])
    plt.title('%s evokeds'%viz_roi_labels[probe])
    plt.subplot(512, sharey = ax)
    if sum(ch_labels == roi_codes['C']) > 0:
        plt.plot(t,lfp_evoked[lfp_ch_labels == roi_codes['C'], :].T)
    plt.axvline(0.0, color='black', linestyle='--')
    plt.axhline(0.0, color='black')
    plt.xlabel('Time (s)')
    plt.ylabel('Microvolts')
    plt.xlim([-0.5, 0.5])
    plt.title('CA evokeds')
    plt.subplot(513, sharey = ax)
    if sum(ch_labels == roi_codes['D']) > 0:
        plt.plot(t,lfp_evoked[lfp_ch_labels == roi_codes['D'], :].T)
    plt.axvline(0.0, color='black', linestyle='--')
    plt.axhline(0.0, color='black')
    plt.xlabel('Time (s)')
    plt.ylabel('Microvolts')
    plt.xlim([-0.5, 0.5])
    plt.title('DG evokeds')
    plt.subplot(514, sharey = ax)
    if sum(ch_labels == roi_codes['T']) > 0:
        plt.plot(t,lfp_evoked[lfp_ch_labels == roi_codes['T'], :].T)
    plt.axvline(0.0, color='black', linestyle='--')
    plt.axhline(0.0, color='black')
    plt.xlabel('Time (s)')
    plt.ylabel('Microvolts')
    plt.xlim([-0.5, 0.5])
    plt.title('TH evokeds')
    plt.subplot(515, sharey = ax)
    if sum(ch_labels == roi_codes['S']) > 0:
        plt.plot(t,lfp_evoked[lfp_ch_labels == roi_codes['S'], :].T)
    plt.axvline(0.0, color='black', linestyle='--')
    plt.axhline(0.0, color='black')
    plt.xlabel('Time (s)')
    plt.ylabel('Microvolts')
    plt.xlim([-0.5, 0.5])
    plt.title('SC evokeds')
    f.set_size_inches(6, 15)
    plt.tight_layout()
    plt.show()

    # Save data for visual regions
    viz_lfp = lfp_mat[lfp_ch_labels == roi_codes['V'], :, :] # (nx_viz, nt, ntrials)
    viz_save = {'x':viz_loc, 'y':viz_lfp, 't':t, 'fs':lfp_sample_rate, 'roi':viz_roi_labels[probe]}
    with open('%s/results/neuropixel_viz_%s_m%s.pkl'%(root_path, probe, m_id),'wb') as f:
        pickle.dump(viz_save, f)



# %%
