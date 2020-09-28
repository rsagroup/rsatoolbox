""" Script to preprocess MEG sample data for Temporal RSA demo notebook

Use of pickle is not recommended in multi-user environments
"""
import pickle #nosec - ignoring pickle security warning
import mne
from mne.datasets import sample

data_path = sample.data_path()

subjects_dir = data_path + '/subjects'
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
tmin, tmax = -0.200, .7500
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2,
            'Visual/Left': 3, 'Visual/Right': 4}

raw = mne.io.read_raw_fif(raw_fname, preload=True)

raw.filter(1, 20)
events = mne.find_events(raw, 'STI 014')

# Set up pick list: EEG + MEG - bad channels (modify to your needs)
raw.info['bads'] += ['MEG 2443', 'EEG 053']  # bads + 2 more

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=('grad', 'eog'), baseline=(None, 0.), preload=True,
                    reject=dict(grad=4000e-13, eog=150e-6), decim=10)
epochs.pick_types(meg=True, exclude='bads')  # remove stim and EOG

epochs.apply_baseline((-.2,.0))

X = epochs.get_data()  # MEG signals: n_epochs, n_meg_channels, n_times
y = epochs.events[:, 2]

pickle.dump({'data': X[:,:,:],
             'times': epochs.times[:],
             'cond_idx': epochs.events[:,2],
             'cond_names': event_id,
             'channel_names': epochs.ch_names},
            open( "meg_sample_data.pkl", "wb" ))
