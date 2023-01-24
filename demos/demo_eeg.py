from os.path import join, expanduser, basename
import glob, copy, json
import numpy, tqdm, mne, pandas
import rsatoolbox
from numpy import atleast_2d
from scipy.spatial.distance import pdist
from matplotlib import pyplot

""" PREPROCESSING """

rawdata = expanduser('~/data/imasem/rawdata')
derivdata = expanduser('~/data/imasemrsa/epochs')

fpaths = glob.glob(join(rawdata, '**/*_eeg.bdf'), recursive=True)
for fpath in tqdm.tqdm(fpaths, smoothing=0):
    raw = mne.io.read_raw_bdf(fpath, preload=True, verbose='error')

    chans_df = pandas.read_csv(
        fpath.replace('_eeg.bdf', '_channels.tsv'),
        sep='\t'
    )

    # drop unused channels
    misc_chans = chans_df[chans_df.type=='MISC'].name.to_list()
    raw = raw.drop_channels(misc_chans)

    # filter
    raw = raw.filter(l_freq=0.1, h_freq=40, verbose='error')

    # rereference
    ref_chans = chans_df[chans_df.type=='REF'].name.to_list()
    raw.set_eeg_reference(ref_channels=ref_chans, verbose='error')

    # TODO: add nsd labels in event dict?
    events = mne.find_events(raw, verbose='error')
    eeg_chans = chans_df[chans_df.type=='EEG'].name.to_list()
    epochs = mne.Epochs(
        raw,
        events,
        decim=8,
        tmin=-0.2,
        tmax=+1.0,
        picks=eeg_chans,
        verbose='error'
    )
    fname = basename(fpath.replace('_eeg.bdf', '_epo.fif'))
    epochs.save(join(derivdata, fname), verbose='error')

""" IMPORTING DATA """

## loading the epoched data into rsatoolbox
datasets_3D = []
for sub in numpy.arange(7)+1:
    runs = []
    for fpath in glob.glob(join(derivdata, f'sub-0{sub}*task-images*.fif')):
        epo = mne.read_epochs(fpath, preload=True, verbose='error')
        ## TODO: attach NSD labels from event_id (ideal) or custom
        run_ds = rsatoolbox.data.TemporalDataset(
            measurements=epo.get_data(),
            descriptors=dict(sub=1), # run=x, ses=x
            obs_descriptors=dict(triggers=epo.events[:, 2]),
            channel_descriptors=dict(names=epo.ch_names),
            time_descriptors=dict(time=epo.times)
        )
        runs.append(run_ds)

    ## concatenate runs
    meas = numpy.concatenate([ds.measurements for ds in runs], axis=0)
    trigs = numpy.concatenate([ds.obs_descriptors['triggers'] for ds in runs], axis=0)
    template_run = runs[0]
    ds = rsatoolbox.data.TemporalDataset(
        measurements=meas,
        descriptors=dict(sub=1),
        obs_descriptors=dict(triggers=trigs),
        channel_descriptors=copy.deepcopy(template_run.channel_descriptors),
        time_descriptors=copy.deepcopy(template_run.time_descriptors)
    )
    datasets_3D.append(ds)

""" DATA RDMS """

#P400
sample_dataset = datasets_3D[0]
timepoints = sample_dataset.time_descriptors['time']
p400_tps = timepoints[numpy.bitwise_and(timepoints > 0.35, timepoints < 0.45)]
## TODO loop this
datasets = sample_dataset.bin_time('time', p400_tps).convert_to_dataset('time')

## calc rdm
data_rdms = rsatoolbox.rdm.calc.calc_rdm_crossnobis(datasets, descriptor='triggers')


""" IMPORTING ANNOTATION MODEL """

annotdata = expanduser('~/data/rsatoolbox/nsd100ann')
fpaths = glob.glob(join(annotdata, '*_annotations.csv'))
df_raw = pandas.read_csv(fpaths[0])
df = pandas.json_normalize(df_raw.label.apply(json.loads).tolist())
df['stim'] = df_raw.stim1_name
df['animacy'] = df.animacy.str.contains('Yes')
df['indoor'] = df.inoutdoors.str.contains('Indoor')
df['r'] = df.color == 'red'
df['g'] = df.color == 'green'
df['b'] = df.color == 'blue'
utv_color = pdist(df[['r', 'g', 'b']].values * 1)
utv_zoom = pdist(atleast_2d(df.distance.astype(float)).T)
utv_animacy = pdist(atleast_2d(df.animacy.astype(float)).T)
utv_inout = pdist(atleast_2d(df.indoor.astype(float)).T)
utvs = numpy.asarray([utv_zoom, utv_color, utv_inout, utv_animacy])
utvs /= utvs.max(axis=1, keepdims=True)
model_rdms = rsatoolbox.rdm.rdms.RDMs(
    dissimilarities=utvs,
    dissimilarity_measure='euclidean',
    rdm_descriptors=dict(model_name=['zoom', 'color', 'inout', 'animacy']),
    pattern_descriptors=dict(nds=df.stim)
)

fig, _, _ = rsatoolbox.vis.show_rdm(model_rdms, rdm_descriptor='model_name')
pyplot.show()

## create weighted feature model
annot_model = rsatoolbox.model.ModelWeighted('WeightedAnnotations', model_rdms)
thetas = annot_model.fit(data_rdms, method='cosine')
# TODO plot thetas bar graph or can do this from result?


## create fixed model from MA data
# TODO

## compare models 
#eval_dual_bootstrap[annots+ma] + plot model comparison