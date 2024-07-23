
from os.path import expanduser
from rsatoolbox.io.fmriprep import find_fmriprep_runs
from rsatoolbox.rdm.rdms import concat
import numpy, pandas, matplotlib.pyplot
from os.path import join
import matplotlib.pyplot as plt
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix, plot_stat_map, plot_glass_brain
from nilearn.glm.first_level import FirstLevelModel

data_dir = expanduser('~/data/rsatoolbox/mur32')

runs = find_fmriprep_runs(data_dir, tasks=['main'])

runs = [run for run in runs if run.boldFile.sub != '02']

run = runs[0]
dims = run.get_data().shape ## bold timeseries: x * y * z * volumes
n_voxels = dims[0]
n_vols = dims[-1]
tr = run.get_meta()['RepetitionTime'] ## TR in seconds

lut_fpath = join(data_dir, 'derivatives', 'fmriprep',
    'desc-aparcaseg_dseg.tsv')
lut_df = pandas.read_csv(lut_fpath, sep='\t')

region_names = [
    'cuneus',
    'pericalcarine',
    'lingual',
    'fusiform',
    'lateraloccipital',
    'inferiortemporal',
    'inferiorparietal'
]
counts = numpy.zeros((len(region_names), n_voxels))
for r, region_name in enumerate(region_names):

    for run in runs:

        aparc = run.boldFile.get_mri_sibling(desc='aparcaseg', suffix='dseg')
        aparc_data = aparc.get_data()

        for hemi in ('r', 'l'):
            full_name = f'ctx-{hemi}h-{region_name}'
            matches = lut_df[lut_df['name']==full_name]
            assert len(matches) == 1, f'None or multiple matches for {full_name}'
            region_id = matches['index'].values[0]
            mask = (aparc_data == float(region_id)).ravel()
            print(f'{full_name} ({region_id}) sub {run.sub} run {run.run}: {mask.sum()}')
            counts[r, :] += (mask*1)

        

### select roi extend based on overlap
### run GLM (mask by union of ROIs)

raise ValueError
for r in [0, 20]:
    run = runs[r]
    mask_fpath = run.boldFile.get_mri_sibling(desc='brain', suffix='mask').fpath
    plot_glass_brain(mask_fpath)

plt.show()

from nilearn.plotting import plot_roi
plot_roi(mask_img)

group_ds = merge_datasets(run_datasets)
group_ds.obs_descriptors.keys()


# In[49]:


run_datasets[0].descriptors


rdms_list = []
for sub in set(group_ds.obs_descriptors['sub']):
    run_indices = [r for (r, ds) in enumerate(run_datasets) if ds.descriptors['sub'] == sub]
    rdms_list.append(
        calc_rdm(
            dataset=merge_datasets([run_datasets[r] for r in run_indices]),
            noise=[run_precs[r] for r in run_indices],
            method='crossnobis',
            descriptor='trial_type',
            cv_descriptor='run',
        )
    )
data_rdms = concat(rdms_list)



fig, _, _ = show_rdm(data_rdms, show_colorbar='panel')
matplotlib.pyplot.show()




a_ds = run_datasets[0]
obj_conds = numpy.unique(a_ds.obs_descriptors['trial_type']).tolist()
INDOOR = ['bagel', 'candle', 'clock', 'glass', 'kettle', 'knife', 'sponge', 'table']
STRAIGHT = ['candle', 'knife', 'sponge', 'table', 'spade', 'ladder', 'brick', 'pedal']
df = pandas.DataFrame([dict(
    trial_type=c,
    indoor=float(c.split('_')[1] in INDOOR),
    straight=float(c.split('_')[1] in STRAIGHT),
    modality=float('image_' in c)
) for c in obj_conds])



model_dataset = Dataset.from_df(df)
model_dataset.channel_descriptors
model_rdms = calc_rdm(
    [model_dataset.split_channel('name')],
    method='euclidean',
    descriptor='trial_type'
)
model_rdms.rdm_descriptors['name'] = model_dataset.channel_descriptors['name']
fig, _, _ = show_rdm(model_rdms, rdm_descriptor='name')
matplotlib.pyplot.show()



from rsatoolbox.model.model import ModelFixed

models = []
for model_name in model_rdms.rdm_descriptors['name']:
    model_rdm = model_rdms.subset('name', model_name)
    models.append(ModelFixed(model_name, model_rdm))




from rsatoolbox.inference.evaluate import eval_dual_bootstrap

eval_result = eval_dual_bootstrap(models, data_rdms)
print(eval_result)


from rsatoolbox.vis.model_plot import plot_model_comparison

fig, _, _ = plot_model_comparison(eval_result, sort=True)
matplotlib.pyplot.show()
