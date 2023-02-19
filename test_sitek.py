"""

Based on Kevin Sitek's code samples in #248 as well as Daniel's tutorial:
https://rsatoolbox.readthedocs.io/en/latest/demo_searchlight.html
"""

from os.path import expanduser, join
from glob import glob
import numpy as np
import nibabel as nib
from rsatoolbox.rdm.rdms import RDMs
from rsatoolbox.model.model import ModelFixed
from rsatoolbox.inference import eval_fixed
from rsatoolbox.util.searchlight import (
    get_volume_searchlight, get_searchlight_RDMs, evaluate_models_searchlight)

data_dir = expanduser('~/data/rsatoolbox/248b_sitek')
mask_fpath = glob(join(data_dir, '*mask-gm.nii.gz'))[0]
image_paths = glob(join(data_dir, '*beta.nii.gz'))


mask_img = nib.load(mask_fpath)
mask_data = mask_img.get_fdata()
mask = ~np.isnan(mask_data) # daniel

## 5mins
centers, neighbors = get_volume_searchlight(mask, radius=5, threshold=0.5)

# loop over all images
x, y, z = mask_data.shape
print('reserving memory for betas..')
data = np.zeros((len(image_paths), x, y, z))
for x, im in enumerate(image_paths):
    print(f'loading image {x+1}/{len(image_paths)}')
    data[x] = nib.load(im).get_fdata()

# reshape data so we have n_observastions x n_voxels
data_2d = data.reshape([data.shape[0], -1])
data_2d = np.nan_to_num(data_2d)
# Get RDMs (13m30)
# only one pattern per image
events = np.arange(len(image_paths))
SL_RDM = get_searchlight_RDMs(data_2d, centers, neighbors, events, method='correlation')

#rsatoolbox/env/lib/python3.10/site-packages/rsatoolbox/rdm/calc.py:209: RuntimeWarning: invalid value encountered in divide
#   ma /= np.sqrt(np.einsum('ij,ij->i', ma, ma))[:, None]

# rsatoolbox/env/lib/python3.10/site-packages/rsatoolbox/data/computations.py:36: RuntimeWarning: invalid value encountered in multiply
#   average = np.nan * np.empty(

## MODELS
tone_rdm = np.array([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
                     [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
                     [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
                     [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
                     [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,],
                     [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,],
                     [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,],
                     [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,],
                     [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,],
                     [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,],
                     [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,],
                     [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,], ])

talker_rdm = np.array([[0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, ],
                       [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, ],
                       [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, ],
                       [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, ],
                       [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, ],
                       [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, ],
                       [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, ],
                       [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, ],
                       [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, ],
                       [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, ],
                       [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, ],
                       [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, ],
                       [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, ],
                       [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, ],
                       [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, ],
                       [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, ], ])

model_rdms = RDMs(
    np.asarray([tone_rdm, talker_rdm]),
    rdm_descriptors={'categorical_model':['tone', 'talker'],},
    dissimilarity_measure='Euclidean'
)
tone_model = ModelFixed( 'Tone RDM', model_rdms.subset('categorical_model', 'tone'))

evaluate_models_searchlight(SL_RDM, tone_model, eval_fixed, method='spearman', n_jobs=4)