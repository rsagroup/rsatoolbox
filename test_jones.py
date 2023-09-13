from os.path import expanduser, join
import numpy
import rsatoolbox
from scipy import io
from rsatoolbox.inference import eval_fixed
from rsatoolbox.searchlight.volume import get_volume_searchlight, get_searchlight_RDMs, evaluate_models_searchlight
data_dir = expanduser('~/data/rsatoolbox/248a_jones-michael-s')


# ----- load data ---------------------------------------------------

# 2D array (nconditions x nvoxels) saved from matlab

dataFile = join(data_dir, 'condition_data.mat')

temp = io.matlab.loadmat(dataFile)
varkey = list(temp)[-1]
data = temp[varkey]
# can convert nan here or convert nan in matlab before the
# file is saved (or both) - you get the same result
data = numpy.nan_to_num(data)

# ---- load model ---------------------------------------------

# 2D array (nconditions x nfeatures) saved from matlab

modelFile = join(data_dir, 'model.mat')

temp = io.matlab.loadmat(modelFile)
varkey = list(temp)[-1]
model_def = temp[varkey]

model_features = [rsatoolbox.data.Dataset(model_def)]
model_RDM = rsatoolbox.rdm.calc_rdm(model_features)
model = rsatoolbox.model.ModelFixed('fixed_model',model_RDM)

# ---- load mask -------------------------------------------

# 0/1 mask saved as 3D array from matlab
# [nx ny nz] where nx*ny*nz = nvoxels
# converted to boolean (although 0/1 seems to work)

maskFile = join(data_dir, 'searchlight_mask.mat')

temp = io.matlab.loadmat(maskFile)
varkey = list(temp)[-1]
imask = temp[varkey]
#mask = imask > 0

## mask based on all-zero patterns
mask_2d = ~numpy.all(data==0, axis=0)
x,y,z = imask.shape
mask = mask_2d.reshape(x, y, z)

# ---- searchlight -----------------------------------------

image_value = numpy.arange(data.shape[0])
centers, neighbors = get_volume_searchlight(mask)
SL_RDM = get_searchlight_RDMs(data, centers, neighbors, image_value, method='correlation')
eval_results = evaluate_models_searchlight(SL_RDM, model, eval_fixed, method='spearman', n_jobs=3)