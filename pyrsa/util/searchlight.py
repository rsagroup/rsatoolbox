import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from joblib import Parallel, delayed
import nibabel as nib

"""
This class was initially inspired by the following :
https://github.com/machow/pysearchlight
"""



def _get_searchlight_neighbors(mask, center, radius=3):
    """Return indices for searchlight where distance 
        between a voxel and their center < radius (in voxels)
    
    Args:
        center (index):  point around which to make searchlight sphere
    
    Returns:
        list: the list of volume indices that respect the 
                searchlight radius for the input center.  
    """
    center = np.array(center)
    mask_shape = mask.shape
    cx, cy, cz = np.array(center)
    x = np.arange(mask_shape[0])
    y = np.arange(mask_shape[1])
    z = np.arange(mask_shape[2])

    # First mask the obvious points
    # - may actually slow down your calculation depending.
    x = x[abs(x-cx) < radius]
    y = y[abs(y-cy) < radius]
    z = z[abs(z-cz) < radius]

    # Generate grid of points
    X, Y, Z = np.meshgrid(x, y, z)
    data = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
    distance = cdist(data, center.reshape(1, -1), 'euclidean').ravel()

    return tuple(data[distance < radius].T.tolist())

def get_volume_searchlight(mask, radius=2, threshold=1):
    """Searches through the non-zero voxels of the mask, selects centers where 
        proportion of sphere voxels >= self.threshold.

    Args:
        mask ([numpy array]): binary brain mask
        radius (int, optional): [description]. Defaults to 2.
        threshold (int, optional): [description]. Defaults to 1.

    Returns:
        [numpy array]: array of centers of size n_centers x 3
        [list]: list of lists with neighbors - the length of the list will correspond to:
                n_centers x 3 x n_neighbors
    """

    centers = list(zip(*np.nonzero(mask)))
    good_centers = []
    good_neighbors = []

    for center in tqdm(centers, desc='Finding searchlights...'):
        neighbors = _get_searchlight_neighbors(mask, center, radius)
        if mask[neighbors].mean() >= threshold:
            good_centers.append(center)
            good_neighbors.append(neighbors)

    good_centers = np.array(good_centers)
    assert good_centers.shape[0] == len(good_neighbors), "number of centers and sets of neighbors do not match"
    print(f'Found {len(good_neighbors)} searchlights')

    return good_centers, good_neighbors


if __name__ == '__main__':
    verbose = True
    data = np.load('/Users/daniel/Dropbox/amster/github/fmri_data/singe_trial_betas.npy')
    events = np.load('/Users/daniel/Dropbox/amster/github/fmri_data/singe_trial_events.npy')
    mask_img = nib.load('/Users/daniel/Dropbox/amster/github/fmri_data/sub-01_ses-01_task-WM_run-1_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz')
    mask = mask_img.get_fdata()

    centers, neighbors = get_volume_searchlight(mask, radius=3, threshold=.7)

    # turn the 3-dim coordinates to array coordinates
    centers_raveled = np.ravel_multi_index(centers.T, mask.shape)
    neighbors_raveled = [np.ravel_multi_index(n, mask.shape) for n in neighbors]

    # make sure the new array coordinates correspond to the old 3-dim coordinates
    # mask2 = mask.copy()
    # mask2[centers[0][0], centers[0][1], centers[0][2]] = 10
    # assert mask2.reshape(-1, 1)[centers_raveled[0]] == 10

    # flatten data
    dims = data.shape
    data_raveled = data.reshape(dims[0], -1)

    # loop over centers, make datasets
    from pyrsa.data.dataset import Dataset
    from pyrsa.rdm.calc import calc_rdm
    
    # we can't run all centers at once, that will take too much memory
    # so lets to some chunking
    n_centers = centers_raveled.shape[0]
    chunked_center = np.split(np.arange(n_centers),
                              np.linspace(0, n_centers,
                              100, dtype=int)[1:-1])
    
    if verbose:
        print(f'\nDivided data into {len(chunked_center)} chunks!\n')
    
    n_conds = len(np.unique(events))
    RDM = np.zeros((n_centers, n_conds * (n_conds-1) // 2))
    for chunk in tqdm(chunked_center, desc='Calculating RDMs...'):
        center_data = []
        for c in chunk:

            center = centers_raveled[c]
            nb = neighbors_raveled[c]

            ds = Dataset(data_raveled[:, nb],
                        descriptors={'center': c},
                        obs_descriptors={'events':events},
                        channel_descriptors={'voxels': nb})
            center_data.append(ds)

        RDM_corr = calc_rdm(center_data,method='correlation', descriptor='events')
        RDM[chunk, :] = RDM_corr.dissimilarities
    

    
    







