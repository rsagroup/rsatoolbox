import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from joblib import Parallel, delayed
import nibabel as nib
from pyrsa.data.dataset import Dataset
from pyrsa.rdm.calc import calc_rdm
from pyrsa.rdm import RDMs

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

    mask = np.array(mask)
    assert mask.ndim == 3, "Mask needs to be a 3-dimensional numpy array"

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

    # turn the 3-dim coordinates to array coordinates
    centers_raveled = np.ravel_multi_index(good_centers.T, mask.shape)
    neighbors_raveled = [np.ravel_multi_index(n, mask.shape) for n in good_neighbors]

    return centers_raveled, neighbors_raveled

def get_searchlight_RDMs(data_raveled, centers_raveled, neighbors_raveled, events,
                        method='correlation', verbose=True):

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

        RDM_corr = calc_rdm(center_data, method=method, descriptor='events')
        RDM[chunk, :] = RDM_corr.dissimilarities
    


    model_rdms = RDMs(RDM,
                      rdm_descriptors={'voxel_index':centers_raveled},
                      dissimilarity_measure=method)


    return model_rdms



if __name__ == '__main__':
    verbose = True
    data = np.load('/Users/daniel/Dropbox/amster/github/fmri_data/singe_trial_betas.npy')
    events = np.load('/Users/daniel/Dropbox/amster/github/fmri_data/singe_trial_events.npy')
    mask_img = nib.load('/Users/daniel/Dropbox/amster/github/fmri_data/sub-01_ses-01_task-WM_run-1_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz')
    mask = mask_img.get_fdata()

    centers_raveled, neighbors_raveled = get_volume_searchlight(mask, radius=3, threshold=.7)

    # flatten data
    data_raveled = data.reshape([data.shape[0], -1])

    RDM = get_searchlight_RDMs(data_raveled, centers_raveled, neighbors_raveled, events)

    best_rdms = RDM.subset('voxel_index', centers_raveled[36309])

    plt.figure(figsize=(10,10))
    pyrsa.vis.show_rdm(best_rdms, do_rank_transform=True)

    x, y, z = mask.shape
    n_conds = len(np.unique(events))
    n_comparisons = n_conds * (n_conds-1) // 2
    RDM_brain = np.zeros([x*y*z, n_comparisons])
    RDM_brain[list(centers_raveled), :] = RDM.dissimilarities
    RDM_brain = RDM_brain.reshape([x, y, z, n_comparisons])

    # make sure the new array coordinates correspond to the old 3-dim coordinates
    # mask2 = mask.copy()
    # mask2[centers[0][0], centers[0][1], centers[0][2]] = 10
    # assert mask2.reshape(-1, 1)[centers_raveled[0]] == 10

    # flatten data
    dims = data.shape
    data_raveled = data.reshape(dims[0], -1)

    # loop over centers, make datasets
    
    
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
    

    
    







