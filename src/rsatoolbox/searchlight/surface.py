"""

Notes Jasper:
- prep_vol .. -> for volume data only, supposed to be faster than DL implementation in volume.py
- prep_surf .. -> does same but for surface
"""
import warnings
import os
from collections.abc import Iterable
from copy import deepcopy
from glob import glob
import numpy as np

from joblib import Parallel, delayed, cpu_count
from sklearn.exceptions import ConvergenceWarning
from sklearn import neighbors
from nilearn import surface
import rsatoolbox.data as rsd
import rsatoolbox.rdm as rsr

from scipy.spatial.distance import cdist
from itertools import compress


class GroupIterator():
    """Group iterator. cf. nilearn.
    Provides group of features for search_light loop
    that may be used with Parallel.
    Parameters
    ----------
    n_features : int
        Total number of features
    %(n_jobs)s
    """
    def __init__(self, n_features, n_jobs=1):
        self.n_features = n_features
        if n_jobs == -1:
            n_jobs = cpu_count()
        self.n_jobs = n_jobs

    def __iter__(self):
        split = np.array_split(np.arange(self.n_features), self.n_jobs)
        for list_i in split:
            yield list_i

def fetch_surf(fspath, targetspace):
    """ find a subject's native surface files

    Args:
        fspath (os.path): the absolute path pointing
                        to the fs subjects directory
        targetspace (string): targetspace of the data preparation
                        e.g. 'fsaverage' if the data is prepared on
                        the fsaverage common space, 'subject01' if
                        the data is prepared on subject01's native
                        surface space.

    Returns:
        surface (list): list of absolute paths for
                        fs subjects inflated left and
                        right surfaces.
    """

    # if usual recon, surfaces are .inflated format
    infl_list = glob(os.path.join(fspath, targetspace, 'surf', '*.inflated'))
    infl_list.sort()

    surfaces= {
        'infl_left': infl_list[0],
        'infl_right': infl_list[1]
    }

    return surfaces



def prepare_surf_indices(fspath, targetspace, radius):
    """prepare searchlight indices to be used to
       sample searchlights from fMRI betas prepared
       in surface space.

    Args:
        fspath (os.path): the path where your freesurfer subject's
                          live. e.g. /opt/freesurfer/subjects
        targetspace (string): what surface space are your betas
                              prepared in? e.g. 'fsaverage', 'subject01'
        radius (int): what radius do you want the searchlight 'spheres'
                      to cover?

    Returns:
        sl_indices: list of searchlight indices for every vertex left
                    and right. This list can then be used to run
                    searchlight RSA, indexing the surface prepared betas
    """

    surf = fetch_surf(fspath, targetspace)

    hemis = ['left', 'right']

    sl_indices = []

    for hemi in hemis:

        # we piggy back on nilearn to get inflated coordinates
        infl_mesh = surf['infl_' + hemi]
        coords, _ = surface.load_surf_mesh(infl_mesh)

        # prepare the nearest neighbours algo
        nn = neighbors.NearestNeighbors(radius=radius)

        # get the list of vertex indices using nearest neighbour
        adjacency = nn.fit(coords).radius_neighbors_graph(coords).tolil()

        # append lists of indices for both hemispheres
        sl_indices.append(adjacency)

    return sl_indices


def _get_searchlight_neighbors(mask, centers, radius=3):
    """Return indices for searchlight where distance
        between a voxel and their center < radius (in voxels)
    Args:
        center (index):  point around which to make searchlight sphere
    Returns:
        list: the list of volume indices that respect the
                searchlight radius for the input center.
    """
    all_indices = []
    for center in centers:
        mask_shape = mask.shape
        cx, cy, cz = center
        x = np.arange(mask_shape[0])
        y = np.arange(mask_shape[1])
        z = np.arange(mask_shape[2])

        # First mask the obvious points
        # - may actually slow down your calculation depending.
        x = x[abs(x - cx) < radius]
        y = y[abs(y - cy) < radius]
        z = z[abs(z - cz) < radius]

        # Generate grid of points
        X, Y, Z = np.meshgrid(x, y, z)
        data = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
        distance = cdist(data, center.reshape(1, -1), 'euclidean').ravel()

        indices = data[distance < radius, :]

        all_indices.append(indices)

    return all_indices


def prepare_vol_indices(mask, radius, threshold, n_jobs=-1, verbose=0):

    # find the mask centers
    centers = np.asarray(list(zip(*np.nonzero(mask))))

    group_iter = GroupIterator(len(centers), n_jobs)

    # get the list of spherical neighbor volume indices
    with warnings.catch_warnings():  # might not converge
        warnings.simplefilter('ignore', ConvergenceWarning)
        neighbors = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_get_searchlight_neighbors)(
                mask,
                centers[list_i, :],
                radius)
            for list_i in group_iter)
    # unpack
    neighbors_list = [item for sublist in neighbors for item in sublist]

    # janitor
    neighbors = neighbors_list
    
    # turn the 3-dim coordinates to array coordinates
    centers = np.ravel_multi_index(centers.T, mask.shape)
    neighbors = [np.ravel_multi_index(n.T, mask.shape) for n in neighbors]

    mask = mask.flatten()

    # now we only keep those spheres that have at least threshold% brain.
    good = [mask[n].mean() >=threshold for n in neighbors]

    good_centers = list(compress(centers, good))

    good_centers = np.array(good_centers)

    good_neighbors = list(compress(neighbors, good))

    assert good_centers.shape[0] == len(good_neighbors),\
        "number of centers and sets of neighbors do not match"
    print(f'Found {len(good_neighbors)} searchlights')

    return good_centers, good_neighbors


def compute_searchlight_rdms(
    indices,
    betas,
    des,
    obs_des,
    method='correlation',
    cv_descriptor=None,
    prior_lambda=1,
    prior_weight=0.1,
    noise=None,
    n_jobs=-1,
    verbose=0
    ):
    """compute searchlight RDMs takes a list of indices
       and maps the betas to compute an RDM for each
       searchlight of surface vertices.

    Args:
        indices (_type_): list of searchliht indices
                    (see prep_surf_indices)
        betas (_type_): betas in shape n_vertices by n_conditions
        des : participant and session details
        obs_des: conditions dictionary e.g. {'conds': 'cond_0',...}
        method: metric for constructing rdm
        for a full description of the arguments, refer to calc_rdm.
        n_jobs (int, optional): number of cpus available.
                    Defaults to -1 (find number of cpus automatically).
        verbose (int, optional): level of shouting. Defaults to 0.

    Returns:
        array: searchlight rdms in the shape
               n_vertices x n_pairwise_comparisons
    """
    # first deal with making datasets for the searchlights
    data = []
    for ind in indices:
        chan_des = {'chans': np.array(['chan_' + str(x) for x in ind])}

        data.append(
            rsd.Dataset(
                measurements=betas[ind, :].T,
                descriptors=des,
                obs_descriptors=obs_des,
                channel_descriptors=chan_des,
            )
        )

    # next we call calc_rdm. we use joblib parallel
    # to distribute it if multiple cpus are present.
    # this might be memory intense dependent on the
    # number of conditions.
    group_iter = GroupIterator(len(data), n_jobs)
    with warnings.catch_warnings():  # might not converge
        warnings.simplefilter('ignore', ConvergenceWarning)
        if noise is None:
            rdms = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(calc_rdm_batch)(
                    np.array(data)[list_i],
                    method=method,
                    descriptor='conds',
                    cv_descriptor=cv_descriptor,
                    prior_lambda=prior_lambda,
                    prior_weight=prior_weight)
                for list_i in group_iter)
        elif isinstance(noise, np.ndarray) and noise.ndim == 2:
            rdms = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(calc_rdm_batch)(
                    np.array(data)[list_i],
                    method=method,
                    descriptor='conds',
                    noise=noise,
                    cv_descriptor=cv_descriptor,
                    prior_lambda=prior_lambda,
                    prior_weight=prior_weight)
                for list_i in group_iter)
        elif isinstance(noise, Iterable):
            rdms = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(calc_rdm_batch)(
                    np.array(data)[list_i],
                    method=method,
                    descriptor='conds',
                    noise=np.array(noise)[list_i],
                    cv_descriptor=cv_descriptor,
                    prior_lambda=prior_lambda,
                    prior_weight=prior_weight)
                for list_i in group_iter)

        # collect rdms in one array
        rdms = np.concatenate(rdms)

        # repack to list of RDMs object with descriptors
        # and chan descriptors.
        # the descriptor here becomes the index of the
        # centre of sphere related to the spherical indices
        rdm_objects = [
            rsr.RDMs(
                dissimilarities=x,
                dissimilarity_measure=method,
                descriptors=des,
                rdm_descriptors=deepcopy(data[y].descriptors),
                pattern_descriptors=obs_des)
                for y, x in enumerate(rdms)
        ]

    return rdm_objects


def calc_rdm_batch(
    data_batch,
    method='correlation',
    descriptor='conds',
    cv_descriptor=None,
    prior_lambda=1,
    prior_weight=0.1,
    noise=None
    ):
    """ calc rdm batch

    Args:
        data_batch (list): list of rsa datasets
        method (str, optional): metric to use. Defaults to 'correlation'.
        descriptor (dict, optional): key in the dataset descriptors object.
                                     Defaults to conds.
        noise (numpy.ndarray or list):
            dataset.n_channel x dataset.n_channel
            precision matrix used to calculate the RDM
            used only for Mahalanobis and Crossnobis estimators
            defaults to an identity matrix, i.e. euclidean distance
        cv_descriptor (string, optional):
            obs_descriptor which determines the cross-validation folds.
            Defaults to None.
        prior_lambda (int, optional):
            prior lambda used in symmetrized KL-divergence. Defaults to 1.
        prior_weight (float, optional):
            prior weight used in symmetrised KL-divergence. Defaults to 0.1.

    Returns:
        rdms (numpy.ndarray): dissimilarities for the batch of datasets.
    """

    rdms = []
    for data in data_batch:
        rdm = rsr.calc_rdm(
            data,
            method=method,
            descriptor=descriptor,
            noise=noise,
            cv_descriptor=cv_descriptor,
            prior_lambda=prior_lambda,
            prior_weight=prior_weight)
        rdms.append(rdm.dissimilarities)
    
    return np.concatenate(rdms)
