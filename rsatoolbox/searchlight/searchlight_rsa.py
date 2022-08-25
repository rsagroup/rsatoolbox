
import warnings
import numpy as np

from joblib import Parallel, delayed, cpu_count
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.optimize import nnls
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold
from sklearn import neighbors
from nilearn import datasets, surface

#def searchlight_rsa(targetspace, radius, betas,  metrix='correlation', n_jobs=-1, verbose=0):

    # prepare surf_indices


    # compute_brain_rdms

#    return brain_rdms

def prepare_surf_indices(targetspace, radius):
    """prepare searchlight indices to be used to
       sample searchlights from fMRI betas prepared
       in surface space.

    Args:
        targetspace (string): what surface space are your betas
                              prepared in? e.g. 'fsaverage'
        radius (int): what radius do you want the searchlight 'spheres'
                      to cover?

    Returns:
        sl_indices: list of searchlight indices for every vertex left
                    and right. This list can then be used to run
                    searchlight RSA, indexing the surface prepared betas
    """

    fsaverage = datasets.fetch_surf_fsaverage(mesh=targetspace)

    hemis = ['left', 'right']

    sl_indices = []

    for hemi in hemis:

        # we piggy back on nilearn to get inflated coordinates
        infl_mesh = fsaverage['infl_' + hemi]
        coords, _ = surface.load_surf_mesh(infl_mesh)

        # prepare the nearest neighbours algo
        nn = neighbors.NearestNeighbors(radius=radius)

        # get the list of vertex indices using nearest neighbour
        adjacency = nn.fit(coords).radius_neighbors_graph(coords).tolil()

        # extend lists of indices for both hemispheres
        sl_indices.append(adjacency)

    return sl_indices


def compute_searchlight_rdms(A, X, n_jobs=-1, verbose=0):
    """compute searchlight RDMs takes a list of indices
       and maps the betas to compute an RDM for each
       searchlight of surface vertices. 

    Args:
        A (_type_): list of searchliht indices 
                    (see prep_surf_indices)
        X (_type_): betas in shape n_vertices by n_conditions
        n_jobs (int, optional): number of cpus available. 
                    Defaults to -1 (find number of cpus automatically).
        verbose (int, optional): level of shouting. Defaults to 0.

    Returns:
        array: searchlight rdms in the shape 
               n_vertices x n_pairwise_comparisons
    """

    group_iter = GroupIterator(A.shape[0], n_jobs)
    with warnings.catch_warnings():  # might not converge
        warnings.simplefilter('ignore', ConvergenceWarning)
        rdms = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(get_distance)(
                A.rows[list_i],
                X)
            for list_i in group_iter)
    return np.concatenate(rdms)


class GroupIterator(object):
    """Group iterator
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


def get_distance(list_rows, X):

    """get_distance returns the correlation distance
       across condition patterns in X
       get_distance uses numpy's einsum

    Args:
        list_rows : array of arrays of int
        adjacency rows. For a voxel with index i in X, list_rows[i] is the list
        of neighboring voxels indices (in X).

        X : array-like of shape at least 2D
        data to fit.

    Returns:
        par_rdms: pairwise distances between condition patterns in X
             (in upper triangular vector form) for the list in list_rows
    """
    n_items = X.shape[1]
    n_comparisons = (n_items*(n_items-1))/2

    par_rdms = np.zeros((len(list_rows), int(n_comparisons)))

    for i, row in enumerate(list_rows):
        ind = np.array(row)
        Xi = np.array(X[ind, :]).T
        par_rdms[i, :] = pdist(Xi, metric='correlation')

    return par_rdms