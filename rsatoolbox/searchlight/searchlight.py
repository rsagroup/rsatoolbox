
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
import rsatoolbox.data as rsd
import rsatoolbox.rdm as rsr


class GroupIterator(object):
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

        # append lists of indices for both hemispheres
        sl_indices.append(adjacency)

    return sl_indices


def compute_searchlight_rdms(
    indices,
    betas,
    des,
    obs_des,
    method='correlation',  cv_descriptor=None, prior_lambda=1,
    prior_weight=0.1, noise=None, n_jobs=-1, verbose=0):
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
    for ind in indices.rows:
        chan_des = {'verts': np.array(['vert_' + str(x) for x in ind])}

        data.append(
            rsd.Dataset(
                measurements=betas[ind, :].T,
                descriptors=des,
                obs_descriptors=obs_des,
                channel_descriptors=chan_des
                )
            )

    # next we call calc_rdm. we use joblib parallel
    # to distribute it if multiple cpus. this might be memory
    # intense dependent on the number of conditions. 
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
    
    # TODO: repack to list of RDMs object with descriptors 
    # and chan descriptors. for now rdms is a n_vertices
    # by n_pairwise_comparisons dissimilarities array

    return np.concatenate(rdms)


def calc_rdm_batch(
    data_batch,     
    method='correlation', descriptor=None, cv_descriptor=None, prior_lambda=1,
    prior_weight=0.1, noise=None, n_jobs=-1, verbose=0):
    """ calc rdm batch

    Args:
        data_batch (_type_): _description_
        method (str, optional): _description_. Defaults to 'correlation'.
        descriptor (_type_, optional): _description_. Defaults to None.
        cv_descriptor (_type_, optional): _description_. Defaults to None.
        prior_lambda (int, optional): _description_. Defaults to 1.
        prior_weight (float, optional): _description_. Defaults to 0.1.
        noise (_type_, optional): _description_. Defaults to None.
        n_jobs (int, optional): _description_. Defaults to -1.
        verbose (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """

    rdms = []
    for data in data_batch:
        rdm = rsr.calc_rdm(
                    data,
                    method=method, 
                    descriptor=descriptor,
                    cv_descriptor=cv_descriptor,
                    prior_lambda=prior_lambda,
                    prior_weight=prior_weight)
        rdms.append(rdm.dissimilarities)
    
    return np.concatenate(rdms)
