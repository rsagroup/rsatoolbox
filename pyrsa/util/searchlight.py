import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from joblib import Parallel, delayed
import nibabel as nib

"""
This class was initially inspired by the following :
https://github.com/machow/pysearchlight
"""

data = np.load('/Users/daniel/Dropbox/amster/github/fmri_data/singe_trial_betas.npy')
events = np.load('/Users/daniel/Dropbox/amster/github/fmri_data/singe_trial_events.npy')
mask_img = nib.load('/Users/daniel/Dropbox/amster/github/fmri_data/sub-01_ses-01_task-WM_run-1_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz')
mask = mask_img.get_fdata()



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

    return data[distance < radius].T.tolist()

def get_volume_searchLight(mask, radius=2, threshold=1):
    """Searches through the non-zero voxels of the mask, selects centers where 
        proportion of sphere voxels >= self.threshold.

    Args:
        mask ([numpy array]): [description]
        radius (int, optional): [description]. Defaults to 2.
        threshold (int, optional): [description]. Defaults to 1.

    Returns:
        [numpy array]: array of centers of size n_centers x 3
        [list]: list of lists with neighbors - the length of the list will correspond to:
                len(good_neighbors) = n_centers
                len(good_neighbors[0]) = 3
                len(good_neighbors[0][0]) = n_neighbors for 
    """

    centers = list(zip(*np.nonzero(mask)))
    good_centers = []
    good_neighbors = []

    for center in tqdm(centers, desc='Finding searchlights...'):
        neighbors = _get_searchlight_neighbors(mask, center, radius)
        if mask[neighbors].mean() >= threshold:
            good_centers.append(center)
            good_neighbors.append(neighbors)

    assert good_centers.shape[0] == len(good_neighbors), "number of centers and sets of neighbors do not match"
    print(f'Found {len(good_neighbors)} searchlights')

    return np.array(good_centers), good_neighbors

    

class VolumeSearchLight():
    def __init__(self, mask, radius=2, threshold=.7, njobs=1, verbose=False):
        """[summary]
        
        Args:
            mask 3D numpy array):   typically imported with nib.load('yourmask.nii').get_data()
                                    3d spatial mask (boolean where brain voxels are set to 1)
            radius (int, optional): radius for spheres around each center (in voxels). 
                                    Defaults to 2.
            threshold (float, optional): proportion of brain voxels within a sphere.
                                    threshold = 1 means we don't accept centers with voxels outside
                                    the brain. Defaults to .7.
            njobs (int, optional): number of cores to distribute the fitting procedure to. Defaults to 1.
            verbose (bool, optional): turn tqdm on. Defaults to False. This reports progress in the 
                                    object initialisation and in fit_rsa(). 
        """
        self.verbose = verbose
        self.mask = mask
        self.njobs = njobs
        self.radius = radius
        self.threshold = threshold
        if self.verbose:
            print('finding centers')
        self.centers = self._findCenters()
        if self.verbose:
            print('finding center indices')
        self.centerIndices = self._findCenterIndices()
        if self.verbose:
            print('finding all sphere indices')
        self.allIndices = self._allSphereIndices()
        self.RDM = None
        self.NaNs = []

    def _findCenters(self):
        """ Find the centers whose 
            proportion of sphere voxels >= self.threshold.
        
        Returns:
            numpy array: valid centers (xyz tuple coordinates)
        """
        # make centers a list of 3-tuple coords
        centers = zip(*np.nonzero(self.mask))
        good_center = []
        for center in centers:
            ind = self.searchlightInd(center)
            if self.mask[ind].mean() >= self.threshold:
                good_center.append(center)
        return np.array(good_center)

    def _findCenterIndices(self):
        """find the subspace indices for the centers 
        
        Returns:
            numpy array: center indices in volume subspace.
        """
        centerIndices = []
        dims = self.mask.shape
        for i, cen in enumerate(self.centers):
            n_done = i/len(self.centers)*100
            if i % 50 == 0 and self.verbose is True:
                print('Converting voxel coordinates of centers to subspace'
                      f'indices {n_done:.0f}% done!', end='\r')
            centerIndices.append(np.ravel_multi_index(np.array(cen), dims))
        print('\n')
        return np.array(centerIndices)

    def _allSphereIndices(self):
        """module for gathering searchlight sphere indices.
           
        Returns:
            SL.allIndices: list of searchlight indices for all centers.
                           If this could be parallelised, we would gain even more speed.  
        """
        allIndices = []
        dims = self.mask.shape
        for i, cen in enumerate(self.centers):
            n_done = i/len(self.centers)*100
            if i % 50 == 0 and self.verbose is True:
                print(f'Finding SearchLights {n_done:.0f}% done!', end='\r')

            # Get indices from center
            ind = np.array(self.searchlightInd(cen))           
            allIndices.append(np.ravel_multi_index(np.array(ind), dims))
        print('\n')
        return allIndices

    def searchlightInd(self, center):
        """Return indices for searchlight where distance 
           between a voxel and their center < radius (in voxels)
        
        Args:
            center (index):  point around which to make searchlight sphere
        
        Returns:
            list: the list of volume indices that respect the 
                    searchlight radius for the input center.  
        """
        center = np.array(center)
        mask_shape = self.mask.shape
        cx, cy, cz = np.array(center)
        x = np.arange(mask_shape[0])
        y = np.arange(mask_shape[1])
        z = np.arange(mask_shape[2])

        # First mask the obvious points
        # - may actually slow down your calculation depending.
        x = x[abs(x-cx) < self.radius]
        y = y[abs(y-cy) < self.radius]
        z = z[abs(z-cz) < self.radius]

        # Generate grid of points
        X, Y, Z = np.meshgrid(x, y, z)
        data = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
        distance = cdist(data, center.reshape(1, -1), 'euclidean').ravel()

        return data[distance < self.radius].T.tolist()

    def checkNaNs(self, X):
        """
        TODO - this function
        """
        pass
        # nans = np.all(np.isnan(X), axis=0)[0]
        # return X[:,~nans]

    def fit_rsa(self, data, wantreshape=True):
        """ Fit RDMs in searchlight spheres
            
            Args:
                data 4D numpy array: (x, y, z, condition vols)
                wantreshape (bool, optional): Defaults to True. 
                                              whether you want the data back in
                                              its volume form, or 2d array of 
                                              distances for every center. 
                
            Return SL model, with RDMs for every center.
        """
        print('Running searchlight RSA')

        # reshape the data to squish the first three dimensions
        x, y, z, nobjects = data.shape

        # now the first dimension of data is directly indexable by
        # subspace index of the searchlight centers
        data = data.reshape((x*y*z, nobjects))

        # test get_distance()
        # for x in self.allIndices:
        #    t = get_distance(data, x)
        # test passed.

        # brain = np.zeros((x, y, z, rdm_size, rdm_size))
        if wantreshape:
            if self.verbose is True:
                distances = Parallel(n_jobs=self.njobs)(
                    delayed(get_rdm)(
                        data, x) for x in tqdm(self.allIndices))
            else:
                distances = Parallel(n_jobs=self.njobs)(
                    delayed(get_rdm)(
                        data, x) for x in self.allIndices)
            distances = np.asarray(distances)
            
            # number of pairwise comparisons
            n_combs = nobjects*(nobjects-1) // 2
            self.RDM = np.zeros((x*y*z, n_combs)).astype(np.float32)
            self.RDM[list(self.centerIndices), :] = distances
            self.RDM = self.RDM.reshape((x, y, z, n_combs))
        else:
            if self.verbose is True:
                self.RDM = Parallel(n_jobs=self.njobs)(
                    delayed(get_rdm)(
                        data, x) for x in tqdm(self.allIndices))
            else:
                self.RDM = Parallel(n_jobs=self.njobs)(
                    delayed(get_rdm)(
                        data, x) for x in self.allIndices)
                self.RDM = np.asarray(self.RDM)