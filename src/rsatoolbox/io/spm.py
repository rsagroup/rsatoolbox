"""Handeling SPM (Statistical Parametric Mapping) fMRI data

Utility object that helps to extract time series data, beta coefficients, and residuals from a GLM stored in a SPM.mat file.

## Usage
```
spm = SpmGlm('path/to/spm')
spm.get_info_from_spm_mat()
[residuals, beta, info] = spm.get_residuals('my_ROI_Mask.nii')
```
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Dict
from os.path import normpath, dirname
import numpy as np
from scipy.io import loadmat
from rsatoolbox.io.optional import import_nitools
if TYPE_CHECKING:
    from numpy.typing import NDArray
    from nibabel.nifti1 import Nifti1Image


class SpmGlm:
    """class for handling first-levels GLMs estimated in SPM

    Attributes:
        path (str):
            paths to directory containing SPM files
    """

    def __init__(self, path: str, nitoolsMock=None):
        self.path = normpath(path)
        self.nt = import_nitools(nitoolsMock)

    def get_info_from_spm_mat(self) -> None:
        """Initializes information for SPM.mat file

        Args:
            spm_mat_path (str): _description_
        """
        SPM = loadmat(f"{self.path}/SPM.mat", simplify_cells=True)['SPM']
        # Get basic information from SPM.mat
        self.nscans = SPM['nscan']
        self.nruns = len(self.nscans)
        # Get the name and information on all the beta files``
        self.beta_files = [v['fname'] for v in SPM['Vbeta']]
        self.beta_names = []
        self.run_number = []
        # Extract run number and condition name from SPM names
        for reg_name in SPM['xX']['name']:
            s = reg_name.split(' ')
            self.run_number.append(int(s[0][3:-1]))
            self.beta_names.append(s[1])
        self.run_number = np.array(self.run_number)
        self.beta_names = np.array(self.beta_names)
        # Get the raw data file name
        self.rawdata_files = [self.relocate_file(p) for p in SPM['xY']['P']]
        # Get the necesssary matrices to reestimate the GLM for getting the residuals
        self.filter_matrices = [k['X0'] for k in SPM['xX']['K']]
        self.reg_of_interest = SPM['xX']['iC']
        self.design_matrix = SPM['xX']['xKXs']['X']  # Filtered and whitened design matrix
        self.eff_df = SPM['xX']['erdf']  # Effective degrees of freedom
        self.weight = SPM['xX']['W']  # Weight matrix for whitening
        self.pinvX = SPM['xX']['pKX']  # Pseudo-inverse of (filtered and weighted) design matrix

    def relocate_file(self, fpath: str) -> str:
        """SPM file entries to current project directory and OS.

        These are file paths suffixed with two spaces and an index number.
        They are stored as absolute paths on the computer where it is
        generated, meaning the project directory, as well as OS-dependent
        path separator may have changed. This method fixes these.

        Args:
            fpath (str): SPM-style file path and number from unknown OS

        Returns:
            str: SPM-style file path
        """
        norm_fpath = fpath.replace('\\', '/')
        base_path = dirname(self.path).replace('\\', '/')
        c = norm_fpath.find('func')
        return base_path + '/' + norm_fpath[c:]

    def get_betas(self, mask: Nifti1Image | NDArray | str) -> Tuple[NDArray, NDArray, Dict]:
        """
        Samples the beta images of an estimated SPM GLM at the mask locations
        also returns the ResMS values, and the obseration descriptors (run and condition) name

        Args:
            mask (ndarray or nibabel nifti1image):
                Indicates which voxels to extract
                Could be a binary 3d-array or a nifti image of the same size as the data
                Or a 3xP array of coordinates to extract (in mm space)
        Returns:
            data (ndarray): N x P array of beta coefficients
            resms (ndarray): 1d array of ResMS values
            obs_descriptors (dict): with lists reg_name and run_number (N long)
        """

        coords = self.nt.get_mask_coords(mask)

        # Generate the list of relevant beta images:
        indx = self.reg_of_interest-1
        beta_files = [f'{self.path}/{self.beta_files[i]}' for i in indx]
        # Get the data from beta and ResMS files
        rms_file = [f'{self.path}/ResMS.nii']
        data = self.nt.sample_images(beta_files + rms_file, coords, use_dataobj=False)
        # Return the data and the observation descriptors
        info = {'reg_name': self.beta_names[indx], 'run_number': self.run_number[indx]}
        return data[:-1, :], data[-1, :], info

    def get_residuals(self, mask: Nifti1Image | NDArray | str) -> Tuple[NDArray, NDArray, Dict]:
        """
        Collects 3d images of a range of GLM residuals
        (typical SPM GLM results) and corresponding metadata
        (scan number + run info) into respective lists

        Args:
            res_range (range): range of to be saved residual images per run
        """
        # Sample the relevant time series data
        coords = self.nt.get_mask_coords(mask)
        data = self.nt.sample_images(self.rawdata_files, coords, use_dataobj=True)

        # Filter and temporal pre-whiten the data
        fdata = self.spm_filter(self.weight @ data)  # spm_filter

        # Estimate the beta coefficients abd residuals
        beta = self.pinvX @ fdata
        residuals = fdata - self.design_matrix @ beta

        # Return the regressors of interest
        indx = self.reg_of_interest-1
        info = {'reg_name': self.beta_names[indx], 'run_number': self.run_number[indx]}
        return residuals, beta[indx, :], info

    def spm_filter(self, data: NDArray) -> NDArray:
        """
        Does high pass-filtering and temporal weighting of the data (indentical to spm_filter)

        Args:
            data (ndarray): 2d array of time series data (TxP)
        Returns:
            data (ndarray): 2d array of time series data (TxP)
        """
        scan_bounds = self.nscans.cumsum()
        scan_bounds = np.insert(scan_bounds, 0, 0)

        fdata = data.copy()
        for i in range(self.nruns):
            Y = fdata[scan_bounds[i]:scan_bounds[i+1], :]
            Y = Y - self.filter_matrices[i] @ (self.filter_matrices[i].T @ Y)
        return fdata
