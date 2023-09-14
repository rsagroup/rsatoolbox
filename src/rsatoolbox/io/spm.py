"""Loading SPM (Statistical Parametric Mapping) fMRI data

SPM stores GLM results as one volume per condition. This module has functions to
load these beta images or their residuals as a single data object. This can then be stored
in a new format.

## Usage
```
betas = rsatoolbox.io.spm.load_betas('/dir/imgs/')
betas.save2combo('/dir/out')
betas.to_dataset() ## not implemented yet
```
"""
from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional
from os.path import join, normpath
from glob import glob
from rsatoolbox.io.optional import import_nibabel
from pandas import DataFrame
from numpy import stack
if TYPE_CHECKING:
    from numpy.typing import NDArray


def load_betas(path: str, stim_ids_dict=None) -> SpmGlm:
    glm = SpmGlm(path)
    glm.load_betas(stim_ids_dict)
    return glm


def load_residuals(path: str, res_range=None) -> SpmGlm:
    glm = SpmGlm(path)
    glm.load_betas(res_range)
    return glm


class SpmGlm:

    path: str
    pooled_data_array: NDArray ## conds * x * y * z
    dim4_descriptors: List[str]
    affine: NDArray

    def __init__(self, path: str):
        self.path = normpath(path)
        self.nibabel = import_nibabel()

    def load_betas(self, stim_ids_dict=None):
        """
        Collects 3d images of prespecified beta coefficients
        (typical SPM GLM results) and corresponding metadata
        (condition + run info) into respective lists

        Args:
            run_dirs (list of str):
                paths to directories containing beta NIfTi files
            stim_ids_dict (dict): {condition : beta coefficient number}
                e.g. {'face': 1, 'house': 2}

        Returns:
            pooled_beta_array (4d numpy array):
                all beta 3d arrays for the keys in stim_ids_dict found in each
                run directory stacked along the fourth dimension
            dim4_descriptors (list of str):
                corresponding descriptors
                e.g. ['cond_face_run_05', 'cond_house_run_30']
        """

        if stim_ids_dict is None:
            n_conds = len(glob(join(self.path, "beta*")))
            keys = [str(cond_num).zfill(4) for cond_num in range(1, n_conds+1)]
            values = [cond_num for cond_num in range(1, n_conds+1)]
            stim_ids_dict = dict(zip(keys, values))

        beta_array_superset = []
        dim4_descriptors = []
        run_counter = 0

        run_counter += 1
        for condition in stim_ids_dict.keys():
            num = stim_ids_dict[condition]
            beta_image_path = join(self.path, "beta_" + str(num).zfill(4))
            beta_image = self.nibabel.nifti1.load(beta_image_path)
            beta_array_superset.append(beta_image.get_fdata())
            dim4_descriptors.append("cond_" + condition + "_run_" +
                                    str(run_counter).zfill(2))
        # Get affine matrix
        self.subject_affine = beta_image.affine.copy()
        pooled_beta_array = stack(beta_array_superset, axis=3)
        return pooled_beta_array, dim4_descriptors

    def load_residuals(self, res_range=None):
        """
        Collects 3d images of a range of GLM residuals
        (typical SPM GLM results) and corresponding metadata
        (scan number + run info) into respective lists

        Args:
            run_dirs (list of str):
                paths to directories containing beta NIfTi files
            res_range (range): range of to be saved residual images per run

        Returns:
            residual_array_superset (list of 3d numpy arrays):
                all residual 3d arrays for scans in res_range
            dim4_descriptors (list of str):
                corresponding descriptors
                e.g. ['res_0001_run_01', 'res_0002_run_01']
        """

        residual_array_superset = []
        dim4_descriptors = []
        self.n_res = len(glob(join(self.path, "Res_*")))
        if isinstance(res_range, range):
            assert self.n_res >= max(res_range), \
                "res_range outside of existing residuals"
        else:
            res_range = range(1, self.n_res+1)

        run_counter = 0

        run_counter += 1
        for res in res_range:
            res_image_path = join(self.path, "Res_" +
                                            str(res).zfill(4))
            res_image = self.nibabel.nifti1.load(res_image_path)
            residual_array_superset.append(res_image.get_fdata())
            dim4_descriptors.append("res_" + str(res).zfill(4) + "_run_" +
                                    str(run_counter).zfill(2))
        # Get affine matrix
        self.subject_affine = res_image.affine.copy()
        pooled_residual_array = stack(residual_array_superset, axis=3)
        return pooled_residual_array, dim4_descriptors

    def save2nifti(self, fpath: Optional[str]=None):
        """
        Converts 4d array to subject-specific 4d NIfTi image
        of beta coeffients or residuals and saves it to your OS

        Args:
            pooled_data_array (4d numpy array):
                all 3d arrays stacked along the fourth dimension
            output_dir (str):
                path to which you want to save your data
            data_type (str):
                part of the naming scheme for the saved data

        """
        fpath = fpath or (self.path + '.nii.gz')
        pooled_data = self.nibabel.nifti1.Nifti1Image(
            self.pooled_data_array,
            self.affine
        )
        self.nifti_filename = join(fpath)
        self.nibabel.nifti1.save(pooled_data, self.nifti_filename)

    def save2csv(self, fpath: Optional[str]=None):
        """
        Saves subject-specific 4d NIfTi image descriptors to a csv file

        Args:
            descriptors (list of str):
                descriptors of fourth a NIfTi file's 4th dimension
            output_dir (str):
                path to which you want to save your data
            data_type (str):
                part of the naming scheme for the saved data
        """
        fpath = fpath or (self.path + '.csv')
        df = DataFrame({'descriptor': self.dim4_descriptors})
        df.to_csv(fpath, header=False)

    def save2combo(self, fpath: Optional[str]=None):
        """
        Combined saving of fmri data and descriptors
        """
        fpath_niigz = fpath or (self.path + '.nii.gz')
        fpath_csv = fpath_niigz.replace('.nii.gz', '.csv')
        self.save2nifti(fpath_niigz)
        self.save2csv(fpath_csv)
