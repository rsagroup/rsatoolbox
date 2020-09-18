#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parsing and importing of fMRI data on BIDS format
"""

import os
import glob
import numpy as np
import pandas as pd
from nibabel import nifti1



class BidsDerivatives:
    """
    BidsDerivatives class.
    Parses a specified subdirectory in the derivatives folder
    of a dataset in BIDS format.

    Args:
        fmri_dir (str): path to derivatives subdirectory
        subject_list (list): names of contained subject folders 
        sessions (list): names of unique sessions
        run_dirs (list): paths to all run folders containing e.g. GLM results

    Returns:
        BidsDerivatives object
    """
    
    def __init__(self, fmri_dir, subject_list = None, sessions = None,
                 run_dirs = None):
        """
        Parses BIDS structured data in fmri_dir
        or inherits said info after subsetting
        """
        assert os.path.exists(fmri_dir), "Specified directory does not exist."
        self.fmri_dir = fmri_dir
        
        # Level 1: Subjects
        if isinstance(subject_list, list):
            self.subject_list = subject_list
            self.sub_dirs = flatten_list(
                [glob.glob(self.fmri_dir + os.sep + sub)
                 for sub in subject_list])

        else:
            self.sub_dirs = glob.glob(self.fmri_dir + os.sep + "sub*")
            self.subject_list = [os.path.basename(sub)
                                 for sub in self.sub_dirs]
        self.subject_list.sort()
        self.n_subs = len(self.subject_list)
        
        # Level 2: Sessions
        if isinstance(sessions, list):
            self.sessions = sessions
        else:
            session_dirs = flatten_list([glob.glob(sub_dir + os.sep + "ses*")
                                         for sub_dir in self.sub_dirs])
            bases = [os.path.basename(sess_dir) for sess_dir
                     in session_dirs]
            self.sessions = np.unique(bases)
        self.session_types = np.unique(
            [''.join([i for i in base if not i.isdigit()])
             for base in self.sessions])
        
        # Level 3: Runs
        if isinstance(run_dirs, list):
            self.run_dirs = run_dirs
        else:
            self.run_dirs = flatten_list([glob.glob(ses_dir + os.sep + "run*")
                                          for ses_dir in session_dirs])
        self.run_dirs.sort()
        self.runs_total = len(self.run_dirs)
        
    
    def __repr__(self):
        """
        Defines string which is printed for the object
        """
        return (f'pyrsa.data.{self.__class__.__name__}(\n\n'
                f'fMRI directory = \n{self.fmri_dir}\n\n'
                f'Subject list = {self.subject_list}\n\n'
                f'Number of subjects = {self.n_subs}\n\n'
                f'Session types = {self.session_types}\n\n'
                f'Total number of runs = {self.runs_total}\n\n\n\n'
                )
    
    def subset_subject(self, sub):
        """
        Creates a smaller BidsDerivatives object for specified subject
        
        Args:
            sub (int or str): subject id (e.g. 1 or 'sub-01')

        Returns:
            Subsetted BidsDerivatives object
        
        """
        assert isinstance(sub, int) or sub in self.subject_list, \
            "Subject number must be integer or a string for an existing \
                subject ID"
        if isinstance(sub, int):
            sub_name = "sub-"+str(sub).zfill(2)
            assert sub_name in self.subject_list, \
                "Subject with this ID does not exist"
        else:
            sub_name = sub
            
        run_dirs_subset = [run for run in self.run_dirs if sub_name in run]
        subset = BidsDerivativesSubject(self.fmri_dir,
                                        subject_list = [sub_name],
                                        run_dirs = run_dirs_subset)
        return subset
    
    def subset_session_type(self, session_type):
        """
        Creates a smaller BidsDerivatives object for specified session
        
        Args:
            sub (int or str): subject id (e.g. 1 or 'sub-01')

        Returns:
            Subsetted BidsDerivatives object
        
        """
        raise NotImplementedError(
            "subset_session_type method not implemented in used class, \
                you must first subset to one subject!")
    
    def load_betas_SPM(self, stim_ids_dict = None):
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
            beta_array_superset (list of 3d numpy arrays):
                all beta 3d arrays for the keys in stim_ids_dict found in each
                run directory
            dim4_descriptors (list of str):
                corresponding descriptors
                e.g. ['cond_face_run_05', 'cond_house_run_30']
        """
        raise NotImplementedError(
            "load_betas_SPM method not implemented in used class, \
                you must first subset to one subject!")
                
    def load_residuals_SPM(self, res_range = None):
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
        raise NotImplementedError(
            "load_residuals_SPM method not implemented in used class, \
                you must first subset to one subject!")
    def get_subjects(self):
        return self.subject_list.copy()
    
    def get_sessions(self):
        return self.sessions.copy()
    
    def get_session_types(self):
        return self.session_types.copy()
    
    def get_runs(self):
        return self.run_dirs.copy() 

                
class BidsDerivativesSubject(BidsDerivatives):
    """
    SubjectData class is a standard version of BidsDerivative.
    It contains data for only one subject
    """
    
    def __repr__(self):
        """
        Defines string which is printed for the object
        """
        return (f'pyrsa.data.{self.__class__.__name__}(\n\n'
                f'Subject ID = {self.subject_list[0]}\n\n'
                f'Subject directory = \n{self.sub_dirs[0]}\n\n'
                f'Session types = {self.session_types}\n\n'
                f'Sessions = {self.sessions}\n\n'
                f'Total number of runs = {self.runs_total}\n\n\n\n'
                )
    
    def subset_session_type(self, session_type):
        """
        Creates a smaller BidsDerivatives object for specified session
        
        Args:
            sub (int or str): subject id (e.g. 1 or 'sub-01')

        Returns:
            Subsetted BidsDerivatives object
        
        """
        assert session_type in self.sessions \
            or session_type in self.session_types, \
            "Session (type) does not exist"
        run_dirs_subset = [run for run in self.get_runs() if session_type in run]
        ses_subset = [ses for ses in self.get_sessions() if session_type in ses]
        subset = BidsDerivativesSubject(self.fmri_dir,
                                        subject_list = self.subject_list,
                                        sessions = ses_subset,
                                        run_dirs = run_dirs_subset)
        return subset
    
    def load_betas_SPM(self, stim_ids_dict = None):
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
        assert len(self.session_types) == 1, \
            "You first need to subset to one type of sessions"
        
        if stim_ids_dict is None:
            n_conds = len(glob.glob(self.run_dirs[0] + os.sep + "beta*"))
            keys = [str(cond_num).zfill(4) for cond_num in range(1, n_conds+1)]
            values = [cond_num for cond_num in range(1, n_conds+1)]
            stim_ids_dict = dict(zip(keys, values))
        
        beta_array_superset = []
        dim4_descriptors = []
        run_counter = 0
        for glm_dir in self.run_dirs:
            run_counter += 1
            for condition in stim_ids_dict.keys():
                num = stim_ids_dict[condition]
                beta_image_path = os.path.join(glm_dir, "beta_" +
                                               str(num).zfill(4))
                beta_image = nifti1.load(beta_image_path)
                beta_array_superset.append(beta_image.get_fdata())
                dim4_descriptors.append("cond_" + condition + "_run_" +
                                        str(run_counter).zfill(2))
        # Get affine matrix
        self.subject_affine = beta_image.affine.copy()
        pooled_beta_array = np.stack(beta_array_superset, axis = 3)
        return pooled_beta_array, dim4_descriptors
    
    def load_residuals_SPM(self, res_range = None):
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
        assert len(self.session_types) == 1, \
            "You first need to subset to one type of sessions"
            
        residual_array_superset = []
        dim4_descriptors = []
        self.n_res = len(glob.glob(os.path.join(self.run_dirs[0], "Res_*")))
        if isinstance(res_range, range):
            assert self.n_res >= max(res_range), \
                "res_range outside of existing residuals"
        else:
            res_range = range(1, self.n_res+1)
            
        run_counter = 0
        for glm_dir in self.run_dirs:
            run_counter += 1
            for res in res_range:
                res_image_path = os.path.join(glm_dir, "Res_" +
                                               str(res).zfill(4))
                res_image = nifti1.load(res_image_path)
                residual_array_superset.append(res_image.get_fdata())
                dim4_descriptors.append("res_" + str(res).zfill(4)+ "_run_" +
                                                    str(run_counter).zfill(2))
        # Get affine matrix
        self.subject_affine = res_image.affine.copy()
        pooled_residual_array = np.stack(residual_array_superset, axis = 3)
        return pooled_residual_array, dim4_descriptors
    
    def save2nifti(self, pooled_data_array, output_dir = None,
                    data_type = "signal"):
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
        assert len(self.session_types) == 1, \
            "You first need to subset to one type of sessions"
        assert isinstance(data_type, str), "specified data type must be \
            a string object"
        assert isinstance(pooled_data_array, np.ndarray) \
            and len(pooled_data_array.shape) == 4, "Wrong type of data provided"
            
        if output_dir is None:
           self.output_dir = self.sub_dirs[0]
        else:
           assert isinstance(output_dir, str) and os.path.isdir(output_dir), \
               "specified output dir object must be a string \
                   to an existing path"
           self.output_dir = output_dir
        
        print("Saving", data_type, "to 4d-NIfTi file in",
              self.output_dir, "...")
        pooled_data = nifti1.Nifti1Image(pooled_data_array,
                                          self.subject_affine)
        self.nifti_filename = os.path.join(
            self.output_dir, self.subject_list[0] + "_" +
            self.session_types[0] + "_" + data_type + ".nii.gz")
        nifti1.save(pooled_data, self.nifti_filename)
        print("Saved as", self.nifti_filename)
    
    def save2csv(self, descriptors, output_dir = None,
                    data_type = "signal"):
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
        assert len(self.session_types) == 1, \
            "You first need to subset to one type of sessions"
        assert isinstance(data_type, str), "specified data type must be \
            a string object"
        assert isinstance(descriptors, list) \
            and isinstance(descriptors[0], str), "Wrong type of data provided"    
            
        if output_dir is None:
           self.output_dir = self.sub_dirs[0]
        else:
           assert isinstance(output_dir, str) and os.path.isdir(output_dir), \
               "specified output dir object must be a string \
                   to an existing path"
           self.output_dir = output_dir
       

        self.csv_filename = os.path.join(
            self.output_dir, self.subject_list[0] + "_" +
            self.session_types[0] + "_" + data_type + ".csv")
        
        df = pd.DataFrame({'descriptor': descriptors})
        df.to_csv(self.csv_filename, header=False)
        print("Saved", data_type, "descriptors csv to", self.csv_filename)
    
    def save2combo(self, pooled_data_array, descriptors, output_dir = None,
                    data_type = "signal"):
        """
        Combined saving of fmri data and descriptors
        """
        self.save2nifti(pooled_data_array, output_dir = output_dir,
                    data_type = data_type)
        self.save2csv(descriptors, output_dir = output_dir,
                    data_type = data_type)


def flatten_list(lst):
    """
    Flattens list of lists to single list

    Args:
        lst (list)

    Returns:
        lst_flattened (list): the flattened version of lst
    """
    assert isinstance(lst, list)
    lst_flattened = [item for sublist in lst for item in sublist]
    return lst_flattened
