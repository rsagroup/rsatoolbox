#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data preparation functions
"""

import os
import glob
import numpy as np
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
            sub_dirs = flatten_list([glob.glob(self.fmri_dir + os.sep + sub)
                                     for sub in subject_list])

        else:
            sub_dirs = glob.glob(self.fmri_dir + os.sep + "sub*")
            self.subject_list = [os.path.basename(sub) for sub in sub_dirs]
        self.subject_list.sort()
        self.n_subs = len(self.subject_list)
        
        # Level 2: Sessions
        if isinstance(sessions, list):
            self.sessions = sessions
        else:
            session_dirs = flatten_list([glob.glob(sub_dir + os.sep + "ses*")
                                         for sub_dir in sub_dirs])
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
                f'Sessions = {self.sessions}\n\n'
                f'Session types = {self.session_types}\n\n'
                f'Total number of runs = {self.runs_total}\n'
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
        subset = BidsDerivatives(self.fmri_dir,
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
        assert session_type in self.sessions \
            or session_type in self.session_types, \
            "Session (type) does not exist"
        run_dirs_subset = [run for run in self.run_dirs if session_type in run]
        subset = BidsDerivatives(self.fmri_dir,
                                 subject_list = self.subject_list,
                                 sessions = [session_type],
                                 run_dirs = run_dirs_subset)
        return subset
    
    def get_subjects(self):
        return self.subject_list.copy()
    
    def get_sessions(self):
        return self.sessions.copy()
    
    def get_runs(self):
        return self.run_dirs.copy() 


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


def load_SPM_beta_images(run_dirs, stim_ids_dict):
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
    beta_array_superset = []
    dim4_descriptors = []
    run_counter = 0
    for glm_dir in run_dirs:
        run_counter += 1
        for condition in stim_ids_dict.keys():
            num = stim_ids_dict[condition]
            beta_image_path = os.path.join(glm_dir, "beta_" +
                                           str(num).zfill(4))
            beta_image = nifti1.load(beta_image_path)
            beta_array_superset.append(beta_image.get_fdata())
            dim4_descriptors.append("cond_" + condition + "_run_" +
                                    str(run_counter).zfill(2))
    return beta_array_superset, dim4_descriptors




