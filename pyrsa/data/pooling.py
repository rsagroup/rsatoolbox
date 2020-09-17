#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data preparation functions
"""

import os
import glob
import numpy as np

class BidsDerivatives:
    def __init__(self, fmri_dir, subject_list = None, sessions = None,
                 run_dirs = None):
        assert os.path.exists(fmri_dir), "Specified directory does not exist."
        self.fmri_dir = fmri_dir
        
        # Level 1: Subjects
        sub_dirs = glob.glob(self.fmri_dir + os.sep + "sub*")
        if isinstance(subject_list, list):
            self.subject_list = subject_list
        else:
            self.subject_list = [os.path.basename(sub) for sub in sub_dirs]
        self.subject_list.sort()
        self.n_subs = len(self.subject_list)
        
        # Level 2: Sessions
        session_dirs = [glob.glob(sub_dir + os.sep + "ses*") 
                             for sub_dir in sub_dirs]
        session_dirs_flat = [item for sublist in session_dirs
                             for item in sublist]
        bases = [os.path.basename(sess_dir) for sess_dir in session_dirs_flat]
        if isinstance(sessions, list):
            self.sessions = sessions
        else:
            self.sessions = np.unique(
            [''.join([i for i in base if not i.isdigit()]) for base in bases])
        
        # Level 3: Runs
        if isinstance(run_dirs, list):
            self.run_dirs = run_dirs
        else:
            run_dirs = [glob.glob(ses_dir + os.sep + "run*")
                         for ses_dir in session_dirs_flat]
            self.run_dirs = [item for sublist in run_dirs
                                  for item in sublist]
        self.run_dirs.sort()
        
    
    def __repr__(self):
        """
        defines string which is printed for the object
        """
        return (f'pyrsa.data.{self.__class__.__name__}(\n\n'
                f'fMRI directory = \n{self.fmri_dir}\n\n'
                f'Number of subjects = {self.n_subs}\n\n'
                f'Session types = {self.sessions}\n\n'
                )
    
    def subset_subject(self, sub):
        assert isinstance(sub, int), "Subject number must be integer."
        sub_name = "sub-"+str(sub).zfill(2)
        assert sub_name in self.subject_list, \
            "Subject with this ID does not exist"
        run_dirs_subset = [run for run in self.run_dirs if sub_name in run]
        subset = BidsDerivatives(self.fmri_dir,
                                 subject_list = [sub_name],
                                 sessions = self.sessions,
                                 run_dirs = run_dirs_subset)
        return subset
    
    def subset_session_type(self, session_type):
        assert session_type in self.sessions, \
            "Session type does not exist"
        run_dirs_subset = [run for run in self.run_dirs if session_type in run]
        subset = BidsDerivatives(self.fmri_dir,
                                 subject_list = self.subject_list,
                                 sessions = [session_type],
                                 run_dirs = run_dirs_subset)
        return subset
            
        

