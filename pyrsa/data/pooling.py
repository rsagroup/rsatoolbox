#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data preparation functions
"""

import os
import glob
import numpy as np

def flatten_list(lst):
    assert isinstance(lst, list)
    lst_flattened = [item for sublist in lst for item in sublist]
    return lst_flattened

class BidsDerivatives:
    def __init__(self, fmri_dir, subject_list = None, sessions = None,
                 run_dirs = None):
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
        defines string which is printed for the object
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
        assert session_type in self.sessions, \
            "Session type does not exist"
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
            
        

