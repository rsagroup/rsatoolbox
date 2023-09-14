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

    def __init__(self, fmri_dir, subject_list=None, sessions=None,
                 run_dirs=None):
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
                                        subject_list=[sub_name],
                                        run_dirs=run_dirs_subset)
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

    def load_betas_SPM(self, stim_ids_dict=None):
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

    def load_residuals_SPM(self, res_range=None):
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

    def __init__(self, *args, **kwargs):
        self.nifti_filename = None
        self.csv_filename = None
        super().__init__(*args, **kwargs)

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
        run_dirs_subset = [run for run in self.get_runs()
                           if session_type in run]
        ses_subset = [ses for ses in self.get_sessions()
                      if session_type in ses]
        subset = BidsDerivativesSubject(self.fmri_dir,
                                        subject_list=self.subject_list,
                                        sessions=ses_subset,
                                        run_dirs=run_dirs_subset)
        return subset



class Nifti2Dataset:
    def __init__(self, bids_dir, derivative, subject_list=None):
        """
        Parses BIDS structured data in fmri_dir
        """
        self.bids_dir = bids_dir
        assert os.path.exists(self.bids_dir), \
            "Specified dataset directory does not exist."
        self.fmri_dir = os.path.join(
            self.bids_dir, "derivatives", derivative)
        assert os.path.exists(self.fmri_dir), \
            "Specified derivatives directory does not exist."

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
        self.niftis = flatten_list(
            [glob.glob(sub_dir + os.sep + "*.nii.gz")
             for sub_dir in self.sub_dirs])
        self.n_niftis = len(self.niftis)
        self.csvs = flatten_list(
            [glob.glob(sub_dir + os.sep + "*.csv")
             for sub_dir in self.sub_dirs])

    def __repr__(self):
        """
        Defines string which is printed for the object
        """
        return (f'pyrsa.data.{self.__class__.__name__}(\n\n'
                f'fMRI directory = \n{self.fmri_dir}\n\n'
                f'Number of subjects = {self.n_subs}\n\n'
                f'Subject list = {self.subject_list}\n\n'
                f'Number of NIfTi files = {self.n_niftis}\n\n\n'
                )

    def set_output_dirs(self):
        self.ds_output_dirs = [os.path.join(
            self.bids_dir, "derivatives", "PyRSA", "datasets", sub)
            for sub in self.subject_list]
        for output_dir in self.ds_output_dirs:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
        self.res_output_dirs = [os.path.join(
            self.bids_dir, "derivatives", "PyRSA", "noise", sub)
            for sub in self.subject_list]
        for output_dir in self.res_output_dirs:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)


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
