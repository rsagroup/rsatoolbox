#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for BidsDerivatives class and subclasses

"""

import unittest
import os
import rsatoolbox.io.fmri as fmri
import numpy as np

def get_fmri_data():
    fmri_dir = os.path.join(
            os.path.dirname(__file__), "data", "BIDS_example", "derivatives",
            "SPM_example")
    fmri_data = fmri.BidsDerivatives(fmri_dir)
    return fmri_data

def get_subject_data():
    fmri_data = get_fmri_data()
    subjects = fmri_data.get_subjects()
    subject_data = fmri_data.subset_subject(subjects[0])                        # subset data for first subject
    return subject_data

def get_subject_session_data():
    subject_data = get_subject_data()
    session_types = subject_data.get_session_types()
    subject_data_s = subject_data.subset_session_type(session_types[0])         # further subset data for first session type
    return subject_data_s

class TestParsing(unittest.TestCase):
    
    def test_full_parsing(self):
        fmri_data = get_fmri_data()
        np.testing.assert_array_equal(
            fmri_data.get_subjects(),
            ['sub-01', 'sub-02'])
        np.testing.assert_array_equal(
            fmri_data.get_session_types(),
            ['ses-perceptionTest', 'ses-perceptionTraining'])
        np.testing.assert_array_equal(
            fmri_data.runs_total, 2)
        
    def test_sub_parsing(self):
        subject_data = get_subject_data()
        np.testing.assert_array_equal(
            subject_data.get_subjects(),
            ['sub-01'])
        np.testing.assert_array_equal(
            subject_data.get_sessions(),
            ['ses-perceptionTest01'])
        np.testing.assert_array_equal(
            subject_data.runs_total, 1)
        np.testing.assert_array_equal(
            [os.path.isdir(run_dir)
             for run_dir in subject_data.get_runs()],
            [True])
        
    def test_session_parsing(self):
        subject_data = get_subject_session_data()
        np.testing.assert_array_equal(
            subject_data.get_sessions(),
            ['ses-perceptionTest01'])
        np.testing.assert_array_equal(
            subject_data.runs_total, 1)
        np.testing.assert_array_equal(
            [os.path.isdir(run_dir)
             for run_dir in subject_data.get_runs()],
                [True])

class TestPoolingSPM(unittest.TestCase):
    
    def test_beta_pooling(self):
        subject_data = get_subject_session_data()
        beta_array, beta_descriptors = subject_data.load_betas_SPM()
        np.testing.assert_array_equal(
            beta_array.shape, (55, 60, 51, 2))
        np.testing.assert_array_equal(
            beta_array.shape[3], len(beta_descriptors))
        beta_descriptors.sort()
        np.testing.assert_array_equal(
            np.unique(beta_descriptors),
            beta_descriptors)
        
    def test_beta_pooling_w_dict(self):
        stim_ids_dict = {"Face": 1, "House": 2}
        subject_data = get_subject_session_data()
        beta_array, beta_descriptors = subject_data.load_betas_SPM(
            stim_ids_dict = stim_ids_dict)
        np.testing.assert_array_equal(
            beta_array.shape, (55, 60, 51, 2))
        np.testing.assert_array_equal(
            beta_array.shape[3], len(beta_descriptors))
        beta_descriptors.sort()
        np.testing.assert_array_equal(
            np.unique(beta_descriptors),
            beta_descriptors)
        
    def test_res_pooling(self):
        subject_data = get_subject_session_data()
        res_array, res_descriptors = subject_data.load_residuals_SPM()
        np.testing.assert_array_equal(
            res_array.shape, (55, 60, 51, 1))
        np.testing.assert_array_equal(
            res_array.shape[3], len(res_descriptors))
        res_descriptors.sort()
        np.testing.assert_array_equal(
            np.unique(res_descriptors),
            res_descriptors)
        
class TestSaving(unittest.TestCase):
    
    def test_saving_combo_signal(self):
        stim_ids_dict = {"Face": 1, "House": 2}
        subject_data = get_subject_session_data()
        beta_array, beta_descriptors = subject_data.load_betas_SPM(
            stim_ids_dict = stim_ids_dict)
        subject_data.save2combo(
            beta_array, beta_descriptors, data_type = "signal")
        assert os.path.isfile(subject_data.nifti_filename)
        assert os.path.isfile(subject_data.csv_filename)
        
    def test_saving_combo_noise(self):
        subject_data = get_subject_session_data()
        res_array, res_descriptors = subject_data.load_residuals_SPM()
        subject_data.save2combo(
            res_array, res_descriptors, data_type = "noise")
        assert os.path.isfile(subject_data.nifti_filename)
        assert os.path.isfile(subject_data.csv_filename)
        
if __name__ == '__main__':
    unittest.main()
