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
