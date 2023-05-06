""" tests for calculation of RDMs
"""
import unittest
from unittest.mock import Mock, patch
from copy import deepcopy
import numpy as np
from parameterized import parameterized
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.spatial.distance import pdist, squareform
import rsatoolbox.rdm as rsr
import rsatoolbox as rsa

ALL_METHODS = [
    'euclidean',
    'correlation',
    'mahalanobis',
    'crossnobis',
    'poisson',
    'poisson_cv',
]
NO_CV_METHODS = [
    'euclidean',
    'correlation',
    'mahalanobis',
    'poisson',
]


class TestCalcRDM(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(0)
        measurements = self.rng.random((20, 5))
        measurements_deterministic = np.array(
            [
                [0.11, 0.12, 0.21, 0.22, 0.30, 0.31],
                [0.13, 0.14, 0.24, 0.21, 0.29, 0.28],
                [0.10, 0.11, 0.24, 0.25, 0.32, 0.33],
            ]
        ).T
        des = {'session': 0, 'subj': 0}
        obs_des = {
            'conds': np.array(
                [0, 0, 1, 1, 2, 2, 2, 3, 4, 5, 0, 0, 1, 1, 2, 2, 2, 3, 4, 5]
            ),
            'fold': np.array(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ),
        }
        self.obs_balanced = {
            'conds': np.array(
                [4, 4, 1, 1, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
            ),
            'fold': np.array(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ),
        }
        chn_des = {'rois': np.array(['V1', 'V1', 'IT', 'IT', 'V4'])}
        self.test_data = rsa.data.Dataset(
            measurements=measurements,
            descriptors=des,
            obs_descriptors=obs_des,
            channel_descriptors=chn_des,
        )
        self.test_data_balanced = rsa.data.Dataset(
            measurements=measurements,
            descriptors=des,
            obs_descriptors=self.obs_balanced,
            channel_descriptors=chn_des,
        )
        self.test_data_deterministic = rsa.data.Dataset(
            measurements=measurements_deterministic,
            descriptors=des,
            obs_descriptors=dict(
                conds=np.array([0, 0, 1, 1, 2, 2]),
            ),
            channel_descriptors=dict(feats=['v1', 'v2', 'v3']),
        )

    def test_calc_euclidean_nconds(self):
        d = self.test_data
        rdm = rsr.calc_rdm([d, d], descriptor='conds', method='euclidean')
        assert rdm.n_cond == 6

    def test_parse_input(self):
        from rsatoolbox.rdm.calc import _parse_input
        data = Mock()
        data.descriptors = {'session': 0, 'subj': 0}
        data.measurements = self.rng.random((6, 5))
        measurements, desc = _parse_input(data, None)
        self.assertIsNone(desc)
        self.assertTrue(np.all(data.measurements == measurements))

    @patch('rsatoolbox.rdm.calc._parse_input')
    def test_calc_euclidean_as_scipy(self, _parse_input):
        from rsatoolbox.rdm import calc_rdm
        data = Mock()
        data.descriptors = {'session': 0, 'subj': 0}
        data.obs_descriptors = dict(conds=np.arange(6))
        data.channel_descriptors = dict()
        data.measurements = self.rng.random((6, 5))
        desc = [0, 1, 2, 3, 4, 5]
        _parse_input.return_value = (data.measurements, desc)
        rdm_expected = pdist(data.measurements) ** 2 / 5
        rdms = calc_rdm(data, descriptor='conds', method='euclidean')
        assert_array_almost_equal(
            rdm_expected, rdms.dissimilarities.flatten())

    @patch('rsatoolbox.rdm.calc._parse_input')
    def test_calc_correlation(self, _parse_input):
        from rsatoolbox.rdm import calc_rdm
        data = Mock()
        data.descriptors = {'session': 0, 'subj': 0}
        data.obs_descriptors = dict(conds=np.arange(6))
        data.channel_descriptors = dict()
        data.measurements = self.rng.random((6, 5))
        desc = [0, 1, 2, 3, 4, 5]
        _parse_input.return_value = (data.measurements, desc)
        rdm_expected = 1 - np.corrcoef(data.measurements)
        rdme = rsr.RDMs(
            dissimilarities=np.array([rdm_expected]),
            dissimilarity_measure='correlation',
            descriptors=data.descriptors,
        )
        rdm = calc_rdm(data, descriptor='conds', method='correlation')
        assert_array_almost_equal(
            rdme.dissimilarities.flatten(), rdm.dissimilarities.flatten()
        )

    def test_calc_list_descriptors(self):
        rdm = rsr.calc_rdm(
            [self.test_data, self.test_data, self.test_data],
            descriptor='conds',
            method='euclidean',
        )
        assert np.all(rdm.rdm_descriptors['subj'] == np.array([0, 0, 0]))

    def test_calc_mahalanobis(self):
        rdm = rsr.calc_rdm(
            self.test_data,
            descriptor='conds',
            method='mahalanobis'
        )
        assert rdm.n_cond == 6
        noise = np.linalg.inv(np.cov(self.rng.standard_normal((10, 5)).T))
        rdm = rsr.calc_rdm(
            self.test_data,
            descriptor='conds',
            method='mahalanobis',
            noise=noise
        )
        assert rdm.n_cond == 6

    def test_calc_crossnobis(self):
        rdm = rsr.calc_rdm_crossnobis(
            self.test_data, descriptor='conds', cv_descriptor='fold'
        )
        assert rdm.n_cond == 6

    def test_calc_crossnobis_order(self):
        measurements = np.zeros((20, 5))
        measurements[3, 1] = 1
        measurements[13, 1] = 1
        obs_des = {
            'conds': np.array(
                [0, 0, 1, 1, 2, 2, 2, 3, 4, 5, 0, 0, 1, 1, 2, 2, 2, 3, 4, 5]
            ),
            'fold': np.array(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ),
        }
        zero_data = rsa.data.Dataset(measurements, obs_descriptors=obs_des)
        rdm1 = rsr.calc_rdm(
            zero_data,
            descriptor='conds',
            cv_descriptor='fold',
            method='crossnobis'
        )
        obs_des2 = {
            'conds': np.array(
                [5, 0, 1, 1, 2, 2, 2, 3, 4, 5, 0, 0, 1, 1, 2, 2, 2, 3, 4, 5]
            ),
            'fold': np.array(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ),
        }
        zero_data2 = rsa.data.Dataset(
            measurements=measurements, obs_descriptors=obs_des2
        )
        rdm2 = rsr.calc_rdm(
            zero_data2,
            descriptor='conds',
            cv_descriptor='fold',
            method='crossnobis'
        )
        np.testing.assert_equal(rdm1.dissimilarities, rdm2.dissimilarities)

    def test_calc_crossnobis_no_descriptors(self):
        rdm = rsr.calc_rdm_crossnobis(
            self.test_data_balanced,
            descriptor='conds'
        )
        assert rdm.n_cond == 5

    def test_calc_crossnobis_noise(self):
        noise = self.rng.standard_normal((10, 5))
        noise = np.matmul(noise.T, noise)
        rdm = rsr.calc_rdm_crossnobis(
            self.test_data_balanced, descriptor='conds', noise=noise
        )
        assert rdm.n_cond == 5

    def test_calc_crossnobis_noise_list(self):
        # generate two positive definite noise matricies
        noise = self.rng.standard_normal((2, 10, 5))
        noise = np.einsum('ijk,ijl->ikl', noise, noise)
        rdm = rsr.calc_rdm_crossnobis(
            self.test_data_balanced,
            cv_descriptor='fold',
            descriptor='conds',
            noise=noise,
        )
        assert rdm.n_cond == 5
        # test with noise list
        noise = [noise[i] for i in range(len(noise))]
        rdm = rsr.calc_rdm_crossnobis(
            self.test_data_balanced,
            cv_descriptor='fold',
            descriptor='conds',
            noise=noise,
        )
        assert rdm.n_cond == 5
        rdm = rsr.calc_rdm_crossnobis(
            self.test_data,
            cv_descriptor='fold',
            descriptor='conds',
            noise=noise
        )
        assert rdm.n_cond == 6

    def test_calc_poisson_6_conditions(self):
        rdm = rsr.calc_rdm(
            self.test_data,
            descriptor='conds',
            method='poisson'
        )
        assert rdm.n_cond == 6

    def test_calc_poisson_extreme_pairs(self):
        """Check the dissimilarities computed with the 'poisson' method

        The closest pair should be that between the condition 1 and itself
        The furthest pair should be that between condition 1 and condition 3
        """
        rdm = rsr.calc_rdm(
            self.test_data_deterministic, descriptor='conds', method='poisson'
        )
        rdm_array = squareform(rdm.get_vectors()[0, :])
        closest_pair_index = np.argmin(rdm_array)
        furthest_pair_index = np.argmax(rdm_array)
        self.assertEqual(closest_pair_index, 0)
        self.assertEqual(furthest_pair_index, 2)

    def test_calc_poisson_cv(self):
        rdm = rsr.calc_rdm(
            self.test_data,
            descriptor='conds',
            cv_descriptor='fold',
            method='poisson_cv',
        )
        assert rdm.n_cond == 6

    def test_calc_crossnobis_sorting(self):
        """check that sorting the data does not change results
        This catches errors due to mix ups of indices.
        """
        data2 = deepcopy(self.test_data_balanced)
        data2.sort_by('conds')
        rdm1 = rsr.calc_rdm_crossnobis(
            self.test_data_balanced, descriptor='conds', cv_descriptor='fold'
        )
        rdm2 = rsr.calc_rdm_crossnobis(
            data2, descriptor='conds', cv_descriptor='fold')
        assert_array_almost_equal(
            rdm1.dissimilarities,
            rdm2.dissimilarities
        )
        assert_array_equal(
            self.test_data_balanced.obs_descriptors['conds'],
            self.obs_balanced['conds']
        )

    @parameterized.expand(NO_CV_METHODS)
    def test_calc_single_ds_passes_on_descriptors(self, method):
        """Calc_rdm() should use dataset descriptors as rdm
        descriptors on the resulting RDMs; as well as
        observation descriptors that have unique values
        for the used patterns as pattern descriptors.
        """
        from rsatoolbox.rdm.calc import calc_rdm
        rdms = calc_rdm(self.test_data, method=method)
        self.assertEqual(rdms.rdm_descriptors.get('session'), [0])
        self.assertEqual(rdms.rdm_descriptors.get('subj'), [0])
        assert_array_equal(
            rdms.pattern_descriptors['conds'],
            self.test_data.obs_descriptors['conds']
        )
        assert_array_equal(
            rdms.pattern_descriptors['fold'],
            self.test_data.obs_descriptors['fold']
        )

    @parameterized.expand(ALL_METHODS)
    def test_calc_wdesc_single_ds_passes_on_descriptors(self, method):
        """When using a target observation descriptor;
        Calc_rdm() should use observation descriptors
        that have unique values for the used patterns as
        pattern descriptors.
        """
        from rsatoolbox.rdm.calc import calc_rdm
        kwargs = dict(method=method, descriptor='conds')
        if method in ('crossnobis', 'poisson_cv'):
            kwargs['cv_descriptor'] = 'fold'
        foo_vals = [10, 10, 11, 11, 12, 12, 12, 13, 14, 15, 10, 10, 11, 11,
                    12, 12, 12, 13, 14, 15]
        self.test_data.obs_descriptors['foo'] = np.array(foo_vals)
        rdms = calc_rdm(self.test_data, **kwargs)
        self.assertEqual(rdms.rdm_descriptors.get('session'), [0])
        self.assertEqual(rdms.rdm_descriptors.get('subj'), [0])
        assert_array_equal(
            rdms.pattern_descriptors['conds'],
            np.array([0, 1, 2, 3, 4, 5])
        )
        assert_array_equal(
            rdms.pattern_descriptors['foo'],
            np.array([10, 11, 12, 13, 14, 15])
        )

    @parameterized.expand(NO_CV_METHODS)
    def test_calc_multi_ds_passes_on_descriptors(self, method):
        """When passing multiple Datasets to calc_rdm,
        it should use the individual dataset descriptors
        as rdm descriptors.
        """
        from rsatoolbox.rdm.calc import calc_rdm
        rdms = calc_rdm(
            self.test_data.split_channel(by='rois'),
            method=method
        )
        assert_array_equal(rdms.rdm_descriptors['session'], [0, 0, 0])
        assert_array_equal(rdms.rdm_descriptors['subj'], [0, 0, 0])
        assert_array_equal(
            rdms.rdm_descriptors['rois'],
            np.array(['V1', 'IT', 'V4'])
        )
        assert_array_equal(
            rdms.pattern_descriptors['conds'],
            self.test_data.obs_descriptors['conds']
        )

    def test_calc_with_descriptors_as_list(self):
        """Can the calc methods deal with descriptors
        that are defined as List.
        """
        conds_list = [str(i) for i in self.test_data.obs_descriptors['conds']]
        self.test_data.obs_descriptors['conds'] = conds_list
        rdms = rsr.calc_rdm(
            self.test_data,
            descriptor='conds',
            method='euclidean'
        )
        assert_array_equal(
            rdms.pattern_descriptors['conds'],
            np.unique(self.test_data.obs_descriptors['conds']),
        )


class TestCalcRDMMovie(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(0)
        measurements_time = self.rng.random((20, 5, 15))
        tim_des = {'time': np.linspace(0, 200, 15)}
        des = {'session': 0, 'subj': 0}
        obs_des = {
            'conds': np.array(
                [0, 0, 1, 1, 2, 2, 2, 3, 4, 5, 0, 0, 1, 1, 2, 2, 2, 3, 4, 5]
            ),
            'fold': np.array(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ),
        }
        obs_balanced = {
            'conds': np.array(
                [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
            ),
            'fold': np.array(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ),
        }
        chn_des = {
            'rois': np.array(['sensor' + str(i) for i in [1, 2, 3, 4, 5]])
        }

        self.test_data_time = rsa.data.TemporalDataset(
            measurements=measurements_time,
            descriptors=des,
            obs_descriptors=obs_des,
            channel_descriptors=chn_des,
            time_descriptors=tim_des,
        )
        self.test_data_time_balanced = rsa.data.TemporalDataset(
            measurements=measurements_time,
            descriptors=des,
            obs_descriptors=obs_balanced,
            channel_descriptors=chn_des,
            time_descriptors=tim_des,
        )

    def test_calc_rdm_movie_mahalanobis(self):
        rdm = rsr.calc_rdm_movie(
            self.test_data_time,
            descriptor='conds',
            method='mahalanobis',
            time_descriptor='time',
        )
        assert rdm.n_cond == 6
        assert len([r for r in rdm]) == 15
        assert rdm.rdm_descriptors['time'][0] == 0.0
        assert len(rdm.rdm_descriptors['time']) == 15

    def test_calc_rdm_movie_euclidean(self):
        rdm = rsr.calc_rdm_movie(
            self.test_data_time,
            descriptor='conds',
            method='euclidean',
            time_descriptor='time',
        )
        assert rdm.n_cond == 6
        assert len([r for r in rdm]) == 15
        assert rdm.rdm_descriptors['time'][0] == 0.0

    def test_calc_rdm_movie_correlation(self):
        rdm = rsr.calc_rdm_movie(
            self.test_data_time,
            descriptor='conds',
            method='correlation',
            time_descriptor='time',
        )
        assert rdm.n_cond == 6
        assert len([r for r in rdm]) == 15
        assert rdm.rdm_descriptors['time'][0] == 0.0

    def test_calc_rdm_movie_crossnobis(self):
        rdm = rsr.calc_rdm_movie(
            self.test_data_time,
            descriptor='conds',
            method='crossnobis',
            time_descriptor='time',
            cv_descriptor='fold',
        )
        assert rdm.n_cond == 6
        assert len([r for r in rdm]) == 15
        assert rdm.rdm_descriptors['time'][0] == 0.0

    def test_calc_rdm_movie_crossnobis_no_descriptors(self):
        rdm = rsr.calc_rdm_movie(
            self.test_data_time_balanced,
            method='crossnobis',
            time_descriptor='time',
            descriptor='conds',
        )
        assert rdm.n_cond == 5

    def test_calc_rdm_movie_crossnobis_noise(self):
        noise = self.rng.standard_normal((10, 5))
        noise = np.matmul(noise.T, noise)
        rdm = rsr.calc_rdm_movie(
            self.test_data_time_balanced,
            method='crossnobis',
            time_descriptor='time',
            descriptor='conds',
            noise=noise,
        )
        assert rdm.n_cond == 5

    def test_calc_rdm_movie_rdm_movie_poisson(self):
        noise = self.rng.standard_normal((10, 5))
        noise = np.matmul(noise.T, noise)
        rdm = rsr.calc_rdm_movie(
            self.test_data_time_balanced,
            method='poisson',
            descriptor='conds',
            noise=noise,
        )
        assert rdm.n_cond == 5

    def test_calc_rdm_movie_binned(self):
        time = self.test_data_time.time_descriptors['time']
        bins = np.reshape(time, [5, 3])
        rdm = rsr.calc_rdm_movie(
            self.test_data_time,
            descriptor='conds',
            method='mahalanobis',
            time_descriptor='time',
            bins=bins,
        )
        assert rdm.n_cond == 6
        assert len([r for r in rdm]) == 5
        assert rdm.rdm_descriptors['time'][0] == np.mean(time[:3])
