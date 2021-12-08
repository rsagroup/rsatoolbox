#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_data
Test for RDM class
@author: baihan
"""

import unittest
import numpy as np
from numpy.testing import assert_array_equal
from scipy.spatial.distance import squareform
import rsatoolbox.rdm as rsr
import rsatoolbox as rsa


class TestRDM(unittest.TestCase):

    def test_rdm3d_init(self):
        dis = np.zeros((8, 5, 5))
        mes = "Euclidean"
        des = {'session': 0, 'subj': 0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes,
                        descriptors=des)
        self.assertEqual(rdms.n_rdm, 8)
        self.assertEqual(rdms.n_cond, 5)

    def test_rdm2d_init(self):
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'session': 0, 'subj': 0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes,
                        descriptors=des)
        self.assertEqual(rdms.n_rdm, 8)
        self.assertEqual(rdms.n_cond, 5)

    def test_rdm3d_get_vectors(self):
        dis = np.zeros((8, 5, 5))
        mes = "Euclidean"
        des = {'session': 0, 'subj': 0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes,
                        descriptors=des)
        v_rdms = rdms.get_vectors()
        self.assertEqual(v_rdms.shape[0], 8)
        self.assertEqual(v_rdms.shape[1], 10)

    def test_rdm2d_get_vectors(self):
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'session': 0, 'subj': 0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes,
                        descriptors=des)
        v_rdms = rdms.get_vectors()
        self.assertEqual(v_rdms.shape[0], 8)
        self.assertEqual(v_rdms.shape[1], 10)

    def test_rdm3d_get_matrices(self):
        dis = np.zeros((8, 5, 5))
        mes = "Euclidean"
        des = {'session': 0, 'subj': 0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes,
                        descriptors=des)
        m_rdms = rdms.get_matrices()
        self.assertEqual(m_rdms.shape[0], 8)
        self.assertEqual(m_rdms.shape[1], 5)
        self.assertEqual(m_rdms.shape[2], 5)

    def test_rdm2d_get_matrices(self):
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'session': 0, 'subj': 0}
        rdms = rsr.RDMs(dissimilarities=dis,
                        dissimilarity_measure=mes,
                        descriptors=des)
        m_rdms = rdms.get_matrices()
        self.assertEqual(m_rdms.shape[0], 8)
        self.assertEqual(m_rdms.shape[1], 5)
        self.assertEqual(m_rdms.shape[2], 5)

    def test_rdm_subset(self):
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'subj': 0}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        rdm_descriptors=rdm_des,
                        dissimilarity_measure=mes,
                        descriptors=des)
        rdms_subset = rdms.subset('session', np.array([0, 1, 2]))
        self.assertEqual(rdms_subset.n_rdm, 4)
        self.assertEqual(rdms_subset.n_cond, 5)
        assert_array_equal(rdms_subset.rdm_descriptors['session'],
                           [0, 1, 2, 2])

    def test_rdm_subset_pattern(self):
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'subj': 0}
        pattern_des = {'type': np.array([0, 1, 2, 2, 4])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        pattern_descriptors=pattern_des,
                        dissimilarity_measure=mes,
                        descriptors=des)
        rdms_subset = rdms.subset_pattern('type', np.array([0, 1, 2]))
        self.assertEqual(rdms_subset.n_rdm, 8)
        self.assertEqual(rdms_subset.n_cond, 4)
        assert_array_equal(rdms_subset.pattern_descriptors['type'],
                           [0, 1, 2, 2])

    def test_rdm_subsample(self):
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'subj': 0}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        rdm_descriptors=rdm_des,
                        dissimilarity_measure=mes,
                        descriptors=des)
        rdms_sample = rdms.subsample('session', np.array([0, 1, 2, 2]))
        self.assertEqual(rdms_sample.n_rdm, 6)
        self.assertEqual(rdms_sample.n_cond, 5)
        assert_array_equal(rdms_sample.rdm_descriptors['session'],
                           [0, 1, 2, 2, 2, 2])

    def test_rdm_subsample_pattern(self):
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'subj': 0}
        pattern_des = {'type': np.array([0, 1, 2, 2, 4])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        pattern_descriptors=pattern_des,
                        dissimilarity_measure=mes,
                        descriptors=des)
        rdms_sample = rdms.subsample_pattern('type',
                                             np.array([0, 1, 2, 2]))
        self.assertEqual(rdms_sample.n_rdm, 8)
        self.assertEqual(rdms_sample.n_cond, 6)
        assert_array_equal(rdms_sample.pattern_descriptors['type'],
                           [0, 1, 2, 2, 2, 2])

    def test_rdm_idx(self):
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'subj': 0}
        pattern_des = {'type': np.array([0, 1, 2, 2, 4])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        pattern_descriptors=pattern_des,
                        dissimilarity_measure=mes,
                        descriptors=des)
        rdms_sample = rdms[2]
        self.assertEqual(rdms_sample.n_rdm, 1)
        assert_array_equal(rdms_sample.dissimilarities[0], dis[2])
        rdms_sample = rdms[3, 4, 5]
        self.assertEqual(rdms_sample.n_rdm, 3)
        assert_array_equal(rdms_sample.dissimilarities[0], dis[3])

    def test_rdm_len(self):
        n_rdm, n_cond = 7, 10
        dis = np.zeros((n_rdm, n_cond, n_cond))
        rdms = rsr.RDMs(dissimilarities=dis,
                        pattern_descriptors={
                            'type': np.array(list(range(n_cond)))},
                        dissimilarity_measure='Euclidean',
                        descriptors={'subj': range(n_rdm)})
        self.assertEqual(len(rdms), n_rdm)

    def test_rdm_idx_len_is_1(self):
        n_rdm, n_cond = 7, 10
        dis = np.zeros((n_rdm, n_cond, n_cond))
        rdms = rsr.RDMs(dissimilarities=dis,
                        pattern_descriptors={
                            'type': np.array(list(range(n_cond)))},
                        dissimilarity_measure='Euclidean',
                        descriptors={'subj': range(n_rdm)})
        self.assertEqual(len(rdms[0]), 1)

    def test_rdm_subset_len(self):
        subset_idxs = [0, 1, 2]
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'subj': 0}
        rdm_des = {'session': np.array([0, 1, 2, 3, 4, 5, 6, 7])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        rdm_descriptors=rdm_des,
                        dissimilarity_measure=mes,
                        descriptors=des)
        rdms_subset = rdms.subset('session', np.array(subset_idxs))
        self.assertEqual(len(rdms_subset), len(subset_idxs))

    def test_rdm_iter(self):
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'subj': 0}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        rdm_descriptors=rdm_des,
                        dissimilarity_measure=mes,
                        descriptors=des)
        i = 0
        for rdm in rdms:
            self.assertIsInstance(rdm, rsr.RDMs)
            self.assertEqual(len(rdm), 1)
            assert_array_equal(rdm.dissimilarities, rdms[i].dissimilarities)
            i += 1
        self.assertEqual(i, rdms.n_rdm)

    def test_transform(self):
        from rsatoolbox.rdm import transform
        dis = np.random.rand(8, 10)
        mes = "Euclidean"
        des = {'subj': 0}
        pattern_des = {'type': np.array([0, 1, 2, 2, 4])}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        rdm_descriptors=rdm_des,
                        pattern_descriptors=pattern_des,
                        dissimilarity_measure=mes,
                        descriptors=des)

        def square(x):
            return x ** 2

        transformed_rdm = transform(rdms, square)
        self.assertEqual(transformed_rdm.n_rdm, rdms.n_rdm)
        self.assertEqual(transformed_rdm.n_cond, rdms.n_cond)

    def test_rank_transform(self):
        from rsatoolbox.rdm import rank_transform
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'subj': 0}
        pattern_des = {'type': np.array([0, 1, 2, 2, 4])}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        rdm_descriptors=rdm_des,
                        pattern_descriptors=pattern_des,
                        dissimilarity_measure=mes,
                        descriptors=des)
        rank_rdm = rank_transform(rdms)
        self.assertEqual(rank_rdm.n_rdm, rdms.n_rdm)
        self.assertEqual(rank_rdm.n_cond, rdms.n_cond)
        self.assertEqual(rank_rdm.dissimilarity_measure, 'Euclidean (ranks)')

    def test_sqrt_transform(self):
        from rsatoolbox.rdm import sqrt_transform
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'subj': 0}
        pattern_des = {'type': np.array([0, 1, 2, 2, 4])}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        rdm_descriptors=rdm_des,
                        pattern_descriptors=pattern_des,
                        dissimilarity_measure=mes,
                        descriptors=des)
        sqrt_rdm = sqrt_transform(rdms)
        self.assertEqual(sqrt_rdm.n_rdm, rdms.n_rdm)
        self.assertEqual(sqrt_rdm.n_cond, rdms.n_cond)

    def test_positive_transform(self):
        from rsatoolbox.rdm import positive_transform
        dis = np.random.rand(8, 10) - 0.5
        mes = "Euclidean"
        des = {'subj': 0}
        pattern_des = {'type': np.array([0, 1, 2, 2, 4])}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        rdm_descriptors=rdm_des,
                        pattern_descriptors=pattern_des,
                        dissimilarity_measure=mes,
                        descriptors=des)
        pos_rdm = positive_transform(rdms)
        self.assertEqual(pos_rdm.n_rdm, rdms.n_rdm)
        self.assertEqual(pos_rdm.n_cond, rdms.n_cond)
        assert np.all(pos_rdm.dissimilarities >= 0)

    def test_rdm_append(self):
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'subj': 0}
        pattern_des = {'type': np.array([0, 1, 2, 2, 4])}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        pattern_descriptors=pattern_des,
                        dissimilarity_measure=mes,
                        descriptors=des,
                        rdm_descriptors=rdm_des)
        rdms.append(rdms)
        self.assertEqual(rdms.n_rdm, 16)

    def test_concat(self):
        from rsatoolbox.rdm import concat
        dis = np.zeros((8, 10))
        dis2 = np.random.rand(8, 10)
        mes = "Euclidean"
        des = {'subj': 0}
        pattern_des = {'type': np.array([0, 1, 2, 2, 4])}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7])}
        rdms1 = rsr.RDMs(dissimilarities=dis,
                         pattern_descriptors=pattern_des,
                         dissimilarity_measure=mes,
                         descriptors=des,
                         rdm_descriptors=rdm_des)
        rdms2 = rsr.RDMs(dissimilarities=dis2,
                         pattern_descriptors=pattern_des,
                         dissimilarity_measure=mes,
                         descriptors=des,
                         rdm_descriptors=rdm_des)
        rdms = concat((rdms1, rdms2))
        self.assertEqual(rdms.n_rdm, 16)
        assert len(rdms.rdm_descriptors['session']) == 16

    def test_concat_varargs_multiple_rdms(self):
        from rsatoolbox.rdm import concat
        dis = np.zeros((8, 10))
        dis2 = np.random.rand(8, 10)
        mes = "Euclidean"
        des = {'subj': 0}
        pattern_des = {'type': np.array([0, 1, 2, 2, 4])}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7])}
        rdms1 = rsr.RDMs(dissimilarities=dis,
                         pattern_descriptors=pattern_des,
                         dissimilarity_measure=mes,
                         descriptors=des,
                         rdm_descriptors=rdm_des)
        rdms2 = rsr.RDMs(dissimilarities=dis2,
                         pattern_descriptors=pattern_des,
                         dissimilarity_measure=mes,
                         descriptors=des,
                         rdm_descriptors=rdm_des)
        rdm_c1 = concat((rdms1, rdms2))
        rdm_c2 = concat(rdms1, rdms2)
        self.assertEqual(rdm_c1.n_rdm, 16)
        self.assertEqual(rdm_c2.n_rdm, 16)
        assert_array_equal(rdm_c1.dissimilarities, rdm_c2.dissimilarities)

    def test_concat_varargs_one_rdm(self):
        from rsatoolbox.rdm import concat
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'subj': 0}
        pattern_des = {'type': np.array([0, 1, 2, 2, 4])}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        pattern_descriptors=pattern_des,
                        dissimilarity_measure=mes,
                        descriptors=des,
                        rdm_descriptors=rdm_des)
        rdm_c1 = concat(rdms)
        self.assertEqual(rdm_c1.n_rdm, 8)
        assert_array_equal(rdm_c1.dissimilarities, rdms.dissimilarities)

    def test_categorical_rdm(self):
        from rsatoolbox.rdm import get_categorical_rdm
        category_vector = [1, 2, 2, 3]
        rdm = get_categorical_rdm(category_vector)
        np.testing.assert_array_almost_equal(
            rdm.dissimilarities,
            np.array([[1., 1., 1., 0., 1., 1.]]))

    def test_reorder(self):
        from rsatoolbox.rdm import RDMs
        rdm = np.array([
            [0., 1., 2., 3.],
            [1., 0., 1., 2.],
            [2., 1., 0., 1.],
            [3., 2., 1., 0.]]
        )
        conds = ['a', 'b', 'c', 'd']
        rdms = RDMs(
            np.atleast_2d(squareform(rdm)),
            pattern_descriptors=dict(conds=conds)
        )
        conds_ordered = ['b', 'a', 'c', 'd']
        new_order = [conds.index(cond_idx)
                     for cond_idx in conds_ordered]
        rdm_reordered = rdm[np.ix_(new_order, new_order)]
        rdms.reorder(new_order)
        assert_array_equal(
            np.atleast_2d(squareform(rdm_reordered)),
            rdms.dissimilarities
        )
        assert_array_equal(
            conds_ordered,
            rdms.pattern_descriptors.get('conds')
        )

    def test_sort_by_alpha(self):
        from rsatoolbox.rdm import RDMs
        rdm = np.array([
            [0., 1., 2., 3.],
            [1., 0., 1., 2.],
            [2., 1., 0., 1.],
            [3., 2., 1., 0.]]
        )
        conds = ['b', 'a', 'c', 'd']
        rdms = RDMs(
            np.atleast_2d(squareform(rdm)),
            pattern_descriptors=dict(conds=conds)
        )
        rdms.sort_by(conds='alpha')
        new_order = np.argsort(conds)
        rdm_reordered = rdm[np.ix_(new_order, new_order)]
        self.assertIsNone(
            assert_array_equal(
                np.atleast_2d(squareform(rdm_reordered)),
                rdms.dissimilarities
            )
        )
        self.assertIsNone(
            assert_array_equal(
                sorted(conds),
                rdms.pattern_descriptors.get('conds')
            )
        )

    def test_sort_by_list(self):
        from rsatoolbox.rdm import RDMs
        rdm = np.array([
            [0., 1., 2., 3.],
            [1., 0., 1., 2.],
            [2., 1., 0., 1.],
            [3., 2., 1., 0.]]
        )
        conds = ['a', 'b', 'c', 'd']
        rdms = RDMs(
            np.atleast_2d(squareform(rdm)),
            pattern_descriptors=dict(conds=conds)
        )
        conds_new_order = ['c', 'd', 'b', 'a']
        rdms.sort_by(conds=conds_new_order)

        new_order = np.array([conds.index(c) for c in conds_new_order])
        rdm_reordered = rdm[np.ix_(new_order, new_order)]
        self.assertIsNone(
            assert_array_equal(
                np.atleast_2d(squareform(rdm_reordered)),
                rdms.dissimilarities
            )
        )
        self.assertIsNone(
            assert_array_equal(
                conds_new_order,
                rdms.pattern_descriptors.get('conds')
            )
        )


class TestSave(unittest.TestCase):
    def test_dict_conversion(self):
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'subj': 0}
        pattern_des = {'type': np.array([0, 1, 2, 2, 4])}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7])}
        rdms = rsa.rdm.RDMs(
            dissimilarities=dis,
            pattern_descriptors=pattern_des,
            dissimilarity_measure=mes,
            descriptors=des,
            rdm_descriptors=rdm_des)
        rdm_dict = rdms.to_dict()
        rdms_loaded = rsa.rdm.rdms_from_dict(rdm_dict)
        assert rdms_loaded.n_cond == rdms.n_cond
        assert np.all(rdms_loaded.pattern_descriptors['type']
                      == pattern_des['type'])
        assert np.all(rdms_loaded.rdm_descriptors['session']
                      == rdm_des['session'])
        assert rdms_loaded.descriptors['subj'] == 0

    def test_save_load(self):
        import io
        f = io.BytesIO()  # Essentially a Mock file
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'subj': 0}
        pattern_des = {'type': np.array([0, 1, 2, 2, 4])}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7])}
        rdms = rsa.rdm.RDMs(
            dissimilarities=dis,
            pattern_descriptors=pattern_des,
            dissimilarity_measure=mes,
            descriptors=des,
            rdm_descriptors=rdm_des)
        rdms.save(f, file_type='hdf5')
        rdms_loaded = rsa.rdm.load_rdm(f, file_type='hdf5')
        assert rdms_loaded.n_cond == rdms.n_cond
        assert np.all(rdms_loaded.pattern_descriptors['type']
                      == pattern_des['type'])
        assert np.all(rdms_loaded.rdm_descriptors['session']
                      == rdm_des['session'])
        assert rdms_loaded.descriptors['subj'] == 0


class TestRDMLists(unittest.TestCase):
    """ checking that descriptors stay lists if they are specified as such"""

    def setUp(self):
        dissimilarities = np.random.rand(3, 15)
        des = {'session': 0, 'subj': np.arange(7)}
        rdm_des = {'test': ['a', np.arange(5), None]}
        pattern_des = {'test': [np.arange(1), np.arange(2), np.arange(3),
                                np.arange(4), np.arange(5), np.arange(6)],
                       'test2': [1, 2, 3, 4, 5, 6],
                       'test3': [np.arange(1), np.arange(2), np.arange(1),
                                 np.arange(1), np.arange(1), 'b']}
        self.test_rdm = rsa.rdm.RDMs(
            dissimilarities=dissimilarities,
            dissimilarity_measure='test',
            descriptors=des,
            rdm_descriptors=rdm_des,
            pattern_descriptors=pattern_des
            )

    def test_rdm_init_list(self):
        assert isinstance(self.test_rdm.rdm_descriptors['index'], list)
        assert isinstance(self.test_rdm.pattern_descriptors['index'], list)
        assert isinstance(self.test_rdm.rdm_descriptors['test'], list)
        assert isinstance(self.test_rdm.pattern_descriptors['test'], list)
        assert isinstance(self.test_rdm.pattern_descriptors['test2'], list)
        assert isinstance(self.test_rdm.pattern_descriptors['test3'], list)

    def test_rdm_indexing(self):
        from copy import deepcopy
        rdm = deepcopy(self.test_rdm)
        rdms = rdm[1, 2]
        rdm = rdm[0]
        assert isinstance(rdms.rdm_descriptors['index'], list)
        assert isinstance(rdms.pattern_descriptors['index'], list)
        assert isinstance(rdms.rdm_descriptors['test'], list)
        assert isinstance(rdms.pattern_descriptors['test'], list)
        assert isinstance(rdms.pattern_descriptors['test2'], list)
        assert isinstance(rdms.pattern_descriptors['test3'], list)
        self.assertEqual(rdms.n_rdm, 2)
        assert isinstance(rdm.rdm_descriptors['index'], list)
        assert isinstance(rdm.pattern_descriptors['index'], list)
        assert isinstance(rdm.rdm_descriptors['test'], list)
        assert isinstance(rdm.pattern_descriptors['test'], list)
        assert isinstance(rdm.pattern_descriptors['test2'], list)
        assert isinstance(rdm.pattern_descriptors['test3'], list)
        self.assertEqual(rdm.n_rdm, 1)

    def test_rdm_subset_list(self):
        sub_rdm = self.test_rdm.subset('index', 0)
        sub_rdm2 = self.test_rdm.subset('index', [0, 1])
        assert isinstance(self.test_rdm.rdm_descriptors['index'], list)
        assert isinstance(self.test_rdm.pattern_descriptors['index'], list)
        assert isinstance(self.test_rdm.rdm_descriptors['test'], list)
        assert isinstance(self.test_rdm.pattern_descriptors['test'], list)
        assert isinstance(self.test_rdm.pattern_descriptors['test2'], list)
        assert isinstance(self.test_rdm.pattern_descriptors['test3'], list)
        assert isinstance(sub_rdm.rdm_descriptors['index'], list)
        assert isinstance(sub_rdm.pattern_descriptors['index'], list)
        assert isinstance(sub_rdm.rdm_descriptors['test'], list)
        assert isinstance(sub_rdm.pattern_descriptors['test'], list)
        assert isinstance(sub_rdm.pattern_descriptors['test2'], list)
        assert isinstance(sub_rdm.pattern_descriptors['test3'], list)
        assert isinstance(sub_rdm2.rdm_descriptors['index'], list)
        assert isinstance(sub_rdm2.pattern_descriptors['index'], list)
        assert isinstance(sub_rdm2.rdm_descriptors['test'], list)
        assert isinstance(sub_rdm2.pattern_descriptors['test'], list)
        assert isinstance(sub_rdm2.pattern_descriptors['test2'], list)
        assert isinstance(sub_rdm2.pattern_descriptors['test3'], list)

    def test_rdm_subsample_list(self):
        sub_rdm = self.test_rdm.subsample('index', 0)
        sub_rdm2 = self.test_rdm.subsample('index', [0, 1, 1])
        assert isinstance(self.test_rdm.rdm_descriptors['index'], list)
        assert isinstance(self.test_rdm.pattern_descriptors['index'], list)
        assert isinstance(self.test_rdm.rdm_descriptors['test'], list)
        assert isinstance(self.test_rdm.pattern_descriptors['test'], list)
        assert isinstance(self.test_rdm.pattern_descriptors['test2'], list)
        assert isinstance(self.test_rdm.pattern_descriptors['test3'], list)
        assert isinstance(sub_rdm.rdm_descriptors['index'], list)
        assert isinstance(sub_rdm.pattern_descriptors['index'], list)
        assert isinstance(sub_rdm.rdm_descriptors['test'], list)
        assert isinstance(sub_rdm.pattern_descriptors['test'], list)
        assert isinstance(sub_rdm.pattern_descriptors['test2'], list)
        assert isinstance(sub_rdm.pattern_descriptors['test3'], list)
        assert isinstance(sub_rdm2.rdm_descriptors['index'], list)
        assert isinstance(sub_rdm2.pattern_descriptors['index'], list)
        assert isinstance(sub_rdm2.rdm_descriptors['test'], list)
        assert isinstance(sub_rdm2.pattern_descriptors['test'], list)
        assert isinstance(sub_rdm2.pattern_descriptors['test2'], list)
        assert isinstance(sub_rdm2.pattern_descriptors['test3'], list)
        self.assertEqual(sub_rdm2.n_rdm, 3)

    def test_pattern_subset_list(self):
        sub_rdm = self.test_rdm.subset_pattern('test2', 4)
        sub_rdm2 = self.test_rdm.subset_pattern('index', [0, 1])
        assert isinstance(self.test_rdm.rdm_descriptors['index'], list)
        assert isinstance(self.test_rdm.pattern_descriptors['index'], list)
        assert isinstance(self.test_rdm.rdm_descriptors['test'], list)
        assert isinstance(self.test_rdm.pattern_descriptors['test'], list)
        assert isinstance(self.test_rdm.pattern_descriptors['test2'], list)
        assert isinstance(self.test_rdm.pattern_descriptors['test3'], list)
        assert isinstance(sub_rdm.rdm_descriptors['index'], list)
        assert isinstance(sub_rdm.pattern_descriptors['index'], list)
        assert isinstance(sub_rdm.rdm_descriptors['test'], list)
        assert isinstance(sub_rdm.pattern_descriptors['test'], list)
        assert isinstance(sub_rdm.pattern_descriptors['test2'], list)
        assert isinstance(sub_rdm.pattern_descriptors['test3'], list)
        assert isinstance(sub_rdm2.rdm_descriptors['index'], list)
        assert isinstance(sub_rdm2.pattern_descriptors['index'], list)
        assert isinstance(sub_rdm2.rdm_descriptors['test'], list)
        assert isinstance(sub_rdm2.pattern_descriptors['test'], list)
        assert isinstance(sub_rdm2.pattern_descriptors['test2'], list)
        assert isinstance(sub_rdm2.pattern_descriptors['test3'], list)
        self.assertEqual(sub_rdm.n_cond, 1)
        self.assertEqual(sub_rdm2.n_cond, 2)

    def test_pattern_subsample_list(self):
        sub_rdm = self.test_rdm.subsample_pattern('test2', 4)
        sub_rdm2 = self.test_rdm.subsample_pattern('index', [0, 1, 1, 2])
        assert isinstance(self.test_rdm.rdm_descriptors['index'], list)
        assert isinstance(self.test_rdm.pattern_descriptors['index'], list)
        assert isinstance(self.test_rdm.rdm_descriptors['test'], list)
        assert isinstance(self.test_rdm.pattern_descriptors['test'], list)
        assert isinstance(self.test_rdm.pattern_descriptors['test2'], list)
        assert isinstance(self.test_rdm.pattern_descriptors['test3'], list)
        assert isinstance(sub_rdm.rdm_descriptors['index'], list)
        assert isinstance(sub_rdm.pattern_descriptors['index'], list)
        assert isinstance(sub_rdm.rdm_descriptors['test'], list)
        assert isinstance(sub_rdm.pattern_descriptors['test'], list)
        assert isinstance(sub_rdm.pattern_descriptors['test2'], list)
        assert isinstance(sub_rdm.pattern_descriptors['test3'], list)
        assert isinstance(sub_rdm2.rdm_descriptors['index'], list)
        assert isinstance(sub_rdm2.pattern_descriptors['index'], list)
        assert isinstance(sub_rdm2.rdm_descriptors['test'], list)
        assert isinstance(sub_rdm2.pattern_descriptors['test'], list)
        assert isinstance(sub_rdm2.pattern_descriptors['test2'], list)
        assert isinstance(sub_rdm2.pattern_descriptors['test3'], list)
        self.assertEqual(sub_rdm2.n_cond, 4)

    def test_rdm_append_list(self):
        from copy import deepcopy
        long_rdm = deepcopy(self.test_rdm)
        long_rdm.append(long_rdm)
        assert isinstance(long_rdm.rdm_descriptors['index'], list)
        assert isinstance(long_rdm.pattern_descriptors['index'], list)
        assert isinstance(long_rdm.rdm_descriptors['test'], list)
        assert isinstance(long_rdm.pattern_descriptors['test'], list)
        assert isinstance(long_rdm.pattern_descriptors['test2'], list)
        assert isinstance(long_rdm.pattern_descriptors['test3'], list)
        self.assertEqual(long_rdm.n_rdm, 6)


if __name__ == '__main__':
    unittest.main()
