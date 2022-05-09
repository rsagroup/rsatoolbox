#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crossvalidation tests
@author: heiko
"""

import numpy as np
import unittest


class TestCrossval(unittest.TestCase):
    """ crossvalidation tests
    """

    def setUp(self):
        from rsatoolbox.rdm import RDMs
        from rsatoolbox.model import ModelFixed
        dis = np.random.rand(11, 190)  # 11 20x20 rdms
        mes = "Euclidean"
        des = {'subj': 0}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7, 7, 7, 7])}
        pattern_des = {'type': np.array([0, 1, 2, 2, 4, 5, 5, 5, 6, 7,
                                         10, 11, 12, 12, 14, 15, 15, 15,
                                         16, 17])}
        rdms = RDMs(dissimilarities=dis,
                    rdm_descriptors=rdm_des,
                    pattern_descriptors=pattern_des,
                    dissimilarity_measure=mes,
                    descriptors=des)
        m = ModelFixed('test', rdms[0])
        self.rdms = rdms
        self.m = m

    def test_crossval(self):
        from rsatoolbox.inference import crossval
        rdms = self.rdms
        m = self.m
        train_set = [(rdms.subset_pattern('type', [0, 1]), np.array([0, 1])),
                     (rdms.subset_pattern('type', [0, 4]), np.array([0, 4])),
                     ]
        test_set = [(rdms.subset_pattern('type', [2, 4]), np.array([2, 4])),
                    (rdms.subset_pattern('type', [1, 2]), np.array([1, 2])),
                    ]
        ceil_set = [(rdms.subset_pattern('type', [2, 4]), np.array([2, 4])),
                    (rdms.subset_pattern('type', [1, 2]), np.array([1, 2])),
                    ]
        crossval(m, rdms, train_set, test_set, ceil_set,
                 pattern_descriptor='type')

    def test_eval_fancy(self):
        from rsatoolbox.inference import eval_fancy
        eval_fancy(self.m, self.rdms, N=10, k_rdm=2, k_pattern=2,
                   pattern_descriptor='type',
                   rdm_descriptor='session')

    def test_bootstrap_crossval(self):
        from rsatoolbox.inference import bootstrap_crossval
        from rsatoolbox.rdm import RDMs
        from rsatoolbox.model import ModelFixed
        dis = np.random.rand(11, 190)  # 11 10x10 rdms
        mes = "Euclidean"
        des = {'subj': 0}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7, 7, 7, 7])}
        pattern_des = {'type': np.array([0, 1, 2, 2, 4, 5, 5, 5, 6, 7,
                                         10, 11, 12, 12, 14, 15, 15, 15,
                                         16, 17])}
        rdms = RDMs(dissimilarities=dis,
                    rdm_descriptors=rdm_des,
                    pattern_descriptors=pattern_des,
                    dissimilarity_measure=mes,
                    descriptors=des)
        m = ModelFixed('test', rdms[0])
        bootstrap_crossval(m, rdms, N=10, k_rdm=2, k_pattern=2,
                           pattern_descriptor='type',
                           rdm_descriptor='session')

    def test_bootstrap_cv_random(self):
        from rsatoolbox.inference import bootstrap_cv_random
        from rsatoolbox.rdm import RDMs
        from rsatoolbox.model import ModelFixed
        dis = np.random.rand(11, 190)  # 11 10x10 rdms
        mes = "Euclidean"
        des = {'subj': 0}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7, 7, 7, 7])}
        pattern_des = {'type': np.array([0, 1, 2, 2, 4, 5, 5, 5, 6, 7,
                                         10, 11, 12, 12, 14, 15, 15, 15,
                                         16, 17])}
        rdms = RDMs(dissimilarities=dis,
                    rdm_descriptors=rdm_des,
                    pattern_descriptors=pattern_des,
                    dissimilarity_measure=mes,
                    descriptors=des)
        m = ModelFixed('test', rdms[0])
        res = bootstrap_cv_random(
            m, rdms, N=10, n_rdm=2, n_pattern=4,
            pattern_descriptor='type',
            rdm_descriptor='session')
        self.assertEqual(res.evaluations.shape[0], 10)

    def test_bootstrap_crossval_pattern(self):
        from rsatoolbox.inference import bootstrap_crossval
        rdms = self.rdms
        m = self.m
        bootstrap_crossval(m, rdms, N=10, k_rdm=2, k_pattern=2,
                           pattern_descriptor='type',
                           rdm_descriptor='session',
                           boot_type='pattern')
        bootstrap_crossval(m, rdms, N=10, k_rdm=2, k_pattern=2,
                           pattern_descriptor='type',
                           boot_type='pattern')

    def test_bootstrap_crossval_rdm(self):
        from rsatoolbox.inference import bootstrap_crossval
        rdms = self.rdms
        m = self.m
        bootstrap_crossval(m, rdms, N=10, k_rdm=2, k_pattern=2,
                           pattern_descriptor='type',
                           rdm_descriptor='session',
                           boot_type='rdm')
        bootstrap_crossval(m, rdms, N=10, k_rdm=2, k_pattern=2,
                           rdm_descriptor='session',
                           boot_type='rdm')

    def test_bootstrap_crossval_k1(self):
        from rsatoolbox.inference import bootstrap_crossval
        rdms = self.rdms
        m = self.m
        bootstrap_crossval(m, rdms, N=10, k_rdm=1, k_pattern=2,
                           pattern_descriptor='type',
                           rdm_descriptor='session')
        bootstrap_crossval(m, rdms, N=10, k_rdm=2, k_pattern=1,
                           pattern_descriptor='type',
                           rdm_descriptor='session')

    def test_bootstrap_crossval_list(self):
        from rsatoolbox.inference import bootstrap_crossval
        from rsatoolbox.model import ModelFixed
        rdms = self.rdms
        m = ModelFixed('test', rdms[0])
        m2 = ModelFixed('test2', rdms[1])
        bootstrap_crossval([m, m2], rdms, N=10, k_rdm=2, k_pattern=2,
                           pattern_descriptor='type',
                           rdm_descriptor='session')

    def test_leave_one_out_pattern(self):
        from rsatoolbox.inference import sets_leave_one_out_pattern
        import rsatoolbox.rdm as rsr
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'subj': 0}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7])}
        pattern_des = {'category': np.array([0, 1, 2, 2, 3])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        rdm_descriptors=rdm_des,
                        dissimilarity_measure=mes,
                        pattern_descriptors=pattern_des,
                        descriptors=des)
        train_set, test_set, ceil_set = \
            sets_leave_one_out_pattern(rdms, 'category')
        assert len(test_set) == 4
        for i_test in test_set:
            assert i_test[0].n_cond <= 2

    def test_leave_one_out_rdm(self):
        from rsatoolbox.inference import sets_leave_one_out_rdm
        import rsatoolbox.rdm as rsr
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'subj': 0}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7])}
        pattern_des = {'category': np.array([0, 1, 2, 2, 3])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        rdm_descriptors=rdm_des,
                        dissimilarity_measure=mes,
                        pattern_descriptors=pattern_des,
                        descriptors=des)
        train_set, test_set, ceil_set = sets_leave_one_out_rdm(rdms)
        for i_test in test_set:
            assert i_test[0].n_rdm == 1

    def test_k_fold_pattern(self):
        from rsatoolbox.inference import sets_k_fold_pattern
        import rsatoolbox.rdm as rsr
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'subj': 0}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7])}
        pattern_des = {'category': np.array([0, 1, 2, 2, 3])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        rdm_descriptors=rdm_des,
                        dissimilarity_measure=mes,
                        pattern_descriptors=pattern_des,
                        descriptors=des)
        train_set, test_set, ceil_set = sets_k_fold_pattern(
            rdms, k=2, pattern_descriptor='category')
        assert test_set[0][0].n_cond == 2
        assert test_set[1][0].n_cond == 3

    def test_k_fold(self):
        from rsatoolbox.inference import sets_k_fold
        import rsatoolbox.rdm as rsr
        dis = np.random.rand(8, 10)
        mes = "Euclidean"
        des = {'subj': 0}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7])}
        pattern_des = {'category': np.array([0, 1, 2, 2, 3])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        rdm_descriptors=rdm_des,
                        dissimilarity_measure=mes,
                        pattern_descriptors=pattern_des,
                        descriptors=des)
        train_set, test_set, ceil_set = sets_k_fold(
            rdms, k_rdm=3, k_pattern=2,
            pattern_descriptor='category', rdm_descriptor='session',
            random=False)
        assert test_set[0][0].n_cond == 2
        assert test_set[1][0].n_cond == 3
        assert ceil_set[0][1] == test_set[0][1]
        assert ceil_set[0][0].n_rdm == train_set[0][0].n_rdm
        assert train_set[0][0].n_cond == 3
        assert train_set[1][0].n_cond == 2
        for train, test in zip(train_set, test_set):
            assert train[0].n_rdm + test[0].n_rdm == 8
            assert train[0].n_cond + test[0].n_cond == 5

    def test_k_fold_rdm(self):
        from rsatoolbox.inference import sets_k_fold_rdm
        import rsatoolbox.rdm as rsr
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'subj': 0}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7])}
        pattern_des = {'category': np.array([0, 1, 2, 2, 3])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        rdm_descriptors=rdm_des,
                        dissimilarity_measure=mes,
                        pattern_descriptors=pattern_des,
                        descriptors=des)
        train_set, test_set, ceil_set = sets_k_fold_rdm(
            rdms, k_rdm=3,
            rdm_descriptor='session',
            random=False)
        assert len(test_set) == 3
        assert len(train_set) == 3
        assert len(test_set[0]) == 2
        assert len(train_set[0]) == 2
        assert test_set[0][0].n_cond == 5
        assert test_set[1][0].n_cond == 5
        assert test_set[0][0].n_rdm == 3
        assert test_set[1][0].n_rdm == 3
        assert test_set[2][0].n_rdm == 2
        assert ceil_set[0][0].n_rdm == 5
        assert ceil_set[1][0].n_rdm == 5
        assert ceil_set[2][0].n_rdm == 6
        assert train_set[0][0].n_rdm == 5
        assert train_set[1][0].n_rdm == 5
        assert train_set[2][0].n_rdm == 6
        for train, test in zip(train_set, test_set):
            assert train[0].n_rdm + test[0].n_rdm == 8
        train_set, test_set, ceil_set = sets_k_fold_rdm(
            rdms, k_rdm=3, random=False)
        for train, test in zip(train_set, test_set):
            assert train[0].n_rdm + test[0].n_rdm == 8

    def test_sets_of_k_pattern(self):
        from rsatoolbox.inference import sets_of_k_pattern
        import rsatoolbox.rdm as rsr
        dis = np.zeros((8, 10))
        mes = "Euclidean"
        des = {'subj': 0}
        rdm_des = {'session': np.array([0, 1, 2, 2, 4, 5, 6, 7])}
        pattern_des = {'category': np.array([0, 1, 2, 2, 3])}
        rdms = rsr.RDMs(dissimilarities=dis,
                        rdm_descriptors=rdm_des,
                        dissimilarity_measure=mes,
                        pattern_descriptors=pattern_des,
                        descriptors=des)
        train_set, test_set, ceil_set = sets_of_k_pattern(
            rdms, k=2,
            pattern_descriptor='category', random=False)
        assert test_set[0][0].n_cond == 2
        assert test_set[1][0].n_cond == 3
        for train, test in zip(train_set, test_set):
            assert train[0].n_cond + test[0].n_cond == 5
