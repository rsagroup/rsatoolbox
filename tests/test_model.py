#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the model subpackage
"""

import unittest
import rsatoolbox.model as model
import numpy as np


class TestModel(unittest.TestCase):
    """ Tests for the Model superclass
    """

    def test_creation(self):
        m = model.Model('Test Model')


class TestModelFixed(unittest.TestCase):
    """ Tests for the fixed model class
    """

    def test_creation(self):
        rdm = np.array(np.ones(6))
        m = model.ModelFixed('Test Model', rdm)
        m.fit([])
        pred = m.predict()
        assert np.all(pred == rdm)

    def test_creation_rdm(self):
        from rsatoolbox.rdm import RDMs
        rdm = np.array(np.ones(6))
        rdm_obj = RDMs(np.array([rdm]))
        m = model.ModelFixed('Test Model', rdm_obj)
        m.fit(rdm_obj)
        pred = m.predict()
        assert np.all(pred == rdm)
        pred_obj = m.predict_rdm()
        assert isinstance(pred_obj, RDMs)


class TestModelSelect(unittest.TestCase):
    """ Tests for the fixed model class
    """

    def test_creation(self):
        rdm = np.random.rand(2, 6)
        m = model.ModelSelect('Test Model', rdm)
        pred = m.predict()
        assert np.all(pred == rdm[0])

    def test_creation_rdm(self):
        from rsatoolbox.rdm import RDMs
        rdm = np.random.rand(2, 6)
        pattern_descriptors = {'test': ['a', 'b', 'c', 'd']}
        rdm_obj = RDMs(rdm, dissimilarity_measure='euclid',
                       pattern_descriptors=pattern_descriptors)
        m = model.ModelSelect('Test Model', rdm_obj)
        pred = m.predict()
        assert np.all(pred == rdm[0])
        pred_obj = m.predict_rdm()
        assert isinstance(pred_obj, RDMs)
        assert pred_obj.n_rdm == 1
        assert pred_obj.pattern_descriptors == pattern_descriptors

    def test_fit(self):
        from rsatoolbox.rdm import RDMs
        rdm = np.random.rand(2, 6)
        pattern_descriptors = {'test': ['a', 'b', 'c', 'd']}
        rdm_descriptors = {'ind': np.array([1, 2])}
        rdm_obj = RDMs(rdm, dissimilarity_measure='euclid',
                       pattern_descriptors=pattern_descriptors,
                       rdm_descriptors=rdm_descriptors)
        m = model.ModelSelect('Test Model', rdm_obj)
        train = rdm_obj.subset('ind', 2)
        theta = m.fit(train)
        assert theta == 1


class TestModelWeighted(unittest.TestCase):
    """ Tests for the fixed model class
    """

    def test_creation(self):
        rdm = np.random.rand(4, 15)
        m = model.ModelWeighted('Test Model', rdm)
        pred = m.predict([1, 0, 0, 0])
        assert np.all(pred == rdm[0])

    def test_creation_rdm(self):
        from rsatoolbox.rdm import RDMs
        rdm = np.random.rand(2, 6)
        pattern_descriptors = {'test': ['a', 'b', 'c', 'd']}
        rdm_obj = RDMs(rdm, dissimilarity_measure='euclid',
                       pattern_descriptors=pattern_descriptors)
        m = model.ModelWeighted('Test Model', rdm_obj)
        pred = m.predict(np.array([1, 0]))
        assert np.all(pred == rdm[0])
        pred_obj = m.predict_rdm()
        assert isinstance(pred_obj, RDMs)
        assert pred_obj.n_rdm == 1
        assert pred_obj.pattern_descriptors == pattern_descriptors

    def test_fit(self):
        from rsatoolbox.rdm import RDMs
        rdm = np.random.rand(2, 6)
        pattern_descriptors = {'test': ['a', 'b', 'c', 'd']}
        rdm_descriptors = {'ind': np.array([1, 2])}
        rdm_obj = RDMs(rdm, dissimilarity_measure='euclid',
                       pattern_descriptors=pattern_descriptors,
                       rdm_descriptors=rdm_descriptors)
        m = model.ModelWeighted('Test Model', rdm_obj)
        train = rdm_obj.subset('ind', 2)
        theta = m.fit(train)


class TestModelInterpolate(unittest.TestCase):
    """ Tests for the fixed model class
    """

    def test_creation(self):
        rdm = np.random.rand(4, 15)
        m = model.ModelInterpolate('Test Model', rdm)
        pred = m.predict([1, 0, 0, 0])
        assert np.all(pred == rdm[0])

    def test_creation_rdm(self):
        from rsatoolbox.rdm import RDMs
        rdm = np.random.rand(2, 6)
        pattern_descriptors = {'test': ['a', 'b', 'c', 'd']}
        rdm_obj = RDMs(rdm, dissimilarity_measure='euclid',
                       pattern_descriptors=pattern_descriptors)
        m = model.ModelInterpolate('Test Model', rdm_obj)
        pred = m.predict(np.array([1, 0]))
        assert np.all(pred == rdm[0])
        pred_obj = m.predict_rdm()
        assert isinstance(pred_obj, RDMs)
        assert pred_obj.n_rdm == 1
        assert pred_obj.pattern_descriptors == pattern_descriptors

    def test_fit(self):
        from rsatoolbox.rdm import RDMs
        rdm = np.random.rand(5, 15)
        pattern_descriptors = {'test': ['a', 'b', 'c', 'd', 'e', 'f']}
        rdm_descriptors = {'ind': np.array([1, 2, 3, 1, 2])}
        rdm_obj = RDMs(rdm, dissimilarity_measure='euclid',
                       pattern_descriptors=pattern_descriptors,
                       rdm_descriptors=rdm_descriptors)
        m = model.ModelInterpolate('Test Model', rdm_obj)
        train = rdm_obj.subset('ind', 2)
        theta = m.fit(train)
        pre = m.predict(theta)
