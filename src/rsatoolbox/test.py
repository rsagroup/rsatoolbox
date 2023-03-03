"""Skeleton test module

The full collection of unit and acceptance tests for rsatoolbox is kept
in a separate package that is not part of our distributables. It can be run
by checking out the rsatoolbox git repository.

The tests in this module are a limited number of basic so-called skeleton
tests, which check that the library and all its dependencies are installed
correctly. It is not exhaustive and assumes that unittests have passed
for other most package formats.
If rsatoolbox is installed, the tests can be run with:

`python -m unittest rsatoolbox.test`

These tests have to:

- not have any dependencies outside of direct rsatoolbox runtime dependencies
- be fast (a few seconds)
- test interfaces that depend on external packages
- test compiled code

In other words they have to check that all the moving parts are there,
without looking at very specific calculation outcomes.
"""
# pylint: disable=import-outside-toplevel, no-self-use
from unittest import TestCase


class SkeletonTests(TestCase):

    def test_calc_compiled(self):
        from numpy import asarray
        from rsatoolbox.data.dataset import Dataset
        from rsatoolbox.rdm.calc_unbalanced import calc_rdm_unbalanced
        rdms = calc_rdm_unbalanced(Dataset(asarray([[0,1,2], [2,3,4]])))
        self.assertEqual(rdms.dissimilarities, asarray([]))

    def test_model_fit(self):
        """

        This covers scipy
        """
        self.fail('todo')

    def test_plotting_with_mpl(self):
        self.fail('todo')

    def test_mds(self):
        """
        Covers sklearn and mpl
        """
        self.fail('todo')

    def test_evaluate(self):
        """
        Covers tqdm
        """
        #eval_bootstrap
        self.fail('todo')

    def test_pandas_io(self):
        self.fail('todo')

    def test_hdf_io(self):
        self.fail('todo')
