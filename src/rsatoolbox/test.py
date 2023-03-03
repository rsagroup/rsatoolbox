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

    def setUp(self):
        from numpy import asarray
        from numpy.testing import assert_almost_equal
        from rsatoolbox.data.dataset import Dataset
        from rsatoolbox.rdm.rdms import RDMs
        self.data = Dataset(asarray([[0, 0], [1, 1], [2.0, 2.0]]))
        self.rdms = RDMs(asarray([[1.0, 2, 3], [3, 4, 5]]))
        self.array = asarray
        self.arrayAlmostEqual = assert_almost_equal

    def test_calc_compiled(self):
        """Covers similarity calculation with compiled code
        """
        from rsatoolbox.rdm.calc_unbalanced import calc_rdm_unbalanced
        rdms = calc_rdm_unbalanced(self.data)
        self.arrayAlmostEqual(rdms.dissimilarities, self.array([[1, 4, 1]]))

    def test_model_fit(self):
        """Covers model fitting with scipy
        """
        from rsatoolbox.model.model import ModelWeighted
        from rsatoolbox.model.fitter import fit_optimize
        theta = fit_optimize(ModelWeighted('F', self.rdms), self.rdms)
        self.arrayAlmostEqual(theta, [0.88, 0.47], decimal=2)

    def test_plotting_with_mpl(self):
        from rsatoolbox.vis.rdm_plot import show_rdm
        show_rdm(self.rdms)

    def test_mds(self):
        """
        Covers sklearn and mpl
        """
        from rsatoolbox.vis.scatter_plot import show_MDS
        show_MDS(self.rdms)

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
