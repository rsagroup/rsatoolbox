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
# pylint: disable=import-outside-toplevel
from unittest import TestCase


class SkeletonTests(TestCase):
    """Toolbox skeleton tests to ensure correct packaging and installation
    """

    def setUp(self):
        """Create basic RDMs and Dataset objects for all tests
        """
        from numpy import asarray, ones
        from numpy.testing import assert_almost_equal
        from rsatoolbox.data.dataset import Dataset
        from rsatoolbox.rdm.rdms import RDMs
        self.data = Dataset(asarray([[0, 0], [1, 1], [2.0, 2.0]]))
        self.rdms = RDMs(asarray([[1.0, 2, 3], [3, 4, 5]]))
        self.larger_rdms = RDMs(ones([3, 10]))
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
        """Covers Matplotlib usage
        """
        from rsatoolbox.vis.rdm_plot import show_rdm
        show_rdm(self.rdms)

    def test_mds(self):
        """Covers sklearn and Matplotlib usage
        """
        from rsatoolbox.vis.scatter_plot import show_MDS
        show_MDS(self.rdms)

    def test_evaluate(self):
        """Covers tqdm usage and evaluate functionality
        """
        from rsatoolbox.inference import eval_fixed
        from rsatoolbox.model import ModelFixed
        model = ModelFixed('G', self.array(list(range(10))))
        result = eval_fixed(model, self.larger_rdms)
        self.assertAlmostEqual(result.test_zero()[0], 0)

    def test_pandas_io(self):
        """Covers pandas usage
        """
        df = self.rdms.to_df()
        self.arrayAlmostEqual(
            self.array(df.loc[:, 'dissimilarity'].values),
            self.rdms.dissimilarities.ravel()
        )

    def test_hdf_io(self):
        """Covers h5py library use
        """
        from io import BytesIO
        from rsatoolbox.rdm.rdms import load_rdm
        fhandle = BytesIO()
        self.rdms.save(fhandle, file_type='hdf5')
        reconstituted_rdms = load_rdm(fhandle, file_type='hdf5')
        self.arrayAlmostEqual(
            self.rdms.dissimilarities,
            reconstituted_rdms.dissimilarities
        )
