from __future__ import annotations
from unittest import TestCase


class Test_model_plot(TestCase):

    methods_supported = [
        "corr", "cosine", "cosine_cov", "spearman", "corr_cov",
        "tau-b", "tau-a", "neg_riem_dist", "rho-a"
    ]

    def test_y_label(self):
        from rsatoolbox.vis.model_plot import _get_y_label
        for this_method in self.methods_supported:
            with self.subTest(msg=f"Testing {this_method}..."):
                y_label = _get_y_label(this_method)
                self.assertIsInstance(y_label, str)

    def test_descr(self):
        from rsatoolbox.vis.model_plot import _get_model_comp_descr
        descr = _get_model_comp_descr(
            "t-test",
            5,
            "fwer",
            0.05,
            1000,
            "boostrap_rdm",
            "ci56",
            "droplets",
            "icicles",
        )
        EXPECTED = (
            'Model comparisons: two-tailed t-test, p < 0.05, '
            'Bonferroni-corrected for 10 model-pair comparisonsError bars '
            'indicate the 56.0% confidence interval.\nOne-sided comparisons '
            'of each model performance against 0 and against the lower-bound '
            'estimate of the noise ceiling are Bonferroni-corrected for '
            '5 models.')
        self.assertEqual(EXPECTED, descr)
