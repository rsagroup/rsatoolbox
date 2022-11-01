from rsatoolbox.model import ModelInterpolate, ModelWeighted
from rsatoolbox.model.fitter import fit_regress, fit_optimize_positive
from rsatoolbox.rdm import concat, compare
from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm
from rsatoolbox.rdm import concat
import numpy as np



rdms = []
for _ in range(5):
    data = np.random.rand(6, 20)
    data_s = Dataset(data)
    rdms.append(calc_rdm(data_s))
rdms = concat(rdms)

#def test_two_rdms_nan

for i_method in ['corr_cov']: #['cosine', 'corr', 'cosine_cov', 'corr_cov']:

    for i in range(200):
        rdms = []
        for _ in range(5):
            data = np.random.rand(6, 20)
            data_s = Dataset(data)
            rdms.append(calc_rdm(data_s))
        rdms = concat(rdms)
        rdms = rdms.subsample_pattern('index', [0, 1, 1, 3, 4, 5])
        model_rdms = concat([rdms[0], rdms[1]])
        model_weighted = ModelWeighted(
            'm_weighted',
            model_rdms)
        model_interpolate = ModelInterpolate(
            'm_interpolate',
            model_rdms)
        # theta_m_i = model_interpolate.fit(rdms, method=i_method)
        theta_m_w = model_weighted.fit(rdms, method=i_method)
        # theta_m_w_pos = fit_optimize_positive(
        #     model_weighted, rdms, method=i_method)
        theta_m_w_linear = fit_regress(model_weighted, rdms, method=i_method)
        # eval_m_i = np.mean(compare(model_weighted.predict_rdm(
        #     theta_m_i), rdms, method=i_method))
        eval_m_w = np.mean(compare(
            model_weighted.predict_rdm(theta_m_w),
            rdms,
            method=i_method
        ))
        # eval_m_w_pos = np.mean(compare(model_weighted.predict_rdm(
        #     theta_m_w_pos), rdms, method=i_method))
        eval_m_w_linear = np.mean(compare(
            model_weighted.predict_rdm(theta_m_w_linear),
            rdms,
            method=i_method
        ))
        #rdiff_wei_int = (eval_m_i - eval_m_w_pos) / eval_m_w_pos
        rdiff_reg_opt = (eval_m_w - eval_m_w_linear) / eval_m_w_linear
        assert np.all(theta_m_w > 0)
        if np.all(theta_m_w > 0):
            # does it only happen with negative theta?
            assert abs(rdiff_reg_opt) < 0.002
        # assert False
        #assert abs(rdiff_reg_opt) < 0.02

