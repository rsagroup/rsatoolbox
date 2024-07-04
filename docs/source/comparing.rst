.. _comparing:

Comparing RDMs
==============

To compare RDMs, we need a measure of their similarity.
There are various measures to compare RDMs, which are chosen based on what aspects the model RDMs are meant to capture.
The strictest measure to use would be to simply compute a distance between a model-RDM and the reference-RDMs.
In virtually all cases we cannot predict the exact magnitude of the distances though,
as the signal to noise ratio varies between subjects and measurement sessions.
Allowing an overall scaling of the RDMs leads to the cosine similarity.
If we additionally drop the assumption that a predicted difference of 0 corresponds to a measured dissimilarity of 0
we can use a correlation between RDMs.
For the cosine similarity and correlation between RDMs whitened variants can take the correlations between the different entries of the RDM
into account.
Finally, we can drop the assumption of a linear relationship between RDMs by using rank correlations like Kendall's tau or Spearman's rho.
For this lowest bar for a relationship Kendall's :math:`\tau_a` or randomised rank breaking for Spearman's :math:`\rho_a` are usually preferred
over a standard Spearman's :math:`\rho` or Kendall's :math:`\tau_b` and :math:`\tau_c`, which all favor RDMs with tied ranks.
As we discuss below there is a direct formula for the expected Spearman's rho under random tiebraking, which we prefer now for computational efficiency reasons.

All comparison methods are implemented in ``rsatoolbox.rdm``. They can each be accessed by passing a method argument to ``rsatoolbox.rdm.compare``
or by using a specific function ``rsatoolbox.rdm.compare_[comparison]``. The comparison functions each take two RDMs objects as input
and return a matrix of all pairwise comparisons.

Cosine similarity
-----------------

The most stringent similarity measure for RDMs is the cosine similarity. For two vectorized RDMs :math:`\mathbf{r}_1` and :math:`\mathbf{r}_2`
it is defined as:

.. math::

    \frac{\mathbf{r}_1^T \mathbf{r}_2}{\sqrt{\mathbf{r}_1^T\mathbf{r}_1\,\mathbf{r}_2^T\mathbf{r}_2}}

This comparison measure can be accessed using ``method='cosine'`` or using ``rsatoolbox.rdm.compare_cosine``.

Pearson Correlation
-------------------
When a dissimilarity of 0 is not interpret-able as indistinguishable, the average dissimilarity can be removed by using the Pearson correlation as a similarity measure.
It is defined as:


.. math::

    \frac{(\mathbf{r}_1- \bar{\mathbf{r}}_1)^T (\mathbf{r}_2- \bar{\mathbf{r}}_2)}{\sqrt{(\mathbf{r}_1- \bar{\mathbf{r}}_1)^T (\mathbf{r}_1- \bar{\mathbf{r}}_1)\,(\mathbf{r}_2 -\bar{\mathbf{r}}_2)^T (\mathbf{r}_2- \bar{\mathbf{r}}_2)}},

where the bar indicates the mean of the vector.

This comparison measure can be accessed using ``method='corr'`` or using ``rsatoolbox.rdm.compare_correlation``.

Whitened comparison measures
----------------------------
We recently derived a formula for the covariance of RDM entries, which arises because all dissimilarities of a single condition are based
on the same measurements of that condition (see Diedrichsen_2021_). Based on a simplified estimate of this covariance :math:`V`
we can then compute a whitened cosine similarity as:


.. math::

    \frac{\mathbf{r}_1^T V^{-1} \mathbf{r}_2}{\sqrt{\mathbf{r}_1^TV^{-1}\mathbf{r}_1\,\mathbf{r}_2^TV^{-1}\mathbf{r}_2}}

and a whitened correlation as:

.. math::

    \frac{(\mathbf{r}_1- \bar{\mathbf{r}}_1)^T V^{-1}(\mathbf{r}_2- \bar{\mathbf{r}}_2)}{\sqrt{(\mathbf{r}_1-\bar{\mathbf{r}}_1)^T V^{-1}(\mathbf{r}_1-\bar{\mathbf{r}}_1)(\mathbf{r}_2-\bar{\mathbf{r}}_2)^T V^{-1}(\mathbf{r}_2-\bar{\mathbf{r}}_2)}}

The cosine similarity measures are exactly equivalent to a linear centered kernel alignment (CKA) and the correlation is equivalent to the cosine similarity after removing the mean.
This equivalent formulation can be computed faster as it avoids the inversion of :math:`V`. Thus, our implementation uses these
equivalent formulation for faster computation in the background.

These comparison measures can be accessed using ``method='corr_cov'`` and ``method=='cosine_cov'`` or using ``rsatoolbox.rdm.compare_correlation_cov_weighted`` and ``rsatoolbox.rdm.compare_cosine_cov_weighted``.

Kendall's tau
-------------
Kendals :math:`\tau_a` is implemented for backward comparisons. It implements a rank correlation, which does not favor with tied ranks.
Consider Spearman's :math:`\rho_a` as a faster alternative.

This comparison measure can be accessed using ``method='tau-a'`` or using ``rsatoolbox.rdm.compare_kendall_tau``.

Spearman's rho
--------------
Spearman's rank-correlation in its original form is higher for predictions with tied ranks, which introduces an unwanted bias into analyses.
As a solution earlier versions recommended the use of Kendall's :math:`\tau_a` to remove this problem. This problem can also be solved by using
the expected Spearman's :math:`\rho` under random tiebreaking as an evaluation criterion instead. This coefficient was called :math:`\rho_a` by Kendall.
For this expectation there is a direct formula based on the rank transformed entries of the two RDMs :math:`\mathbf{x}` and :math:`\mathbf{y}`:

.. math::

    \rho_a(\mathbf{x},\mathbf{y})
    &=&\mathop{\mathbb{E}_{\substack{
    \tilde{\mathbf{a}}=\tilde{\mathbf{x}}-\frac{1}{n}\sum_{i=1}^{n}{i},\tilde{\mathbf{x}} \sim Rae(\mathbf{x})\\
    \tilde{\mathbf{b}}=\tilde{\mathbf{y}}-\frac{1}{n}\sum_{i=1}^{n}{i},\tilde{\mathbf{y}} \sim Rae(\mathbf{y})}}
    \biggl[
    \frac{
    \tilde{\mathbf{a}}^\top\tilde{\mathbf{b}}}
    {\|\tilde{\mathbf{a}}\|_2\|\tilde{\mathbf{b}}\|_2}
    \biggr]}\\
    &=&\frac{12}{n^3-n}\mathop{\mathbb{E}_{\tilde{\mathbf{a}}}
    [ \tilde{\mathbf{a}}]^\top}
    \mathop{\mathbb{E}_{\tilde{\mathbf{b}}}
    [ \tilde{\mathbf{b}}] }\\
    &=& \frac{12\mathbf{x}^\top\mathbf{y}}{n^3-n} - \frac{3(n+1)}{n-1}

Using :math:`\rho_a` is much faster to compute and the best average RDM for a set of data RDMs is easily computed, which are two important advantages.
Thus, we generally recommend using this :math:`\rho_a` measure now.

This comparison measure can be accessed using ``method='rho-a'`` or using ``rsatoolbox.rdm.compare_rho_a``.

Bures's rho
--------------
These are a realted similarity measure and distance introduced by harvey_2024_ , based on double centered kernel matrices :math:`K_1` and :math:`K_2`.
The normalized Bures similarity (NBS) is defined as:

.. math::

    NBS(K_1, K_2) = \frac{\mathcal{F}(K_1, K_2)}{\sqrt{\operatorname{Tr}[K_1] \operatorname{Tr}[K_2]}}
    \mathcal{F}(K_1, K_2) = \operatorname{Tr}[(K_1^{1/2}K_2K_1^{1/2})^{1/2}]

and :math:`\mathcal{F}` is known as the fidelity.

and relatedly the Bures distance :math:`\mathcal{B}`, a proper metric is defined as:

.. math::
    \mathcal{B}^2(K_1, K_2) = \operatorname{Tr}[K_1] \operatorname{Tr}[K_2] - 2 \operatorname{Tr}[(K_1^{1/2}K_2K_1^{1/2})^{1/2}]



.. _Diedrichsen_2021: https://arxiv.org/abs/2007.02789
.. _harvey_2024: https://proceedings.mlr.press/v243/harvey24a
