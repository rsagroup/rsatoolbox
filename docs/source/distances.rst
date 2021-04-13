.. _distances:

Estimating dissimilarities
==========================
The second step in RSA is to calculate a dissimilarity of distance measure from the data.

To calculate all pairwise distance, one needs to define the appropriate dataset (see :ref:`datasets`) and use ``rdm.calc.calc_rdm`` to caluclate the appropriate rdm.

.. sourcecode:: python

    data = pyrsa.data.Dataset(xxxx)
    rdm = pyrsa.rdm.calc_rdm(data, method='euclidean', descriptor=None, noise=None)

By default pyrsa calculates the squared distances for the distances where both the squared and non-squared distances are commonly used.
To calculate the non-squared RDM use sqrt_transform provided in pyrsa.rdm as follows:

.. sourcecode:: python

    rdm_nonsquare = pyrsa.rdm.sqrt_transform(rdm)

For more introduction to the calculation of RDMs in pyrsa look at the `example_dissimilarities.ipynb` notebook in the demos folder.

.. _EuclideanDist:

Euclidean Distance
------------------

The standard Euclidean distance is calculated between the average activity patterns each pair of conditions. We denote here the average estimated activity pattern as :math:`\bar{\mathbf{d}_i}`, which is the average of the individual estimates over the *M* data partition

.. math::
    \begin{equation}
    \bar{\mathbf{b}}_i=\frac{1}{M}\sum_{m}^M \hat{\mathbf{b}}_{i,m},
    \end{equation}

The Euclidean distance is then

.. math::
    \begin{equation}
    d_{i,j}=\sqrt{(\bar{\mathbf{b}}_i - \bar{\mathbf{b}}_j) (\bar{\mathbf{b}}_i - \bar{\mathbf{b}}_j)^ T/P}
    \end{equation}

Note that the distances are normalized by the number of channels ($P$). This enables comparison of distances across region with different number of voxels or neurons.


.. _MahalanobisDist:

Mahalanobis Distance
--------------------

The Mahalanobis distance rescales the measurement channels in the distance calculations to achieve a more stable RDM estimate.
To do so, requires a covariance estimate for the measurement noise :math:`\Sigma`. With this the Mahalanobis distance is defined as:

.. math::
    \begin{equation}
    d_{i,j}^2=(\bar{\mathbf{b}}_i - \bar{\mathbf{b}}_j) \Sigma^{-1} (\bar{\mathbf{b}}_i - \bar{\mathbf{b}}_j)^ T/P
    \end{equation}

This dissimilarity is normalized for the number of channels just as the Euclidean distance.
Also, both math:`d_{i,j}^2` and math:`d_{i,j}` have been used to produce RDMs. Pyrsa returns math:`d_{i,j}^2` by default. Use pyrsa.rdm.sqrt_transform to get math:`d_{i,j}`.

In pyrsa the computing the Mahalanobis distances is achieved by passing method='mahalanobis' and the noise precision math:`\Sigma^{-1}` as 'noise'.
The precision is the inverse of the noise covariance, which we use for computational efficiency reasons.

The Mahalanobis can be substantially more reliable than the standard Euclidean distance, especially if different channels are differently reliable or correlated with each other.
The main question for applying the Mahalanobis distance is how to estimate :math:`\Sigma`.
There are two sources of information about the noise covariance, which have been used for estimating :math:`\Sigma`:

* When a first level regression was used to estimate the :math:`\mathbf{b}_i`, the residuals from this analysis can be used to estimate the noise covariance.
* When multiple measurements are available for each stimulus, the deviations of the measurements :math:`\mathbf{b}_{im}` from the mean :math:`\bar{\mathbf{b}}_i` can be used to estimate the noise covariance.

Which one of these is used is usually based on which source provides more samples for a more reliable estimate.

Nonetheless, the large numbers of measurement channels compared to the number of samples available for the estimation of :math:`\Sigma` usually makes the raw sample covariance a bad estimate.
To solve this problem there are two methods used in the literature:

**Univariate noise normalisation**: By using only the diagonal part of the sample covariance, the Mahalanobis distance effectively normalises each channel by its variance and is equivalent to scaling the channels by their standard deviation before calculating the Euclidean distance.

**Shrinkage estimates**: Instead the covariance estimate can be 'shrunk' towards a simpler covariance estimate. Concretely, this means that our covariance estimate is:

.. math::
    \begin{equation}
    \hat{\Sigma} = \lambda S + (1-\lambda) C
    \end{equation}

where :math:`S` is either a multiple of the identity or the diagonal matrix of sample variances and :math:`C` is the full sample covariance.

These estimates are helpful, because they are guaranteed to be positive definite and thus invertible, which is necessary for the distance computation.
Also :math:`\lambda` can be chosen adaptively and automatically based on the data, such that no tuning is necessary.

In pyrsa, the noise computations are implemented as pyrsa.data.noise.prec_from_residuals and pyrsa.data.noise.prec_from_measurements to allow estimation of the noise precision based on
either arbitrary residuals or a dataset object whose pattern means are subtracted before calculating the covariance.
To switch between the different estimates of the covariance pass the methods


.. _Crossnobis:

Crossnobis dissimilarity
------------------------
The cross-validated squared Mahalanobis distance (short: crossnobis distance) is an unbiased distance. It only multiplies pattern estimates across runs, but never within a single run. Technically, the crossnobis

.. math::
    \begin{equation}
    d_{i,j}=\frac{1}{M (M-1)}\sum_{m}^M \sum_{n \neq m}^M (\hat{\mathbf{b}}_{i,m} - \hat{\mathbf{b}}_{j,m}) (\hat{\mathbf{b}}_{i,n} - \hat{\mathbf{b}}_{j,n})^T /P
    \end{equation}


The really big advantage of this dissimilarity measure is that it is unbiased. If the true distance is zero (i.e. if two patterns only differ by noise), the average estimated distance will be zero. If there is not information in a set of activity patterns, then half the distance estimates will be positive, and half the estimates will be **negative**. This is not the case for the non-crossvalidated distances, which will always be positive, even if the two patterns are not different at all. Because the crossnobis dissimilarity can become negative, it is technically not a distance anymore (which need to be non-negative). However, it can be shown that it an **unbiased estimator of the square Mahalanobis distance** (Walther et al, 2016; Diedrichsen et al. 2020). Having an unbiased distance estimator has three advantages:

* You can perform a t-test of the crossnobis estimates against zero, exactly like you would test the classification accuracy of a decoder against chance performance. Thus, you do not need to perform a decoding analysis to determine where there is reliable information, and then conduct an RSA analysis to make inferences about the shape of the representation. By using the crossnobis dissimilarity, you can do so in one step (see also \ref{unbiasedDistanceCorrelation}.

* Unequal noise across conditions can severely bias RDMs computed with normal distances. For example when you have less trials for one condition than another, the pattern for that condition :math:`\bar{\mathbf{b}}_i` has higher noise variance. Therefore the distance to other conditions will be higher, even though the condition only differs by noise. When two conditions are estimated with correlated noise (for example when they are acquired in close temrpora; proximity with fMRI), their distance will be smaller than when they are collected with independent noise. For example, one cannot compare 'normal' distances within an imaging run to distances across imaging runs (due to correlated noise, the former will be usually smaller). Cross-validation removes these biases, making inference much more robust.

* Having a meaningful zero point (i.e. the true patterns are not different) can help in model comparison, as it provides another informative point (Diedrichsen et al., 2020). To exploit this, it is recommended to use the cosine similarity instead of the Pearson correlation for RDM comparison.


.. _CorrelationDist:

Correlation Distance
--------------------

The correlation distance quantifies the dissimilarity between two patterns as :math:`1-r` based on the pearson correlation between the patterns `r`, i.e.:

.. math::
    \begin{equation}
    d_{i,j}= 1-r_{ij} = 1 - \frac{1}{\sigma_{b_i}\sigma_{b_j}}(\mathbf{b}_i - \mu_i)^T (\mathbf{b}_j - \mu_j)
    \end{equation}

where :math:`\mu` and :math:`\sigma` are the mean and standard deviation of the respective pattern over channels.

The correlation distance is part of RSA since the start and was found to be similarly reliable as the Euclidean-like dissimilarities (Walther et. al 2016).
The interpretation of correlation distances is harder than for the euclidean types though. The Euclidean-like dissimilarities all depend only on the difference between the two patterns.
In contrast, the correlation additionally depends on shared overall activations, such that an additional shared activity which does not impair decoding still reduces the correlation distance.


.. _PoissonDist:

Poisson Symmetrized KL-divergence
---------------------------------

The symmetrized-Kullback-Leibler distance was conceived to produce a better dissimilarity measure to be based on spike counts.
This dissimilarity measures the dissimilarity of two spike rates as the symmetrized KL-divergence between poisson distributions with those spike rates. Fortunately this results in the following simple formula for the dissimilarity:

.. math::
    \begin{align}
    d_{i,j}&= \frac{1}{2P}\sum_{k=1}^P KL(Poisson(\lambda_{ik})||Poisson(\lambda_{jk})) + KL(Poisson(\lambda_{jk})||Poisson(\lambda_{ik})) \\
    &= \frac{1}{2P}\sum_{k=1}^P (\lambda_{ik}-\lambda_{jk}) (\log(\lambda_{ik})-\log(\lambda_{jk}))
    \end{align}

Under the assumption of poisson noise this is clearly a sensible dissimilarity. The measure can be easily calculated without this assumption though and always measures a form of dissimilarity,
which weighs differences between large firing rates less strongly than difference between small firing rates.

One issue with this formulation is that zero firing rates cannot be allowed. To avoid this problem pyrsa adds a prior to the estimation of the firing rate.
The parameters of this prior can be passed as `prior_lambda` and `prior_weight`. The first parameter specifies the prior mean, which is 1 by default and the second specifies the weight relative to an observation, which is 0.1 by default.

.. _PoissonCVDist:

Cross-validated Poisson KL-divergence
-------------------------------------

The poisson symmetrized KL-divergence can be cross-validated in an analogue way to the Mahalanobis distance.
Given :math:`M` multiple measurements for each pattern the cross-validated poisson KL-divergence is:

.. math::
    \begin{align}
    d_{i,j}&= \frac{1}{2P}\frac{1}{M(M-1)}\sum_{m=1}^M\sum_{n=1}^M\sum_{k=1}^P KL(Poisson(\lambda_{imk})||Poisson(\lambda_{jnk})) + KL(Poisson(\lambda_{jmk})||Poisson(\lambda_{ink})) \\
    &= \frac{1}{2P}\frac{1}{M(M-1)}\sum_{m=1}^M\sum_{n=1}^M\sum_{k=1}^P (\lambda_{imk}-\lambda_{jmk}) (\log(\lambda_{ink})-\log(\lambda_{jnk}))
    \end{align}

This inherits the same advantages of cross-validation as the crossnobis dissimilarity defined above.
