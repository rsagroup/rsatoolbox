.. _distances:

Estimating dissimilarities
==========================
The second step in RSA is to calculate a dissimilarity of distance measure from the data. 

To calculate all pairwise distance, one needs to define the appropriate dataset (see :ref:`dataset`) and use ``rdm.calc.calc_rdm`` to caluclate the appropriate rdm. 

.. sourcecode:: python

    data = pyrsa.data.Dataset(xxxx)
    pyrsa.rdm.calc.calc_rdm(data, method='euclidean', descriptor=None, noise=None)

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

Note that the distances are normalized by the number of channels ($P$). This enables comparision of distances across region with different number of voxels or neurons. 


.. _MahalanobisDist:

Mahalanobis Distance
--------------------

TO DO 

.. _CorrelationDist:

Correlation Distance
--------------------

TO DO 

.. _Crossnobis:

Crossnobis dissimilarity
------------------------
The crossvalidated squared Mahalanobis distance (short: crossnobis distance) is an unbiased distance. It only multiplies pattern estimates across runs, but never within a single run. Techncally, the crossnobis 

.. math::
    \begin{equation}
    d_{i,j}=\frac{1}{M (M-1)}\sum_{m}^M \sum_{n \neq m}^M (\hat{\mathbf{b}}_{i,m} - \hat{\mathbf{b}}_{j,m}) (\hat{\mathbf{b}}_{i,n} - \hat{\mathbf{b}}_{j,n})^T /P
    \end{equation}


The really big advanatage of this dissimilarity measure is that it is unbiased. If the true distance is zero (i.e. if two patterns only differ by noise), the average estimated distance will be zero. If there is not information in a set of activity patterns, then half the distance estimates will be positive, and half the estimates will be **negative**. This is not the case for the non-crossvalidated distances, which will always be positive, even if the two patterns are not different at all. Because the crossnobis dissimilarity can become negative, it is technically not a distance anymore (which need to be non-negative). However, it can be shown that it an **unbiased estimator of the square Mahalanobis distance** (Walther et al, 2016; Diedrichsen et al. 2020). Having an unbiased distance estimator has three advantages: 

* You can perform a t-test of the crossnobis estimates against zero, exactly like you would test the classification accuracy of a decoder against chance performance. Thus, you do not need to perform a decoding analysis to determine where there is reliable information, and then conduct an RSA analysis to make inferences about the shape of the representation. By using the crossnobis dissimilarity, you can do so in one step (see also \ref{unbiasedDistanceCorrelation}. 

* Unequal noise across conditions can serverly bias RDMs computed with normal distances. For example when you have less trials for one condition than another, the pattern for that condition :math:`\bar{\mathbf{b}}_i` has higher noise variance. Therefore the distance to other conditions will be higher, even though the condition only differs by noise. When two conditions are estimated with correlated noise (for example when they are acquired in close temrpora; proximity with fMRI), their distance will be smaller than when they are collected with indepenent noise. For example, with normal distances you can never compare distance with a imaging run to distances across imaging runs (due to correlated noise, the former will be usually smaller). Crossvalidation removes these biases, making inference much more robust. 

* Having a meaningful zero point (i.e. the true patterns are not different) can help in model comparision, as it provides another informative point (Diedrichsen et al., 2020). To exploit this, it is recommended to use the cosine similarity instead of the Peason correlation for RDM comparision. 