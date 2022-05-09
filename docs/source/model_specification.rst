.. _model:

Model Specification
===================

To run a RSA analysis, we need to define the models to be evaluated. For this the rsatoolbox provides a class with subclasses for different
model types. Any type of model needs to define three functions ``predict``, ``predict_rdm``, and ``fit``. The ``predict`` and ``predict_rdm`` functions
return the prediction of the model, taking a model parameter vector ``theta`` as input. ``predict`` returns a vectorized numpy array format for efficient computation, ``predict_rdm`` returns a RDMs object. For flexible models, the ``fit`` function estimates the parameter vector (``theta``) based on some data RDMs (see :ref:`model fitting <model_fitting>`).

Model types are defined in ``rsatoolbox.model``. Most of them can be initialized with a name and an RDMs object, which defines the RDM(s), which
are combined into a prediction.

Fixed models
------------
.. _ModelFixed:

The simplest type of model are *fixed models*, which simply predict a single RDM and don not have any parameters to be fit. They are available
as ``rsatoolbox.model.ModelFixed`` which can be created based on a single RDM. To generate a model for a RDM saved in rdm with name 'test'
could use the following code:


.. code-block:: python

    import rsatoolbox
    model = rsatoolbox.model.ModelFixed('test', rdm)


To extract the prediction of this model, which will always be the RDM provided at its creation, you can use its ``predict`` and ``predict_rdm``
functions. The ``fit`` function dos nothing.

.. code-block:: python

    pred = model.predict() # returns a numpy vectorized format
    pred_rdm = model.predict_rdm() # returns a RDMs object

These methods also take a ``theta`` argument, which corresponds to the parameter vector of the model. For fixed models, this input is ignored however.

Weighted models
---------------
.. _ModelWeighted:

The first type of flexible models we handle are *weighted models*, which are available as ``rsatoolbox.model.ModelWeighted``. These models
predict the RDM as a weighted sum of a set of given RDMs. The typical use case for these models is feature weighting, i.e. when a theory
contains multiple features or parts which contribute to the measured dissimilarities, whose relative weighting is not known a priori.
Typical sources for the RDMs are the feature dimensions, sets of them like DNN layers, or the RDMs of completely separate model parts,
which are thought to be mixed by the measurement.

To generate a model for a set of RDMs saved in rdm with name 'test' could use the following code:

.. code-block:: python

    import rsatoolbox
    model = rsatoolbox.model.ModelWeighted('test', rdms)

The simplest method for fitting this kind of model is an unconstrained linear fit, which maximizes the chosen RDM similarity metric,
allowing both negative and positive weights for the RDM. More correctly, the weights for each feature component should be constrained
to be postive. See :ref:`model fitting <model_fitting>` for more information.

Fitting this type of model generally works better with continuous RDM comparison measures than with the rank correlations.


Selection models
----------------
.. _ModelSelect:

*Selection models* are models which predict that the true RDM is one of a set of given RDMs. They are available as ``rsatoolbox.model.ModelSelect``.
Fitting this model is simply done by choosing the RDM, which is closest to the training data RDM as implemented in ``rsatoolbox.model.fit_select``.

If there are discrete different versions of the model, which represent the same theory this represents this uncertainty best. This model can also
be used to represent any other uncertainty about the RDM approximately. To do so, sample the range of possible RDMs and let a selection model
choose the best setting for you.

Interpolation models
--------------------
.. _ModelInterpolate:

*Interpolation models* predict that the RDM is a linear interpolation between two consecutive RDMs in the list given to the model. They are available as ``rsatoolbox.model.ModelInterpolate``.
Fitting this model is done by doing a bisection optimization on each line segment as implemented in ``rsatoolbox.model.fit_interpolate``.

These models' primary use is to represent nonlinear effects of a single changed parameter. When the RDMs given to the model are generated
by computing a model RDM under different settings of the parameter the interpolation effectively implements an approximation to the nonlinear
1D manifold of RDMs that the model can produce under arbitrary settings of the parameter.


Noise ceiling models
--------------------
.. _Model_nc:

The computation of a noise ceiling is often conceptualized as evaluating a model, which can arbitrarily set all distances of the RDM.
As the ``rsatoolbox`` currently computes the noise ceiling using analytic methods and does not explicitly create this model, it currently does not
provide an implementation of this maximally flexible model.
