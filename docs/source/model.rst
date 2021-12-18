.. _model:

Model Specification
===================

To run a RSA analysis we need to define the models to be evaluated. For this the rsatoolbox provides a class with subclasses for different
types. Any type of model needs to define three functions ``predict``, ``predict_rdm``, and ``fit``. The ``predict`` and ``predict_rdm`` functions
return the prediction of the model, taking a model parameter vector ``theta`` as input. ``predict`` returns a vectorized numpy array format for efficient computation,
``predict_rdm`` returns a RDMs object. The ``fit`` function takes a data RDMs object as input
and returns a parameter value. Additionally, every model object requires a name.

Model types are defined in ``rsatoolbox.model``. Most of them can be initialized with a name and an RDMs object, which defines the RDM(s), which
are combined into a prediction. All models come equipped with a default fitting function. Other fitting methods can be found in ``rsatoolbox.model.fitter``.
To use them you can either apply them manually and pass them along separately, or you may set them as the default of a model by settings its
``default_fitter`` property.

Fitting methods
---------------
.. _modelfit:

All fitting methods take the following inputs: A ``model`` object, ``data``, a data RDMs object to fit to, ``method``, a string that defines which similarity
measure is optimized, and a ``pattern_idx``, ``pattern_descriptor`` combination that defines which subset of the RDM the provided
data RDMs correspond to. Additionally, they might take a ``ridge_weight``, which adds a penalty on the squared norm of the weights vector to the objective
and ``sigma_k``, which is required for adjusting the whitened correlation and cosine similarity measures to dependent measurements.

For simple fitting of a single model simply apply the fit function to the model and data as shown in the following example:

.. code-block:: python

    import rsatoolbox
    # generate 2 random model RDMs of 10 conditions
    model_features = [rsatoolbox.data.Dataset(np.random.rand(10, 7)) for i in range(2)]
    model_rdms = rsatoolbox.rdm.calc_rdm(model_features)
    model = rsatoolbox.model.ModelWeighted('test', model_rdms)

    # generate 5 random data RDMs of 10 conditions
    data = [rsatoolbox.data.Dataset(np.random.rand(10, 7)) for i in range(5)]
    data_rdms = rsatoolbox.rdm.calc_rdm(data)

    # fit model to data to maximize cosine similarity using its default fitter
    theta = model.fit(data_rdms, method='cosine')

    # explicitly use the fit_optimize function to do the fit
    theta2 = rsatoolbox.model.fitter.fit_optimize(model, data_rdms, method='cosine')

To provide an object, which fixes some parameters of the fitting function, rsatoolbox provides the fitter object. This type of object
is defined as ``rsatoolbox.model.fitter.Fitter`` and takes a fitting function and additional keyword arguments as inputs.
It then behaves as the original fitting function, with defaults changed to given keyword arguments.

To create a fitting function which sets the ``ridge_weight`` to 1 by default you could use the following code for example:

.. code-block:: python

    # fix some parameter of the function by using a fitter object:
    fitter = rsatoolbox.model.fitter.Fitter(rsatoolbox.model.fit_optimize, ridge_weight=1)
    theta3 = fitter(model, data_rdms, method='cosine')

Observe that this does indeed slightly change the fitted parameters compared to ``theta`` and ``theta2``

Both the fitting functions themselves and the fitter objects can be used as inputs to the crossvalidation and bootstrap-crossvalidation
methods to change how models are fit to data.

Fixed models
------------
.. _ModelFixed:

The simplest type of model are *fixed models*, which simply predict a single RDM and don't have any parameters to be fit. They are available
as ``rsatoolbox.model.ModelFixed`` which can be created based on a single RDM. To generate a model for a RDM saved in rdm with name 'test'
could use the following code:


.. code-block:: python

    import rsatoolbox
    model = rsatoolbox.model.ModelFixed('test', rdm)


To extract the prediction of this model, which will always be the RDM provided at its creation, you can use its ``predict`` and ``predict_rdm``
functions.

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

The simplest method for fitting this kind of model is an unconstrained linear fit, which maximizes the chosen RDM similarity metric.
For achieving this the rsatoolbox provides two separate methods: ``rsatoolbox.fitter.fit_optimize``, which uses a general purpose optimization method
and ``rsatoolbox.fitter.fit_regress`` which employs linear algebra analytic solutions.

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
As the ``rsatoolbox`` currently computes the noise ceiling using analytic methods and does not explicily create this model, it currently does not
provide an implementation of this maximally flexible model.
