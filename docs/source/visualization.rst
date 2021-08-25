.. _visualization:

Visualization
=============


Plotting RDMs
-------------

The main function for showing RDMs is ``rsatoolbox.vis.show_rdm``. It is illustrated in :doc:`demo_rdm_vis`. It allows relatively detailed plotting of RDMs
It takes a RDMs object as the main input.


Multidimensional Scaling
------------------------

To be documented.

.. _model plot:

Ploting model evaluations
-------------------------

Results objects can be plotted into the typical bar plot of model evaluations using the ``rsatoolbox.vis.plot_model_comparison``.
It takes a :doc:`Result object<Results objects>` as input and does all necessary inferences based on the uncertainties stored in the results object.
It provides many options for changing the layout and the inferences performed.
