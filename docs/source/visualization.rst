.. _visualization:

Visualization
=============


Plotting RDMs
-------------

The main function for showing RDMs is :func:`rsatoolbox.vis.rdm_plot.show_rdm`.
It is illustrated in :doc:`demo_rdm_vis`. It allows relatively detailed 
plotting of both individual RDMs, as well as combined figures with
multiple RDMs.


Scatter plots
-------------

Sometimes it may be helpful to display an RDM using a two-dimensional
scatter plot. This requires that the multi-dimensional structure of the RDM
is reduced. RSAtoolbox offers various functions for making such plots:

- :func:`show_MDS <rsatoolbox.vis.scatter_plot.show_MDS>`
- :func:`show_tSNE <rsatoolbox.vis.scatter_plot.show_tSNE>`
- :func:`show_iso <rsatoolbox.vis.scatter_plot.show_iso>`

.. _model plot:

Ploting model evaluations
-------------------------

Results objects can be plotted into the typical bar plot of model evaluations 
using ``rsatoolbox.vis.plot_model_comparison``. It takes 
a :doc:`Result object<Results objects>` as input and does all necessary
inferences based on the uncertainties stored in the results object. It
provides many options for changing the layout and the inferences performed.
