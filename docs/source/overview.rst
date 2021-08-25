.. _overview:

Toolbox overview
================


.. figure:: _static/rsatoolbox_workflow.png

    *Overview over subpackages and work flow in rsatoolbox.*

The Figure above shows the most important subpackages (blue), classes (gray), modules (yellow) and auxillary materials (orange) of the RSA toolbox.
A common use of the RSA toolbox involves the following steps:

* Extract the data that you want to analyzed. The data is stored in the format of a ``rsatoolbox.data.Dataset`` object, see :ref:`datasets`.
* Use functions from the module ``rsatoolbox.rdm.calc`` to calculate a RDM from the data, with many options for different dissimilarity measures, see :ref:`distances`.
* Define RSA models by defining objects of the ``rsatoolbox.model`` class. For information on the different model types, see :ref:`model`.
* Models can be fitted to the data and then evaluated using the ``rsatoolbox.inference.evaluate`` module. Results of the evaluation is stored in a ``rsatoolbox.inference.results`` object.
* Dataset, RDMs, Models, and results can be visualized using the ``rsatoolbox.vis`` subpackage.
* For simulation of artificial data sets, you can used the ``rsatoolbox.sim.simulation`` module.

For an example of a complete workflow, see the "getting started with RSAtoolbox" Notebook, :ref:`demos`.