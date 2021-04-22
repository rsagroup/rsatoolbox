.. _rescale_partials:

How to get an average RDM from a stack of partial RDMs
======================================================

Here's an example on how to combine RDMs that do not cover all conditions.
One example of this is the trials of the Multiple Arrangements / inverse MDS task.
Another is where each participant only takes part in a subset of conditions.

.. code-block:: python

    from numpy import array
    from rsatoolbox.rdm.rdms import RDMs
    from rsatoolbox.rdm.combine import from_partials, rescale
    rdms1 = RDMs(
        array(1, 2, 3),
        pattern_descriptors={'conds': ['a', 'b', 'c']}
    )
    rdms2 = RDMs(
        array(6, 7, 8),
        pattern_descriptors={'conds': ['b', 'c', 'd']}
    )
    ## first, all rdms should have the same number of conditions, with NaNs for missing data
    partials = from_partials([rdms1, rdms2])
    ## then we rescale/align these based on pairs in common (put them in the same space)
    rescaledPartials = rescale(partials, method='evidence')
    ## then we can take a weighted average:
    meanRDM = rescaledPartials.mean(weights='rescalingWeights')
