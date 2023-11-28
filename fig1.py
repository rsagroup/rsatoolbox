"""rdm plot figure
"""

# imports
import matplotlib.pyplot as plt
import matplotlib
from rsatoolbox import vis
from rsatoolbox import rdm
import numpy as np
import os
import inspect
import scipy.io
from collections import defaultdict

# supporting functions and constants
DEMO_DIR = 'demos'
NEURON_DIR = os.path.join(DEMO_DIR, "92imageData")


def neuron_2008_icons(**kwarg):
    """ Load Krigeskorte et al. (2008, Neuron) images as Icon instances."""
    mat_path = os.path.join(NEURON_DIR, "Kriegeskorte_Neuron2008_supplementalData.mat")
    mat = scipy.io.loadmat(mat_path)
    colors = plt.get_cmap('Accent', lut=16).colors
    markers = list(matplotlib.markers.MarkerStyle('').markers.keys())
    icons = defaultdict(list)
    for this_struct in mat["stimuli_92objs"][0]:
        # we're going to treat the 4 binary indicators (human, face, animal, natural) as a
        # base2 binary string, and index into this array of colors accordingly.
        index = int("".join(
                     [
                         str(this_struct[this_key][0,0]) for this_key in (
                     'human', 'face', 'animal', 'natural')
                     ]
                 ),
                                  base=2)
        this_color = colors[index]
        icons['image'].append(vis.Icon(image=this_struct['image'],
                                       color=this_color,
                                       circ_cut='cut',
                                       border_type='conv',
                                       border_width=5,
                                       **kwarg))
        icons['string'].append(vis.Icon(string=this_struct['category'][0],
                                        color=this_color,
                                        font_color=this_color,
                                        **kwarg))
        icons['marker'].append(vis.Icon(marker=markers[index],
                                       color=this_color,
                                       **kwarg))
    return icons


def neuron_2008_rdms_fmri(**kwarg):
    """ Load Kriegeskorte et al. (2008, Neuron) fMRI RDMs as RDMs instance.
    All key-word arguments are passed to neuron_2008_images."""
    mat_path = os.path.join(NEURON_DIR, "92_brainRDMs.mat")
    mat = scipy.io.loadmat(mat_path)
    icons = neuron_2008_icons(**kwarg)
    # insert leading dim to conform with rsatoolbox nrdm x ncon x ncon convention
    return rdm.concat(
        [
            rdm.RDMs(
                dissimilarities=this_rdm["RDM"][None, :, :],
                dissimilarity_measure="1-rho",
                rdm_descriptors=dict(
                    zip(["ROI", "subject", "session"], this_rdm["name"][0].split(" | ")),
                    name=this_rdm["name"][0]
                ),
                pattern_descriptors=icons
            )
            for this_rdm in mat["RDMs"].flatten()
        ]
    )

rdms = neuron_2008_rdms_fmri()
rdms.dissimilarity_measure = '1-rho'
rdms_subset = rdms.subset_pattern('index', np.arange(1, 92, 4))
rdms_subset[0].rdm_descriptors['name']
#rdms_subset = rdm.rank_transform(rdms_subset)

# font.size: 9
# font.style: normal
# axes.grid: True
# axes.titlesize: 10
# axes.titleweight: bold
# axes.labelsize: 10
# axes.labelweight: normal
# grid.color: white
# xtick.major.size: 0
# xtick.labelsize: 9
# ytick.labelsize: 9
# ytick.major.size: 0

fig, ax, ret_val = vis.show_rdm(rdms_subset[0],
    pattern_descriptor='image',
    num_pattern_groups=3,
    icon_spacing=1.8, #1.1,
    show_colorbar='panel',
    figsize=(10, 10)
)
#fig.tight_layout()
fig.savefig('fig1.png', bbox_inches='tight', dpi=300)
plt.close('all')