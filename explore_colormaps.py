"""Script to showcase various colormap options

- no neutral / bi-directional
- suitable for white/gray background
- perceptually uniform
- at least based on luminance (for colorblindness)

### todo's

- poll could be an OrderSort

some useful guides:

https://seaborn.pydata.org/tutorial/color_palettes.html
https://matplotlib.org/stable/tutorials/colors/colormaps.html#lightness-of-matplotlib-colormaps
"""
from os.path import join
from pkg_resources import resource_filename
import numpy
from matplotlib.pyplot import subplots, imshow, show
from scipy.io import loadmat
from rsatoolbox.rdm.rdms import RDMs, get_categorical_rdm
from rsatoolbox.vis.rdm_plot import show_rdm_panel, _rdm_colorbar

# various sample RDMs
data_92images = loadmat(join('demos', '92imageData', "92_brainRDMs.mat"))
data_sessions = RDMs(
    dissimilarities=numpy.array([s[0] for s in data_92images['RDMs'].flatten()])
)
data = data_sessions.mean()
model = get_categorical_rdm([0]*5 + [1]*5 + [0]*5 + [1]*5 + [0]*20)
noise = RDMs(dissimilarities=numpy.random.rand(780))
rdm_variants = [noise, model, data]

cm_options = [None, #'flare', 'crest'] #,
    'magma', 'viridis', 'plasma', 'inferno', 'cividis']
fig, axes = subplots(
    nrows=len(cm_options),
    ncols=len(rdm_variants)+1
)
for c, colormap in enumerate(cm_options):
    for r, rdm in enumerate(rdm_variants):
        img = show_rdm_panel(rdm, ax=axes[c, r], cmap=colormap)
    ax = axes[c, len(rdm_variants)]
    ax.imshow(numpy.tile(numpy.arange(100), [50, 1]).T, cmap=colormap)

show()