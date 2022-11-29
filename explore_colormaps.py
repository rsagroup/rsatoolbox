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
import numpy
from matplotlib.pyplot import subplots, close
from scipy.io import loadmat
from seaborn import color_palette
from rsatoolbox.rdm.rdms import RDMs, get_categorical_rdm
from rsatoolbox.vis.rdm_plot import show_rdm_panel
from rsatoolbox.vis.colors import rdm_colormap

# various sample RDMs
data_92images = loadmat(join('demos', '92imageData', "92_brainRDMs.mat"))
data_sessions = RDMs(
    dissimilarities=numpy.array([s[0] for s in data_92images['RDMs'].flatten()])
)
data = data_sessions.mean()
model = get_categorical_rdm([0]*5 + [1]*5 + [0]*5 + [1]*5 + [0]*20)
noise = RDMs(dissimilarities=numpy.random.rand(780))
rdm_variants = [model, noise, data]

cm_options = [None, 'flare_r', 'crest_r',
    'magma', 'viridis', 'plasma', 'inferno', 'cividis']
fig, axes = subplots(
    ncols=len(rdm_variants)+1,
    nrows=len(cm_options),
    figsize=(len(rdm_variants)+1, len(cm_options))
)
for c, cmap in enumerate(cm_options):
    if cmap is None:
        colormap, cmap_name = rdm_colormap(), 'current'
    else:
        colormap, cmap_name = color_palette(cmap, as_cmap=True), cmap
    for r, rdm in enumerate(rdm_variants):
        img = show_rdm_panel(rdm, ax=axes[c, r], cmap=colormap)
    subplot_ax = axes[c, len(rdm_variants)]
    sample_fig, sample_axes = subplots(figsize=(2, 2))
    sample = numpy.tile(numpy.arange(100), [100, 1]).T
    for ax in [subplot_ax, sample_axes]:
        ax.imshow(sample, cmap=colormap)
        ax.axis('off')
        ax.set_title(cmap_name, fontsize=10)
    sample_fig.savefig(f'{cmap_name}.png', dpi=100)

fig.tight_layout()
fig.savefig('rdm_colormaps.png', dpi=300)
close('all')