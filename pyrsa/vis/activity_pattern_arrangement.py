import numpy as np
from scipy import io
from sklearn.manifold import MDS
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
import pyrsa
from pyrsa.vis.icon import icons_from_folder



def activity_pattern_arrangement(rdm, condition_icons, method, significance=None):
    """ DISCRIPTION


    Args:
        rdm(PyRSA Object):
        condition_icons():
        method(String): 'mds', 'isomap', 'tsne'
        significance()

    Returns:
        None
    """


    rdm = rdm.dissimilarities

    if method == 'mds':
        mds = MDS(n_components=2)
        transformed_rdm = mds.fit_transform(rdm)

    elif method == 'isomap':
        isomap = Isomap(n_components=2)
        transformed_rdm = isomap.fit_transform(rdm)

    elif method == 'tsne':
        tsne = TSNE(n_components=2)
        transformed_rdm = tsne.fit_transform(rdm)


    # Obejects

    fig, ax = plt.subplots()


    for key, value in zip(images.keys(), images.values()):
        i = int(key.split('.')[0]) - 1
        value.image.thumbnail((128, 128), Image.ANTIALIAS)
        npImage = np.array(value.image)
        h, w = value.image.size

        alpha = Image.new('L', value.image.size, 0)
        draw = ImageDraw.Draw(alpha)
        draw.pieslice([0, 0, h, w], 0, 360, fill=255)

        npAlpha = np.array(alpha)
        npImage = np.dstack((npImage, npAlpha))


        imagebox = OffsetImage(npImage)
        imagebox.image.axes = ax
        ab = AnnotationBbox(imagebox, transformed_rdm[i], frameon=False)

        plt.autoscale(tight=True)

        ax.add_artist(ab)
        fig = plt.gcf()
        fig.set_size_inches(20, 20)
        ax.set_xlim(transformed_rdm.min(), transformed_rdm.max())
        ax.set_ylim(transformed_rdm.min(), transformed_rdm.max())

    plt.axis('off')
    plt.show()

    ## Circles
    labels = np.array([int(k.split('.')[0]) - 1 for k in images.keys()])
    labels[0:27] = 0
    labels[27:65] = 1
    labels[64:99] = 2
    labels[100:123] = 3
    labels[124:155] = 4

    plt.figure()
    target_name = ['animal', 'object', 'scene', 'people', 'face']
    colors = ['cyan', 'blue', 'red', 'green', 'yellow']

    for i, target_name in zip([0, 1, 2, 3, 4], target_name):
        plt.scatter(transformed_rdm[labels == i, 0],
                    transformed_rdm[labels == i, 1],
                    label=target_name, s=800, c=[colors[i]])


    plt.legend(loc='best', scatterpoints=1)
    fig = plt.gcf()
    fig.set_size_inches(20, 20)
    ax.set_xlim(transformed_rdm.min(), transformed_rdm.max())
    ax.set_ylim(transformed_rdm.min(), transformed_rdm.max())

    plt.axis('off')
    plt.show()


images_folder = 'Stimuli Set'
rdm = io.matlab.loadmat('RDM.mat')['RDM']
pyrsa_rdm = pyrsa.rdm.RDMs(rdm)
images = icons_from_folder(images_folder)

# plot
activity_pattern_arrangement(pyrsa_rdm, condition_icons=None, method='mds', significance=None)
activity_pattern_arrangement(pyrsa_rdm, condition_icons=None, method='isomap', significance=None)
activity_pattern_arrangement(pyrsa_rdm, condition_icons=None, method='tsne', significance=None)