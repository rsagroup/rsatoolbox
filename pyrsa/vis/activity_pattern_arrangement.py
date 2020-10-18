import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE


def activity_pattern_arrangement(rdm, condition_icons, method=None):
    """ visualize the relationships among a set of experimental conditions
    (activity patterns) as captured by an RDM by arranging the conditions
    with MDS, isomap, or t-SNE
    Args:
        rdm(pyrsa.rdm.RDMs): an RDMs class object
        condition_icons(dictionary): a dictionary of Icons for all images
        method(string): 'mds', 'isomap', or 'tsne'
        significance()
    Returns:
        None
    """
    
    if len(condition_icons) != rdm.n_cond:
        raise TypeError("The number of conditions in RDM does not match the number of condition icons")
    
    rdmm = rdm.get_matrices()
    
    
    if method == 'mds' or method is None:
        emb = MDS(n_components=2, dissimilarity='precomputed')
    elif method == 'isomap':
        emb = Isomap(n_components=2)
    elif method == 'tsne':
        emb = TSNE(n_components=2)
    else:
        raise TypeError("Incorrect method")
        
    
    for i in np.arange(rdms.n_rdm):
        drs = emb.fit_transform(rdmm[i, :, :])
        
        fig, ax = plt.subplots()
        fig.set_size_inches(30, 30)
        ax.set_xlim(drs.min() - 0.01, drs.max() + 0.01)
        ax.set_ylim(drs.min() - 0.01, drs.max() + 0.01)
        ax.axis('off')

        for j,icon in zip(np.arange(rdm.n_cond),condition_icons.values()):
            imagebox = OffsetImage(icon.image.resize((100,100)))
            ab = AnnotationBbox(imagebox, (drs[j,0],drs[j,1]), frameon=False)
            ax.add_artist(ab)
            
