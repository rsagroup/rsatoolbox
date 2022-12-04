"""Script to analyze colormap preference ratings
"""
from os.path import expanduser, join
import pandas

fname = 'Meadows_rsatoolbox_color_maps_v_v3_annotations.csv'
fpath = join(expanduser('~/data/rsatoolbox/colormaps/'), fname)
df = pandas.read_csv(fpath)
df['nmoves'] = [int(l.split('_')[0]) for l in df.label.values]
df['order'] = ['_'.join(l.split('_')[1:]).split('-') for l in df.label.values]
votes = dict([(c, list()) for c in df.iloc[0].order])
for _, row in df[df.nmoves > 0].iterrows():
    for rank, cmap in enumerate(row['order']):
        votes[cmap].append(rank)
print(sorted(votes.items(), key=lambda cmap_tuple: sum(cmap_tuple[1])))

""" 3 votes:
[('bone', [1, 0, 0]),
 ('gray', [0, 1, 1]),
 ('inferno', [6, 3, 3]),
 ('cividis', [3, 5, 5]),
 ('magma', [7, 4, 2]),
 ('crest_r', [2, 6, 8]),
 ('current', [5, 2, 9]),
 ('viridis', [4, 9, 4]),
 ('plasma', [8, 8, 6]),
 ('flare_r', [9, 7, 7])]
"""