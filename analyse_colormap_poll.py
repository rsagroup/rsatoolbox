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
[('bone', [1, 0, 0, 0, 1, 2]),
 ('gray', [0, 4, 1, 1, 2, 0]),
 ('cividis', [3, 1, 5, 5, 3, 1]),
 ('crest_r', [2, 3, 6, 8, 0, 3]),
 ('magma', [7, 5, 4, 2, 5, 7]),
 ('inferno', [6, 6, 3, 3, 6, 8]),
 ('flare_r', [9, 2, 7, 7, 4, 6]),
 ('viridis', [4, 8, 9, 4, 8, 4]),
 ('current', [5, 9, 2, 9, 9, 5]),
 ('plasma', [8, 7, 8, 6, 7, 9])]
"""