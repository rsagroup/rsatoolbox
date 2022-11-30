"""Script to analyze colormap preference ratings
"""
from os.path import expanduser, join
import pandas

fname = 'Meadows_rsatoolbox_color_maps_v_v2_annotations.csv'
fpath = join(expanduser('~/data/rsatoolbox/colormaps/'), fname)
df = pandas.read_csv(fpath)
df['nmoves'] = [int(l.split('_')[0]) for l in df.label.values]
df['order'] = ['_'.join(l.split('_')[1:]).split('-') for l in df.label.values]
votes = dict([(c, list()) for c in df.iloc[0].order])
for _, row in df[df.nmoves > 0].iterrows():
    for rank, cmap in enumerate(row['order']):
        votes[cmap].append(rank)
print(sorted(votes.items(), key=lambda cmap_tuple: sum(cmap_tuple[1])))

"""
Out[20]:
[('inferno', [0, 1]),
 ('magma', [2, 0]),
 ('current', [3, 2]),
 ('plasma', [1, 7]),
 ('flare_r', [4, 5]),
 ('cividis', [7, 3]),
 ('crest_r', [6, 4]),
 ('viridis', [5, 6])]
"""