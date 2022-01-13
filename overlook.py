# %%
import json
import numpy as np
import pandas as pd
import plotly.express as px

from tqdm.auto import tqdm
from pathlib import Path

# %%
# List json files in ./annotations
json_files = [(e.name, e.as_posix())
              for e in Path.cwd().joinpath('./annotations/').iterdir()
              if e.name.endswith('.json')]

json_files = pd.DataFrame(
    json_files, columns=['name', 'fullPath']).set_index('name', drop=False)

print('Found json files:')
print(json_files)

# %%
# Read the json file
_path = json_files.loc['instances_val2014.json', 'fullPath']
obj = json.load(open(_path))
print('Read {} for keys: {}'.format(_path, [e for e in obj]))

# %%
# Convert the annotations into the DataFrame format
annotations = pd.DataFrame(obj['annotations'])
print('The "Annotation" contains the records')
print(annotations)

# %%
categories = pd.DataFrame(obj['categories'])
print('The "Categories" contains the records')
print(categories)

# %%
# Compute the area of bbox


def _area(e):
    return e[2] * e[3]


annotations['bboxArea'] = annotations['bbox'].map(_area)
annotations

# %%
# Say something
print('There are {} images'.format(len(annotations['image_id'].unique())))
print('There are {} categories'.format(
    len(annotations['category_id'].unique())))

# %%
# fig = px.box(df, x='category_id', y='bboxArea')
# fig.show()


# %%
# Make Occipital Count Heat Map,
# A pixel-map of Occipital Overlapping Count

# Assume the Image size is 800 x 800
# Make sure it is bigger than the pictures

# --------------------------------------------------------------------------------
# The mat is the count map of every images
image_id_table = dict()

count = 0
for e in sorted(annotations['image_id'].unique()):
    image_id_table[e] = count
    count += 1

mat = np.zeros((count, 800, 800), dtype=np.uint8)

print(mat.shape)

# --------------------------------------------------------------------------------
# Count the count for every record
for j in tqdm(range(len(annotations))):
    b = [int(e) for e in annotations.loc[j, 'bbox']]
    i = image_id_table[annotations.loc[j, 'image_id']]
    mat[i, b[0]:b[0]+b[2], b[1]:b[1]+b[3]] += 1

print(mat.shape)

# %%
max_mat = np.max(mat, axis=0)

# %%
fig = px.imshow(max_mat, title='Occipital Count Heat Map')
# fig.write_html(Path.cwd().joinpath('Occipital Count Heat Map.html'))
fig.show()

# %%
