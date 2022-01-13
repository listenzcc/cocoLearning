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
# Fix the annotations
categories['idx'] = range(len(categories))
categories.set_index('id', drop=False, inplace=True)


def _full(e):
    return '{}-{}'.format(e['supercategory'], e['name'])


categories['full'] = categories.apply(_full, axis=1)
categories

# %%
# Say something
n_images = len(annotations['image_id'].unique())
n_categories = len(annotations['category_id'].unique())
print('There are {} images'.format(n_images))
print('There are {} categories'.format(n_categories))

# %%
# fig = px.box(df, x='category_id', y='bboxArea')
# fig.show()


# %%
'''
# Make Occipital Count Heat Map,
# A pixel-map of Occipital Overlapping Count

# Assume the Image size is 800 x 800
# Make sure it is bigger than the pictures
'''

img_max_size = (640, 640)

mat_images = np.zeros((n_images, img_max_size[0], img_max_size[1]),
                      dtype=np.uint8)

mat_categories = np.zeros((n_categories, img_max_size[0], img_max_size[1]),
                          dtype=np.uint16)

print(mat_images.shape, mat_categories.shape)

# %%
# --------------------------------------------------------------------------------
# The mat is the count map of every images
image_id_table = dict()

count = 0
for e in tqdm(sorted(annotations['image_id'].unique()), 'Building image_id_table'):
    image_id_table[e] = count
    count += 1


# --------------------------------------------------------------------------------
# Count the count for every record
for j in tqdm(range(len(annotations)), 'Iter Records'):
    # Compute bbox
    b = [int(e) for e in annotations.loc[j, 'bbox']]
    i = image_id_table[annotations.loc[j, 'image_id']]
    mat_images[i, b[0]:b[0]+b[2], b[1]:b[1]+b[3]] += 1

    # Compute category_id
    c = int(annotations.loc[j, 'category_id'])
    i = categories.loc[c, 'idx']
    mat_categories[i, b[0]:b[0]+b[2], b[1]:b[1]+b[3]] += 1

print(mat_images.shape, mat_categories.shape)

# %%
max_mat = np.max(mat_images, axis=0)

# %%
name = 'Occlude Count Heat Map'
fig = px.imshow(max_mat, title=name)
path = Path.cwd().joinpath('largeFiles', '{}.html'.format(name))
fig.write_html(path)
fig.write_image('{}.png'.format(path))
fig.show()


# %%

for j in tqdm(range(n_categories), 'Draw Categories'):
    name = categories.iloc[j]['full']
    path = Path.cwd().joinpath('largeFiles', '{}.html'.format(name))

    fig = px.imshow(mat_categories[j], title=name)
    fig.write_html(path)
    fig.write_image('{}.png'.format(path))
    pass


# %%
mat = mat_categories.copy().astype(np.float32)

for j in range(len(mat)):
    mat[j] = mat[j] / np.max(mat[j])

# %%
mean = np.mean(mat, axis=0)
std = np.std(mat, axis=0)

title = 'Std - Categories'
fig = px.imshow(std, title=title)
path = Path.cwd().joinpath('largeFiles', '{}.html'.format(title))
fig.write_html(path)
fig.write_image('{}.png'.format(path))
fig.show()

title = 'Mean - Categories'
fig = px.imshow(mean, title=title)
path = Path.cwd().joinpath('largeFiles', '{}.html'.format(title))
fig.write_html(path)
fig.write_image('{}.png'.format(path))
fig.show()

# %%
title = 'Area - BoxGraph'
path = Path.cwd().joinpath('largeFiles', '{}.html'.format(title))
fig = px.box(annotations, title=title, y='area',
             x='category_id', log_y=True, points=False)
fig.write_html(path)
fig.write_image('{}.png'.format(path))
fig.show()

# %%
