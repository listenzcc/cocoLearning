
# %%
import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from minisom import MiniSom

from pathlib import Path
from tqdm.auto import tqdm

from read_annotations import annotations, image_id_table, categories, n_images, n_categories

# %%


def save_fig(fig, path):
    '''
    Save plotly figure as .html and .png format.

    Input:
    - fig: The plotly figure;
    - path: The path of the html format.
    '''
    fig.write_html('{}.html'.format(path))
    fig.write_image('{}.png'.format(path))
    print('Saved into path: {}'.format(path))
    fig.show()
    pass


save_path = Path.cwd().joinpath('largeFiles', 'som_analysis')
Path.mkdir(save_path, exist_ok=True)

# %%
# Build the MAT as zero matrix,
# it is the hot-coding of the categories of the images
MAT = np.zeros((n_categories, n_images), dtype=np.float64)
MATArea = np.zeros((n_categories, n_images), dtype=np.float64)

# %%
# --------------------------------------------------------------------------------
# Count the count for every record
for j in tqdm(range(len(annotations)), 'Iter Records'):
    # Get the idx of image
    img_idx = image_id_table[annotations.loc[j, 'image_id']]
    area = annotations.loc[j, 'area']

    # Get the category_id as c
    # Get the index of category by c as cat_idx
    c = int(annotations.loc[j, 'category_id'])
    cat_idx = categories.loc[c, 'idx']

    MAT[cat_idx, img_idx] = 1
    MATArea[cat_idx, img_idx] = area

MAT.shape, MATArea.shape


# %%
select = np.sum(MAT, axis=0) > 1
mat = MAT[:, select]
mat_area = MATArea[:, select]

label_idx_max = np.argmax(MATArea[:, select], axis=0)
labels_max_area = [categories.iloc[int(e)]['supercategory']
                   for e in tqdm(label_idx_max, 'Max Area Labeling')]

count_categories = np.sum(mat, axis=0, dtype=np.float64)

mat.shape, count_categories.shape, len(labels_max_area)

# %%
som = MiniSom(50, 50, 80, sigma=0.8, learning_rate=0.5)
som.train(mat.transpose(), 1000, verbose=True)


# %%
winners = [som.winner(mat.transpose()[j])
           for j in tqdm(range(mat.shape[1]), 'Winners of SOM')]

# %%
df = pd.DataFrame(winners, columns=['x', 'y'])
df['count'] = count_categories
df['label'] = labels_max_area
df

# %%
title = 'SOM project (Label by max area)'
fig = px.scatter_3d(df, title=title, x='x', y='y', z='count', color='label')

fig.update_traces(marker=dict(size=2,
                              line=dict(width=0,
                                        color='white')),
                  selector=dict(mode='markers'))

fig.update_layout(dict(scene={'aspectmode': 'data'}, width=600, height=600))

path = save_path.joinpath(title)
save_fig(fig, path)

# %%
title = 'SOM project (Count of categories)'
fig = px.scatter_3d(df, title=title, x='x', y='y', z='count', color='count')

fig.update_traces(marker=dict(size=2,
                              line=dict(width=0,
                                        color='white')),
                  selector=dict(mode='markers'))

fig.update_layout(dict(scene={'aspectmode': 'data'}, width=600, height=600))

path = save_path.joinpath(title)
save_fig(fig, path)


# %%
