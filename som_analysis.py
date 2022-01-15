
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


save_path = Path.cwd().joinpath('largeFiles', 'category_space_analysis')
Path.mkdir(save_path, exist_ok=True)

# %%
# Build the MAT as zero matrix,
# it is the hot-coding of the categories of the images
MAT = np.zeros((n_categories, n_images), dtype=np.uint8)
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

label_idx_max = np.argmax(MATArea[:, select], axis=0)
labels_max_area = [categories.iloc[int(e)]['supercategory']
                   for e in tqdm(label_idx_max, 'Max Area Labeling')]

count_categories = np.sum(mat, axis=0, dtype=np.float64)

mat.shape, count_categories.shape, len(labels_max_area)

# %%
som = MiniSom(100, 100, 80, sigma=0.8, learning_rate=0.5)
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
fig = px.scatter(df, x='x', y='y', color='label')
fig.show()

# %%
fig = px.scatter(df, x='x', y='y', color='count')
fig.show()


# %%
