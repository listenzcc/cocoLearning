# %%
import numpy as np
import pandas as pd
import plotly.express as px

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
    fig.write_html(path)
    fig.write_image('{}.png'.format(path))
    print('Saved into path: {}'.format(path))
    fig.show()
    pass


# %%
# Build the MAT as zero matrix,
# it is the hot-coding of the categories of the images
MAT = np.zeros((n_categories, n_images), dtype=np.uint8)

# %%
# --------------------------------------------------------------------------------
# Count the count for every record
for j in tqdm(range(len(annotations)), 'Iter Records'):
    # Get the idx of image
    img_idx = image_id_table[annotations.loc[j, 'image_id']]

    # Get the category_id as c
    # Get the index of category by c as cat_idx
    c = int(annotations.loc[j, 'category_id'])
    cat_idx = categories.loc[c, 'idx']

    MAT[cat_idx, img_idx] = 1

MAT.shape

# %%

forward_mat = np.zeros((n_categories, n_categories), dtype=np.float64)

prob_vec = np.zeros(n_categories, dtype=np.float64)

for cat_idx in tqdm(range(n_categories), 'Fill cov_mat'):
    mat = MAT[:, MAT[cat_idx] > 0]
    mean = np.mean(mat, axis=1)
    forward_mat[cat_idx] = mean
    prob_vec[cat_idx] = mat.shape[1] / n_images

np.fill_diagonal(forward_mat, 0)

# %%
title = 'Base prob.'
df = pd.DataFrame(prob_vec, columns=['prob'])
df['name'] = categories['name'].to_list()
df['super'] = categories['supercategory'].to_list()
fig = px.bar(df, title=title, y='name', x='prob',
             log_x=True, color='super', height=600)
path = Path.cwd().joinpath('largeFiles', '{}.html'.format(title))
save_fig(fig, path)

# %%
title = 'Forward prob.'
fig = px.imshow(forward_mat, title=title, height=600, width=600,
                x=categories['name'], y=categories['full'])
path = Path.cwd().joinpath('largeFiles', '{}.html'.format(title))
save_fig(fig, path)

# %%
