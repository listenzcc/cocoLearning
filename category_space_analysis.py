
# %%
import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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

mat.shape, len(labels_max_area)

# %%
pca = PCA(n_components=6)
features = pca.fit_transform(mat.transpose())
features = features[:, :3]
features.shape

# %%
df = pd.DataFrame(features, columns=['pc1', 'pc2', 'pc3'])
df['labelCount'] = np.sum(mat, axis=0, dtype=np.float64)
df['labelMaxArea'] = labels_max_area
df

# ---- %%
title = 'PCA on categories (labelCount)'

fig = px.scatter_3d(df, title=title, x='pc1', y='pc2',
                    z='pc3', color='labelCount')

fig.update_traces(marker=dict(size=2,
                              line=dict(width=0,
                                        color='white')),
                  selector=dict(mode='markers'))

fig.update_layout(dict(scene={'aspectmode': 'data'}, width=600, height=600))

path = save_path.joinpath(title)
save_fig(fig, path)

# ---- %%
title = 'PCA on categories (labelMaxArea)'

fig = px.scatter_3d(df, title=title, x='pc1', y='pc2',
                    z='pc3', color='labelMaxArea')

fig.update_traces(marker=dict(size=2,
                              line=dict(width=0,
                                        color='white')),
                  selector=dict(mode='markers'))

fig.update_layout(dict(scene={'aspectmode': 'data'}, width=600, height=600))

path = save_path.joinpath(title)
save_fig(fig, path)

# %%
df = pd.DataFrame(pca.components_.transpose(),
                  columns=['pc{}'.format(e+1)
                           for e in range(pca.components_.shape[0])])
df.columns.name = 'pcs'
df['idx'] = df.index
df = df.reset_index().melt(id_vars='idx', value_name='weight')
df = df[df['pcs'] != 'index']
df

# %%
title = 'PCA weights'
fig = px.line(df, title=title, y='weight', x='idx', color='pcs')

path = save_path.joinpath(title)
save_fig(fig, path)

# %%
path = save_path.joinpath('tsne-f3-{}.npy'.format(mat.shape[1]))

if path.is_file():
    with open(path, 'rb') as f:
        tsne_f3 = np.load(f)
    print('Loaded tsne_f3 matrix: {}'.format(tsne_f3.shape))
else:
    tsne = TSNE(n_components=3, n_jobs=32)
    tsne_f3 = tsne.fit_transform(mat.transpose())
    with open(path, 'wb') as f:
        np.save(f, tsne_f3)
    print('Saved tsne_f3 matrix: {}'.format(tsne_f3.shape))


# %%
df = pd.DataFrame(tsne_f3, columns=['x', 'y', 'z'])
df['labelCount'] = np.sum(mat, axis=0, dtype=np.float64)
df['labelMaxArea'] = labels_max_area

# %%
title = 'TSNE space (labelCount)'

fig = px.scatter_3d(df, title=title, x='x', y='y', z='z',
                    width=600, height=600, color='labelCount')

fig.update_traces(marker=dict(size=1,
                              line=dict(width=0,
                                        color='white')),
                  selector=dict(mode='markers'))

fig.update_layout(dict(scene={'aspectmode': 'data'}, width=600, height=600))

path = save_path.joinpath(title)
save_fig(fig, path)

# %%
title = 'TSNE space (labelMaxArea)'

fig = px.scatter_3d(df, title=title, x='x', y='y', z='z',
                    width=600, height=600, color='labelMaxArea')

fig.update_traces(marker=dict(size=1,
                              line=dict(width=0,
                                        color='white')),
                  selector=dict(mode='markers'))

fig.update_layout(dict(scene={'aspectmode': 'data'}, width=600, height=600))

path = save_path.joinpath(title)
save_fig(fig, path)

# %%
title = 'TSNE space (joint)'

fig = px.scatter_3d(df, title=title, x='x', y='y', z='z',
                    width=600, height=600,
                    size='labelCount', size_max=4, color='labelMaxArea')

fig.update_traces(marker=dict(line=dict(width=0,
                                        color='white')),
                  selector=dict(mode='markers'))

fig.update_layout(dict(scene={'aspectmode': 'data'}, width=600, height=600))

path = save_path.joinpath(title)
save_fig(fig, path)
# %%
