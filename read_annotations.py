
# %%
import json
import pandas as pd

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
# Convert the annotations and categories into the DataFrame format
annotations = pd.DataFrame(obj['annotations'])
categories = pd.DataFrame(obj['categories'])

# %%
# Build the image_id_table,
# the key is image_id in annotations,
# the value is its natural index.

image_id_table = dict()

count = 0
for e in tqdm(sorted(annotations['image_id'].unique()), 'Building image_id_table'):
    image_id_table[e] = count
    count += 1
    pass

# %%
# Fix the annotations
categories['idx'] = range(len(categories))
categories.set_index('id', drop=False, inplace=True)


def _full(e):
    return '{}-{}'.format(e['supercategory'], e['name'])


categories['full'] = categories.apply(_full, axis=1)
categories

# %%
n_images = len(annotations['image_id'].unique())
n_categories = len(annotations['category_id'].unique())

# %%
# Say something

print('The "Annotation" contains the records')
print(annotations)

print('The "Categories" contains the records')
print(categories)

print('There are {} images'.format(n_images))
print('There are {} categories'.format(n_categories))

# %%
