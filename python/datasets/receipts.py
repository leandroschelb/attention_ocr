import datasets.fsns as fsns

import os
DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(__file__), 'data', 'receipts')

DEFAULT_CONFIG = {
    'name':
        'number_plates', # you can change the name if you want.
    'splits': {
        'train': {
            'size': 1100, # change according to your own train-test split
            'pattern': 'train.tfrecord'
        },
        'test': {
            'size': 152, # change according to your own train-test split
            'pattern': 'test.tfrecord'
        }
    },
    'charset_filename':
        'charset-labels.txt',
    'image_shape': (150,400,3),#(max_width, max_height, 3), << devo confiar nesse comentario?? acho que não, no fnsn é (150, 600, 3) então acredito que seja height, width 
    'num_of_views':
        1,
    'max_sequence_length':
        50, # TO BE CONFIGURED
    'null_code':
        65,
    'items_to_descriptions': {
        'image':
            'A 3 channel color image.',
        'label':
            'Characters codes.',
        'text':
            'A unicode string.',
        'length':
            'A length of the encoded text.',
        'num_of_views':
            'A number of different views stored within the image.'
    }
}


def get_split(split_name, dataset_dir=None, config=None):
  if not dataset_dir:
    dataset_dir = DEFAULT_DATASET_DIR
  if not config:
    config = DEFAULT_CONFIG

  return fsns.get_split(split_name, DEFAULT_DATASET_DIR, DEFAULT_CONFIG)
