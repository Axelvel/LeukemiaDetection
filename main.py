import os
import kaggle
import tensorflow as tf

import preprocessing



def import_dataset():
    """
    Downloads and extracts files from the
    'andrewmvd/leukemia-classification' Kaggle dataset.
    """
    dataset_name = 'andrewmvd/leukemia-classification'
    kaggle.api.dataset_download_files(dataset_name, unzip=True)
    print('Import done')


DATASET_PATH = 'C-NMC_Leukemia/'

if __name__ == '__main__':
    print('Tensorflow:', tf.__version__)
    if not os.path.isdir(DATASET_PATH):
        import_dataset()

    preprocessing.show_image_samples()


