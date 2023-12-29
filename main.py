import os
import kaggle
import tensorflow as tf

import preprocessing
import data_augment



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

    train_data0, train_data1, train_data2, val_data, test_data = preprocessing.data_loading()
    preprocessing.plot_histogram(train_data0,"fold_0")
    preprocessing.plot_histogram(train_data1,"fold_1")
    preprocessing.plot_histogram(train_data2,"fold_2")
    preprocessing.plot_histogram(val_data,"val")
    preprocessing.plot_histogram(test_data,"test")
    
