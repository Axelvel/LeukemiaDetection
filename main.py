import os
import kaggle
import tensorflow as tf
from preprocessing import data_loading, plot_histogram
from model import LeukemiaDetector
from train import training


import preprocessing
import data_augment

DATASET_PATH = './C-NMC_Leukemia/' if os.name == 'nt' else 'C-NMC_Leukemia/'


def import_dataset():
    """
    Downloads and extracts files from the
    'andrewmvd/leukemia-classification' Kaggle dataset.
    """
    dataset_name = 'andrewmvd/leukemia-classification'
    kaggle.api.dataset_download_files(dataset_name, unzip=True)
    print('Import done')


if __name__ == '__main__':
    print('Tensorflow:', tf.__version__)
    if not os.path.isdir(DATASET_PATH):
        import_dataset()
    
    for i in range(3):
        data_augment.augment_data(f"{DATASET_PATH}training_data/fold_{i}/hem/")

    # Creating dataloaders
    train_loader, test_loader, val_loader = data_loading(dataset_path=DATASET_PATH, batch_size=16)

    # Plotting class distribution
    plot_histogram(train_loader)

    # Training loop
    INPUT_SIZE = 450 * 450 * 3
    OUTPUT_SIZE = 2
    model = LeukemiaDetector(INPUT_SIZE, OUTPUT_SIZE)
    training(model, train_loader)
