import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


def concat_dataloader(head, rest):
    if not rest:
        return head
    current = head
    remaining = rest[1:]
    current = current.concatenate(rest[0])
    return concat_dataloader(current, remaining)


def data_loading(dataset_path, batch_size=32):

    TRAIN_PATH = dataset_path + 'training_data/'
    TEST_PATH = dataset_path + 'testing_data/'
    VAl_PATH = dataset_path + 'validation_data/'

    # Loading training set
    FOLDS = sorted(os.listdir(TRAIN_PATH))

    total_dataset = []

    for fold in FOLDS:
        loaded_imgs = tf.keras.utils.image_dataset_from_directory(
            TRAIN_PATH + fold,
            batch_size=batch_size,
            image_size=(450, 450))
        total_dataset.append(loaded_imgs)

    train_loader = concat_dataloader(total_dataset[0], total_dataset[1:])

    # Loading testing set
    test_loader = tf.keras.utils.image_dataset_from_directory(
        TEST_PATH,
        batch_size=batch_size,
        image_size=(450, 450),
        label_mode=None)

    # Loading validation set
    val_labels = pd.read_csv(VAl_PATH + 'C-NMC_test_prelim_phase_data_labels.csv')
    labels_list = val_labels['labels'].astype(int).tolist()
    val_loader = tf.keras.utils.image_dataset_from_directory(
        VAl_PATH,
        batch_size=batch_size,
        image_size=(450, 450),
        label_mode='int',
        labels=labels_list,
        shuffle=False
    )

    return train_loader, test_loader, val_loader


def plot_histogram(data_loader, display=False):
    labels = []
    for _, target in data_loader:
        labels.extend(target.numpy().tolist())
    plt.clf()
    _, y = np.unique(labels, return_counts=True)
    plt.bar(["all", "hem"], y)
    plt.savefig('./histogram/' + 'class_distribution' + '.png')
    if display:
        plt.show()
