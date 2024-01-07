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
    VAl_PATH = dataset_path + 'validation_data/'

    # Loading training set
    FOLDS = sorted(os.listdir(TRAIN_PATH))
    FOLDS = [fold for fold in FOLDS if not fold.startswith('.')]

    # We are using the first 2 folds for training and the last one for testing
    testing_fold = FOLDS[-1]
    TEST_PATH = TRAIN_PATH + testing_fold

    
    a_dataset = []
    b_dataset = []
    c_dataset = []
    a_val_dataset = []
    b_val_dataset = []
    c_val_dataset = []

    for i,fold in enumerate(FOLDS):
        loaded_imgs = tf.keras.utils.image_dataset_from_directory(
            TRAIN_PATH + fold,
            batch_size=batch_size,
            image_size=(450, 450),
            color_mode="rgb")
        if i == 0:
            print("0")
            a_dataset.append(loaded_imgs)
            b_dataset.append(loaded_imgs)
            c_val_dataset.append(loaded_imgs)
        elif i ==1:
            print("1")
            b_dataset.append(loaded_imgs)
            c_dataset.append(loaded_imgs)
            a_val_dataset.append(loaded_imgs)
        else:
            print("aled")
            c_dataset.append(loaded_imgs)
            a_dataset.append(loaded_imgs)
            b_val_dataset.append(loaded_imgs)


    a_loader = concat_dataloader(a_dataset[0], a_dataset[1:])
    b_loader = concat_dataloader(b_dataset[0], b_dataset[1:])
    c_loader = concat_dataloader(c_dataset[0], c_dataset[1:])
    a_val_loader = concat_dataloader(a_val_dataset[0], a_val_dataset[1:])
    b_val_loader = concat_dataloader(b_val_dataset[0], b_val_dataset[1:])
    c_val_loader = concat_dataloader(c_val_dataset[0], c_val_dataset[1:])

    # Loading testing set
    test_loader = tf.keras.utils.image_dataset_from_directory(
        TEST_PATH,
        batch_size=batch_size,
        image_size=(450, 450))

    # Loading validation set
    # val_labels = pd.read_csv(VAl_PATH + 'C-NMC_test_prelim_phase_data_labels.csv')
    # labels_list = val_labels['labels'].astype(int).tolist()
    # val_loader = tf.keras.utils.image_dataset_from_directory(
    #     VAl_PATH,
    #     batch_size=batch_size,
    #     image_size=(450, 450),
    #     label_mode='int',
    #     labels=labels_list,
    #     shuffle=False
    # )

    # Converting to Grayscale
    a_loader = a_loader.map(lambda x, y: (tf.image.rgb_to_grayscale(x), y))
    b_loader = b_loader.map(lambda x, y: (tf.image.rgb_to_grayscale(x), y))
    c_loader = c_loader.map(lambda x, y: (tf.image.rgb_to_grayscale(x), y))
    a_val_loader = a_val_loader.map(lambda x, y: (tf.image.rgb_to_grayscale(x), y))
    b_val_loader = b_val_loader.map(lambda x, y: (tf.image.rgb_to_grayscale(x), y))
    c_val_loader = c_val_loader.map(lambda x, y: (tf.image.rgb_to_grayscale(x), y))
    test_loader = test_loader.map(lambda x, y: (tf.image.rgb_to_grayscale(x), y))

    # Normalizing the tensors
    a_loader = a_loader.map(lambda x, y: (tf.divide(x, 255), y))
    b_loader = b_loader.map(lambda x, y: (tf.divide(x, 255), y))
    c_loader = c_loader.map(lambda x, y: (tf.divide(x, 255), y))
    a_val_loader = a_val_loader.map(lambda x, y: (tf.divide(x, 255), y))
    b_val_loader = b_val_loader.map(lambda x, y: (tf.divide(x, 255), y))
    c_val_loader = c_val_loader.map(lambda x, y: (tf.divide(x, 255), y))
    test_loader = test_loader.map(lambda x, y: (tf.divide(x, 255), y))

    return a_loader, b_loader, c_loader,a_val_loader, b_val_loader, c_val_loader, test_loader


def plot_histogram(data_loader, display=False):
    labels = []
    for _, target in data_loader:
        labels.extend(target.numpy().tolist())
    plt.clf()
    _, y = np.unique(labels, return_counts=True)
    plt.bar(['all' + '\n' + str(y[0]), 'hem' + '\n' + str(y[1])], y)
    plt.savefig('./histogram/' + 'class_distribution' + '.png')
    if display:
        plt.show()


def plot_folds(path):
    FOLDS = sorted(os.listdir(path))
    for fold in FOLDS:
        # Count the number of data in each training folder, and then create an histogram with the data balance
        x = ['all', 'hem']
        y = [len(os.listdir(path + fold + '/' + x[0])), len(os.listdir(path + fold + '/' + x[1]))]
        for i in range(len(x)):
            x[i] = x[i] + '\n' + str(y[i])
        plt.clf()
        plt.bar(x, y)
        plt.savefig("./histogram/" + fold + ".png")
