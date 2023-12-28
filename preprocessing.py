import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

# Set batch size and dimensions of the images
batch_size = 32
img_height = 180
img_width = 180

# Initialize the ImageDataGenerator for rescaling the image arrays
train_datagen = ImageDataGenerator(rescale=1./255)
val_labels = pd.read_csv('./C-NMC_Leukemia/validation_data/C-NMC_test_prelim_phase_data_labels.csv')
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Train data generators
train_data_gen_fold_0 = train_datagen.flow_from_directory(
    directory='./C-NMC_Leukemia/training_data/fold_0/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

train_data_gen_fold_1 = train_datagen.flow_from_directory(
    directory='./C-NMC_Leukemia/training_data/fold_1/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

train_data_gen_fold_2 = train_datagen.flow_from_directory(
    directory='./C-NMC_Leukemia/training_data/fold_2/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Check the class indices
print(train_data_gen_fold_0.class_indices)

# Validation data generators
validation_data_gen = val_datagen.flow_from_dataframe(
    dataframe=val_labels,
    directory='./C-NMC_Leukemia/validation_data/C-NMC_test_prelim_phase_data',
    x_col='new_names',
    y_col='labels',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

# Test data generators
# ToDo: the testing data has no labels
test_data_gen = test_datagen.flow_from_directory(
    directory='./C-NMC_Leukemia/testing_data/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False) 