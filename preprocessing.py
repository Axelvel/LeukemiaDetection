import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

batch_size = 32
img_height = 180
img_width = 180

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data_gen = train_datagen.flow_from_directory(
    directory='./C-NMC_Leukemia/training_data/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_data_gen = val_datagen.flow_from_directory(
    directory='./C-NMC_Leukemia/validation_data/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

test_data_gen = test_datagen.flow_from_directory(
    directory='./C-NMC_Leukemia/testing_data/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)  # Important for evaluation, so the prediction order matches the file order.