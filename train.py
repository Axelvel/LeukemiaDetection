import os
import tensorflow as tf
import matplotlib.pyplot as plt
from model import LeukemiaDetector
#from sklearn.model_selection import train_test_split

DATASET_PATH = 'C-NMC_Leukemia/'

TRAIN_PATH = DATASET_PATH + 'training_data/'
FOLDS = sorted(os.listdir(TRAIN_PATH))

total_dataset = []

for fold in FOLDS:
    loaded_imgs = tf.keras.utils.image_dataset_from_directory(TRAIN_PATH + fold, image_size=(450, 450))
    total_dataset.append(loaded_imgs)

    #count the number of data in each training folder, and then create an histogram with the data balance
    x = os.listdir(TRAIN_PATH + fold)
    y = [len(os.listdir(TRAIN_PATH + fold+"/all/")),len(os.listdir(TRAIN_PATH + fold+"/hem/"))]
    for i in range(len(x)):
        x[i] = x[i]+"\n"+str(y[i])
    plt.clf()
    plt.bar(x, y)
    plt.savefig("./histogram/"+fold+".png")



def concat_dataloader(head, rest):
    if not rest:
        return head
    current = head
    remaining = rest[1:]
    current = current.concatenate(rest[0])
    return concat_dataloader(current, remaining)


loader = concat_dataloader(total_dataset[0], total_dataset[1:])

INPUT_SIZE = 450 * 450 * 3
OUTPUT_SIZE = 2

model = LeukemiaDetector(INPUT_SIZE, OUTPUT_SIZE)

# Training Loop

for inputs, labels in loader:
    outputs = model(inputs)
    print(outputs.shape)