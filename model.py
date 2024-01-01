import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


class LeukemiaDetector(tf.keras.Model):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Convolutional layers
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_size)
        self.maxpool1 = MaxPooling2D((2, 2))
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
        self.maxpool2 = MaxPooling2D((2, 2))

        # Classification layers
        self.flatten = Flatten()
        self.dense = Dense(128, activation='relu')
        self.output_layer = Dense(output_size, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.output_layer(x)
