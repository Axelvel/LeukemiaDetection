import tensorflow as tf


class LeukemiaDetector(tf.keras.Model):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.flatten = tf.keras.layers.Flatten()
        #self.dense1 = tf.keras.layers.Dense(input_size)
        #self.dense2 = tf.keras.layers.Dense(output_size)

    def call(self, x):
        x = self.flatten(x)
        #x = self.dense1(x)
        #x = self.dense2(x)
        return x