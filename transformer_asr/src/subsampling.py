import tensorflow as tf



class Subsampling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Subsampling, self).__init__(**kwargs)
        self.conv_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2,
                           padding="SAME", activation="relu")
        self.conv_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2,
                           padding="SAME", activation="relu")
    
    def call(self, inputs):
        inputs = self.conv_1(inputs)
        return self.conv_2(inputs)