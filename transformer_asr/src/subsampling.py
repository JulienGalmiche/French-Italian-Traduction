import tensorflow as tf



class Subsampling(tf.keras.layers.Layer):
    def __init__(self, d_att, **kwargs):
        super(Subsampling, self).__init__(**kwargs)
        self.d_att = d_att
        self.conv_1 = tf.keras.layers.Conv2D(filters=self.d_att, kernel_size=3, strides=2,
                           padding="SAME", activation="relu", input_shape=[None,129,1])
        self.conv_2 = tf.keras.layers.Conv2D(filters=self.d_att, kernel_size=3, strides=2,
                           padding="SAME", activation="relu")
        self.flatten = tf.keras.layers.Flatten()
    
    def call(self, inputs):
        inputs = self.conv_1(inputs)
        outputs = self.conv_2(inputs)
        reshape_dim_0 = outputs.shape[0]
        n_sub = outputs.shape[1]*outputs.shape[2] # 4.2 from paper
        return tf.reshape(outputs,[reshape_dim_0, n_sub, self.d_att])