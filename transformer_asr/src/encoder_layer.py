import tensorflow as tf

from transformer_asr.src.multiheadattention import MultiHeadAttention
from transformer_asr.src.tools import point_wise_feed_forward_network

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_head_att, dff,  rate=0.1):
        super(EncoderLayer, self).__init__()
        self.multihead = MultiHeadAttention(d_model, n_head_att)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.norm_1 = tf.keras.layers.LayerNormalization()
        self.feedforward = point_wise_feed_forward_network(d_model, dff)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.norm_2 = tf.keras.layers.LayerNormalization()
            
        
    def call(self, x, training, mask=None):
        x_multi_head = self.multihead(x,x,x,mask)
        output_dropout = self.dropout1(x_multi_head, training=training)
        output_norm = self.norm_1(output_dropout+x)
        print(output_dropout.shape)
        print("***")
        print(x.shape)
        output_feedforward = self.feedforward(output_norm)
        output_dropout = self.dropout2(output_feedforward, training=training)
        output_norm_2 = self.norm_2(output_dropout+output_norm)
        return output_norm_2
        
        
        
