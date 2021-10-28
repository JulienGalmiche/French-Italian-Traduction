import tensorflow as tf

from transformer_asr.src.multiheadattention import MultiHeadAttention
from transformer_asr.src.tools import point_wise_feed_forward_network

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_head_att, dff,  rate=0.1):
        super(DecoderLayer, self).__init__()
        self.multihead_masked = MultiHeadAttention(d_model, n_head_att)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.norm_1 = tf.keras.layers.LayerNormalization()
        self.multihead= MultiHeadAttention(d_model, n_head_att)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.norm_2 = tf.keras.layers.LayerNormalization()
        self.feedforward = point_wise_feed_forward_network(d_model, dff)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        self.norm_3 = tf.keras.layers.LayerNormalization()
            
        
    def call(self, x, x_enc_output, training, mask_1, mask_2=None):
        x_multi_head = self.multihead(x,x,x,mask_1)
        output_dropout = self.dropout1(x_multi_head, training=training)
        output_norm = self.norm_1(output_dropout+x)
         
        x_multi_head = self.multihead(output_norm,x_enc_output,x_enc_output,mask_2)
        output_dropout = self.dropout2(x_multi_head, training=training)
        output_norm = self.norm_2(output_dropout+x)
        
        output_feedforward = self.feedforward(output_norm)
        output_dropout = self.dropout3(output_feedforward, training=training)
        output = self.norm_3(output_dropout+output_norm)
        return output
        
        
        
