import tensorflow as tf

from transformer_asr.src.encoder_layer import EncoderLayer
from transformer_asr.src.subsampling import Subsampling
from transformer_asr.src.positional_encoding import positional_encoding
from transformer_asr.src.multiheadattention import MultiHeadAttention

class Encoder(tf.keras.layers.Layer):
    def __init__(self, d_model, n_head_att, dff, num_layers, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.subsampling = Subsampling(d_model)
        self.enc_layers = [EncoderLayer(d_model, n_head_att, dff, rate)
                       for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

        
    def call(self, x, training, mask):
        
        x = self.subsampling(x)
        pos_emb = positional_encoding(x.shape[1], self.d_model) 
        x += pos_emb 
        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        
        return x