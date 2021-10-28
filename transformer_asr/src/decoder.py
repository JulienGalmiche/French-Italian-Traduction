import tensorflow as tf

from transformer_asr.src.decoder_layer import DecoderLayer
from transformer_asr.src.subsampling import Subsampling
from transformer_asr.src.positional_encoding import positional_encoding
from transformer_asr.src.multiheadattention import MultiHeadAttention


class Decoder(tf.keras.layers.Layer):
    def __init__(self, d_model, n_head_att, dff, n_dec_layers,target_vocab_size, rate=0.1):
        super(Decoder, self).__init__()
        self.n_dec_layers = n_dec_layers
        self.d_model = d_model
        self.dec_layers = [DecoderLayer(d_model, n_head_att, dff, rate)
                       for _ in range(n_dec_layers)]

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        
        #self.bert_model = TFAutoModel.from_pretrained(model_name)
        self.dropout = tf.keras.layers.Dropout(rate)
        
    def call(self, x, x_enc, training, mask_1, mask_2=None):
        x = self.embedding(x)
        pos_emb = positional_encoding(x.shape[1], self.d_model) #seq_len, 768 (embeddings dimension from BERT)
        x += pos_emb
        
        x = self.dropout(x, training=training)
        
        for i in range(self.n_dec_layers):
            x = self.dec_layers[i](x, x_enc, training, mask_1, mask_2)
            
        return x
        