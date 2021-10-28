from transformer_asr.src.tools  import create_look_ahead_mask
from transformer_asr.src.encoder import Encoder
from transformer_asr.src.decoder import Decoder
import tensorflow as tf

class Transformer(tf.keras.Model):
    def __init__(self,d_model, n_head_att, dff, num_layers, n_dec_layers,target_vocab_size, voc_total_size, rate=0.1):
        super(Transformer, self).__init__()
        self.decoder = Decoder(d_model, n_head_att, dff, n_dec_layers,target_vocab_size,  rate)
        self.encoder = Encoder(d_model, n_head_att, dff, num_layers, rate)
        self.output_layer =  tf.keras.layers.Dense(voc_total_size)
        
    def call(self, inputs, training):
        inp, tar = inputs
        output_enc = self.encoder(inp, training, None)
        mask_dec_1 = create_look_ahead_mask(tar.shape[1]) #sequence length
        output_dec = self.decoder(tar, output_enc, training, mask_dec_1, None)
        final_output = self.output_layer(output_dec)
        
        return final_output