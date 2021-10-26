import tensorflow as tf



class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, n_head_att):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_head_att = n_head_att
        
        assert d_model % n_head_att == 0 
        
        self.d_att_proj = d_model//n_head_att
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.linear_output = tf.keras.layers.Dense(d_model)
    
    
    def call(self, q, k ,v, mask=None):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self._projection(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self._projection(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self._projection(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
    
        scaled_attention = self._scaled_dot_product_attention(
        q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.linear_output(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output          
    
    def _projection(self,x, batch_size):
        x = tf.reshape(x, (batch_size, -1 , self.n_head_att, self.d_att_proj))               
        return tf.transpose(x, perm=[0, 2, 1, 3]) 
        #for q, k multiplication for the scaled_dot_product, we need matrix with seq_length*d_att_proj dimensions.
                                                   
    
    def _scaled_dot_product_attention(self, q, k, v, mask=None):
        output = tf.linalg.matmul(q, k, transpose_b=True)
        output /= tf.math.sqrt(tf.cast(q.shape[1], dtype=tf.float32)) #need "cast into tf.float32 for sqrt.
        if mask is not None: #need "is not None" otherwise ValueError rises
            mask *= -1e9
            output += mask
        output = tf.nn.softmax(output)
        return tf.linalg.matmul(output, v)




        
        
