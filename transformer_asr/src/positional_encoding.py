import numpy as np
import tensorflow as tf
import pdb

def get_angles(pos, i, d_model):
    angle_rates = 1 /10000**(2 * i / d_model)
    return pos * angle_rates

def positional_encoding(position, d_model):
    if d_model % 2==1:
        d_model +=1 #d_model has to be even
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    #pdb.set_trace()
    return tf.cast(pos_encoding, dtype=tf.float32)
