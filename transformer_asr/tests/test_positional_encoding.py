import tensorflow as tf
import numpy as np
from src.positional_encoding import get_angles, positional_encoding

class UnetTest(tf.test.TestCase):
    
    def setUp(self):
        super(UnetTest, self).setUp()
        self.pos = 3
        self.dim_even = 4

    def test_get_angles(self):
        result = get_angles(np.arange(self.pos)[:, np.newaxis],
                      np.arange(self.dim_even)[np.newaxis, :],
                      self.dim_even)
        self.expected = np.array([[0,0,0,0],
                                  [1, 1.e-2, 1.e-4, 1.e-6],
                                  [2, 2.e-2, 2.e-4, 2.e-6]])
        self.assertTrue((result == self.expected).all())
        
    def test_positional_encoding(self):
            result = positional_encoding(self.pos, self.dim_even)
            self.expected = tf.reshape(tf.constant(np.array([[      0   ,  1.        ,     0    ,  1.        ],
       [8.41470985e-01 ,  0.99995   ,     9.99999998e-05     ,  1.        ],
       [9.09297427e-01,  0.99980001,  1.99999999e-04,  1.        ]]), dtype=tf.float32), (1, 3,4))
            tf.assert_equal(result, self.expected)
