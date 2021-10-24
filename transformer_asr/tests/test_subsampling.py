import tensorflow as tf
import numpy as np
from src.subsampling import Subsampling

class SubsamplingTest(tf.test.TestCase):
    
    def setUp(self):
        super(UnetTest, self).setUp()
        self.subsampling = Subsampling(256)

    
    def _test_call(self):
        x= tf.constant(0, shape=(16, 200,100, 1), dtype=tf.float32)
        shape_result = tf.TensorShape([16, 1250, 256])
        output = self.subsampling.call(x)
        tf.assert_equal(shape_result, output.shape)
      