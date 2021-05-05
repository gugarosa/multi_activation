import copy

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape

class MultiActivation(Layer):
    """Applies a multi activation transformation to the incoming data.

    Example:

    Args:

    Input shape:

    Output shape:

    """
    
    def __init__(self, activation, strategy='mean', **kwargs):
        super(MultiActivation, self).__init__(**kwargs)

        self.activation = activation
        self.strategy = strategy

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `MultiActivation` layer with non-floating point '
                            'dtype %s' % (dtype,))

    def call(self, inputs):
        if self.strategy == 'concat':
            return tf.concat([a(inputs) for a in self.activation], -1)

        return tf.math.reduce_mean(tf.stack([a(inputs) for a in self.activation]), axis=0)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        output_shape = copy.copy(input_shape)

        if self.strategy == 'mean':
            output_shape[-1] = len(self.activation) * output_shape[-1]
        
        return tensor_shape.TensorShape(output_shape)
        
    def get_config(self):
        config = {
            'activation': self.activation,
            'strategy': self.strategy
        }
        base_config = super(MultiActivation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

if __name__ == "__main__":
    m = MultiActivation(activation=(tf.keras.activations.linear, tf.keras.activations.linear), strategy='concat')

    a = tf.random.normal((128, 32))
    print(m.compute_output_shape(a.shape))
