import copy

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.framework import dtypes, tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K


class MultiActivation(Layer):
    """Applies a multi-activation transformation to the incoming data.

    Example:

        >>> model = tf.keras.models.Sequential()
        >>> model.add(tf.keras.Input(shape=(16,)))
        >>> model.add(MultiActivation(activation=('linear', 'sigmoid'), strategy='mean'))
        >>> print(model.output_shape)
        (None, 16)

    Args:
        activation: Tuple of activations, such as `tf.nn.relu`, or string name of
            built-in activation function, such as "relu".
        strategy: Output tensor strategy.

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the batch axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as input, except for `concat` strategy, which outputs
        number of activations * input_shape on the last dimension.

    """
    
    def __init__(self, activation, strategy='mean', **kwargs):
        super(MultiActivation, self).__init__(**kwargs)

        if not isinstance(activation, tuple):
            raise TypeError(
                'Invalid activation type {!r}, should be tuple'.format(
                    type(activation).__name__
                )
            )
        self.activation = (activations.get(a) for a in activation)
    
        valid_strategy_strings = {'concat', 'mean'}
        if not isinstance(strategy, str):
            raise TypeError(
                'Invalid strategy type {!r}, should be string'.format(
                    type(strategy).__name__
                )
            )
        if strategy not in valid_strategy_strings:
            raise ValueError(
                'Invalid strategy string {!r}, should be one of {}'.format(
                    strategy, valid_strategy_strings
                )
            )
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
            'activation': (activations.serialize(a) for a in self.activation),
            'strategy': self.strategy
        }
        base_config = super(MultiActivation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
