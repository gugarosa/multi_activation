# Multi-Activation Layers

## Structure
 * `tf`
   * `multi_activation.py`: Tensorflow-based Multi-Activation layer;
 * `torch`
   * `multi_activation.py`: PyTorch-based Multi-Activation layer.

---

## Installation

Copy the desired layer file (PyTorch or Tensorflow) into your related project and instantiate the `MultiActivation` class in the same way as any layer.

---

## Usage

The `MultiActivation` layer applies a multi-activation transformation over the incoming data. Essentially, it has two arguments: `activation` and `strategy`.

### PyTorch

In the PyTorch-based version, the `activation` arguments correspond to a tuple of activation-based classes, such as `ReLU()`, `Sigmoid()`, among others, followed by the `strategy` argument (`mean` for averaging the result or `concat` for concatenating the results).

```Python
m = MultiActivation((ReLU(), Sigmoid()), strategy='mean')
input = torch.randn(128, 32)
output = m(input)
print(output.size()) # torch.Size([128, 32])
```

### Tensorflow

In the Tensorflow-based version, the former `activation` argument corresponds to a tuple of activation functions to be used (built-in function or string name), followed by the `strategy` argument (`mean` for averaging the result or `concat` for concatenating the results).

```Python
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(16,)))
model.add(MultiActivation(activation=('linear', 'sigmoid'), strategy='mean'))
print(model.output_shape) # (None, 16)
```

---