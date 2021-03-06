# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Alexnet model configuration.

References:
  Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton
  ImageNet Classification with Deep Convolutional Neural Networks
  Advances in Neural Information Processing Systems. 2012
"""

import tensorflow as tf
from . import model


class AlexnetModel(model.Model):
  """Alexnet cnn model."""

  def __init__(self):
    super(AlexnetModel, self).__init__('alexnet', 224 + 3, 512, 0.005)

  def add_inference(self, cnn):
    # def conv(self, num_out_channels, k_height, k_width, d_height=1, d_width=1, mode='SAME', ...)
    # Note: VALID requires padding the images by 3 in width and height
    # LRN: self, depth_radius, bias, alpha, beta
    cnn.conv(96, 11, 11, 4, 4, 'VALID')   # Originally 64 output channels
    cnn.lrn(depth_radius=5, bias=1, alpha=0.0001, beta=0.75)   # Was not here originally
    cnn.mpool(3, 3, 2, 2)
    cnn.conv(256, 5, 5)                   # Originally, 192 output channels.
    cnn.lrn(5, 1, 0.0001, 0.75)              # Was not here originally
    cnn.mpool(3, 3, 2, 2)
    cnn.conv(384, 3, 3)
    cnn.conv(384, 3, 3)
    cnn.conv(256, 3, 3)
    cnn.mpool(3, 3, 2, 2)
    cnn.reshape([-1, 256 * 6 * 6])
    cnn.affine(4096)
    cnn.dropout()
    cnn.affine(4096)
    cnn.dropout()


class AlexnetOWTModel(model.Model):
  """Alexnet cnn model."""

  def __init__(self):
    super(AlexnetOWTModel, self).__init__('alexnet_owt', 224 + 3, 512, 0.005)

  def add_inference(self, cnn):
    # def conv(self, num_out_channels, k_height, k_width, d_height=1, d_width=1, mode='SAME', ...)
    # Note: VALID requires padding the images by 3 in width and height
    # LRN: self, depth_radius, bias, alpha, beta
    cnn.conv(64, 11, 11, 4, 4, 'VALID')   # Originally 64 output channels
    cnn.mpool(3, 3, 2, 2)
    cnn.conv(192, 5, 5)                   # Originally, 192 output channels.
    cnn.mpool(3, 3, 2, 2)
    cnn.conv(384, 3, 3)
    cnn.conv(256, 3, 3)
    cnn.conv(256, 3, 3)
    cnn.mpool(3, 3, 2, 2)
    cnn.reshape([-1, 256 * 6 * 6])
    cnn.affine(4096)
    cnn.dropout()
    cnn.affine(4096)
    cnn.dropout()

"""
class AlexnetModel(model.Model):
  # Alexnet cnn model.

  def __init__(self):
    super(AlexnetModel, self).__init__('alexnet', 224 + 3, 512, 0.005)

  def add_inference(self, cnn):
    # Note: VALID requires padding the images by 3 in width and height
    cnn.conv(64, 11, 11, 4, 4, 'VALID')
    cnn.mpool(3, 3, 2, 2)
    cnn.conv(192, 5, 5)
    cnn.mpool(3, 3, 2, 2)
    cnn.conv(384, 3, 3)
    cnn.conv(384, 3, 3)
    cnn.conv(256, 3, 3)
    cnn.mpool(3, 3, 2, 2)
    cnn.reshape([-1, 256 * 6 * 6])
    cnn.affine(4096)
    cnn.dropout()
    cnn.affine(4096)
    cnn.dropout()
"""

class AlexnetCifar10Model(model.Model):
  """Alexnet cnn model for cifar datasets.

  The model architecture follows the one defined in the tensorflow tutorial
  model.

  Reference model: tensorflow/models/tutorials/image/cifar10/cifar10.py
  Paper: http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
  """

  def __init__(self):
    super(AlexnetCifar10Model, self).__init__('alexnet', 32, 128, 0.1)

  def add_inference(self, cnn):
    cnn.conv(64, 5, 5, 1, 1, 'SAME', stddev=5e-2)
    cnn.mpool(3, 3, 2, 2, mode='SAME')
    cnn.lrn(depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    cnn.conv(64, 5, 5, 1, 1, 'SAME', bias=0.1, stddev=5e-2)
    cnn.lrn(depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    cnn.mpool(3, 3, 2, 2, mode='SAME')
    shape = cnn.top_layer.get_shape().as_list()
    flat_dim = shape[1] * shape[2] * shape[3]
    cnn.reshape([-1, flat_dim])
    cnn.affine(384, stddev=0.04, bias=0.1)
    cnn.affine(192, stddev=0.04, bias=0.1)

  def get_learning_rate(self, global_step, batch_size):
    num_examples_per_epoch = 50000
    num_epochs_per_decay = 100
    decay_steps = int(num_epochs_per_decay * num_examples_per_epoch /
                      batch_size)
    decay_factor = 0.1
    return tf.train.exponential_decay(
        self.learning_rate, global_step, decay_steps, decay_factor,
        staircase=True)
