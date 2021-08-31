# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

"""Faster RCNN Keras-based Resnet V1 FPN Feature Extractor."""

import tensorflow as tf

from absl import logging
from six.moves import range
from six.moves import zip

from object_detection.meta_architectures import probabilistic_two_stage_meta_arch
from object_detection.models import bidirectional_feature_pyramid_generators as bifpn_generators
from object_detection.utils import ops
from object_detection.utils import shape_utils
from object_detection.utils import tf_version

if tf_version.is_tf2():
  from official.vision.image_classification.efficientnet import efficientnet_model


_EFFICIENTNET_LEVEL_ENDPOINTS = {
    1: 'stack_0/block_0/project_bn',
    2: 'stack_1/block_1/add',
    3: 'stack_2/block_1/add',
    4: 'stack_4/block_2/add',
    5: 'stack_6/block_0/project_bn',
}


class _EfficientNetBiFPN(tf.keras.layers.Layer):

  def __init__(self,
               efficientnet_backbone,
               bifpn_generator,
               pad_to_multiple,
               output_layer_alias
               ):
    super(_EfficientNetBiFPN, self).__init__()
    self._efficientnet_backbone = efficientnet_backbone
    self._bifpn_generator = bifpn_generator
    self._pad_to_multiple = pad_to_multiple
    self._output_layer_alias = output_layer_alias

  def call(self, inputs):
    preprocessed_inputs = shape_utils.check_min_image_dim(
      129, inputs)

    base_feature_maps = self._efficientnet_backbone(
      ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple))

    output_feature_map_dict = self._bifpn_generator(
      list(zip(self._output_layer_alias, base_feature_maps)))

    return list(output_feature_map_dict.values())


class ProbabilisticTwoStageEfficientNetBiFPNKerasFeatureExtractor(
    probabilistic_two_stage_meta_arch.ProbabilisticTwoStageKerasFeatureExtractor):
  """Faster RCNN Feature Extractor using Keras-based Resnet V1 FPN features."""

  def __init__(self,
               is_training,
               first_stage_features_stride,
               conv_hyperparams,
               min_depth,
               bifpn_min_level,
               bifpn_max_level,
               bifpn_num_iterations,
               bifpn_num_filters,
               bifpn_combine_method,
               efficientnet_version,
               freeze_batchnorm,
               pad_to_multiple=32,
               weight_decay=0.0,
               name=None
               ):
    """Constructor.

    Args:
      is_training: See base class.
      resnet_v1_base_model: base resnet v1 network to use. One of
        the resnet_v1.resnet_v1_{50,101,152} models.
      resnet_v1_base_model_name: model name under which to construct resnet v1.
      first_stage_features_stride: See base class.
      conv_hyperparams: a `hyperparams_builder.KerasLayerHyperparams` object
        containing convolution hyperparameters for the layers added on top of
        the base feature extractor.
      batch_norm_trainable: See base class.
      pad_to_multiple: An integer multiple to pad input image.
      weight_decay: See base class.
      fpn_min_level: the highest resolution feature map to use in FPN. The valid
        values are {2, 3, 4, 5} which map to Resnet v1 layers.
      fpn_max_level: the smallest resolution feature map to construct or use in
        FPN. FPN constructions uses features maps starting from fpn_min_level
        upto the fpn_max_level. In the case that there are not enough feature
        maps in the backbone network, additional feature maps are created by
        applying stride 2 convolutions until we get the desired number of fpn
        levels.
      additional_layer_depth: additional feature map layer channel depth.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams`.

    Raises:
      ValueError: If `first_stage_features_stride` is not 8 or 16.
    """
    if first_stage_features_stride != 8 and first_stage_features_stride != 16:
      raise ValueError('`first_stage_features_stride` must be 8 or 16.')

    super(ProbabilisticTwoStageEfficientNetBiFPNKerasFeatureExtractor, self).__init__(
        is_training=is_training,
        first_stage_features_stride=first_stage_features_stride,
        freeze_batchnorm=freeze_batchnorm,
        weight_decay=weight_decay)

    self._bifpn_min_level = bifpn_min_level
    self._bifpn_max_level = bifpn_max_level
    self._bifpn_num_iterations = bifpn_num_iterations
    self._bifpn_num_filters = max(bifpn_num_filters, min_depth)
    self._bifpn_node_params = {'combine_method': bifpn_combine_method}
    self._efficientnet_version = efficientnet_version
    self._pad_to_multiple = pad_to_multiple
    self._conv_hyperparams = conv_hyperparams
    self._freeze_batchnorm = freeze_batchnorm

    self.classification_backbone = None

    logging.info('EfficientDet EfficientNet backbone version: %s',
                 self._efficientnet_version)
    logging.info('EfficientDet BiFPN num filters: %d', self._bifpn_num_filters)
    logging.info('EfficientDet BiFPN num iterations: %d',
                 self._bifpn_num_iterations)

    self._backbone_max_level = min(
      max(_EFFICIENTNET_LEVEL_ENDPOINTS.keys()), self._bifpn_max_level)
    self._output_layer_names = [
      _EFFICIENTNET_LEVEL_ENDPOINTS[i]
      for i in range(self._bifpn_min_level, self._backbone_max_level + 1)]
    self._output_layer_alias = [
      'level_{}'.format(i)
      for i in range(self._bifpn_min_level, self._backbone_max_level + 1)]

    efficientnet_base = efficientnet_model.EfficientNet.from_name(
      model_name=self._efficientnet_version,
      overrides={'rescale_input': False})
    outputs = [efficientnet_base.get_layer(output_layer_name).output
               for output_layer_name in self._output_layer_names]
    self.classification_backbone = tf.keras.Model(
      inputs=efficientnet_base.inputs, outputs=outputs)
    self._bifpn_stage = bifpn_generators.KerasBiFpnFeatureMaps(
      bifpn_num_iterations=self._bifpn_num_iterations,
      bifpn_num_filters=self._bifpn_num_filters,
      fpn_min_level=self._bifpn_min_level,
      fpn_max_level=self._bifpn_max_level,
      input_max_level=self._backbone_max_level,
      is_training=self._is_training,
      conv_hyperparams=self._conv_hyperparams,
      freeze_batchnorm=self._freeze_batchnorm,
      bifpn_node_params=self._bifpn_node_params,
      name='bifpn')

    self.proposal_feature_extractor_model = _EfficientNetBiFPN(efficientnet_backbone=self.classification_backbone,
                                                              bifpn_generator=self._bifpn_stage,
                                                              pad_to_multiple=self._pad_to_multiple,
                                                              output_layer_alias=self._output_layer_alias)


  def preprocess(self, inputs):
    """SSD-Style preprocessing

    Args:
      inputs: A [batch, height_in, width_in, channels] float32 tensor
        representing a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: A [batch, height_out, width_out, channels] float32
        tensor representing a batch of images.

    """
    if inputs.shape.as_list()[3] == 3:
      # Input images are expected to be in the range [0, 255].
      channel_offset = [0.485, 0.456, 0.406]
      channel_scale = [0.229, 0.224, 0.225]
      return ((inputs / 255.0) - [[channel_offset]]) / [[channel_scale]]
    else:
      return inputs

  def get_proposal_feature_extractor_model(self, name=None):
    """Returns a model that extracts first stage RPN features.

    Extracts features using the Resnet v1 FPN network.

    Args:
      name: A scope name to construct all variables within.

    Returns:
      A Keras model that takes preprocessed_inputs:
        A [batch, height, width, channels] float32 tensor
        representing a batch of images.

      And returns rpn_feature_map:
        A list of tensors with shape [batch, height, width, depth]
    """

    return self.proposal_feature_extractor_model



  def get_box_classifier_feature_extractor_model(self, name=None):
    """Returns a model that extracts second stage box classifier features.

    Args:
      name: A scope name to construct all variables within.

    Returns:
      A Keras model that takes proposal_feature_maps:
        A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, crop_height, crop_width, depth]
        representing the feature map cropped to each proposal.

      And returns proposal_classifier_features:
        A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, 1, 1, 512]
        representing box classifier features for each proposal.
    """

    box_classifier_model_conv = tf.keras.models.Sequential([
      tf.keras.layers.SeparableConv2D(filters=128,
                                      kernel_size=[3, 3],
                                      strides=2,
                                      activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(units=1024, activation='relu'),
      tf.keras.layers.Dense(units=512, activation='relu'),
      tf.keras.layers.Dense(units=256, activation='relu'),
      tf.keras.layers.Reshape((1, 1, 256))
    ])
    return box_classifier_model_conv



class ProbabilisticTwoStageEfficientNetB0BiFPNKerasFeatureExtractor(
    ProbabilisticTwoStageEfficientNetBiFPNKerasFeatureExtractor):
  """Faster RCNN with EfficientNet B0 BiFPN feature extractor."""

  def __init__(self,
               is_training,
               first_stage_features_stride,
               conv_hyperparams,
               freeze_batchnorm,
               min_depth,
               bifpn_min_level,
               bifpn_max_level,
               bifpn_num_iterations,
               bifpn_num_filters,
               bifpn_combine_method='fast_attention',
               pad_to_multiple=32,
               weight_decay=0.0,
               name='EfficientDet-D0'
               ):

    super(ProbabilisticTwoStageEfficientNetB0BiFPNKerasFeatureExtractor, self).__init__(
        is_training=is_training,
        first_stage_features_stride=first_stage_features_stride,
        conv_hyperparams=conv_hyperparams,
        freeze_batchnorm=freeze_batchnorm,
        min_depth=min_depth,
        bifpn_min_level=bifpn_min_level,
        bifpn_max_level=bifpn_max_level,
        bifpn_num_iterations=bifpn_num_iterations,
        bifpn_num_filters=bifpn_num_filters,
        bifpn_combine_method=bifpn_combine_method,
        efficientnet_version='efficientnet-b0',
        pad_to_multiple=pad_to_multiple,
        weight_decay=weight_decay,
        name=name
    )
