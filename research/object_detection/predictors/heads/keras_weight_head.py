from object_detection.predictors.heads import head
import tensorflow as tf


class MaskRCNNWeightHead(head.KerasHead):
  """
  Weight prediction head.

  """

  def __init__(self,
               is_training,
               num_classes,
               fc_hyperparams,
               freeze_batchnorm,
               use_dropout,
               dropout_keep_prob,
               name=None):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      fc_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
        containing hyperparameters for fully connected dense ops.
      freeze_batchnorm: Whether to freeze batch norm parameters during
        training or not. When training with a small batch size (e.g. 1), it is
        desirable to freeze batch norm update and use pretrained batch norm
        params.
      use_dropout: Option to use dropout or not.  Note that a single dropout
        op is applied here prior to both box and class predictions, which stands
        in contrast to the ConvolutionalBoxPredictor below.
      dropout_keep_prob: Keep probability for dropout.
        This is only used if use_dropout is True.
      share_box_across_classes: Whether to share boxes across classes rather
        than use a different box for each class.
      name: A string name scope to assign to the weight head. If `None`, Keras
        will auto-generate one from the class name.
    """
    super(MaskRCNNWeightHead, self).__init__(name=name)
    self._is_training = is_training
    self._num_classes = num_classes
    self._fc_hyperparams = fc_hyperparams
    self._freeze_batchnorm = freeze_batchnorm
    self._use_dropout = use_dropout
    self._dropout_keep_prob = dropout_keep_prob

    self._weight_predictor_layers = [tf.keras.layers.Flatten()]

    if self._use_dropout:
      self._weight_predictor_layers.append(
          tf.keras.layers.Dropout(rate=1.0 - self._dropout_keep_prob))

    self._number_of_boxes = 1

    self._weight_predictor_layers.append(
        tf.keras.layers.Dense(self._number_of_boxes * 1,
                              name='WeightPredictor_dense'))
    self._weight_predictor_layers.append(
        fc_hyperparams.build_batch_norm(training=(is_training and
                                                  not freeze_batchnorm),
                                        name='WeightPredictor_batchnorm'))

  def _predict(self, features):
    """Predicts box encodings.

    Args:
      features: A float tensor of shape [batch_size, height, width,
        channels] containing features for a batch of images.

    Returns:
      weight_predictions: A float tensor of shape
        [batch_size, 1, 1] representing the weights of the
        objects.
    """
    spatial_averaged_roi_pooled_features = tf.reduce_mean(
        features, [1, 2], keepdims=True, name='AvgPool')
    net = spatial_averaged_roi_pooled_features
    for layer in self._weight_predictor_layers:
      net = layer(net)
    weight_predictions = tf.reshape(net,
                               [-1, 1])
    return weight_predictions


class WeightSharedConvolutionalWeightHead(head.KerasHead):

	def __init__(self,
	             num_predictions_per_location,
	             conv_hyperparams,
	             kernel_size=3,
	             use_dropout=False,
	             dropout_keep_prob=0.8,
	             use_depthwise=False,
	             apply_conv_hyperparams_to_heads=False,
	             name=None):
		super(WeightSharedConvolutionalWeightHead, self).__init__(name=name)
		self._num_predictions_per_location = num_predictions_per_location
		self._kernel_size = kernel_size
		self._use_dropout = use_dropout
		self._dropout_keep_prob = dropout_keep_prob
		self._use_depthwise = use_depthwise
		self._apply_conv_hyperparams_to_heads = apply_conv_hyperparams_to_heads

		self._weight_predictor_layers = []

		if self._use_dropout:
			self._weight_predictor_layers.append(
				tf.keras.layers.Dropout(rate=1.0 - self._dropout_keep_prob)
			)
		if self._use_depthwise:
			kwargs = conv_hyperparams.params(use_bias=True)
			if self._apply_conv_hyperparams_to_heads:
				kwargs['depthwise_regularizer'] = kwargs['kernel_regularizer']
				kwargs['depthwise_initializer'] = kwargs['kernel_initializer']
				kwargs['pointwise_regularizer'] = kwargs['kernel_regularizer']
				kwargs['pointwise_initializer'] = kwargs['kernel_initializer']
			self._weight_predictor_layers.append(
				tf.keras.layers.SeparableConv2D(
					filters=self._num_predictions_per_location,
					kernel_size=[self._kernel_size, self._kernel_size],
					padding='SAME',
					name='WeightPredictor',
					**kwargs
				)
			)
		else:
			self._weight_predictor_layers.append(
				tf.keras.layers.Conv2D(
					filters=self._num_predictions_per_location,
					kernel_size=[self._kernel_size, self._kernel_size],
					padding='SAME',
					name='WeightPredictor',
					**conv_hyperparams.params(use_bias=True)
				)
			)

	def _predict(self, features):
		"""Predicts weight.

		    Args:
		      features: A float tensor of shape [batch_size, height, width, channels]
		        containing image features.

		    Returns:
		      weight_prediction: A float tensor of shape [batch_size, num_anchors, 1] representing
		      the weight of a fruit in a anchor.
		    """
		weight_predictions = features
		for layer in self._weight_predictor_layers:
			weight_predictions = layer(weight_predictions)
		batch_size = features.get_shape().as_list()[0]
		if batch_size is None:
			batch_size = tf.shape(features)[0]

		weight_predictions = tf.reshape(weight_predictions, [batch_size, -1, 1])
		return weight_predictions
