from object_detection.predictors.heads import head
import tensorflow as tf

class WeightSharedConvolutionalWeightHead(head.KerasHead):

	def __init__(self,
	             conv_hyperparams,
	             kernel_size=3,
	             use_dropout=False,
	             dropout_keep_prob=0.8,
	             use_depthwise=False,
	             apply_conv_hyperparams_to_heads=False,
	             name=None):
		super(WeightSharedConvolutionalWeightHead, self).__init__(name=name)
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
					filters=1,
					kernel_size=[self._kernel_size, self._kernel_size],
					padding='SAME',
					name='WeightPredictor',
					**kwargs
				)
			)
		else:
			self._weight_predictor_layers.append(
				tf.keras.layers.Conv2D(
					filters=1,
					kernel_size=[self._kernel_size, self._kernel_size],
					padding='SAME',
					name='WeightPredictor',
					**conv_hyperparams.params(use_bias=True)
				)
			)
		self._weight_predictor_layers.append(
			tf.keras.layers.GlobalAveragePooling2D()
		)

	def _predict(self, features):
		"""Predicts weight.

		    Args:
		      features: A float tensor of shape [batch_size, height, width, channels]
		        containing image features.

		    Returns:
		      weight_prediction: A float tensor of shape [batch_size, 1] representing
		      the weight of the fruits in each image.
		    """
		weight_predictions = features
		for layer in self._weight_predictor_layers:
			weight_predictions = layer(weight_predictions)
		batch_size = features.get_shape().as_list()[0]
		if batch_size is None:
			batch_size = tf.shape(features)[0]

		weight_predictions = tf.reshape(weight_predictions, [batch_size, 1])
		return weight_predictions
