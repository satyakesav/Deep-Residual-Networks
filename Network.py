import tensorflow as tf

"""This script defines the network.
"""


class ResNet(object):

    def __init__(self, resnet_version, resnet_size, num_classes,
                 first_num_filters):
        """Define hyperparameters.

        Args:
            resnet_version: 1 or 2. If 2, use the bottleneck blocks.
            resnet_size: A positive integer (n).
            num_classes: A positive integer. Define the number of classes.
            first_num_filters: An integer. The number of filters to use for the
                first block layer of the model. This number is then doubled
                for each subsampling block layer.
        """
        self.resnet_version = resnet_version
        self.resnet_size = resnet_size
        self.num_classes = num_classes
        self.first_num_filters = first_num_filters

    def __call__(self, inputs, training):
        """Classify a batch of input images.

        Architecture (first_num_filters = 16):
        layer_name      | start | stack1 | stack2 | stack3 | output
        output_map_size | 32x32 | 32×32  | 16×16  | 8×8    | 1x1
        #layers 		| 1 	| 2n/3n	 | 2n/3n  | 2n/3n  | 1
        #filters 		| 16 	| 16(*4) | 32(*4) | 64(*4) | num_classes

        n = #residual_blocks in each stack layer = self.resnet_size
        The standard_block has 2 layers each.
        The bottleneck_block has 3 layers each.

        Example of replacing:
        standard_block 		conv3-16 + conv3-16
        bottleneck_block	conv1-16 + conv3-16 + conv1-64

        Args:
            inputs: A Tensor representing a batch of input images.
            training: A boolean. Used by operations that work differently
                in training and testing phases.

        Returns:
            A logits Tensor of shape [<batch_size>, self.num_classes].
        """

        outputs = self._start_layer(inputs, training)

        if self.resnet_version == 1:
            block_fn = self._standard_block_v1
        else:
            block_fn = self._bottleneck_block_v2

        for i in range(3):
            filters = self.first_num_filters * (2 ** i)
            strides = 1 if i == 0 else 2
            with tf.variable_scope('scope__' + str(i), reuse=tf.AUTO_REUSE):
                outputs = self._stack_layer(outputs, filters, block_fn, strides, training)

        outputs = self._output_layer(outputs, training)

        return outputs

    ################################################################################
    # Blocks building the network
    ################################################################################
    def _batch_norm_relu(self, inputs, training):
        """Perform batch normalization then relu."""

        ### YOUR CODE HERE
        with tf.variable_scope('bnorm', reuse=tf.AUTO_REUSE):
            outputs = tf.layers.batch_normalization(inputs=inputs, axis=3, training=training, trainable=training)
            outputs = tf.nn.relu(outputs)
        ### END CODE HERE

        return outputs

    def _start_layer(self, inputs, training):
        """Implement the start layer.

        Args:
            inputs: A Tensor of shape [<batch_size>, 32, 32, 3].
            training: A boolean. Used by operations that work differently
                in training and testing phases.

        Returns:
            outputs: A Tensor of shape [<batch_size>, 32, 32, self.first_num_filters].
        """

        ### YOUR CODE HERE
        # initial conv1
        with tf.variable_scope('start_layer', reuse=tf.AUTO_REUSE):
            outputs = tf.layers.conv2d(inputs=inputs,
                                       filters=self.first_num_filters,
                                       kernel_size=3,
                                       strides=1,
                                       padding="same",
                                       use_bias=False,
                                       kernel_initializer=tf.variance_scaling_initializer(),
                                       trainable=training)
        ### END CODE HERE

        # We do not include batch normalization or activation functions in V2
        # for the initial conv1 because the first block unit will perform these
        # for both the shortcut and non-shortcut paths as part of the first
        # block's projection.
        if self.resnet_version == 1:
            outputs = self._batch_norm_relu(outputs, training)

        return outputs

    def _output_layer(self, inputs, training):
        """Implement the output layer.

        Args:
            inputs: A Tensor of shape [<batch_size>, 8, 8, channels].
            training: A boolean. Used by operations that work differently
                in training and testing phases.

        Returns:
            outputs: A logits Tensor of shape [<batch_size>, self.num_classes].
        """

        # Only apply the BN and ReLU for model that does pre_activation in each
        # bottleneck block, e.g. resnet V2.
        if self.resnet_version == 2:
            inputs = self._batch_norm_relu(inputs, training)

        ### YOUR CODE HERE
        with tf.variable_scope('output_layer', reuse=tf.AUTO_REUSE):
            outputs = tf.reduce_mean(inputs, [1, 2], keepdims=True)
            outputs = tf.squeeze(outputs, [1, 2])

            outputs = tf.layers.dense(inputs=outputs, units=self.num_classes)
        ### END CODE HERE

        return outputs

    def _stack_layer(self, inputs, filters, block_fn, strides, training):
        """Creates one stack of standard blocks or bottleneck blocks.

        Args:
            inputs: A Tensor of shape [<batch_size>, height_in, width_in, channels].
            filters: A positive integer. The number of filters for the first
                convolution in a block.
            block_fn: 'standard_block' or 'bottleneck_block'.
            strides: A positive integer. The stride to use for the first block. If
                greater than 1, this layer will ultimately downsample the input.
            training: A boolean. Used by operations that work differently
                in training and testing phases.

        Returns:
            outputs: The output tensor of the block layer.
        """

        filters_out = filters * 4 if self.resnet_version == 2 else filters

        def projection_shortcut(inputs):
            ### YOUR CODE HERE
            return tf.layers.conv2d(inputs=inputs,
                                    filters=filters_out,
                                    kernel_size=1,
                                    strides=strides,
                                    padding="same",
                                    use_bias=False,
                                    kernel_initializer=tf.variance_scaling_initializer(),
                                    trainable=training)
            ### END CODE HERE

        ### YOUR CODE HERE
        # Only the first block per stack_layer uses projection_shortcut
        with tf.variable_scope('stack_layer', reuse=tf.AUTO_REUSE):
            outputs = block_fn(inputs, filters, training, projection_shortcut, strides)

        for i in range(self.resnet_size - 1):
            with tf.variable_scope('stack_layer' + str(i), reuse=tf.AUTO_REUSE):
                outputs = block_fn(outputs, filters, training, None, 1)
        ### END CODE HERE

        return outputs

    def _standard_block_v1(self, inputs, filters, training, projection_shortcut, strides):
        """Creates a standard residual block for ResNet v1.

        Args:
            inputs: A Tensor of shape [<batch_size>, height_in, width_in, channels].
            filters: A positive integer. The number of filters for the first
                convolution.
            training: A boolean. Used by operations that work differently
                in training and testing phases.
            projection_shortcut: The function to use for projection shortcuts
                  (typically a 1x1 convolution when downsampling the input).
            strides: A positive integer. The stride to use for the block. If
                greater than 1, this block will ultimately downsample the input.

        Returns:
            outputs: The output tensor of the block layer.
        """

        shortcut = inputs

        if projection_shortcut is not None:
            ### YOUR CODE HERE
            with tf.variable_scope('v1-shortcut', reuse=tf.AUTO_REUSE):
                shortcut = projection_shortcut(shortcut)
                shortcut = tf.layers.batch_normalization(inputs=shortcut, axis=3, training=training, trainable=training)
            ### END CODE HERE

        ### YOUR CODE HERE
        with tf.variable_scope('v1', reuse=tf.AUTO_REUSE):
            outputs = tf.layers.conv2d(inputs=inputs,
                                       filters=filters,
                                       kernel_size=3,
                                       strides=strides,
                                       padding="same",
                                       use_bias=False,
                                       kernel_initializer=tf.variance_scaling_initializer())
            outputs = self._batch_norm_relu(outputs, training)
            outputs = tf.layers.conv2d(inputs=outputs,
                                       filters=filters,
                                       kernel_size=3,
                                       strides=1,
                                       padding="same",
                                       use_bias=False,
                                       kernel_initializer=tf.variance_scaling_initializer())
            outputs = tf.layers.batch_normalization(inputs=outputs, axis=3, training=training, trainable=training)
            outputs = tf.add(outputs, shortcut)
            outputs = tf.nn.relu(outputs)
        ### END CODE HERE

        return outputs

    def _bottleneck_block_v2(self, inputs, filters, training, projection_shortcut, strides):
        """Creates a bottleneck block for ResNet v2.

        Args:
            inputs: A Tensor of shape [<batch_size>, height_in, width_in, channels].
            filters: A positive integer. The number of filters for the first
                convolution. NOTE: filters_out will be 4xfilters.
            training: A boolean. Used by operations that work differently
                in training and testing phases.
            projection_shortcut: The function to use for projection shortcuts
                  (typically a 1x1 convolution when downsampling the input).
            strides: A positive integer. The stride to use for the block. If
                greater than 1, this block will ultimately downsample the input.

        Returns:
            outputs: The output tensor of the block layer.
        """

        ### YOUR CODE HERE
        with tf.variable_scope('v2_0', reuse=tf.AUTO_REUSE):
            shortcut = inputs

            if projection_shortcut is not None:
                shortcut = projection_shortcut(shortcut)

            outputs = self._batch_norm_relu(inputs, training)
            outputs = tf.layers.conv2d(inputs=outputs,
                                       filters=filters,
                                       kernel_size=1,
                                       strides=strides,
                                       padding="same",
                                       use_bias=False,
                                       kernel_initializer=tf.variance_scaling_initializer())

        with tf.variable_scope('v2_1', reuse=tf.AUTO_REUSE):
            outputs = self._batch_norm_relu(outputs, training)
            outputs = tf.layers.conv2d(inputs=outputs,
                                       filters=filters,
                                       kernel_size=3,
                                       strides=1,
                                       padding="same",
                                       use_bias=False,
                                       kernel_initializer=tf.variance_scaling_initializer())

        with tf.variable_scope('v2_2', reuse=tf.AUTO_REUSE):
            outputs = self._batch_norm_relu(outputs, training)
            outputs = tf.layers.conv2d(inputs=outputs,
                                       filters=filters*4,
                                       kernel_size=1,
                                       strides=1,
                                       padding="same",
                                       use_bias=False,
                                       kernel_initializer=tf.variance_scaling_initializer())

            outputs = tf.add(outputs, shortcut)
            outputs = tf.nn.relu(outputs)
        ### END CODE HERE

        return outputs
