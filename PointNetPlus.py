import tensorflow as tf


def pointnet_layer(inputs, npoints, nsample, mlp, scope):
    """
    PointNet layer with max pooling and grouping operations.
    """
    with tf.name_scope(scope):
        # randomly sample npoints from inputs
        indices = tf.random.uniform(shape=(npoints,), maxval=tf.shape(inputs)[1], dtype=tf.int32)
        sampled_points = tf.gather(inputs, indices, axis=1)

        # perform grouping operation on sampled points
        distances = tf.reduce_sum(tf.square(tf.expand_dims(inputs, 2) - tf.expand_dims(sampled_points, 1)), axis=-1)
        indices = tf.math.top_k(-distances, k=nsample)[1]
        grouped_points = tf.gather(inputs, indices, axis=1)

        # apply MLP to grouped points
        for units in mlp:
            grouped_points = tf.keras.layers.Conv1D(units, 1, activation=tf.nn.relu)(grouped_points)

        # perform max pooling over grouped points
        max_pooled = tf.reduce_max(grouped_points, axis=1)

        # apply MLP to sampled points
        for units in mlp:
            sampled_points = tf.keras.layers.Conv1D(units, 1, activation=tf.nn.relu)(sampled_points)

        # concatenate max pooled and sampled points
        outputs = tf.concat([max_pooled, sampled_points], axis=-1)
        return outputs


def pointnet(inputs, is_training):
    """
    PointNet++ architecture.
    """
    # inputs shape: (batch_size, num_points, num_features)
    net = inputs

    # PointNet layer 1
    net = pointnet_layer(net, npoints=512, nsample=32, mlp=[64, 64, 128], scope='layer1')

    # PointNet layer 2
    net = pointnet_layer(net, npoints=128, nsample=64, mlp=[128, 128, 256], scope='layer2')

    # PointNet layer 3
    net = pointnet_layer(net, npoints=None, nsample=None, mlp=[256, 512, 1024], scope='layer3')

    # global max pooling
    net = tf.reduce_max(net, axis=1, keepdims=True)

    # fully connected layer
    net = tf.keras.layers.Dense(512, activation=tf.nn.relu)(net)
    net = tf.keras.layers.Dropout(rate=0.5)(net, training=is_training)

    # output layer
    net = tf.keras.layers.Dense(10, activation=None)(net)
    return net
