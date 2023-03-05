"""
# Reference
- [Inverted Residuals and Linear Bottlenecks Mobile Networks for
   Classification, Detection and Segmentation]
   (https://arxiv.org/abs/1801.04381)
"""

import tensorflow as tf

import h5py
from imblearn.over_sampling import SMOTE
import numpy as np
from keras.utils import to_categorical
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
# Define ReLU6 activation
relu6 = tf.keras.layers.ReLU(6.)

def _conv_block(inputs, filters, kernel, strides):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
    # Returns
        Output tensor.
    """

    x = tf.keras.layers.Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    return relu6(x)


def _bottleneck(inputs, filters, kernel, t, s, r=False):
    """Bottleneck
    This function defines a basic bottleneck structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        r: Boolean, Whether to use the residuals.
    # Returns
        Output tensor.
    """

    tchannel = inputs.shape[-1] * t

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = tf.keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = relu6(x)

    x = tf.keras.layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if r:
        x = tf.keras.layers.add([x, inputs])
    return x


def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
    """

    x = _bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, True)

    return x


def MobileNetV2(input_shape, k, plot_model=False):
    """MobileNetv2
    This function defines a MobileNetv2 architecture.
    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
        k: Integer, number of classes.
        plot_model: Boolean, whether to plot model architecture or not
    # Returns
        MobileNetv2 model.
    """

    inputs = tf.keras.layers.Input(shape=input_shape, name='input')
    x = _conv_block(inputs, 32, (3, 3), strides=(2, 2))

    x = _inverted_residual_block(x, 16, (3, 3), t=1, strides=1, n=1)
    x = _inverted_residual_block(x, 24, (3, 3), t=6, strides=2, n=2)
    x = _inverted_residual_block(x, 32, (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, 64, (3, 3), t=6, strides=2, n=4)
    x = _inverted_residual_block(x, 96, (3, 3), t=6, strides=1, n=3)
    x = _inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, 320, (3, 3), t=6, strides=1, n=1)

    x = _conv_block(x, 1280, (1, 1), strides=(1, 1))
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Reshape((1, 1, 1280))(x)
    x = tf.keras.layers.Dropout(0.3, name='Dropout')(x)
    x = tf.keras.layers.Conv2D(k, (1, 1), padding='same')(x)
    x = tf.keras.layers.Activation('softmax', name='final_activation')(x)
    output = tf.keras.layers.Reshape((k,), name='output')(x)
    model = tf.keras.models.Model(inputs, output)
    model.summary()
    if plot_model:
        tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

    return model

if __name__ == "__main__":
    input_shape = (16,16,16)
    NUM_CLASSES = 10


    oversample = SMOTE()
    with h5py.File("data_voxel_"+str(NUM_CLASSES)+".h5", "r") as hf:
        X_train = hf["X_train"][:]
        X_train = np.array(X_train)

        targets_train = hf["y_train"][:]

        X_test = hf["X_test"][:]
        X_test = np.array(X_test)

        targets_test = hf["y_test"][:]
        test_y = targets_test
        # Determine sample shape
        sample_shape = (16, 16, 16)

        X_train, targets_train = oversample.fit_resample(X_train, targets_train)
        X_train = np.array(X_train)

        X_test, targets_test = oversample.fit_resample(X_test, targets_test)

        X_train = X_train.reshape(X_train.shape[0], 16, 16, 16)
        X_test = X_test.reshape(X_test.shape[0], 16, 16, 16)

        targets_train = to_categorical(targets_train).astype(np.int32)
        targets_test = to_categorical(targets_test).astype(np.int32)

    NUM_EPOCH = 50
    is_training = True
    if is_training:
        model = MobileNetV2(input_shape=input_shape, k=NUM_CLASSES)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(X_train, targets_train, epochs=NUM_EPOCH, verbose=1,
                            validation_split=0.2)

        model.save('models/mobilenetv2_modelnet'+str(NUM_CLASSES)+'.h5', save_format='h5')
        hist_df = pd.DataFrame(history.history)

        # or save to csv:
        hist_csv_file = 'history/history_mobilenetv2_modelnet'+str(NUM_CLASSES)+'.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

        # plt.plot(history.history['loss'], label='Categorical crossentropy (training data)')
        # plt.plot(history.history['val_loss'], label='Categorical crossentropy (validation data)')
        # plt.title('Model performance for 3D Voxel Keras Conv2D (Loss)')
        # plt.ylabel('Loss value')
        # plt.xlabel('No. epoch')
        # plt.legend(['train', 'test'], loc="upper left")
        # plt.show()
        #
        # # # Plot history: Categorical Accuracy
        # plt.plot(history.history['accuracy'], label='Accuracy (training data)')
        # plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
        # plt.title('Model performance for 3D Voxel Keras Conv2D (Accuracy)')
        # plt.ylabel('Accuracy value')
        # plt.xlabel('No. epoch')
        # plt.legend(['train', 'test'], loc="upper left")
        # plt.show()
    else:
        model = tf.keras.models.load_model("models/mobilenetv2_modelnet" + str(NUM_CLASSES) + ".h5")

    loss, accuracy = model.evaluate(X_test, targets_test)

    print(loss, accuracy)

    labels = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
    pred = model.predict(X_test)
    pred = np.argmax(pred, axis=1)

    y = np.argmax(targets_test, axis=1)

    report = metrics.classification_report(y, pred, target_names=labels)
    print(report)


