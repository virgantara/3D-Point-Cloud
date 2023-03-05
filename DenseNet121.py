import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense
from tensorflow.keras.layers import AvgPool2D, GlobalAveragePooling2D, MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ReLU, concatenate
import tensorflow.keras.backend as K

from sklearn import metrics
import h5py
from imblearn.over_sampling import SMOTE
import numpy as np
from keras.utils import to_categorical
import pandas as pd
import matplotlib.pyplot as plt
import os

from path import Path

# Creating Densenet121
def densenet(input_shape, n_classes, filters = 32):

    #batch norm + relu + conv
    def bn_rl_conv(x, filters, kernel=1, strides=1):
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters, kernel, strides=strides, padding='same')(x)
        return x


    def dense_block(x, repetition):
        for _ in range(repetition):
            y = bn_rl_conv(x, 4 * filters)
            y = bn_rl_conv(y, filters, 3)
            x = concatenate([y, x])
        return x


    def transition_layer(x):
        x = bn_rl_conv(x, K.int_shape(x)[-1] // 2)
        x = AvgPool2D(2, strides=2, padding='same')(x)
        return x


    input = Input(input_shape)
    x = Conv2D(64, 7, strides=2, padding='same')(input)
    x = MaxPool2D(3, strides=2, padding='same')(x)

    for repetition in [6, 12, 24, 16]:
        d = dense_block(x, repetition)
        x = transition_layer(d)

    x = GlobalAveragePooling2D()(d)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(input, output)
    return model




if __name__ == "__main__":
    input_shape = (16,16,16)

    BASEDATA_PATH = "/media/virgantara/DATA1/Penelitian/Datasets"
    DATA_DIR = os.path.join(BASEDATA_PATH, "ModelNet40")
    path = Path(DATA_DIR)
    folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path / dir)]
    classes = {folder: i for i, folder in enumerate(folders)};
    NUM_CLASSES = np.array(folders).shape[0]
    oversample = SMOTE()
    with h5py.File("data_voxel_"+str(NUM_CLASSES)+"_16.h5", "r") as hf:
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
        model = densenet(input_shape=input_shape, n_classes=NUM_CLASSES)
        model.summary()
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(X_train, targets_train, epochs=NUM_EPOCH, verbose=1,
                            validation_split=0.2)

        model.save('densenet121_modelnet'+str(NUM_CLASSES)+'.h5', save_format='h5')
        hist_df = pd.DataFrame(history.history)

        # or save to csv:
        hist_csv_file = 'history_densenet121_modelnet'+str(NUM_CLASSES)+'.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

        plt.plot(history.history['loss'], label='Categorical crossentropy (training data)')
        plt.plot(history.history['val_loss'], label='Categorical crossentropy (validation data)')
        plt.title('Model performance for 3D Voxel Keras Conv2D (Loss)')
        plt.ylabel('Loss value')
        plt.xlabel('No. epoch')
        plt.legend(['train', 'test'], loc="upper left")
        plt.show()

        # # Plot history: Categorical Accuracy
        plt.plot(history.history['accuracy'], label='Accuracy (training data)')
        plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
        plt.title('Model performance for 3D Voxel Keras Conv2D (Accuracy)')
        plt.ylabel('Accuracy value')
        plt.xlabel('No. epoch')
        plt.legend(['train', 'test'], loc="upper left")
        plt.show()
    else:
        model = tf.keras.models.load_model("models/densenet121_modelnet10.h5")

    loss, accuracy = model.evaluate(X_test, targets_test)

    print(loss, accuracy)

    labels = folders
    pred = model.predict(X_test)
    pred = np.argmax(pred, axis=1)

    y = np.argmax(targets_test, axis=1)

    report = metrics.classification_report(y, pred, target_names=labels)
    print(report)


