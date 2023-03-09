import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Conv2D,Add
from tensorflow.keras.layers import SeparableConv2D,ReLU
from tensorflow.keras.layers import BatchNormalization,MaxPool2D
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras import Model

from sklearn import metrics
import h5py
from imblearn.over_sampling import SMOTE
import numpy as np
from keras.utils import to_categorical
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from path import Path
import os
def conv_bn(x, filters, kernel_size, strides=1):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same',
               use_bias=False)(x)
    x = BatchNormalization()(x)
    return x


def sep_bn(x, filters, kernel_size, strides=1):
    x = SeparableConv2D(filters=filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding='same',
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    return x


def entry_flow(x):
    x = conv_bn(x, filters=32, kernel_size=3, strides=2)
    x = ReLU()(x)
    x = conv_bn(x, filters=64, kernel_size=3, strides=1)
    tensor = ReLU()(x)

    x = sep_bn(tensor, filters=128, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=128, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=128, kernel_size=1, strides=2)
    x = Add()([tensor, x])

    x = ReLU()(x)
    x = sep_bn(x, filters=256, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=256, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=256, kernel_size=1, strides=2)
    x = Add()([tensor, x])

    x = ReLU()(x)
    x = sep_bn(x, filters=728, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=728, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=728, kernel_size=1, strides=2)
    x = Add()([tensor, x])
    return x


# middle flow

def middle_flow(tensor):
    for _ in range(8):
        x = ReLU()(tensor)
        x = sep_bn(x, filters=728, kernel_size=3)
        x = ReLU()(x)
        x = sep_bn(x, filters=728, kernel_size=3)
        x = ReLU()(x)
        x = sep_bn(x, filters=728, kernel_size=3)
        x = ReLU()(x)
        tensor = Add()([tensor, x])

    return tensor


# exit flow

def exit_flow(tensor, n_classes=1000):
    x = ReLU()(tensor)
    x = sep_bn(x, filters=728, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=1024, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=1024, kernel_size=1, strides=2)
    x = Add()([tensor, x])

    x = sep_bn(x, filters=1536, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=2048, kernel_size=3)
    x = GlobalAvgPool2D()(x)

    x = Dense(units=n_classes, activation='softmax')(x)

    return x


if __name__ == "__main__":
    # model code




    BASEDATA_PATH = "/media/virgantara/DATA1/Penelitian/Datasets"
    # DATA_DIR = "dataset/45Deg_merged"
    DATA_DIR = os.path.join(BASEDATA_PATH, "ModelNet10")
    path = Path(DATA_DIR)
    folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path / dir)]
    classes = {folder: i for i, folder in enumerate(folders)};
    NUM_CLASSES = np.array(folders).shape[0]
    NUM_EPOCH = 50
    # oversample = SMOTE()
    VOXEL_X = 16
    VOXEL_Y = 16
    VOXEL_Z = 16

    X_train, X_test, targets_train, targets_test = read_voxel_our(vx=VOXEL_X, vy=VOXEL_Y, vz=VOXEL_Z)
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
    is_training = True
    if is_training:
        input = Input(shape=(16, 16, 16))
        x = entry_flow(input)
        x = middle_flow(x)
        output = exit_flow(x, n_classes=NUM_CLASSES)
        model = Model(inputs=input, outputs=output)
        model.summary()
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(X_train, targets_train, epochs=NUM_EPOCH, verbose=1,
                            validation_split=0.2)
        hist_df = pd.DataFrame(history.history)

        # or save to csv:
        # hist_csv_file = 'history/history_xception_our_pose' + str(NUM_CLASSES) + '.csv'
        # with open(hist_csv_file, mode='w') as f:
        #     hist_df.to_csv(f)

        # model.save('xception_our_pose'+str(NUM_CLASSES)+'.h5', save_format='h5')

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
        model = tf.keras.models.load_model("models/xception_our_pose.h5")



    loss, accuracy = model.evaluate(X_test, targets_test)

    print(loss, accuracy)

    labels = folders
    pred = model.predict(X_test)
    pred = np.argmax(pred, axis=1)

    y = np.argmax(targets_test, axis=1)

    report = metrics.classification_report(y, pred, target_names=labels)
    print(report)


