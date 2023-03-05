import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn import metrics
import h5py
from imblearn.over_sampling import SMOTE
import numpy as np
from keras.utils import to_categorical
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

def get_model(input_shape=(16, 16, 3), n_classes=10):
    model = Sequential()
    model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(1, 1), strides=(1, 1)))
    model.add(Flatten())

    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=n_classes, activation="softmax"))

    return model




if __name__ == "__main__":
    # model code
    NUM_CLASSES = 10

    input_shape = (16,16,16)


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
        model = get_model(input_shape=input_shape, n_classes=NUM_CLASSES)
        model.summary()
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(X_train, targets_train, epochs=NUM_EPOCH, verbose=1,
                            validation_split=0.3)

        model.save('models/vgg16_modelnet'+str(NUM_CLASSES)+'.h5', save_format='h5')
        hist_df = pd.DataFrame(history.history)

        # or save to csv:
        hist_csv_file = 'history/history_vgg16_modelnet'+str(NUM_CLASSES)+'.csv'
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
        model = tf.keras.models.load_model("models/vgg16_modelnet"+str(NUM_CLASSES)+".h5")

    loss, accuracy = model.evaluate(X_test, targets_test)

    print(loss, accuracy)
    labels = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
    pred = model.predict(X_test)
    pred = np.argmax(pred, axis=1)

    y = np.argmax(targets_test, axis=1)

    report = metrics.classification_report(y, pred, target_names=labels)
    print(report)

