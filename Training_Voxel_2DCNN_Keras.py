import keras
from keras.models import *
from keras.layers import *
from keras.utils import to_categorical
import h5py
import numpy as np
import matplotlib.pyplot as plt
from keras import layers

from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import h5py
import seaborn as sns
sns.set_style('white')

from imblearn.over_sampling import SMOTE

# -- Preparatory code --
# Model configuration
batch_size = 32
no_epochs = 50
learning_rate = 0.001
no_classes = 10
validation_split = 0.2
verbosity = 1


# -- Process code --
# Load the HDF5 data file
with h5py.File("data_voxel_10.h5", "r") as hf:

    # Split the data into training/test features/targets
    X_train = hf["X_train"][:]
    X_train = np.array(X_train)

    targets_train = hf["y_train"][:]
  
    X_test = hf["X_test"][:] 
    X_test = np.array(X_test)

    targets_test = hf["y_test"][:]
    test_y = targets_test
    # Determine sample shape
    sample_shape = (16, 16, 16)

    #Over sampling
    oversample = SMOTE()
    # print(oversample)
    X_train, targets_train = oversample.fit_resample(X_train, targets_train)
    # print(X_train.shape)
    X_train = np.array(X_train)

    X_train = X_train.reshape(X_train.shape[0], 16, 16, 16)
    X_test  = X_test.reshape(X_test.shape[0], 16, 16, 16)

    # print(X_train.shape)
    # print(X_test.shape)


    # # # Convert target vectors to categorical targets
    targets_train = to_categorical(targets_train).astype(np.int32)
    targets_test = to_categorical(targets_test).astype(np.int32)


    # #Create the model

    model = Sequential()

    ########### MODEL 1
    model.add(Conv2D(32, (3, 3), activation='relu',
              padding='same', input_shape=sample_shape))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    #model.add(Dropout(0.6))
    model.add(Flatten())
    model.add(Dense(100, 'relu'))
    #model.add(Dropout(0.7))
    #model.add(Dense(64, 'relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(no_classes, 'softmax'))

    model.summary()
    ###### MODEL 2
    # model.add(ZeroPadding2D(padding=(3,3),input_shape=sample_shape))
    # model.add(Conv2D(16, (5, 5), strides=(3,3), activation='relu'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(32, (3, 3), strides=(1,1), activation='relu'))
    # model.add(ZeroPadding2D(padding=(1,1)))
    # model.add(Conv2D(64, (3, 3), strides=(1,1), activation='relu'))
    # # Flatten feature map to Vector with 576 element.
    # model.add(Dropout(0.6))
    # model.add(Flatten())
    # # Fully Connected Layer
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dropout(0.7))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.7))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(no_classes,'softmax'))

    # model.summary()

    # # Compile the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])
    # Fit data to model
    history = model.fit(X_train, targets_train,
                batch_size=batch_size,
                epochs=no_epochs,
                verbose=verbosity,
                validation_split=validation_split)

    model.save('model_2d_40.h5')

    pred = model.predict(X_test)
    pred = np.argmax(pred, axis=1)
    array = confusion_matrix(test_y, pred)
    cm = pd.DataFrame(array, index = range(40), columns = range(40))
    plt.figure(figsize=(20,20))
    sns.heatmap(cm, annot=True)
    plt.show()
    # # Generate generalization metrics
    score = model.evaluate(X_test, targets_test, verbose=0)
    #print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    print('Test accuracy: %.2f%% Test loss: %.3f' % (score[1]*100, score[0]))

    # # Plot history: Categorical Loss
    plt.plot(history.history['loss'], label='Categorical crossentropy (training data)')
    plt.plot(history.history['val_loss'], label='Categorical crossentropy (validation data)')
    plt.title('Model performance for 3D Voxel Keras Conv2D (Loss)')
    plt.ylabel('Loss value')
    plt.xlabel('No. epoch')
    plt.legend(['train','test'],loc="upper left")
    plt.show()

    # # Plot history: Categorical Accuracy
    plt.plot(history.history['accuracy'], label='Accuracy (training data)')
    plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
    plt.title('Model performance for 3D Voxel Keras Conv2D (Accuracy)')
    plt.ylabel('Accuracy value')
    plt.xlabel('No. epoch')
    plt.legend(['train','test'],loc="upper left")
    plt.show()

