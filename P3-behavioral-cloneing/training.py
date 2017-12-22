# Train the model

import pickle
import numpy as np
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam


# Reload the data

def reload_data(pickle_file):
    print('reload: ', pickle_file)
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)
        X_train = pickle_data['X_train']
        y_train = pickle_data['y_train']
        del pickle_data  # Free up memory
    print('X_train shape: ', X_train.shape, 'y_train shape: ',y_train.shape)
    return X_train, y_train

X_train1, y_train1 = reload_data('./pre-data/1.pickle')
X_train2, y_train2 = reload_data('./pre-data/2.pickle')
X_train3, y_train3 = reload_data('./pre-data/3.pickle')
X_train4, y_train4 = reload_data('./pre-data/4.pickle')
X_train5, y_train5 = reload_data('./pre-data/5.pickle')
X_train6, y_train6 = reload_data('./pre-data/6.pickle')


# Combine the data set
print('combine data set ...')
X_train = np.concatenate((X_train1, X_train2, X_train3, X_train4, X_train5, X_train6))
y_train = np.concatenate((y_train1, y_train2, y_train3, y_train4, y_train5, y_train6))
X_train, y_train = shuffle(X_train, y_train)
print('X_train shape: ', X_train.shape, 'y_train shape: ',y_train.shape)


# ## Model architecture

# ### model 2: Nvidia

nvidia = Sequential()
nvidia.add(Lambda(lambda x: x/255. - 0.5, input_shape=(80, 160, 3)))
nvidia.add(Cropping2D(cropping=((35, 13), (0, 0))))
nvidia.add(Convolution2D(24, 3, 3, subsample=(2, 2), activation='relu'))
nvidia.add(Convolution2D(36, 3, 3, subsample=(2, 2), activation='relu'))
nvidia.add(Convolution2D(48, 3, 3, activation='relu'))
nvidia.add(Convolution2D(64, 3, 3, activation='relu'))
nvidia.add(Convolution2D(64, 3, 3, activation='relu'))
nvidia.add(Dropout(0.5))
nvidia.add(Flatten())
nvidia.add(Dense(100))
nvidia.add(Dense(50))
nvidia.add(Dense(10))
nvidia.add(Dense(1))


# ## Training method

# Hyperparameters
LEARNING_RATE = 1e-4
EPOCHS = 5

# Training
nvidia.compile(loss='mse', optimizer=Adam(LEARNING_RATE))
nvidia.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=EPOCHS)
nvidia.save('model.h5')
