"""
model.py

This is the top level file to run the project

- Loads the captured training data and processes it
- Augments the training data
- Creates a Keras model
- Trains the model on the training data
- Saves the model to 'model.h5'
"""
import csv
from tempfile import NamedTemporaryFile
import shutil
import cv2
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Conv2D, Dropout, Reshape
from keras.utils import print_summary
from keras.backend import resize_images
from keras.callbacks import ModelCheckpoint

from process_data import process_data

TRAINING_DATA_PATH = "training_data/run1/"

# Load and process the training data if it exists
if TRAINING_DATA_PATH is not None:
    process_data(TRAINING_DATA_PATH, overwrite_csv=True)
    with open(TRAINING_DATA_PATH + "data.p", "rb") as f:
        X_train, y_train = pickle.load(f)
else:
    # Load the sample data located in /opt/carnd-p3/data/
    process_data("/opt/carnd_p3/data/", overwite_csv=True)
    with open("/opt/carnd_p3/data/data.p", "rb") as f:
        X_train, y_train = pickle.load(f)

# Print the dimensions of the labels and features set
print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)

# Create a modified version of Nvidia's DAVE-2 architecture in Keras
model = Sequential()
input_shape=(160, 320, 3)

# Input normalisation layer
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape))

# Cropping layer to remove 20 pixels each from top and bottom of the image
model.add(Cropping2D(cropping=((50,20), (0,0))))

# Resize image to 66x200

# 3 Conv2D layers - 5x5 kernel, 2x2 stride, 'valid' padding, 'relu' activation
model.add(Conv2D(24, kernel_size=5, strides=(2,2), padding="valid", activation="relu"))
model.add(Conv2D(36, kernel_size=5, strides=(2,2), padding="valid", activation="relu"))
model.add(Conv2D(48, kernel_size=5, strides=(2,2), padding="valid", activation="relu"))

# 2 Conv2D layers - 3x3 kernel, 1x1 stride, 'valid' padding, 'relu' activation
model.add(Conv2D(64, kernel_size=3, strides=(1,1), padding="valid", activation="relu"))
model.add(Conv2D(64, kernel_size=3, strides=(1,1), padding="valid", activation="relu"))

# Flatten
model.add(Flatten())

# 3 Dense layers with Dropouts
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))

# Output layer with tanh activation
model.add(Dense(1, activation='tanh'))

# Print model summary
print_summary(model)

# Compile the model
model.compile(optimizer="adam", loss="mse")

# Add checkpoint to save the best model trained after each epoch
checkpoint = ModelCheckpoint("model.h5", monitor="val_loss", 
                             verbose=1, save_best_only=True, mode='min')


model.fit(X_train, y_train, epochs=10, batch_size=64, callbacks=[checkpoint],
         validation_split=0.2, shuffle=True, verbose=0)

# Save the model
model.save("model.h5")

