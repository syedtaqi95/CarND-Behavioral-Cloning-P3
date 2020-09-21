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
from math import ceil
import cv2
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Conv2D, Dropout
from keras.utils import print_summary
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os.path

from process_data import process_csv, data_generator

TRAINING_DATA_PATH = "/opt/carnd_p3/training_data/run5_counterclockwise/"
CSV_FILE = TRAINING_DATA_PATH + "driving_log.csv"

# Process the CSV first so it points to the images correctly
try:
    process_csv(DATA_PATH=TRAINING_DATA_PATH, original_char="\\", new_char="/")
except:
    # Load the default data located in /opt/carnd-p3/data/
    process_csv(DATA_PATH="/opt/carnd_p3/data/", original_char="/", new_char="/")

# Save samples in memory
samples = []
with open(CSV_FILE) as csvfile:
    reader = csv.reader(csvfile)
    # Skip header row
    next(reader)
    for line in reader:
        samples.append(line)    

# Split the data into training and validation sets
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Set batch size
batch_size=64

# Compile and train the model using the generator
train_generator = data_generator(train_samples, batch_size=batch_size)
validation_generator = data_generator(validation_samples, batch_size=batch_size)

# If the model exists already then use it
if os.path.isfile("model.h5"):
    model = load_model("model.h5")
else:
    # Create a modified version of Nvidia's DAVE-2 architecture in Keras
    model = Sequential()
    input_shape=(160, 320, 3)

    # Input normalisation layer
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape))

    # Cropping layer to remove pixels each from top and bottom of the image
    model.add(Cropping2D(cropping=((40,20), (0,0))))

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

# Fit the model on the training data
history_object = model.fit_generator(train_generator, 
                                     steps_per_epoch=ceil(len(train_samples)/batch_size), 
                                     validation_data=validation_generator, 
                                     validation_steps=ceil(len(validation_samples)/batch_size), 
                                     callbacks=[checkpoint], 
                                     epochs=10, 
                                     verbose=2)

# Plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig(TRAINING_DATA_PATH + "model_history.png")

# Save the model
model.save("model.h5")
print("Model saved.")
