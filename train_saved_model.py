"""
train_saved_model.py

Loads a saved model and train it on new training data
"""

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Conv2D, Dropout
from keras.utils import print_summary
from keras.callbacks import ModelCheckpoint

# Load the model
model = load_model('model.h5')