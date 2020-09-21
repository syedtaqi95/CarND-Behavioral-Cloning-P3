"""
process_data.py - Functions used to process the training/validation data for the Keras model

process_csv():
- Used to modify the image location delimiter to allow data exchange between Windows and Unix systems
- Windows convention is "\", Unix convention is "/"
- Changes image path in CSV from old location to new location

process_data(): # OLD METHOD (WITHOUT GENERATOR)
- Renames image locations in the CSV to match those in the workspace (for hassle-free import from a local machine)
- Saves the images and measurements to a pickle file to be used in model.py

data_generator(): # NEW METHOD (USING GENERATOR)
- Loads lines from the CSV in batches
- Sets X_train and y_train in batches
"""

import csv
from tempfile import NamedTemporaryFile
import shutil
import cv2
import numpy as np
from sklearn.utils import shuffle


def process_csv(DATA_PATH, original_char="/", new_char="/"):
    IMG_PATH = DATA_PATH + "IMG" + new_char
    CSV_FILE = DATA_PATH + "driving_log.csv"
    tempfile = NamedTemporaryFile('w+t', delete=False)
    
    # Open the CSV
    with open(CSV_FILE) as csvfile, tempfile:
        reader = csv.reader(csvfile)
        writer = csv.writer(tempfile)
        
        # Skip the header row
        next(reader)

        # Loop through each line
        for line in reader:
            # Rename centre, left and right images
            # NOTE Change the split character to \ or / depending on the OS file path convention
            for i in range(3):
                filename = line[i].split(original_char)[-1]
                line[i] = IMG_PATH + filename
            writer.writerow(line)

    shutil.move(tempfile.name, CSV_FILE)
    print("CSV modified and saved to", CSV_FILE)

    
def process_data(DATA_PATH, overwrite_csv=False, file_char="/"):
    print("Processing data located in", DATA_PATH)
    IMG_PATH = DATA_PATH + "IMG/"
    CSV_FILE = DATA_PATH + "driving_log.csv"
    tempfile = NamedTemporaryFile('w+t', delete=False)

    images = []
    measurements = []

    # Open the CSV
    with open(CSV_FILE) as csvfile, tempfile:
        reader = csv.reader(csvfile)
        writer = csv.writer(tempfile)
        
        # Skip the header row
        next(reader)

        # Loop through each line
        for line in reader:
            # Rename centre, left and right images
            # NOTE Change the split character to \ or / depending on the OS file path convention
            for i in range(3):
                filename = line[i].split(file_char)[-1]
                line[i] = IMG_PATH + filename
            writer.writerow(line)

            # Read the centre camera image
            image = cv2.imread(line[0])
            images.append(image)

            # Save the steering angle measurement
            measurement = float(line[3])
            measurements.append(measurement)

    # Save the temp CSV to the original file
    if overwrite_csv == True:
        shutil.move(tempfile.name, CSV_FILE)
        print("CSV modified and saved to", CSV_FILE)

    # Convert labels and features to np.array and save them
    X_train = np.array(images)
    y_train = np.array(measurements, dtype=np.float32)

    with open(DATA_PATH + "data.p", "wb") as f:
        pickle.dump([X_train, y_train], f)
        print("X_train and y_train saved to", DATA_PATH + "data.p")

        
def data_generator(samples, batch_size=32):
    while 1:
        # Shuffle the data
        shuffle(samples)
        
        # Load labels and features in batches
        for offset in range(0, len(samples), batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []            
            
            # Append images and steering angles for each sample
            CAMERA_OFFSET = 0.2
            for batch_sample in batch_samples:
                # Save centre, left and right images per sample to augemnt data
                for i in range(0,3):
                    image = cv2.imread(batch_sample[i])
                    angle = float(batch_sample[3])
                    if i == 1: angle = angle + CAMERA_OFFSET  # Left image, increase steering
                    if i == 2: angle = angle - CAMERA_OFFSET  # Right image, decrease steering
                    images.append(image)
                    angles.append(angle)
                    
                    # Save the flipped image and steering angle for even more data augmentation
                    image_flipped = np.fliplr(image)
                    angle_flipped = -angle
                    images.append(image_flipped)
                    angles.append(angle_flipped)
            
            # Convert labels and features to numpy arrays for faster processing
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

            
if __name__ == "__main__":
    # Old method (without using generator)
    # process_data(DATA_PATH="/opt/carnd_p3/data/", overwrite_csv=False, file_char="/")
    
    # New method (using generator)
    process_csv(DATA_PATH="/opt/carnd_p3/data/", original_char="/", new_char="/")