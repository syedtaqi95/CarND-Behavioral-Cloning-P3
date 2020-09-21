"""
process_data.py

Use this file to process the training/validation data from the Udacity simulator

- Renames image locations in the CSV to match those in the workspace (for hassle-free import from a local machine)
- Saves the images and measurements to a pickle file to be used in model.py
"""
import csv
from tempfile import NamedTemporaryFile
import shutil
import cv2
import numpy as np
import pickle

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
    
if __name__ == "__main__":
    process_data(DATA_PATH="training_data/run1/", overwrite_csv=True)
    