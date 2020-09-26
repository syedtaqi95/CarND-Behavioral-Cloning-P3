# **Behavioral Cloning Project Writeup** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[centre_lane]: ./writeup_images/centre_lane.jpg (centre_lane)
[counter]: ./writeup_images/counter.jpg (counter)
[recovery_1]: ./writeup_images/recovery_1.jpg (recovery_1)
[recovery_2]: ./writeup_images/recovery_2.jpg (recovery_2)
[smooth_curve]: ./writeup_images/smooth_curve.jpg (smooth_curve)
[augment_left]: ./writeup_images/augment_left.jpg (augment_left)
[augment_centre]: ./writeup_images/augment_centre.jpg (augment_centre)
[augment_right]: ./writeup_images/augment_right.jpg (augment_right)
[flipped_augment_left]: ./writeup_images/flipped_augment_left.jpg (flipped_augment_left)
[flipped_augment_centre]: ./writeup_images/flipped_augment_centre.jpg (flipped_augment_centre)
[flipped_augment_right]: ./writeup_images/flipped_augment_right.jpg (flipped_augment_right)
[model_history]: ./writeup_images/model_history.png (model_history)

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 showing the model driving the car around the track

#### 2. Submission includes functional code
Using the Udacity provided simulator and the drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py and process_data.py files contain the code for training and saving the convolution neural network. process_data.py was contains functions to pre-process the data and also contains the generator used to feed data into the model, while model.py shows the pipeline I used for training and validating the model, and contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used a modified version of NVIDIA's DAVE-2 architecture ([link to paper](https://arxiv.org/pdf/1604.07316v1.pdf)). 

The first layer is a Keras lambda layer for input normalisation. I normalised the input images to have zero mean and equal variance (model.py line 73).

The next layer is a *Cropping2D* layer to remove the top 40 pixels and bottom 20 pixels of the image. I did this to discard the parts of the frame that are not relevant to driving a vehicle (e.g. car bonnet, sky etc.). The output of this layer is a normalised 100x320 image (model.py line 76).

Next there are 3 convolutional layers with 5x5 kernel size, 2x2 strides, 'valid' padding, ReLU activation and depth sizes between 24 and 48 (model.py lines 79-81).

Then there are 2 convolutional layers with 3x3 kernal size, 1x1 stride, 'valid' padding, ReLU activation and 64 depth size (model.py 84-85).

I flattened the output of the convolutional layers and implemented 3 fully connected layers of sizes 100, 50 and 10 and using Relu activation units. I also added dropout layers between each fully connected layer with a 0.5 dropout rate to reduce overfitting (model.py lines 88-96).

The output layer uses tanh activation instead of ReLU to limit the range of the output between -1 and +1 as this works better for an output like steering angle (model.py line 99).

#### 2. Attempts to reduce overfitting in the model

As explained above, the model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. This consisted of the following runs:
- 2 laps of centre lane driving
- 1 lap of counter-clockwise driving
- 1 lap of recovery driving
- 1 lap of smooth curve driving

 The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (model.py line 102).

For training, I used a batch size of 64 and trained for 10 epochs.

#### 4. Appropriate training data

TThe model was trained and validated on different data sets to keep the vehicle on the track. This consisted of the following runs:
- 2 laps of centre lane driving
- 1 lap of counter-clockwise driving
- 1 lap of recovery driving
- 1 lap of smooth curve driving

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My overall strategy for deriving a model architecture was to consider popular convolutional network architectures from academia and industry, and modify it for my own use case. Some of these include the NVIDIA DAVE-2 architecture, LeNet-5, VGGNet etc.

I based my network on NVIDIA's DAVE-2 architecture as this model predicts steering angle with great accuracy using images as input. I used the Keras *Sequential* API to achieve this.

I split my data into training and validation sets and ran this model without modification. I found the mean-squared error was low on the training set and high on the validation set, i.e. my model was overfitting. To combat this, I added dropout layers.

I then normalised my data to improve prediction accuracy and shuffled the data. To generate more data, I used some data augmentation techniques in my generator. One was to use the centre, left and right images with some camera offset to simulate recovery driving. The other technique was to flip the images and steering angles to simulate counter clockwise driving.

The vehicle was driving fairly well after this, but my code was saving the model after each epoch. So I created a *ModelCheckpoint* callback to save the model with the least *val_loss* only.

To reduce memory usage, I implemented a Python generator (process_data.py lines 99-132). It loads the samples in batches from the CSV so my code didn't have to load the complete training dataset in memory (which was over 500 MB at this point!). 

I also added the capability to load the model if it was saved previously (model.py lines 54-56 and 63-65). I found the Adam optimiser was resetting the learning rate if I loaded a saved model (which was retraining the model from scratch), so I used the ```initial_epochs```  parameter in my ```model.fit_generator()``` function.

At the end of the process, the vehicle was able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

|Layer (type)             |    Output Shape          |    Param #   |
|-------------------------|--------------------------|--------------|
|lambda_1 (Lambda)        |    (None, 160, 320, 3)   |    0         |
|cropping2d_1 (Cropping2D)|    (None, 100, 320, 3)   |    0         |
|conv2d_1 (Conv2D)        |    (None, 48, 158, 24)   |    1824      |
|conv2d_2 (Conv2D)        |    (None, 22, 77, 36)    |    21636     |
|conv2d_3 (Conv2D)        |    (None, 9, 37, 48)     |    43248     |
|conv2d_4 (Conv2D)        |    (None, 7, 35, 64)     |    27712     |
|conv2d_5 (Conv2D)        |    (None, 5, 33, 64)     |    36928     |
|flatten_1 (Flatten)      |    (None, 10560)         |    0         |
|dense_1 (Dense)          |    (None, 100)           |    1056100   |
|dropout_1 (Dropout)      |    (None, 100)           |    0         |
|dense_2 (Dense)          |    (None, 50)            |    5050      |
|dropout_2 (Dropout)      |    (None, 50)            |    0         |
|dense_3 (Dense)          |    (None, 10)            |    510       |
|dropout_3 (Dropout)      |    (None, 10)            |    0         |
|dense_4 (Dense)          |    (None, 1)             |    11        |

Total parameters: 1,193,019

#### 3. Creation of the Training Set & Training Process

As mentioned previously, I used the following runs to generate my training data:

1. Two laps of centre lane driving, which looked like this:

![centre_lane]

2. One lap of counter-clockwise driving:

![counter]

3. 1 lap of recovery driving. The left image is the start of the recovery and the right image is the recovered position:

![recovery_1] ![recovery_2]

4. 1 lap of smooth curve driving. The image below shows the vehicle driven by myself staying central while navigating a sharp turn:

![smooth_curve]

I used a generator ```train_generator()``` to feed training data to the model. I also used a ```valid_generator()``` to feed validation data to the model which works identically to the training generator. It consists of the following steps:

- Shuffling the data.

- Loading data in batches of 64 samples.

- I used the centre, left and right images with a +/-0.2 steering angle offset to augment the training data. An example:

![augment_left] ![augment_centre] ![augment_right]

- I flipped the images and angles thinking that this would generate even more data to simulate counter-clockwise driving. For example, here are the above images flipped:

![flipped_augment_left] ![flipped_augment_centre] ![flipped_augment_right]

I used a training/validation split of 80:20 using ```sklearn```'s ```train_test_split()``` API and fed this data into the model for 10 epochs. As the training history below shows, the validation loss didn't improve after 8 epochs hence I used 10 epochs to be safe. Using the ```ModelCheckpoint``` callback, I only saved the model with the lowest *val_loss*.

![model_history]

Since I used an Adam optimiser, I didn't need to manually tune the learning rate.

### Simulation

#### 1. Correct navigation on the test data

*video.mp4* shows the vehicle driving on track 1 using the trained model. As the video shows, the vehicle completes a full lap around the track while staying on the road.

