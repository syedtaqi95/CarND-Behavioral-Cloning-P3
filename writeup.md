# **Behavioral Cloning Project Writeup** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

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

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 102).

For training, I used a batch size of 64 and 10 epochs.

#### 4. Appropriate training data

TThe model was trained and validated on different data sets to keep the vehicle on the track. This consisted of the following runs:
- 2 laps of centre lane driving
- 1 lap of counter-clockwise driving
- 1 lap of recovery driving
- 1 lap of smooth curve driving

### Model Architecture and Training Strategy

#### 1. Solution Design Approach



---
The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
