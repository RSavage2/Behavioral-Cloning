# Behavioral-Cloning

Udacity Self Driving Car Engineer Nanodegree - Project 3

The goal of this project is to train a vehicle to drive autonomously in a simulator by taking camera images as inputs to a deep neural network and outputting a predicted steering angle based on the image.

The project has 4 main parts:

 1: Data Collection
 2: Image Preprocessing and Augmentation
 3: Building and Training a Neural Network
 4: Testing Model in Simulator

Data Collection

When training in the simulator, data is collected by recording images through three cameras on the vehicle; one in the center, one on the left side and one on the right. For each image which is recorded, the simulator also records an associated steering angle which is used as a label for the images.

https://d17h27t6h515a5.cloudfront.net/topher/2017/January/588ab788_driving-log-output/driving-log-output.png


 I recorded images for Controlled Driving.

While driving I did my best tp maintain the vehicle close to the center of the driving lane as it went through the course. For the controlled driving part of the dataset I used all three images (center, right, left) and I manufactured appropriate labels for the side camera images by adding a small amount of right turn to the left image labels and a small amount of left turn to the right image labels. The result of this is that when the car begins to drift from the center of the lane and finds itself in the approximate spot where the side camera had recorded the image, it will automatically know to steer itself back towards the center.

Controlled Driving
Controlled

Image Preprocessing 

The images outputted by the simulator are 160 by 320 pixels. First,  I chose to resize the images to 32 by 64. The final size of each image, and the input shape for my neural network, is 20 by 64 with 3 color channels.

The last preprocessing step was to create a validation set made up of 5% of random images/labels from the full set. The validation set was used to monitor the performance of the neural network in training.

Building and Training a Neural Network

For this problem I chose to use a convolutional neural network because this is an image classification problem. 
To come up with a model architecture, I used Nvidia's architecture as a starting pont and went from there. They trained a convolutional neural network for a similar type of problem.  My model also introduces a dropout layer after the first fully connected layer which helps to prevent overfitting to the training data.

My architecture model is as follows:

Batch Normalization (input shape = (20, 64, 3))
2D Convolutional Layer 1: (depth = 16, kernel = 3 x 3, stride = 2 X 2, border mode = valid, activation = ReLu, output shape = (None, 9, 31, 16))
2D Convolutional Layer 2: (depth = 24, kernel = 3 x 3, stride = 1 X 2, border mode = valid, activation = ReLu, output shape = (None, 7, 15, 24))
2D Convolutional Layer 3: (depth = 36, kernel = 3 x 3, border mode = valid, activation = ReLu, output shape = (None, 5, 13, 36))
2D Convolutional Layer 4: (depth = 48, kernel = 2 x 2, border mode = valid, activation = ReLu, output shape = (None, 4, 12, 48))
2D Convolutional Layer 5: (depth = 48, kernel = 2 x 2, border mode = valid, activation = ReLu, output shape = (None, 3, 11, 48))
Flatten Layer
Dense Layer 1: (512 neurons)
Dropout Layer: (keep prob = .5)
Relu Activation
Dense Layer 2: (10 neurons)
Relu Activation
Output Layer
The model was compiled with an adam optimizer (learning rate = .0001), and was set to train for 15 epochs.  When training is complete the model and weights are saved to be used for autonomous driving in the simulator.

Testing model in the sumulator

The script drive.py takes in a constant stream of images, resizes and crops them to the input shape for the model, and passes the transformed image array to the model which predicts an appropriate steering angle based on the image. The steering angle is then passed to the car as a control and the car steers itself accordingly. Hopefully the goal is for the car to replicate or even surpass the driving skills of the a user of the simulator. 

Autonomous Driving
Controlled
The data collection and preprocessing techniques and model architecture outlined above were sufficient to build a model which drives safely around the course for multiple laps without hitting the curbs or drifting off of the road.
