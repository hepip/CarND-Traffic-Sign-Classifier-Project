## **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_img/train_hist.png "Training Data"
[image2]: ./writeup_img/test_hist.png "Test Data"
[image3]: ./writeup_img/Sample_Data.png "Sample Data"
[image4]: ./writeup_img/ClassesCount.png "Images per class"
[image5]: ./writeup_img/placeholder.png "Traffic Sign 2"
[image6]: ./writeup_img/placeholder.png "Traffic Sign 3"
[image7]: ./writeup_img/placeholder.png "Traffic Sign 4"
[image8]: ./writeup_img/placeholder.png "Traffic Sign 5"

### Rubric Points
#### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup 

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Please find the detailed writeup of my experimentation below! and here is a link to my [project code](https://github.com/hepip/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 27839
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. Below are the two histograms for the training and the test data. As seen in the plots, there is a lot of differences in count for images for each of the 43 classes.

![alt text][image1]
![alt text][image2]

Also, I plotted the data to visually see the different signs and count for each of the classes.

![alt text][image3]
![alt text][image4]


### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the seventh cell of the IPython notebook.

I decided to normalize the data using Min-Max Scaling Technique.

Here is an example of a traffic sign image before and after normalization.

Before:
![alt text][image5]

After:
![alt text][image6]


#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the first code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by using train_test_split function present in sklearn.model_selection. I shuffled the training data as well.

My final training set had X number of images. My validation set and test set had Y and Z number of images.

Augmenting additional data to classes is also possible by doing things like flipping and translating. 


#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the ninth cell of the ipython notebook. 

My final model consisted of the following layers and is based on LeNet Architecture:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 	    	| 1x1 stride, Valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, Valid padding, outputs 16x16x64	|
| Convolution 		    | 1x1 stride, Valid padding, outputs 28x28x6    |
| RELU					| 		    									|
| Max pooling	      	| 2x2 stride, Valid padding, outputs 16x16x64	|
| Fully Connected		| 400 Input and 120 Outputs						|
| RELU					| 		    									|
| Fully Connected		| 120 Input and 84 Outputs						|
| RELU					| 		    									|
| Fully Connected		| 84 Input and 10 Outputs						|
 


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the _______ cell of the ipython notebook. 

I have used Adam Optimizer. The training function uses 50 Epochs for training with a batch size of 128 and learning rate of 0.001.


#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ___________ cell of the Ipython notebook.

My final model results were:

* validation set accuracy of 0.988
* test set accuracy of 0.927

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

I started with the famous [LeNet architecture](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) 

* What were some problems with the initial architecture?

The accuracy for test set was not going beyond 90%

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

To improve the training, i have tried playing with the learning rate and number of epochs. At an earlier stage, i was trying to train it using lower value for EPOCH, which was not good enough
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7] ![alt text][image8] ![alt text][image9] 
![alt text][image10] ![alt text][image11]



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess _ of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 92.7%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


