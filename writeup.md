**Traffic Sign Recognition Project**

This is my writeup for the traffic sign classification project. 
My code is available [on github](https://github.com/roneyal/CarND-Traffic-Sign-Classifier-Project)


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image4]: ./sign1.jpg "Traffic Sign 1"
[image5]: ./sign2.jpg "Traffic Sign 2"
[image6]: ./sign3.jpg "Traffic Sign 3"
[image7]: ./sign4.jpg "Traffic Sign 4"
[image8]: ./sign5.jpg "Traffic Sign 5"

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 images
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32 x 32 x 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

I created a histogram to show the frequency of each class and to get a notion of how balanced the data set is.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, roughly normalized each image to improve the model performance. I estimated the normalization by subtracting 128 from each pixel and dividing by 128.
This eventually led to results that were good enough.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is heavily based on the LeNet architecture along with dropout to improve the model generalization.

| Layer         		|     Description				| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image				| 
| Convolution 5x5     		| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU				|						|
| Dropout			| Keep probability 0.9				|
| Max pooling	      		| 2x2 stride,  outputs 14x14x6			|
| Convolution 5x5	    	| 1x1 stride, valid padding, outputs 10x10x16	|
| RELU				|						|
| Dropout			| Keep probability 0.9				|
| Max pooling	      		| 2x2 stride,  outputs 5x5x16			|
| Flattening			| outputs 400					|
| Fully Connected		| 400 input nodes, 120 outputs			|
| RELU				| 						|
| Dropout			| Keep probability 0.9				|
| Fully Connected		| 120 input nodes, 84 outputs			|
| RELU				| 						|
| Dropout			| Keep probability 0.9				|
| Fully connected		| 84 inputs, 43 outputs				|
| Softmax			| 						|
|				|						|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer with learning rate 0.001 and 200 epochs with batch size 128. 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.7%
* validation set accuracy of 94.2%
* test set accuracy of 92.9%

As suggested, I relied on the LeNet architecture. The basic architecture reached around 89% accuracy on the validation set and almost 100% on the training set.
It seemed like overfitting so I though that I could get better results using regularization techniques.
I tried adding dropout, layer after layer and saw that it gradually improves the performance, and by tuning the dropout probability I was able to pass 93%. 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The fourth image seems the most difficult to classify due to the angle at which the sign is shown.
I thought others might be difficult as well since they are not the most common signs.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction for new traffic signs I found online:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General caution      		| General caution   									| 
| Yield     			| Yield 										|
| Priority road			| Priority road											|
| Pedestrians      		| End of all speed and passing limits	|
| Right-of-way at the next intersection		| Right-of-way at the next intersection      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. The angle at which the sign is seen seems important.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)



| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| General caution 			|
| 1.0     			| Yield					|
| 1.0					| Priority road												|
| 0.233	      			| End of all speed and passing limits					 				|
| 1.0				    | Right-of-way at the next intersection      							|


The model was very certain of images 1,2,3,5 and it was correct. For image 4, for which the classification was wrong, the model was very uncertain.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


