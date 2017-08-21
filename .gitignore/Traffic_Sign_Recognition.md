**Traffic Sign Recognition** 

The document records the process for Project 'Traffic Sign Recognition' in Self-Driving Car on Udacity.



**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

\* Load the data set (see below for links to the project data set)

\* Explore, summarize and visualize the data set

\* Design, train and test a model architecture

\* Use the model to make predictions on new images

\* Analyze the softmax probabilities of the new images

\* Summarize the results with a written report



1.Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic signs data set:

\* The size of training set is 34799.

\* The size of the validation set is 4410.

\* The size of test set is 12630.

\* The shape of a traffic sign image is (32,32,3)

\* The number of unique classes/labels in the data set is 43 (Here use pandas)

2.Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![sample image from training set][train_sample.png]

*Design and Test a Model Architecture*

\####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

\*As a first step, I decided to convert the images to grayscale because I adopted LeNet architecture, and the input of this architecture is (32,32,1).

I adpot the function of tensorflow 'tf.image.rgb_to_grayscale()' to convert the color image to gray image.

Because the type of this function's return is tensor, so I turn the tensor to numpy.ndarray for the input of LeNet.

\*Then, I normalized the image data, so that the data has mean zero  and equal variance.

I used (pixel - 128.0)/128.0 to realized this and the values after normalized will be in the scope [-1,1]

\####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         						|Description    			        			 | 

|:------------------------------------------------:|:--------------------------------------------------------------:| 

| Input         						| 32x32x1 Gray image   		 			 | 

| Convolution 1x1     				| 1x1 stride, valid padding, outputs 28x28x6  |

| RELU							|										 |

| Max pooling	      				| 2x2 stride,  outputs 14x14x6 	     			 |

| Convolution 1x1	    			| 1x1 stride, valid padding, outputs 10x10x16|

| RELU                 					|                             					                 |	

| Max pooling					| 2x2 stride,  outputs 5x5x16  				 |

| Flatten							| 1D instead 3D		   	  				 |

| Full Connected        				| outputs is 120						   	 |

| RELU                  					|                                               				 |

| Full Connected        				| outputs is 84                                 			 |

| RELU                  					|                                               				 |

| Dropout						| dropout 0.8

| Full Connected        				| outputs is 43                                			 |

\####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, the cross entroy, mean loss are used and then optimizer (Adam optimizer) .

epoches = 80

batch_size = 128

rate = 0.004

\####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

\* validation set accuracy of 4410 is 93.3 %

\* test set accuracy of 12630 samples is 91.4%

If an iterative approach was chosen:

\* What was the first architecture that was tried and why was it chosen?

\* What were some problems with the initial architecture?

\* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

\* Which parameters were tuned? How were they adjusted and why?

I adjusted the learning rate, and epoch.

For learning rate, I decreased its value while increasing epoch.

\* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:

\* What architecture was chosen?

LeNet was chosen.

\* Why did you believe it would be relevant to the traffic sign application?

It has been applied in handwritten digit recognition.

\* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

The validation accuracy is 93.3%.

\###Test a Model on New Images

\####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

(1) children crossing

(2) general caution

(3) road work

(4) speed limit (70km/h)

(5) trun right ahead



\####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                             			|     Prediction	        				       | 

|:------------------------------------------------:|:-------------------------------------------------------------:| 

| Children crossing                 		| Right-of-way at the next intersection  		| 

| General caution                   		| General caution 						|

| Road work                         			| Road work								|

| Speed limit (70km/h)      			| Speed limit (70km/h)					|

| Turn right ahead                 		 | urn right ahead						|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 



\####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in section 'Output Top 5 Softmax Probabilities For Each Image Found on the Web'.



\### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

\####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?