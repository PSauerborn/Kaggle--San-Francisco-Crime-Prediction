# Kaggle--San-Francisco-Crime-Prediction
Kaggle competition to classify crimes types in the san fransico area

The following code segments are used to construct a model to predict the category of crime committed based on the given data set, which can 
be found at 

https://www.kaggle.com/c/sf-crime/leaderboard

The competition uses log loss as a metric; the model proposed in these files achieves a log loss of ~2.16, as evaluated on the validation 
sets used.



main.py
----------------------------------------------

file that is used to instantiate and execute the model 

model.py
----------------------------------------------

The model.py file contains the model itself; note that the model is essentially an MLP constructed with the TensorFlow low level API. The 
network istelf consists of a single hidden layer with 128 neurons and the reLu activation function. The model uses the softmax cross entropy
cost function and uses a Adam optimizer to minimize the cost function. Drop out is used as a regularization method. The batch size is set to
250 data elements per batch

processing.py
-----------------------------------------------

contains the functions used to process that data into a usable format. Note that the latitude and longitude are standardized, and all cyclic
data elements (such as time of day and day of the month) are all converted to a datax, datay format using datax = cos(2*pi*x/T),
datay = sin(2*pi*x/T) in order to fully capture the cyclic nature of such variables. 

