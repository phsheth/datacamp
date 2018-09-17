## Capstone Exercise for Introduction to TensorFlow API in Python
### Description:
### Bringing it all together:
In this exercise, we will create a Multi Layer neural network. The model will contain an input layer, two hidden layers and an output layer. Furhter set the epochs. Then setup the weights and biases for the first layer. After that add the activation function (Sigmoid and RELU are usual functions.). Then setup the weights and biases for the second layer. The output layer will use softmax activation, this is used where binary decisions are to be taken as the predicted label. Setup the cost function, optimizer and the accuracy calculation function. Finally, write the function for starting the session and training the network.

### Instructions:

In this exercise, we collate all the functions together and build a multi layer neural network. For this, the following steps are performed

1. Define the input layer, the hidden layers and the output layer.
2. Define the weights and biases for the first and second layer, the activation function after the first layer and the activation function for the output layer.
3. Setup the cost function, optimizer and accuracy calculation.
4. Write the for loop for starting the session and training the network, also printing the epoch, accuracy and predicted v/s actual variables.

### Exercise Code:


```python
import tensorflow as tf

# STEP #1:
tr_epochs = 500
n_neurons_in_h1 = 60
n_neurons_in_h2 = 60
learning_rate = 0.01

# STEP #2: 
X = tf.placeholder(tf.float32, [None, n_features], name='features')
Y = tf.placeholder(tf.float32, [None, n_classes], name='labels')

W1 = tf.Variable(tf.truncated_normal([n_features, n_neurons_in_h1], mean=0, stddev=1 / np.sqrt(n_features)), name='weights1')
b1 = tf.Variable(tf.truncated_normal([n_neurons_in_h1],mean=0, stddev=1 / np.sqrt(n_features)), name='biases1')

y1 = tf.nn.tanh((tf.matmul(X, W1)+b1), name='activationLayer1')

#network parameters(weights and biases) are set and initialized(Layer2)
W2 = tf.Variable(tf.random_normal([n_neurons_in_h1, n_neurons_in_h2],mean=0,stddev=1/np.sqrt(n_features)),name='weights2')
b2 = tf.Variable(tf.random_normal([n_neurons_in_h2],mean=0,stddev=1/np.sqrt(n_features)),name='biases2')
#activation function(sigmoid)
y2 = tf.nn.sigmoid((tf.matmul(y1,W2)+b2),name='activationLayer2')

#output layer weights and biasies
Wo = tf.Variable(tf.random_normal([n_neurons_in_h2, n_classes], mean=0, stddev=1/np.sqrt(n_features)), name='weightsOut')
bo = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=1/np.sqrt(n_features)), name='biasesOut')
#activation function(softmax)
a = tf.nn.softmax((tf.matmul(y2, Wo) + bo), name='activationOutputLayer')

# STEP #3: 
# invoke cost function
cross_entropy = cost_function(args)
# optimizer
train_step = train_function(args)

#compare predicted value from network with the expected value/target
correct_prediction =  calc_corr_prediction(args)
#accuracy determination
accuracy = calc_accuracy(args)

# initialization of all variables
initial = tf.global_variables_initializer()

# STEP #4:
train_model_function(args)
view_confusion_matrix(args)
```

