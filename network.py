import tensorflow as tf
import numpy as np
import random

# Loading the dataset
from sklearn.datasets import load_iris

# Converting the output to vector form
def vector(data):
    vectors = []
    for element in data:
        x =np.array([0,0,0])
        x[element] += 1
        vectors.append(x) 
    return vectors 

# Creating the neural network
def network(x):
    y = tf.layers.dense(x, 7, tf.nn.relu)
    y = tf.layers.dense(y, 3)
    return y
    
# Shuffling the dataset (set up)
dataset = load_iris()
dataset = list(zip(dataset.data, dataset.target))
random.shuffle(dataset)
x,y = zip(*dataset)

# Allocating the train and test data
train_x = x[:100]
test_x = x[100:]
train_y = vector(y[:100])
test_y = vector(y[100:])

# Setting up the graph
features = tf.placeholder(tf.float32, shape=(None, 4))
labels = tf.placeholder(tf.float32, shape=(None, 3))

prediction = tf.nn.softmax(network(features))

# Loss and optimizer functions
loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(prediction), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(loss)

# Predictions and accuracy calculations
values = tf.equal(tf.argmax(prediction, 1),tf.argmax(labels ,1))
accuracy = tf.reduce_mean(tf.cast(values, tf.float32))

#Training the data
session = tf.Session()
session.run(tf.global_variables_initializer())
session.run(tf.local_variables_initializer())

iterations = 5000

for i in range(iterations):

    #Diversifying the data each iteration
    run_data_x, run_data_y = [], []
    for i in range(50):
        i = random.randint(0, 99)
        run_data_x.append(train_x[i])
        run_data_y.append(train_y[i])

    _, l, a = session.run([optimizer, loss, accuracy], feed_dict={features: run_data_x, labels: run_data_y})
    print("Loss: " + str(l), " Accuracy: " + str(a))

# Testing the accuracy of the trained neural network
a = session.run(accuracy, feed_dict={features: test_x, labels: test_y})
print("Test accuracy: " + str(a))