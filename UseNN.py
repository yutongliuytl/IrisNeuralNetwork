import tensorflow as tf
import numpy as np

# Creating the neural network
def network(x):
    y = tf.layers.dense(x, 7, tf.nn.relu)
    y = tf.layers.dense(y, 3)
    return y

tf.reset_default_graph()

# Setting up the graph
features = tf.placeholder(tf.float32, shape=(None, 4))
labels = tf.placeholder(tf.float32, shape=(None, 3))
prediction = tf.nn.softmax(network(features))

# Reloading the model
session = tf.Session()
saver = tf.train.Saver()

# Restore variables from disk
saver.restore(sess = session, save_path = "/tmp/model.ckpt")
print("Model restored.")

# Predictions
predict_dataset = [
[5.0, 3.3, 1.4, 0.2,], # Setosa: 0
[6.0, 3.0, 4.8, 1.8,], # Virginica: 2
[6.9, 3.1, 4.9, 1.5]   # Versicolor: 1
]

classification = session.run(tf.argmax(prediction, 1), feed_dict={features: predict_dataset})
print(classification)