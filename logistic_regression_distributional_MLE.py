import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels


## inspect the training data
# print(trY[1])
# plt.imshow(trX[1].reshape(28,28))
# plt.show()
num_train = trY.shape[0]
num_test = teY.shape[0]

# hyper parameters
batch_size = 100
learning_rate = 0.01
num_epochs = 25

# define placeholder variables

X = tf.placeholder(tf.float32,[None,784])
Y = tf.placeholder(tf.float32,[None,10])

W = tf.get_variable("W",shape=[784,10])
b = tf.get_variable("b",shape=[10])


# define the model
scores = tf.nn.softmax(tf.matmul(X,W) + b)

# define the loss function
loss = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(scores), reduction_indices=1)) 

# define the optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# init the tensorflow session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):

        num_batches = int(num_train/batch_size)
        
        for i in range(num_batches):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _,c = sess.run([optimizer,loss], feed_dict={X:batch_xs,Y:batch_ys})

            # print("loss = {}".format(c/batch_size))


# test model
    correct_prediction = tf.equal(tf.argmax(scores, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("test accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))



