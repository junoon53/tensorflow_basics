import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_minibatch():
    ''' This is called every time the optimizer needs a minibatch. Since the points
    are generated randomly, it's like having an infinite amount of data'''
    x_val = np.random.uniform(-10,10,size=100)
    y_val = 10 * np.square(x_val) - 5 

    return x_val,y_val


## build computational graph

# define placeholder variables
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w = tf.get_variable("w", shape=[3,1])

# compute the estimate
f = tf.stack([tf.square(x), x, tf.ones_like(x)], 1)
yhat = tf.squeeze(tf.matmul(f, w), 1)

# compute the loss
loss = tf.nn.l2_loss(yhat - y) + 0.1 * tf.nn.l2_loss(w)

# define the optimizer
# train_op = tf.train.AdamOptimizer(0.1).minimize(loss)
# train_op = tf.train.AdadeltaOptimizer(0.01).minimize(loss)
train_op = tf.train.AdagradOptimizer(learning_rate=1.0).minimize(loss)
# train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

# define the tensorflow session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

## define stopping criteria

# a) directly specify the minimum loss value 
min_loss = 1
loss_val = 1e10
weights = np.array([3,1])

# b) directly specify the minimum loss update 
min_update = 0.1
update = 1e5

# c) minimum update ratio
min_loss_update_ratio = 1e-5
min_param_update_ratio = 1e-5

# d) fractional loss criteria

def fn_loss_update_ratio(val,old_val):

    return np.abs(val - loss_val)/old_val

# e) fractional weight update criteria

def fn_iterate_update_ratio(val,old_val):

    return np.linalg.norm(np.array(val) - np.array(old_val))/np.linalg.norm(old_val)

# collect some statistics
num_iter = 0
loss_vals = []

# while loss_val > min_loss and np.abs(update) > min_update:
while True:
    num_iter+=1
    x_val, y_val = get_minibatch()
    # print(sess.run([tf.shape(f)],{x:x_val}))
    # print(sess.run([tf.shape(yhat)],{x:x_val,y:y_val}))
    old_loss_val = loss_val
    _, loss_val = sess.run([train_op, loss], {x: x_val, y: y_val})
    old_weights = weights
    weights = sess.run([w])
    loss_vals.append(loss_val)
    # update_ratio = fn_loss_update_ratio(loss_val,old_loss_val)
    iterate_update_ratio = fn_iterate_update_ratio(weights,old_weights)
    if iterate_update_ratio < min_param_update_ratio:
        print("converged in {} iterations".format(num_iter))
        print("weights: ",sess.run([w]))
        break
    # update = old_loss - loss_val
    print("loss: ",loss_val,iterate_update_ratio)


plt.plot(np.arange(num_iter),loss_vals)
plt.show()
plt.savefig('fig_train_loss.png')

