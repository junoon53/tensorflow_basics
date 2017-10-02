import tensorflow as tf
import numpy as np

N = 20000

x = tf.placeholder(tf.float32,[N,N])
y = tf.placeholder(tf.float32,[N,N])

z = tf.matmul(x, y)

def generate_data():
    x_val = np.random.normal(0.0,1,[N,N])
    y_val = np.random.normal(0.0,1,[N,N])
    return x_val,y_val

sess = tf.Session()
x_val,y_val = generate_data()
z_val = sess.run(z, {x:x_val,y:y_val})

print(z_val)
