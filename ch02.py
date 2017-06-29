# ch02
# Ensuring TensorFlow works (p.25)
import tensorflow as tf
import numpy as np

m1 = [[1.0, 2.0],
      [3.0, 4.0]]

m2 = np.array(m1, dtype=np.float32)

m3 = tf.constant(m1)

print(type(m1))
print(type(m2))
print(type(m3))

t1 = tf.convert_to_tensor(m1, dtype=tf.float32)
t2 = tf.convert_to_tensor(m2, dtype=tf.float32)
t3 = tf.convert_to_tensor(m3, dtype=tf.float32)

print(type(t1))
print(type(t2))
print(type(t3))


# Creating operators (p.31)
x = tf.constant([[1.0, 2.0]])  # x is a 1 x 2 array.
neg_op = tf.negative(x)

# Executing operators with sessions (p.32)
with tf.Session() as sess:
    result = sess.run(neg_op)
print(result)

# Alternatively
sess = tf.InteractiveSession()

neg_x = tf.negative(x)
result = neg_x.eval()
print(result)

sess.close()


# Session configurations
x = tf.constant([[1.0, 2.0]])  # x is a 1 x 2 array.
neg_x = tf.negative(x)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    result = sess.run(neg_x)
print(neg_x)


# Using variables (p.38)










