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
sess = tf.InteractiveSession()

raw_data = [1., 2., 8., -1., 0., 5.5, 6., 13]
spike = tf.Variable(False)
spike.initializer.run()

for i in range(1, len(raw_data)):
    if raw_data[i] - raw_data[i - 1] > 5:
        updater = tf.assign(spike, True)
        updater.eval()
    else:
        updater = tf.assign(spike, False)
        updater.eval()
    print("Spike", spike.eval())

sess.close()

# Saving and Loading Variables (p.40)
sess = tf.InteractiveSession()

raw_data = [1., 2., 8., -1., 0., 5.5, 6., 13]
spikes = tf.Variable([False] * len(raw_data), name='spikes')
spikes.initializer.run()

saver = tf.train.Saver()

for i in range(1, len(raw_data)):
    if raw_data[i] - raw_data[i-1] > 5:
        spikes_val = spikes.eval()
        spikes_val[i] = True
        updater = tf.assign(spikes, spikes_val)
        updater.eval()

print(spikes)

sess.close()


# Visualizing data using TensorBoard (p.41)
raw_data = np.random.normal(10, 1, 100)

alpha = tf.constant(0.05)
curr_value = tf.placeholder(tf.float32)
prev_avg = tf.Variable(0.)
update_avg = alpha * curr_value + (1 - alpha) * prev_avg

avg_hist = tf.summary.scalar("running_average", update_avg)
value_hist = tf.summary.scalar("incoming_values", curr_value)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs")
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(len(raw_data)):
        summary_str, curr_avg = sess.run([merged, update_avg], feed_dict={curr_value: raw_data[i]})
        sess.run(tf.assign(prev_avg, curr_avg))
        print(raw_data[i], curr_avg)
        writer.add_summary(summary_str, i)
















