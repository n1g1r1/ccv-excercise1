import tensorflow as tf

# Define the graph
## Operation 1
x = tf.constant('Hello world!')

## Constants for operation 2
y1 = tf.constant(2, dtype=tf.int32)
y2 = tf.constant(1, dtype=tf.int32)

## Operation 2
z = y1 + y2

# Make new Session
sess = tf.Session()

# Run first operation
x_eval = sess.run(x)
print(x_eval)

# Run second operation
z_eval = sess.run(z)
print(z_eval)

# Output the two tensor constants as a rank 2 tensor
y1_eval , y2_eval = sess.run([y1,y2])
print(y1_eval , y2_eval)
