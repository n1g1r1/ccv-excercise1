import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# SETTINGS #############################################

# Image file
img_filename = 'cameraman.png'
gaussian_sigma = 2


# HELPER FUNCTIONS #####################################

# Get the gaussian kernel
def get_gaussian_kernel(sigma, size=None):
    if not size:
        size = int(round(sigma*3*2+1))|1
    C = cv2.getGaussianKernel(size, sigma)
    C = np.outer(C,C).astype(np.float32)
    C /= np.sum(C)
    return C

# Open image, handle format
with Image.open(img_filename) as img:
    global img_arr
    img_arr = np.array(img).astype(np.float32)/255.0


# GRAPH #################################################

# Get the dimensions of the image
img_shape_x, img_shape_y = img_arr.shape

# Reshape image
img_reshaped = tf.reshape(img_arr, shape=[1, img_shape_x, img_shape_y, 1], name="image")

# Get kernel filter
kernel = get_gaussian_kernel(gaussian_sigma)
# Reshape kernel filter
kernel_reshaped = tf.reshape(kernel, shape=list(kernel.shape + (1,1)), name="kernel")

# Convolution layer
convolution = tf.nn.conv2d(input=img_reshaped, filter=kernel_reshaped,strides=[1,1,1,1], padding="SAME")


# SESSION ###############################################

with tf.Session() as sess:

    # Start session, run operations
    summary_writer = tf.summary.FileWriter(logdir='./ass1_graph', graph=sess.graph)

    # Run convolution
    result = sess.run(convolution)
    # reshape the result to the dimensions of an image
    result_img_arr = np.reshape(result, newshape=[256,256])

    # Display the result side by side
    f, axarray = plt.subplots(1,2)
    axarray[0].imshow(img_arr, cmap='gray')
    axarray[1].imshow(result_img_arr, cmap='gray')
    plt.show()