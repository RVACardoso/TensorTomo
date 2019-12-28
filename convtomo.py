# Import libraries
import numpy as np
# np.random.seed(0)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# %matplotlib inline
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu
import sys

#######################################
# CNN

learning_rate = 0.0001
batch_size = 256

training_iters = 10000
batch_count = 30

n_input = 32


def conv2d_transpose(x, W, b, out_shape, strides=2):
    # Conv2D_transpose wrapper, with bias and relu activation
    x = tf.nn.conv2d_transpose(x, W, output_shape=out_shape, #[2*x.get_shape().as_list()[0], 2*x.get_shape().as_list()[1], x.get_shape().as_list()[2]],
                               strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


box_shape = [-1, 15, 15, 15]
n = box_shape[1]*box_shape[2]*box_shape[3]
d = 15
k = 5  # filter size
l = 1
n_output = box_shape[1]*(2**2)

# both placeholders are of type float
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output, n_output, 1])

# init_method = tf.contrib.layers.xavier_initializer()
# init_method = tf.contrib.layers.variance_scaling_initializer()
# init_method = tf.keras.initializers.glorot_normal()
init_method = tf.glorot_uniform_initializer()  # good!!!
# init_method = tf.initializers.random_normal()
# init_method = tf.initializers.random_uniform()

weights = {
    'wd1': tf.get_variable('W0', shape=(n_input, n), initializer=init_method),
    'wd2': tf.get_variable('W1', shape=(n, n), initializer=init_method),
    # 'wtc1': tf.get_variable('W2', shape=(k, k, d, d), initializer=tf.contrib.layers.xavier_initializer()),
    'wtc2': tf.get_variable('W3', shape=(k, k, d, d), initializer=init_method),
    'wtc3': tf.get_variable('W4', shape=(k, k, d, d), initializer=init_method),
    'wc1': tf.get_variable('W5', shape=(l, l, d, 1), initializer=init_method),
}

init_method = tf.keras.initializers.Zeros  # good!!!

biases = {
    'bd1': tf.get_variable('B0', shape=(n), initializer=init_method),
    'bd2': tf.get_variable('B1', shape=(n), initializer=init_method),
    # 'btc1': tf.get_variable('B2', shape=(d), initializer=tf.contrib.layers.xavier_initializer()),
    'btc2': tf.get_variable('B3', shape=(d), initializer=init_method),
    'btc3': tf.get_variable('B4', shape=(d), initializer=init_method),
    'bc1': tf.get_variable('B5', shape=(1), initializer=init_method),
}


def conv_net(x, weights, biases):

    fc1 = tf.add(tf.matmul(x, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    box = tf.reshape(fc2, shape=box_shape)
    # tconv1 = conv2d_transpose(box, weights['wtc1'], biases['btc1'], out_shape=[batch_size, 12, 12, 15])
    tconv2 = conv2d_transpose(box, weights['wtc2'], biases['btc2'], out_shape=[batch_size, 30, 30, 15])
    tconv3 = conv2d_transpose(tconv2, weights['wtc3'], biases['btc3'], out_shape=[batch_size, 60, 60, 15])
    conv1 = conv2d(tconv3, weights['wc1'], biases['bc1'])
    return conv1


pred = conv_net(x, weights, biases)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# diff_loss = tf.losses.absolute_difference(predictions=pred, labels=y)
cost = tf.losses.absolute_difference(predictions=pred, labels=y)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999).minimize(cost)

ssim = tf.reduce_mean(tf.image.ssim(y, pred, max_val=1.0))

####################################
# Tomogram Generation

batch_size_tf = tf.constant([batch_size])

mu_x = tf.placeholder("float", [None, 1])
mu_y = tf.placeholder("float", [None, 1])
var_x = tf.placeholder("float", [None, 1])
var_y = tf.placeholder("float", [None, 1])

x_array = tf.linspace(start=-0.09, stop=0.09, num=n_output)   # change to 50!!!!
y_array = tf.linspace(start=-0.09, stop=0.09, num=n_output)
x_mesh, y_mesh = tf.meshgrid(x_array, y_array)
x_mesh = tf.reshape(x_mesh, shape=[-1])
y_mesh = tf.reshape(y_mesh, shape=[-1])
x_mat = tf.reshape(tf.tile(x_mesh, batch_size_tf), shape=[ batch_size_tf[0], tf.shape(x_mesh)[0] ])
y_mat = tf.reshape(tf.tile(y_mesh, batch_size_tf), shape=[ batch_size_tf[0], tf.shape(y_mesh)[0] ])

x_numerator = tf.math.square(tf.math.subtract(x=x_mat, y=mu_x))
x_denominator = tf.math.scalar_mul(scalar=2.0, x=tf.math.square(var_x))
x_power = tf.math.truediv(x=x_numerator, y=x_denominator)
x_expon = tf.math.exp(tf.math.negative(x_power))

y_numerator = tf.math.square(tf.math.subtract(x=y_mat, y=mu_y))
y_denominator = tf.math.scalar_mul(scalar=2.0, x=tf.math.square(var_y))
y_power = tf.math.truediv(x=y_numerator, y=y_denominator)
y_expon = tf.math.exp(tf.math.negative(y_power))

pre_img = tf.math.multiply(x=x_expon, y=y_expon)
flat_tomo = tf.math.scalar_mul(scalar=1.0, x=pre_img)
tomo = tf.reshape(flat_tomo, shape=[-1, n_output, n_output, 1])


###################################
# Detectors computation

geo_matrix_np = np.load('geo_test60-px1000.npy')
geo_matrix = tf.cast(tf.transpose(geo_matrix_np), dtype=tf.float32)
# geo_matrix = tf.random.uniform(shape=[n_output*n_output, 32]) # it's trasposed compare to usual geometric matrix!!!<
detectors = tf.linalg.matmul(flat_tomo, geo_matrix)
detectors = tf.math.scalar_mul(scalar=1e9, x=detectors)


# Initializing the variables
init = tf.global_variables_initializer()
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

fig = plt.gcf()
fig.show()
fig.canvas.draw()

with tf.Session() as sess:
    sess.run(init)
    # saver.restore(sess, "./models_iter2/model.ckpt")

    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    for i in range(training_iters):

        for batch in range(batch_count):

            # var = sess.run(pred, feed_dict={mu_x: np.random.uniform(size=(batch_size, 1), low=-0.05, high=0.05),
            #                                                            mu_y: np.random.uniform(size=(batch_size, 1), low=-0.05, high=0.05),
            #                                                            var_x: np.random.uniform(size=(batch_size, 1), high=0.02),
            #                                                            var_y: np.random.uniform(size=(batch_size, 1), high=0.02)})
            # print(var.shape)

            # if i < 5:
            #     mu_lim = 0.01 # 0.05
            #     var_low_lim = 0.015  # 0.005
            #     var_up_lim = 0.02
            # else:
            #     mu_lim = 0.05  # 0.05
            #     var_low_lim = 0.005  # 0.005
            #     var_up_lim = 0.02

            mu_lim = 0.05  # 0.01
            var_low_lim = 0.01  # 0.015
            var_up_lim = 0.03

            batch_x, batch_y = sess.run([detectors, tomo], feed_dict={mu_x: np.random.uniform(size=(batch_size, 1), low=-mu_lim, high=mu_lim),
                                                                       mu_y: np.random.uniform(size=(batch_size, 1), low=-mu_lim, high=mu_lim),
                                                                       var_x: np.random.uniform(size=(batch_size, 1), low=var_low_lim, high=var_up_lim),
                                                                       var_y: np.random.uniform(size=(batch_size, 1), low=var_low_lim, high=var_up_lim)})


            # # data check
            # plt.figure()
            # plt.subplot(223)
            # plt.imshow(np.reshape(batch_y[0], [60, 60]))
            # # print(np.sort(batch_y[0].flatten()))
            # cam_nr = np.arange(16)
            # plt.subplot(221)
            # plt.bar(cam_nr, batch_x[0][:16])
            # plt.xticks(cam_nr, cam_nr)
            # plt.title("TOP")
            # plt.subplot(224)
            # plt.bar(cam_nr, batch_x[0][16:32])
            # plt.xticks(cam_nr, cam_nr)
            # plt.title("OUTER")
            # plt.show()


            # var = sess.run(ssim, feed_dict={x: batch_x, y: batch_y})
            # print(var.shape)


            # print(batch_x.shape)
            # print(pred_np.shape)


        #     # Run optimization op (backprop) and Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            loss, acc = sess.run([cost, ssim], feed_dict={x: batch_x, y: batch_y})

        # Save the variables to disk.
        # save_path = saver.save(sess, "./models/model.ckpt")
        # print("Model saved in path: %s" % save_path)
        print("\n Iter " + str(i) + "\n Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

        train_loss.append(loss)
        train_accuracy.append(acc)

        if i % 5 == 0:

            # show training data            # # data check
            # plt.figure()
            plt.subplot(141)
            plt.imshow(np.reshape(batch_y[0], [60, 60]), aspect='auto')
            plt.title("Training")
            # print(np.sort(batch_y[0].flatten()))
            # cam_nr = np.arange(16)
            # plt.subplot(231)
            # plt.bar(cam_nr, batch_x[0][:16])
            # plt.xticks(cam_nr, cam_nr)
            # plt.title("TOP")
            # plt.subplot(235)
            # plt.bar(cam_nr, batch_x[0][16:32])
            # plt.xticks(cam_nr, cam_nr)
            # plt.title("OUTER")

            # show predicted (first of batch)
            plt.subplot(142)
            pred_np = sess.run(pred, feed_dict={x: batch_x, y: batch_y})
            plt.imshow(np.reshape(pred_np[0], [60, 60]), aspect='auto')
            plt.title("Predicted")

            # show loss and accuracy

            plt.subplot(143)
            plt.plot(range(len(train_loss)), train_loss, 'b')
            # plt.plot(range(len(train_loss)), test_loss, 'r', label='Test loss')
            plt.title('Training loss')
            plt.xlabel('Epochs ', fontsize=16)
            plt.ylabel('Loss', fontsize=16)

            plt.subplot(144)
            plt.plot(range(len(train_accuracy)), train_accuracy, 'r')
            plt.title('Training accuracy')
            plt.xlabel('Epochs ', fontsize=16)
            plt.ylabel('accuracy', fontsize=16)
            plt.pause(1e-7)

            fig.canvas.draw()

        # Calculate accuracy for all 10000 mnist test images

    #     print("Testing Accuracy:","{:.5f}".format(test_acc))
    # summary_writer.close()




    plt.show()