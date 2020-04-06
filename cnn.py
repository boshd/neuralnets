'''
Convolutional Neural Network
@author: Kareem Arab

- refs //
    - ...
'''
import sys, pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os
# from scipy.misc import toimage
import scipy

from PIL import Image

np.set_printoptions(threshold=sys.maxsize)
file_ = open('output_q2','w')

class CONVNetwork(object):

    def __init__(self, data, nf, batch_size, epochs, weight_shapes):
        self.trX, self.trY, self.teX, self.teY = data
        self.nf = nf
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_shapes = weight_shapes

        self.depth = 3
        self.test_size = 256
        self.output_size = 10
        self.image_size = len(self.trX[0])

        self.X = tf.compat.v1.placeholder(tf.float32, [None, self.image_size, self.image_size, self.depth], name='image')
        self.Y = tf.compat.v1.placeholder(tf.float32, [None, self.output_size], name='label')

        self.feature_map_image = tf.placeholder("float", [None, 32, 32, 1])

        self.p_keep_conv = tf.compat.v1.placeholder(tf.float32)
        self.p_keep_hidden = tf.compat.v1.placeholder(tf.float32)

        self.model_accuracy = []
        self.weights = {}

        for i in range(0, len(weight_shapes)):
            self.weights['w_'+str(i+1)] = self.init_weights(self.weight_shapes[i])

        p = self.weight_shapes[len(weight_shapes) - 1][3]
        self.weights['w_fc'] = self.init_weights([nf * nf * p, 625])
        self.weights['w_o'] = self.init_weights([625, self.output_size])

        self.y_pred, self.l1a = self.computational_graph()

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred, labels=self.Y))
        self.train_op = tf.compat.v1.train.RMSPropOptimizer(0.001, 0.9).minimize(self.cost)
        self.predict_op = tf.argmax(self.y_pred, 1)

    def init_weights(self, shape):
        return tf.Variable(tf.random.normal(shape, stddev=0.01))

    def computational_graph(self):
        '''
        This graph is built based on how many convolutional layers are needed. 1-3 inclusive.

        The model follows this progression:
        {[conv(with relu) -> max_pool]x1-3 -> dense layer -> [output(train), softmax(main predictions)]}
        '''
        if len(self.weight_shapes) == 1:
            print('1 conv layer')
            conv_layer_1 = tf.nn.relu(tf.nn.conv2d(self.X, self.weights['w_1'], strides = [1, 1, 1, 1], padding = 'SAME'))
            pool_layer_1 = tf.nn.max_pool2d(conv_layer_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_layer_1 = tf.nn.dropout(pool_layer_1, self.p_keep_conv)

            shape = pool_layer_1.get_shape().as_list()
            dense_layer = tf.reshape(pool_layer_1, [-1, shape[1] * shape[2] * shape[3]])
            dense1 = tf.nn.relu(tf.matmul(dense_layer, self.weights['w_fc']))
            dense1 = tf.nn.dropout(dense1, self.p_keep_hidden)

            return tf.matmul(dense1, self.weights['w_o']), conv_layer_1

        elif len(self.weight_shapes) == 2:
            print('2 conv layer')
            conv_layer_1 = tf.nn.relu(tf.nn.conv2d(self.X, self.weights['w_1'], strides = [1, 1, 1, 1], padding = 'SAME'))
            pool_layer_1 = tf.nn.max_pool2d(conv_layer_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_layer_1 = tf.nn.dropout(pool_layer_1, self.p_keep_conv)

            conv_layer_2 = tf.nn.relu(tf.nn.conv2d(pool_layer_1, self.weights['w_2'], strides = [1, 1, 1, 1], padding = 'SAME'))
            pool_layer_2 = tf.nn.max_pool2d(conv_layer_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_layer_2 = tf.nn.dropout(pool_layer_2, self.p_keep_conv)

            shape = pool_layer_2.get_shape().as_list()
            dense_layer = tf.reshape(pool_layer_2, [-1, shape[1] * shape[2] * shape[3]])

            dense1 = tf.nn.relu(tf.matmul(dense_layer, self.weights['w_fc']))
            dense1 = tf.nn.dropout(dense1, self.p_keep_hidden)

            return tf.matmul(dense1, self.weights['w_o'])

        else:
            print('3 conv layer')
            conv_layer_1 = tf.nn.relu(tf.nn.conv2d(self.X, self.weights['w_1'], strides = [1, 1, 1, 1], padding = 'SAME'))
            pool_layer_1 = tf.nn.max_pool2d(conv_layer_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_layer_1 = tf.nn.dropout(pool_layer_1, self.p_keep_conv)

            conv_layer_2 = tf.nn.relu(tf.nn.conv2d(pool_layer_1, self.weights['w_2'], strides = [1, 1, 1, 1], padding = 'SAME'))
            pool_layer_2 = tf.nn.max_pool2d(conv_layer_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_layer_2 = tf.nn.dropout(pool_layer_2, self.p_keep_conv)

            conv_layer_3 = tf.nn.relu(tf.nn.conv2d(pool_layer_2, self.weights['w_3'], strides = [1, 1, 1, 1], padding = 'SAME'))
            pool_layer_3 = tf.nn.max_pool2d(conv_layer_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_layer_3 = tf.nn.dropout(pool_layer_3, self.p_keep_conv)

            shape = pool_layer_3.get_shape().as_list()
            dense_layer = tf.reshape(pool_layer_3, [-1, shape[1] * shape[2] * shape[3]])

            dense1 = tf.nn.relu(tf.matmul(dense_layer, self.weights['w_fc']))
            dense1 = tf.nn.dropout(dense1, self.p_keep_hidden)

            return tf.matmul(dense1, self.weights['w_o'])

    def run(self):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            for i in range(self.epochs):
                training_batch = zip(range(0, len(self.trX), self.batch_size), range(self.batch_size, len(self.trX)+1, self.batch_size))
                for start, end in training_batch:
                    sess.run([self.train_op], feed_dict={
                        self.X: self.trX[start:end],
                        self.Y: self.trY[start:end],
                        self.p_keep_conv: 0.8,
                        self.p_keep_hidden: 0.5
                    })

                test_indices = np.arange(len(self.teX))
                np.random.shuffle(test_indices)
                test_indices = test_indices[0:self.test_size]

                epoch_accuracy = np.mean(np.argmax(self.teY[test_indices], axis=1) == sess.run(self.predict_op, feed_dict = {
                    self.X: self.teX[test_indices],
                    self.p_keep_conv: 1.0,
                    self.p_keep_hidden: 1.0
                }))

                self.model_accuracy.append(epoch_accuracy*100)

                print('epoch //', i+1, '//', epoch_accuracy*100)

def prepData():
    trX = []
    trY = []
    for batch_i in range(1, 6):
        with open('cifar-10-batches-py/data_batch_' + str(batch_i), 'rb') as file:
            batch = pickle.load(file, encoding='latin1')
            trX += [i for i in batch['data']]
            trY += [j for j in batch['labels']]
    with open('cifar-10-batches-py/test_batch', 'rb') as file:
        batch = pickle.load(file, encoding='latin1')
        teX = [i for i in batch['data']]
        teY = [j for j in batch['labels']]

    trX = np.array(trX).astype('float32')
    teX = np.array(teX).astype('float32')
    trX /= 255
    teX /= 255

    trX = np.array(trX).reshape(-1, 32, 32, 3)
    teX = np.array(teX).reshape(-1, 32, 32, 3)

    trY = tf.keras.utils.to_categorical(trY, 10)
    teY = tf.keras.utils.to_categorical(teY, 10)

    return trX, trY, teX, teY

shapes = [
    [
        [5, 5, 3, 6],
        # [5, 5, 6, 6],
        # [5, 5, 6, 6]
    ],
    # [
    #     [5, 5, 3, 16],
    #     # [5, 5, 16, 16],
    #     # [5, 5, 16, 16]
    # ],
    # [
    #     [5, 5, 3, 32],
    #     # [5, 5, 32, 32],
    #     # [5, 5, 32, 32]
    # ]
]

nfs = [16]

fm_accuracies = []

for fm, nf in zip(shapes, nfs):
    print(fm)
    convnet = CONVNetwork(
        data=prepData(),
        nf=nf,
        batch_size=128,
        epochs=2,
        weight_shapes=fm
    )
    convnet.run()

    # plt.plot(convnet.model_accuracy)
    # plt.show()

    fm_accuracies.append(convnet.model_accuracy)

    print(convnet.model_accuracy)

# xaxis = [i for i in range(1, 16)]
# plt.plot(xaxis, fm_accuracies[0], 'r', label='6 (5x5) f-maps')
# plt.plot(xaxis, fm_accuracies[1], 'b', label='16 (5x5) f-maps')
# plt.plot(xaxis, fm_accuracies[2], 'g', label='32 (5x5) f-maps')
# plt.legend(loc='best')
# plt.title('ConvNet Accuracy w/ 1 Convolutional Layers')
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.show()

# file_.close()