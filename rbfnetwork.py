'''
Feed Forward RBF Neural Network w/ K-means Clustering
@author: Kareem Arab
'''
import sys, os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from mnist import MNIST

from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.model_selection import KFold

np.random.seed(42)
np.set_printoptions(threshold=sys.maxsize)
file_ = open('output_q2','w')

class RBFNetwork(object):

    def __init__(self, data, labels, centroids, centroid_labels):
        print('\nInitialzing RBF Network')
        self.data = data
        self.labels = labels
        self.centroids = centroids
        self.centroid_labels = centroid_labels

        self.size_input_layer = 784
        self.size_output_layer = 10

        self.beta = self.getBetaCoefficients()

        tf.compat.v1.reset_default_graph()

        self.X = tf.compat.v1.placeholder(tf.float32, shape=[None, self.size_input_layer], name='X')
        self.Y = tf.compat.v1.placeholder(tf.float32, shape=[None, self.size_output_layer], name='Y')

        self.y_pred = self.computational_graph(self.X)

        self.pred_op = tf.argmax(self.y_pred, 1)
        self.true_pr = tf.argmax(self.Y, 1)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred, labels=self.Y))
        self.reduced_comparator = tf.reduce_mean(tf.cast(tf.equal(self.pred_op, self.true_pr), tf.float32))

        self.train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)

        self.saver = tf.compat.v1.train.Saver()

        self.fold_acc = []

    def run(self):
        print('Training network with {} neurons (centroids)'.format(len(self.centroids)))

        BATCH_SIZE = 50
        kf = KFold(n_splits=5)

        fold_index = 0
        for train_index, test_index in kf.split(self.data):
            print('TRAIN:', len(train_index), 'TEST:', len(test_index))
            trX, teX = self.data[train_index], self.data[test_index]
            trY, teY = self.labels[train_index], self.labels[test_index]

            run_acc = []
            with tf.compat.v1.Session() as sess:
                tf.compat.v1.global_variables_initializer().run()
                epoch_accuracies = []
                for i in range(1):
                    print('epoch //', i)
                    for start, end in zip(range(0, len(trX), BATCH_SIZE), range(BATCH_SIZE, len(trY) + 1, BATCH_SIZE)):
                        sess.run(self.train_op, feed_dict={self.X: trX[start:end], self.Y: trY[start:end]})

                    s_accuracies = []
                    for start, end in zip(range(0, len(teX), BATCH_SIZE), range(BATCH_SIZE, len(teX) + 1, BATCH_SIZE)):
                        accuracy = sess.run(self.reduced_comparator, feed_dict={self.X: teX[start:end], self.Y: teY[start:end]})
                        s_accuracies.append(accuracy)

                    epoch_accuracies.append(np.sum(s_accuracies) / len(s_accuracies))

                run_acc.append(np.sum(epoch_accuracies) / len(epoch_accuracies))

                print('fold acc //', (np.sum(run_acc) / len(run_acc)))

            self.fold_acc.append(np.sum(run_acc) / len(run_acc) * 100)
            fold_index += 1

    def getFoldAcc(self):
        return self.fold_acc

    def init_weights(self, shape):
        return tf.Variable(tf.random.normal(shape, stddev=1), dtype=tf.float32)

    def computational_graph(self, X):
        '''
        RBF Gaussian Activation Function

        φ(x) = e^(-β * ||x - µ||^2)

        µ is the prototype vector stored at the centre of the curve
        x is the point at the boundary of the convex structure (gaussian, sombrero, etc.)
        ||x - µ||^2 is the squared euclidean distance

        β coefficient
            - β = 1/2σ^2

        σ is the average distance between all points in the cluster and the cluster center
            - σ = 1/m * Σ [i=1 → m] ||x_i - µ||

        m is the number of points in the cluster being represented by the hidden neuron

        '''
        w_o = self.init_weights([len(self.centroids), 10])

        mu = tf.Variable(self.centroids)
        beta = tf.Variable(self.beta)

        x = tf.to_float(tf.tile(tf.expand_dims(X, 1), [1, len(centroids), 1]))
        mean = tf.to_float(tf.reshape(tf.tile(mu, [tf.shape(X)[0], 1]), [tf.shape(X)[0], len(centroids), self.size_input_layer]))
        euclidean_distance = tf.square(tf.norm(tf.subtract(x, mean), axis=-1))
        phi = tf.exp(tf.multiply(tf.to_float(tf.negative(beta)), euclidean_distance))

        return tf.matmul(phi, w_o)

    def getBetaCoefficients(self):
        '''
        cluster_center: the prototype vector, centroid (784 dimensional point in space)
        cluster_data: list of vectors; all the points within the cluster specified by the centroid
        '''
        betas = np.zeros([len(self.centroids)])
        for i in range(len(self.centroids)):
            cluster_points = self.data[np.where(self.centroid_labels == i)]

            sigma = np.zeros([784])
            for j in range(len(cluster_points)):
                sigma += np.linalg.norm(cluster_points[j] - self.centroids[i])

            sigma = np.multiply((1/len(cluster_points)), sigma)

            beta = np.divide(1, np.multiply(2, np.square(sigma)))

            betas[i] = beta[0]

        return betas

class Kmeans(object):

    def __init__(self, data, k):
        self.k = k
        self.data = data

    def computeCentroids(self):
        print('\nComputing centroids')
        self.kmeans = KMeans(init='random', n_clusters=self.k, n_init=1).fit(self.data)
        self.centroids = self.kmeans.cluster_centers_
        self.centroid_labels = self.kmeans.labels_

        return self.centroids, self.centroid_labels

    def visualize(self):
        # do not attempt, data not reduced... use PCa first
        plt_data = plt.scatter(self.data[:, 0], self.data[:, 1], c=self.kmeans.labels_, cmap=plt.cm.get_cmap('Spectral', 10))
        plt.colorbar()
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], marker='x')
        labels = ['{0}'.format(i) for i in range(10)]
        for i in range (10):
            xy=(self.centroids[i, 0], self.centroids[i, 1])
            plt.annotate(labels[i],xy, horizontalalignment='right', verticalalignment='top')
        plt.show()

def prepData():
    print('Preparing data')

    (trX, trY), (teX, teY) = tf.keras.datasets.mnist.load_data()

    trX = trX.reshape(60000, 784)
    teX = teX.reshape(10000, 784)
    trX = trX.astype('float32')
    teX = teX.astype('float32')

    trX /= 255
    teX /= 255

    trY = tf.keras.utils.to_categorical(trY, 10)
    teY = tf.keras.utils.to_categorical(teY, 10)

    data = np.append(trX, teX, axis=0)
    labels = np.append(trY, teY, axis=0)

    return data, labels

# prep data
data, labels = prepData()

centroid_accuracies = []

for centroid_size in [50, 60, 70, 80, 90, 100]:
    kmeans = Kmeans(data, centroid_size)
    centroids, centroid_labels = kmeans.computeCentroids()

    network = RBFNetwork(data, labels, centroids, centroid_labels)
    network.run()
    f = network.getFoldAcc()

    centroid_accuracies.append(np.sum(f) / len(f))

plt.plot([50, 60, 70, 80, 90, 100], centroid_accuracies)
plt.xlabel('Number of centroids')
plt.ylabel('Accuracy %')
plt.title('Centroid Accuracy')
plt.show()

file_.close()