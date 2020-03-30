'''
Hopfield Network

@author: Kareem Arab

refs //
    - https://stackoverflow.com/questions/51202181/how-do-i-select-only-a-specific-digit-from-the-mnist-dataset-provided-by-keras
'''
import sys, os
import numpy as np
import tensorflow as tf
from sklearn import datasets
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

class HopfieldNetwork(object):

    def __init__(self, data, rule):
        self.data = data
        self.rule = rule

    def storkey_local_field(self, digit):
        return np.add(np.outer(digit, np.dot(self.W, digit)), np.outer(np.dot(self.W, digit), digit)) / len(self.data[0][0])

    def minimize_energy(self, v, size, u_s=True):
        while u_s:
            w = []
            for i in range(0, size):
                w.append(i)
            np.random.shuffle(w)
            v_ = np.copy(v)
            u_s = False
            for i in range(0, size):
                neuron = w.pop()
                w_at_neuron = self.W[neuron]
                s = np.dot(v_, w_at_neuron)
                v_[neuron] = 1 if s > 0 else -1
                u_s = (not v[neuron] == v_[neuron]) or u_s
            v = v_
        return v

    def compute_learning(self):
        self.W = np.zeros((len(self.data[0][0]), len(self.data[0][0])), int)
        for digit, label in self.data:
            d = np.array(digit).reshape(784, 1)
            self.W = np.add(self.W, (d @ d.T))

        if self.rule == 'hebbian':
            np.fill_diagonal(self.W, 0)
        elif self.rule == 'storkey':
            for digit, label in self.data:
                self.W = self.W - self.storkey_local_field(digit)
            np.fill_diagonal(self.W, 0)

def prepData():
    (trX, trY), (teX, teY) = tf.keras.datasets.mnist.load_data()

    trX = trX.reshape(60000, 784)
    teX = teX.reshape(10000, 784)
    trX = trX.astype('float32')
    teX = teX.astype('float32')

    train_filter = np.where((trY == 1) | (trY == 5))
    test_filter = np.where((teY == 1) | (teY == 5))

    train_filter_ones = np.where((trY == 1))
    train_filter_fives = np.where((trY == 5))
    test_filter_ones = np.where((teY == 1))
    test_filter_fives = np.where((teY == 5))

    ones_trX = trX[train_filter_ones]
    ones_teX = teX[test_filter_ones]
    fives_trX = trX[train_filter_fives]
    fives_teX = teX[test_filter_fives]

    ones_tr = [[-1 if pixel == 0 else 1 for pixel in digit] for digit in ones_trX]
    fives_tr = [[-1 if pixel == 0 else 1 for pixel in digit] for digit in fives_trX]
    ones_te = [[-1 if pixel == 0 else 1 for pixel in digit] for digit in ones_teX]
    fives_te = [[-1 if pixel == 0 else 1 for pixel in digit] for digit in fives_teX]

    ones_tr = [(digit, 1) for digit in ones_trX]
    fives_tr = [(digit, 5) for digit in fives_trX]
    ones_te = [(digit, 1) for digit in ones_teX]
    fives_te = [(digit, 5) for digit in fives_teX]

    # for x in [ones_tr, fives_tr, ones_te, fives_te]:
    np.random.shuffle(ones_tr)
    np.random.shuffle(fives_tr)
    np.random.shuffle(ones_te)
    np.random.shuffle(fives_te)

    return ones_tr, fives_tr, ones_te, fives_te

def runNetwork(rule):
    ones_tr, fives_tr, ones_te, fives_te = prepData()
    o = ones_te[:140]
    f = fives_te[:140]
    te_ones_fives = o + f
    np.random.shuffle(te_ones_fives)
    ac = []
    np.random.shuffle(ones_tr)
    np.random.shuffle(fives_tr)

    for index in range(1, 18):
        d = (len(ones_tr[:index]))
        tr_ones_fives = ones_tr[:index] + fives_tr[:index]
        A = {}
        np.random.shuffle(tr_ones_fives)
        hopfield = HopfieldNetwork(tr_ones_fives, rule)
        hopfield.compute_learning()
        agg = 0
        df = []
        sd = 0

        for te_digit, te_label in te_ones_fives:
            # n = float('inf')
            predicted_digit = np.array(hopfield.minimize_energy(te_digit, len(te_digit)))
            n = 0
            for digit_o, label_o in ones_tr[:index]:
                norm = np.linalg.norm(predicted_digit - np.array(digit_o))
                if norm < n:
                    n = norm
                    w = label_o
            for digit_f, label_f in fives_tr[:index]:
                norm = np.linalg.norm(predicted_digit - np.array(digit_f))
                if norm < n:
                    n = norm
                    w = label_f
            if (w == te_label):
                sd += 1
        print(d, (sd / len(te_ones_fives) * 100))

runNetwork('storkey') # or 'hebbian'
