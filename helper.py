import matplotlib.pyplot as plt
import numpy as np
import os
import gzip

def get_class(prediction):
    prediction = list(prediction[0].A1)
    return prediction.index(max(prediction))


def get_second_class(prediction):
    prediction = list(prediction[0].A1)
    max_index = prediction.index(max(prediction))
    prediction.remove(max(prediction))
    second_max_index = prediction.index(max(prediction))

    if second_max_index >= max_index:
        return second_max_index + 1
    return second_max_index


def normalize(data):
    data = data/np.linalg.norm(data)
    data -= np.mean(data)
    return data
def plot(data):
        img_size = int(len(data)**0.5)
        plt.imshow(np.matrix(data).reshape(img_size, img_size), cmap='gray')

def plot2(data, size):
    plt.imshow(np.matrix(data).reshape(size[0], size[1]), cmap='gray')

def get_expected_output(label):
    z = np.ones(10) *-1
    z[label] = 1.0
    return z