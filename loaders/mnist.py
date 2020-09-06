import matplotlib.pyplot as plt
import numpy as np
import os
import gzip

class Loader:
    def load(self, path, kind='train'):
        labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
        images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

        return images, labels


    def plot(self, data):
        img_size = int(len(data)**0.5)
        plt.imshow(np.matrix(data).reshape(img_size, img_size), cmap='gray')

    def plot2(self, data, size):
        plt.imshow(np.matrix(data).reshape(size[0], size[1]), cmap='gray')

    def get_expected_output(self, label):
        z = np.ones(10) *-1
        z[label] = 1.0
        return z


    def get_class(self, prediction):
        prediction = list(prediction[0].A1)
        return prediction.index(max(prediction))


    def get_second_class(self, prediction):
        prediction = list(prediction[0].A1)
        max_index = prediction.index(max(prediction))
        prediction.remove(max(prediction))
        second_max_index = prediction.index(max(prediction))

        if second_max_index >= max_index:
            return second_max_index + 1
        return second_max_index


    def normalize(self, data):
        data = data/np.linalg.norm(data)
        data -= np.mean(data)
        return data