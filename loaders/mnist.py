import matplotlib.pyplot as plt
import numpy as np
import os
import gzip

def get_network_constrains():
    return (784, 10)

def plot(data):
    img_size = int(len(data)**0.5)
    plt.imshow(np.matrix(data).reshape(img_size, img_size), cmap='gray')

def _normalize(data):
    data = data/np.linalg.norm(data)
    data -= np.mean(data)
    return data

def load(path, kind='train'):
    kind = 't10k' if kind == 'test' else 'train'
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    
    images = [_normalize(image) for image in images]
    
    return images, labels