import matplotlib.pyplot as plt
import numpy as np
import os, random
import gzip

def get_network_constrains():
    return (2, 2)

def plot(data):
    plt.plot(data[0], data[1], 'h')

def _normalize(data):
    data = data*2
    data = data-1
    return data/np.linalg.norm(data)

def load(path, kind='train'):
    N = 1000
    features = [[random.randint(0,1), random.randint(0,1)] for _ in range(N)]
    labels = [f[0]^f[1] for f in features]
    features = [_normalize(np.array(f)) for f in features]
    return features, labels