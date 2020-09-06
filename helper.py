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

def get_expected_output(label, size):
    z = np.ones(size) *-1
    z[label] = 1.0
    return z