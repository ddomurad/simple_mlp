import pickle
import numpy as np


class FFN:
    def __init__(self):
        self.A = []
        self.Z = []

        self.W = []
        self.B = []

        self.layers_size = None
        self.layers_count = None
        self.activation_type = None
        self.activation_fnc = None
        self.activation_fnc_d = None
        self.input_vector_size = None
        self.output_vector_size = None

    def predict(self, i):
        i = np.matrix(i)

        # for each layer
        for l in range(self.layers_count):
            # for input layer
            if l == 0:
                self.A[l] = i
            # for any hidden layer or output layer
            else:
                self.Z[l] = self.A[l-1]*np.transpose(self.W[l]) + self.B[l]
                self.A[l] = self.activation_fnc(self.Z[l])

        # return output layer activation matrix
        return self.A[-1]

    def train(self, i, y, alpha):
        i = np.matrix(i)
        y = np.matrix(y)

        self.predict(i)

        # delta[l+1]
        next_layer_delta = None

        # derivatives placeholders  
        dc_db = []
        dc_dw = []

        # for each layer in reverse order (skip layer 0)
        for l in reversed(range(1, self.layers_count)):
            delta = None

            # compute delta
            if l == self.layers_count - 1:
                # for output layer
                delta = np.multiply(self.A[l] - y, self.activation_fnc_d(self.Z[l]))
            else:
                # for any hidden layer
                delta = np.multiply(next_layer_delta*self.W[l+1], self.activation_fnc_d(self.Z[l]))

            dc_db.append(delta)
            dc_dw.append(np.transpose(np.transpose(self.A[l - 1]) * delta))

            # store delta as delta[l+1]
            next_layer_delta = delta

        for l in reversed(range(1, self.layers_count)):
            self.B[l] -= alpha * dc_db[self.layers_count - l - 1]
            self.W[l] -= alpha * dc_dw[self.layers_count - l - 1]

        return 0.5 * np.square(np.sum((y - self.A[-1])))


class Activations:
    @staticmethod
    def relu(z):
        return np.multiply(z, (z >= 0)) + np.multiply(0.1 * z, (z < 0))

    @staticmethod
    def relu_d(z):
        return (1 * (z >= 0)) + (0.1 * (z < 0))

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def tanh_d(z):
        return 1.0 - np.square(np.tanh(z))


class Builder:

    @staticmethod
    def new_ffn(layers_size, activation_type) -> FFN:
        ffn = FFN()

        # initialize layer description
        ffn.layers_size = layers_size[:]
        ffn.layers_count = len(ffn.layers_size)
        ffn.input_vector_size = ffn.layers_size[0]
        ffn.output_vector_size = ffn.layers_size[-1]

        # apply activation function
        Builder._apply_activation_function(ffn, activation_type)

        for (layer_index, l_size) in enumerate(layers_size):
            # input layer
            if layer_index == 0:
                # initialize A,Z,B,W for input layer (Z,B,W are not used in input the layer)
                ffn.A.append(np.zeros((1, ffn.input_vector_size)))
                ffn.Z.append(None)
                ffn.W.append(None)
                ffn.B.append(None)
            # hidden or output layer
            else:
                # prepare random W matrix with uniform distributions in range [-1, 1]
                random_w_matrix = np.random.rand(l_size, layers_size[layer_index-1])
                random_w_matrix = random_w_matrix*2 - 1

                # prepare random B matrix with uniform distributions in range [-1, 1]
                random_b_matrix = np.random.rand(1, l_size)
                random_b_matrix = random_b_matrix*2 - 1

                ffn.W.append(random_w_matrix)
                ffn.B.append(random_b_matrix)

                # prepare Z, A matrices for further usage
                ffn.Z.append(np.zeros((1, l_size)))
                ffn.A.append(np.zeros((1, l_size)))

        return ffn

    @staticmethod
    def save_ffn(file_path, ffn):
        dump_data = [ffn.layers_size, ffn.activation_type, ffn.W, ffn.B]
        with open(file_path, 'wb') as out_file:
            pickle.dump(dump_data, out_file)

    @staticmethod
    def load_ffn(file_path):
        with open(file_path, 'rb') as in_file:
            dump_data = pickle.load(in_file)
            ffn = Builder.new_ffn(dump_data[0], dump_data[1])
            ffn.W = dump_data[2]
            ffn.B = dump_data[3]

            return ffn

    @staticmethod
    def _apply_activation_function(ffn, activation_type):
        ffn.activation_type = activation_type
        if activation_type == "relu":
            ffn.activation_fnc = Activations.relu
            ffn.activation_fnc_d = Activations.relu_d
        elif activation_type == "tanh":
            ffn.activation_fnc = Activations.tanh
            ffn.activation_fnc_d = Activations.tanh_d
        else:
            raise ValueError("Invalid network activation type")
