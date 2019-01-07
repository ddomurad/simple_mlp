import pickle
import numpy as np


class Mlp:
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


class MlpActivations:
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


class MlpBuilder:

    @staticmethod
    def new_mlp(layers_size, activation_type) -> Mlp:
        mlp = Mlp()

        # initialize layer description
        mlp.layers_size = layers_size[:]
        mlp.layers_count = len(mlp.layers_size)
        mlp.input_vector_size = mlp.layers_size[0]
        mlp.output_vector_size = mlp.layers_size[-1]

        # apply activation function
        MlpBuilder._apply_activation_function(mlp, activation_type)

        for (layer_index, l_size) in enumerate(layers_size):
            # input layer
            if layer_index == 0:
                # initialize A,Z,B,W for input layer (Z,B,W are not used in input the layer)
                mlp.A.append(np.zeros((1, mlp.input_vector_size)))
                mlp.Z.append(None)
                mlp.W.append(None)
                mlp.B.append(None)
            # hidden or output layer
            else:
                # prepare random W matrix with uniform distributions in range [-1, 1]
                random_w_matrix = np.random.rand(l_size, layers_size[layer_index-1])
                random_w_matrix = random_w_matrix*2 - 1

                # prepare random B matrix with uniform distributions in range [-1, 1]
                random_b_matrix = np.random.rand(1, l_size)
                random_b_matrix = random_b_matrix*2 - 1

                mlp.W.append(random_w_matrix)
                mlp.B.append(random_b_matrix)

                # prepare Z, A matrices for further usage
                mlp.Z.append(np.zeros((1, l_size)))
                mlp.A.append(np.zeros((1, l_size)))

        return mlp

    @staticmethod
    def save_mlp(file_path, mlp):
        dump_data = [mlp.layers_size, mlp.activation_type, mlp.W, mlp.B]
        with open(file_path, 'wb') as out_file:
            pickle.dump(dump_data, out_file)

    @staticmethod
    def load_mlp(file_path):
        with open(file_path, 'rb') as in_file:
            dump_data = pickle.load(in_file)
            mlp = MlpBuilder.new_mlp(dump_data[0], dump_data[1])
            mlp.W = dump_data[2]
            mlp.B = dump_data[3]

            return mlp

    @staticmethod
    def _apply_activation_function(mlp, activation_type):
        mlp.activation_type = activation_type
        if activation_type == "relu":
            mlp.activation_fnc = MlpActivations.relu
            mlp.activation_fnc_d = MlpActivations.relu_d
        elif activation_type == "tanh":
            mlp.activation_fnc = MlpActivations.tanh
            mlp.activation_fnc_d = MlpActivations.tanh_d
        else:
            raise ValueError("Invalid network activation type")
