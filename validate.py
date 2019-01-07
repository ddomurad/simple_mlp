import sys
from random import randint
import mnist_helper as mnist
from mlp import MlpBuilder
from tqdm import tqdm
import matplotlib.pyplot as plt

MNIST_DIR = None
MLP_FILE_NAME = None

if len(sys.argv) < 2:
    print("INVALID ARGS. EXPECTED ARGS: '[mnist_dir]  [mlp_input_file]'")
    print("EXAMPLE: 'python train.py ./DIGITS/ ./test_mlp'")
    exit()
else:
    MNIST_DIR = sys.argv[1]
    MLP_FILE_NAME = sys.argv[2]


print("PARAMS:")
print("MNIST_DIR", MNIST_DIR)
print("MLP_FILE_NAME", MLP_FILE_NAME)

# load validation samples
v_features, v_labels = mnist.load(MNIST_DIR, kind="t10k")

# create new mlp
mlp = MlpBuilder.load_mlp(MLP_FILE_NAME)

validation_performance = 0
for i in tqdm(range(len(v_labels))):
    x = mnist.normalize(v_features[i])

    mlp_prediction = mlp.predict(x)
    predicted_label = mnist.get_class(mlp_prediction)

    if v_labels[i] == predicted_label:
        validation_performance += 1

validation_performance /= len(v_labels)
print("Validation performance = {}%".format(validation_performance*100))

while True:
    plt.figure()
    
    for i in range(8):
        validation_index = randint(1, len(v_labels) - 1)
        x = mnist.normalize(v_features[validation_index])
        y_label = v_labels[validation_index]

        y = mnist.get_expected_output(y_label)
        mlp_prediction = mlp.predict(x)

        class1 = mnist.get_class(mlp_prediction)
        class2 = mnist.get_second_class(mlp_prediction)

        plt.subplot(241 + i)

        plt.title('e: {}, p: {}(s: {})'.format(y_label, class1, class2))
        mnist.plot(x)

    plt.show()


