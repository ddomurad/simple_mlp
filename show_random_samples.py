import sys
import mnist_helper as mnist
import matplotlib.pyplot as plt
from random import randint

MNIST_DIR = "./DIGITS/"
MNIST_SAMPLES_TYPE = "t10k"

if len(sys.argv) < 2:
    print("INVALID ARGS. EXPECTED ARGS: [minst_dir] [sample_type]")
    exit()

MNIST_DIR = sys.argv[1]
MNIST_SAMPLES_TYPE = sys.argv[2]

print("PARAMS:")
print("MNIST_DIR", MNIST_DIR)
print("MNIST_SAMPLES_TYPE", MNIST_SAMPLES_TYPE)

features, labels = mnist.load(MNIST_DIR, MNIST_SAMPLES_TYPE)
samples_count = len(labels)


plt.figure()
plt.suptitle("Random {}-{} samples.".format(MNIST_DIR, MNIST_SAMPLES_TYPE))
for i in range(1, 9):
    plt.subplot(240 + i)

    random_index = randint(0, samples_count-1)
    preprocessed_data = mnist.normalize(features[random_index])

    mnist.plot(preprocessed_data)
    plt.title("label: {}".format(labels[random_index]))

plt.show()
