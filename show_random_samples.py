import sys, argparse
import loaders
import matplotlib.pyplot as plt
from random import randint

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True, help="Sample databse dir [eg. '-m ./DATA/MNIST/']")
ap.add_argument("-s", "--sample_type", required=True, help="Sample type [eg. '-s t10k']")
ap.add_argument("--loader", required=False, help="data loader name [eg. '--laoder mnist']", default="mnist")
args = vars(ap.parse_args())

DB_DIR = args['db']
MNIST_SAMPLES_TYPE = args['sample_type']
DATA_LOADER = args['loader']

print("PARAMS:")
print("DB_DIR", DB_DIR)
print("MNIST_SAMPLES_TYPE", MNIST_SAMPLES_TYPE)
print("DATA_LOADER", DATA_LOADER)

loader = loaders.get_loader(DATA_LOADER)
features, labels = loader.load(DB_DIR, MNIST_SAMPLES_TYPE)
samples_count = len(labels)


plt.figure()
plt.suptitle("Random {}-{} samples.".format(DB_DIR, MNIST_SAMPLES_TYPE))
for i in range(1, 9):
    plt.subplot(240 + i)

    random_index = randint(0, samples_count-1)
    preprocessed_data = loader.normalize(features[random_index])

    loader.plot(preprocessed_data)
    plt.title("label: {}".format(labels[random_index]))

plt.show()
