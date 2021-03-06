import sys, argparse
from random import randint
import loaders, helper
from tqdm import tqdm
import matplotlib.pyplot as plt
import ffn

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--network", required=True, help="network file name eg. ['-f ./test_network']")
ap.add_argument("-d", "--db", required=False, help="Sample databse dir [eg. '-m ./DATA/MNIST/']", default="")
ap.add_argument("--loader", required=False, help="data loader name [eg. '--laoder mnist']", default="mnist")
args = vars(ap.parse_args())

DB_DIR = args['db']
NETWORK_FILE_NAME = args['network']
DATA_LOADER = args['loader']

print("PARAMS:")
print("DB_DIR", DB_DIR)
print("NETWORK_FILE_NAME", NETWORK_FILE_NAME)
print("DATA_LOADER", DATA_LOADER)

loader = loaders.get_loader(DATA_LOADER)
input_size, output_size = loader.get_network_constrains()
# load validation samples
v_features, v_labels = loader.load(DB_DIR, kind="test")

# load newtwork from file
network = ffn.Builder.load_ffn(NETWORK_FILE_NAME)

validation_performance = 0
for i in tqdm(range(len(v_labels))):
    x = v_features[i]

    prediction = network.predict(x)
    predicted_label = helper.get_class(prediction)

    if v_labels[i] == predicted_label:
        validation_performance += 1

validation_performance /= len(v_labels)
print("Validation performance = {}%".format(validation_performance*100))

while True:
    plt.figure()
    
    for i in range(8):
        validation_index = randint(1, len(v_labels) - 1)
        x = v_features[validation_index]
        y_label = v_labels[validation_index]

        y = helper.get_expected_output(y_label, output_size)
        prediction = network.predict(x)

        class1 = helper.get_class(prediction)
        class2 = helper.get_second_class(prediction)

        plt.subplot(241 + i)

        plt.title('e: {}, p: {}(s: {})'.format(y_label, class1, class2))
        loader.plot(x)

    plt.show()


