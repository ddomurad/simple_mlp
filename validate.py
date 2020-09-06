import sys, argparse
from random import randint
import loaders
from tqdm import tqdm
import matplotlib.pyplot as plt
from mlp import MlpBuilder

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--network", required=True, help="network file name eg. ['-f ./test_mlp']")
ap.add_argument("-d", "--db", required=True, help="Sample databse dir [eg. '-m ./DATA/MNIST/']")
ap.add_argument("--loader", required=False, help="data loader name [eg. '--laoder mnist']", default="mnist")
args = vars(ap.parse_args())

DB_DIR = args['db']
MLP_FILE_NAME = args['network']
DATA_LOADER = args['loader']

print("PARAMS:")
print("DB_DIR", DB_DIR)
print("MLP_FILE_NAME", MLP_FILE_NAME)
print("DATA_LOADER", DATA_LOADER)

loader = loaders.get_loader(DATA_LOADER)
# load validation samples
v_features, v_labels = loader.load(DB_DIR, kind="t10k")

# create new mlp
mlp = MlpBuilder.load_mlp(MLP_FILE_NAME)

validation_performance = 0
for i in tqdm(range(len(v_labels))):
    x = loader.normalize(v_features[i])

    mlp_prediction = mlp.predict(x)
    predicted_label = loader.get_class(mlp_prediction)

    if v_labels[i] == predicted_label:
        validation_performance += 1

validation_performance /= len(v_labels)
print("Validation performance = {}%".format(validation_performance*100))

while True:
    plt.figure()
    
    for i in range(8):
        validation_index = randint(1, len(v_labels) - 1)
        x = loader.normalize(v_features[validation_index])
        y_label = v_labels[validation_index]

        y = loader.get_expected_output(y_label)
        mlp_prediction = mlp.predict(x)

        class1 = loader.get_class(mlp_prediction)
        class2 = loader.get_second_class(mlp_prediction)

        plt.subplot(241 + i)

        plt.title('e: {}, p: {}(s: {})'.format(y_label, class1, class2))
        loader.plot(x)

    plt.show()


