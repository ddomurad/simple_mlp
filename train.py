import sys, argparse
import mnist_helper as mnist
from tqdm import tqdm
from mlp import MlpBuilder

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="new - create new mlp network model, load - load existing network model. [eg. '-n new']")
ap.add_argument("-n", "--network", required=True, help="network file name eg. ['-f ./test_mlp']")
ap.add_argument("-d", "--mnist_db", required=True, help="MNIST databse dir [eg. '-m ./DIGITS/']")
ap.add_argument("-f", "--func", required=False, help="activation function type (tanh, ...). [eg. '-a tanh']")
ap.add_argument("-l", "--layer", required=False, help="add hidden layers of giver size [eg. '-l 10,20'] - two hidden layers of size 10 and 20")
ap.add_argument("-e", "--epoche", required=True, help="training epoche count [eg. '-e 10']")
ap.add_argument("-a", "--alpha", required=True, help="training alpha value [eg. '-a 0.1']")
args = vars(ap.parse_args())

MLP_LOAD = args['model']
MNIST_DIR = args['mnist_db']
MLP_FILE_NAME = args['network']
ACT_TYPE = args['func'] if args['func'] is not None else 'tanh'
HIDDEN_LAYERS = [int(l.tring()) for l in args['layer'].split(',')] if args['layer'] is not None else None
EPOCH_COUNT = int(args['epoche'])
MAX_ALPHA = float(args['alpha'])

print("PARAMS:")
print("MLP_LOAD", MLP_LOAD)
print("MNIST_DIR", MNIST_DIR)
print("MLP_FILE_NAME", MLP_FILE_NAME)
print("ACT_TYPE", ACT_TYPE)
print("HIDDEN_LAYERS", HIDDEN_LAYERS)
print("EPOCH_COUNT", EPOCH_COUNT)
print("MAX_ALPHA", MAX_ALPHA)

# load training and validation samples
t_features, t_labels = mnist.load(MNIST_DIR, kind="train")
v_features, v_labels = mnist.load(MNIST_DIR, kind="t10k")

# create new mlp
mlp = None
if MLP_LOAD == "new":
    if HIDDEN_LAYERS is None:
        mlp = MlpBuilder.new_mlp((784, 10), ACT_TYPE)
    else:
        mlp = MlpBuilder.new_mlp((784, *HIDDEN_LAYERS, 10), ACT_TYPE)
else:
    mlp = MlpBuilder.load_mlp(MLP_FILE_NAME)


def calc_mlp_performance():
    performance = 0

    for i in tqdm(range(len(v_labels))):
        vx = mnist.normalize(v_features[i])
        mlp_prediction = mlp.predict(vx)
        predicted_label = mnist.get_class(mlp_prediction)

        if v_labels[i] == predicted_label:
            performance += 1

    return performance / len(v_labels)


max_validation_performance = calc_mlp_performance()
alpha = MAX_ALPHA

print("Start with performance = {}%".format(max_validation_performance*100))

for t in range(EPOCH_COUNT):
    print("---------- EPOCH: {} ----------------------------------- ".format(t))
    print("Training (a={})".format(alpha))
    training_avg_error = 0

    for i in tqdm(range(len(t_labels))):
        x = mnist.normalize(t_features[i])
        y = mnist.get_expected_output(t_labels[i])
        training_avg_error += mlp.train(x, y, alpha)

    training_avg_error /= len(t_labels)
    alpha = alpha*0.80

    print("Training avg error = {}".format(training_avg_error))
    print("Validation")

    validation_performance = calc_mlp_performance()
    print("Validation performance = {}%".format(validation_performance*100))

    if validation_performance > max_validation_performance:
        max_validation_performance = validation_performance
        MlpBuilder.save_mlp(MLP_FILE_NAME, mlp)
