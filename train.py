import sys
import mnist_helper as mnist
from mlp import MlpBuilder
from tqdm import tqdm

MLP_LOAD = None
MNIST_DIR = None
MLP_FILE_NAME = None
ACT_TYPE = None
HIDDEN_LAYER_SIZE = 0
EPOCH_COUNT = 0
MAX_ALPHA = 0

if len(sys.argv) < 8:
    print("INVALID ARGS. EXPECTED ARGS: '[mlp_load] [mnist_dir]  [mlp_output_file]  [activation_type] [hidden_layer_size] [training_epoch_count] [max_alpha]'")
    print("EXAMPLE: 'python train.py new ./DIGITS/ ./test_mlp tanh 0 10 0.1'")
    exit()
else:
    MLP_LOAD = sys.argv[1]
    MNIST_DIR = sys.argv[2]
    MLP_FILE_NAME = sys.argv[3]
    ACT_TYPE = sys.argv[4]
    HIDDEN_LAYER_SIZE = int(sys.argv[5])
    EPOCH_COUNT = int(sys.argv[6])
    MAX_ALPHA = float(sys.argv[7])


print("PARAMS:")
print("MLP_LOAD", MLP_LOAD)
print("MNIST_DIR", MNIST_DIR)
print("MLP_FILE_NAME", MLP_FILE_NAME)
print("ACT_TYPE", ACT_TYPE)
print("HIDDEN_LAYER_SIZE", HIDDEN_LAYER_SIZE)
print("EPOCH_COUNT", EPOCH_COUNT)
print("MAX_ALPHA", MAX_ALPHA)

# load training and validation samples
t_features, t_labels = mnist.load(MNIST_DIR, kind="train")
v_features, v_labels = mnist.load(MNIST_DIR, kind="t10k")

# create new mlp
mlp = None
if MLP_LOAD == "new":
    if HIDDEN_LAYER_SIZE == 0:
        mlp = MlpBuilder.new_mlp((784, 10), ACT_TYPE)
    else:
        mlp = MlpBuilder.new_mlp((784, HIDDEN_LAYER_SIZE, 10), ACT_TYPE)
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
