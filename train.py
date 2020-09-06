import sys, argparse, random
import loaders, helper
from tqdm import tqdm
from mlp import MlpBuilder

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="new - create new mlp network model, load - load existing network model. [eg. '-n new']")
ap.add_argument("-n", "--network", required=True, help="network file name eg. ['-f ./test_mlp']")
ap.add_argument("-d", "--db", required=False, help="Sample databse dir [eg. '-m ./DATA/MNIST/']", default="")
ap.add_argument("-f", "--func", required=False, help="activation function type (tanh, relu). [eg. '-a tanh']", default="tanh")
ap.add_argument("-l", "--layer", required=False, help="add hidden layers of giver size [eg. '-l 10,20'] - two hidden layers of size 10 and 20", default="")
ap.add_argument("-e", "--epoche", required=False, help="training epoche count [eg. '-e 10']", default="10")
ap.add_argument("-a", "--alpha", required=False, help="training alpha value [eg. '-a 0.1']", default="0.1")
ap.add_argument("-g", "--goal", required=False, help="training goal [eg. '-g 99']", default="100")
ap.add_argument("--alpha_decay", required=False, help="training alpha decay value [eg. '--alpha_decay 0.8']", default="0.8")
ap.add_argument("--loader", required=False, help="data loader name [eg. '--laoder mnist']", default="mnist")
args = vars(ap.parse_args())


MLP_LOAD = args['model']
DB_DIR = args['db']
MLP_FILE_NAME = args['network']
ACT_TYPE = args['func']
HIDDEN_LAYERS = [int(l.strip()) for l in args['layer'].split(',') if l]
MAX_EPOCH_COUNT = int(args['epoche'])
MAX_ALPHA = float(args['alpha'])
ALPHA_DEACY = float(args['alpha_decay'])
DATA_LOADER = args['loader']
TRAINING_GOAL = float(args['goal'])

print("PARAMS:")
print("MLP_LOAD", MLP_LOAD)
print("DB_DIR", DB_DIR)
print("MLP_FILE_NAME", MLP_FILE_NAME)
print("ACT_TYPE", ACT_TYPE)
print("HIDDEN_LAYERS", HIDDEN_LAYERS)
print("MAX_EPOCH_COUNT", MAX_EPOCH_COUNT)
print("TRAINING_GOAL", TRAINING_GOAL)
print("MAX_ALPHA", MAX_ALPHA)
print("ALPHA_DEACY", ALPHA_DEACY)
print("DATA_LOADER", DATA_LOADER)

# load training and validation samples
loader = loaders.get_loader(DATA_LOADER)

input_size, output_size = loader.get_network_constrains()
t_features, t_labels = loader.load(DB_DIR, kind="train")
v_features, v_labels = loader.load(DB_DIR, kind="test")

# create new mlp
mlp = None
if MLP_LOAD == "new":
    NETWORK_SHAPE = (input_size, *HIDDEN_LAYERS, output_size)
    print("NEW NETWORK_SHAPE", NETWORK_SHAPE)
    mlp = MlpBuilder.new_mlp(NETWORK_SHAPE, ACT_TYPE)
else:
    mlp = MlpBuilder.load_mlp(MLP_FILE_NAME)


def calc_mlp_performance():
    performance = 0

    for i in tqdm(range(len(v_labels))):
        # vx = loader.normalize(v_features[i])
        vx = v_features[i]
        mlp_prediction = mlp.predict(vx)
        predicted_label = helper.get_class(mlp_prediction)

        if v_labels[i] == predicted_label:
            performance += 1

    return performance / len(v_labels)


max_validation_performance = calc_mlp_performance()
alpha = MAX_ALPHA

print("Start with performance = {}%".format(max_validation_performance*100))

for t in range(MAX_EPOCH_COUNT):
    if max_validation_performance*100 >= TRAINING_GOAL:
        print("Performance goal achieved")
        exit()
    print("---------- EPOCH: {} ----------------------------------- ".format(t))
    print("Training (a={})".format(alpha))
    training_avg_error = 0
    
    trainign_indices = list(range(len(t_labels))) 
    random.shuffle(trainign_indices)

    for i in tqdm(trainign_indices):
        x = t_features[i]
        y = helper.get_expected_output(t_labels[i], output_size)
        training_avg_error += mlp.train(x, y, alpha)

    training_avg_error /= len(t_labels)
    alpha = alpha*ALPHA_DEACY

    print("Training avg error = {}".format(training_avg_error))
    print("Validation")

    validation_performance = calc_mlp_performance()
    print("Validation performance = {}%".format(validation_performance*100))

    if validation_performance > max_validation_performance:
        max_validation_performance = validation_performance
        MlpBuilder.save_mlp(MLP_FILE_NAME, mlp)
