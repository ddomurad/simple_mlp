rem python .\show.py -d .\DATA\MNIST\ -s train --loader mnist

rem python .\train.py -m new -n ./DATA/MODELS/mnist_model_tanh_5 -d .\DATA\MNIST\ -f tanh -l 5 -e 15 -a 0.1 --loader mnist --alpha_decay 0.8
rem python .\train.py -m new -n ./DATA/MODELS/mnist_model_tanh_10 -d .\DATA\MNIST\ -f tanh -l 10 -e 15 -a 0.1 --loader mnist --alpha_decay 0.8
rem python .\train.py -m new -n ./DATA/MODELS/mnist_model_tanh_15 -d .\DATA\MNIST\ -f tanh -l 15 -e 15 -a 0.1 --loader mnist --alpha_decay 0.8
rem python .\train.py -m new -n ./DATA/MODELS/mnist_model_tanh_20 -d .\DATA\MNIST\ -f tanh -l 20 -e 15 -a 0.1 --loader mnist --alpha_decay 0.8

python .\train.py -m new -n ./DATA/MODELS/mnist_model_relu_5 -d .\DATA\MNIST\ -f relu -l 5 -e 15 -a 0.1 --loader mnist --alpha_decay 0.8
python .\train.py -m new -n ./DATA/MODELS/mnist_model_relu_10 -d .\DATA\MNIST\ -f relu -l 10 -e 15 -a 0.1 --loader mnist --alpha_decay 0.8
python .\train.py -m new -n ./DATA/MODELS/mnist_model_relu_15 -d .\DATA\MNIST\ -f relu -l 15 -e 15 -a 0.1 --loader mnist --alpha_decay 0.8
python .\train.py -m new -n ./DATA/MODELS/mnist_model_relu_20 -d .\DATA\MNIST\ -f relu -l 20 -e 15 -a 0.1 --loader mnist --alpha_decay 0.8
python .\train.py -m new -n ./DATA/MODELS/mnist_model_relu_20 -d .\DATA\MNIST\ -f relu -l 50 -e 150 -a 0.1 --loader mnist --alpha_decay 0.8 -g 100

python .\validate.py -n mnist_model -d .\DATA\MNIST\ --loader mnist