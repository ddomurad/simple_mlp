rem python .\show.py -d .\DATA\MNIST\ -s train --loader mnist
python .\train.py -m new -n mnist_model -d .\DATA\MNIST\ -f tanh -l 15 -e 15 -a 0.1 --loader mnist --alpha_decay 0.8
python .\validate.py -n mnist_model -d .\DATA\MNIST\ --loader mnist