# Machine Learning Project: Task1 - Neural Networks

## Crossvalidate
```
usage: crossvalidate.py [-h] [--learning_rate LEARNING_RATE]
                        [--momentum MOMENTUM] [--batch_size BATCH_SIZE]
                        [--epochs EPOCHS] [--epslon EPSLON] [--seed SEED]
                        nn k

positional arguments:
  nn                    the network to operate with: ['basic', 'lenet5']
  k                     the number of folds to use

optional arguments:
  -h, --help            show this help message and exit
  --learning_rate LEARNING_RATE
                        the learning rate
  --momentum MOMENTUM   the momentum
  --batch_size BATCH_SIZE
                        the batch size
  --epochs EPOCHS       the epochs to train
  --epslon EPSLON       the convergence criteria
  --seed SEED           the seed to consider in random numbers generation for
                        reproducibility
```

## Validate
```
usage: validate.py [-h] [--learning_rate LEARNING_RATE] [--momentum MOMENTUM]
                   [--batch_size BATCH_SIZE] [--epochs EPOCHS]
                   [--epslon EPSLON] [--seed SEED] [--plot]
                   nn s

positional arguments:
  nn                    the network to operate with: ['basic', 'lenet5']
  s                     the split fraction used for the validation set

optional arguments:
  -h, --help            show this help message and exit
  --learning_rate LEARNING_RATE
                        the learning rate
  --momentum MOMENTUM   the momentum
  --batch_size BATCH_SIZE
                        the batch size
  --epochs EPOCHS       the epochs to train
  --epslon EPSLON       the convergence criteria
  --seed SEED           the seed to consider in random numbers generation for
                        reproducibility
  --plot                whether the results should be plotted
```

## Generate
```
usage: generate.py [-h] [--learning_rate LEARNING_RATE] [--momentum MOMENTUM]
                   [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--seed SEED]
                   [--plot]
                   nn s p

positional arguments:
  nn                    the network to operate with: ['basic', 'lenet5']
  s                     the split fraction used for the validation set
  p                     the patience factor used in early stopping

optional arguments:
  -h, --help            show this help message and exit
  --learning_rate LEARNING_RATE
                        the learning rate
  --momentum MOMENTUM   the momentum
  --batch_size BATCH_SIZE
                        the batch size
  --epochs EPOCHS       the epochs to train
  --seed SEED           the seed to consider in random numbers generation for
                        reproducibility
  --plot                whether the results should be plotted
```

## Test
```
usage: test.py [-h] nn

positional arguments:
  nn          the network .pt model path

optional arguments:
  -h, --help  show this help message and exit
```