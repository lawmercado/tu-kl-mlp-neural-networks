import argparse
import torch
import numpy as np
from operations import test
from data.dataset import get_strange_symbols_test_data

"""
Small program to apply the generated neural network into the test set

"""

parser = argparse.ArgumentParser()
parser.add_argument('nn', type=str, help='the network .pt model path')

args = parser.parse_args()

if __name__ == '__main__':
    print('[TEST] model saved in', args.nn)
    print()

    net = torch.load(args.nn)
    print(net)
    print()

    test_data = get_strange_symbols_test_data()

    output = test(net, test_data, normalize=True)

    idx = np.zeros((len(output), 1))

    for i in range(1, len(output) + 1):
        idx[i - 1] = i

    np.savetxt('results.csv', np.hstack((idx, output)), delimiter=',')

    print('[TEST] Saved test set results in \'results.csv\'')


