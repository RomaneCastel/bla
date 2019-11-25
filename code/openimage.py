import argparse
import torch
from networks import FullyConnected, Conv, Normalization
from transformers import TransformedNetwork, upper_lower
import torch.nn as nn
from torch import optim
from common import loadNetwork
import time
import matplotlib.pyplot as plt

DEVICE = 'cpu'
INPUT_SIZE = 28
VERBOSE = True

# TODO figure out why any image is always certified...
# Possible improvements:
# 1) Use a more advanced optimizer than SGD, like Adam
# 2) Optimize the number of iterations
MODE = "NO DEBUG"


def main():
    parser = argparse.ArgumentParser(description='See image')
    parser.add_argument('--spec', type=str, required=True, help='Test case to visualise.')
    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])


    inputs = torch.FloatTensor(pixel_values).view(INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    print("true_label: %d" % true_label)
    print("eps: %f" %  eps)
    plt.imshow(inputs.numpy())
    # plt.show()
    plt.savefig("image.png")


if __name__ == '__main__':
    main()
