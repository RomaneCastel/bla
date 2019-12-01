import argparse
import torch
from networks import FullyConnected, Conv, Normalization
from transformers import TransformedNetwork, upper_lower
import torch.nn as nn
from torch import optim
from common import loadNetwork
import time

DEVICE = 'cpu'
INPUT_SIZE = 28
VERBOSE = True

# TODO figure out why any image is always certified...
# Possible improvements:
# 1) Use a more advanced optimizer than SGD, like Adam
# 2) Optimize the number of iterations
MODE = "DEBUG"


def analyze(net, inputs, eps, true_label, slow, it):
    beginning = time.time()
    transformed_net = TransformedNetwork(net, eps, INPUT_SIZE)
    parameters = list(transformed_net.get_params())
    optimizer = optim.SGD(transformed_net.parameters(), lr=0.001, momentum=0.9)

    shouldContinue = True
    i = 0
    while shouldContinue:
        torch.autograd.set_detect_anomaly(True)
        output_zonotope = transformed_net.forward(inputs)
        upper, lower = upper_lower(output_zonotope)
        # we want to prove that the lower bound for the true label is smaller than the
        # max upper bound for all the other labels, because this means the true label value
        # will always be bigger than the other labels, and so the classification will be correct
        lower_bound = lower[0, true_label]
        upper[0, true_label] = -1000000  # Ignore upper bound of the true label
        upper_bound = torch.max(upper)
        if upper_bound <= lower_bound:
            return 1

        # we want to minimize the max of upper bounds (mean used as max not really
        # differentiable (same issue as L1 norm) and maximize the lower bound of
        # the real class
        loss = torch.mean(upper) - lower_bound
        loss.backward()
        optimizer.step()

        if mode == "DEBUG":
            print("Before clipping")
            for m in transformed_net.layers:
                print(m)
                try:
                    print(m.lambda_.grad)
                except:
                    print('no weight')

        transformed_net.clip_lambdas()

        if MODE == "DEBUG":
            print("After clipping")
            # few sanity checks
            parameters = transformed_net.assert_only_relu_params_changed(parameters)
            transformed_net.assert_valid_lambda_values()


            for m in transformed_net.layers:
                print(m)
                try:
                    print(m.lambda_.grad)
                except:
                    print('no weight')

            print(loss)

            print("Failed: " + str((upper_bound - lower_bound).item()))
            print(transformed_net.get_mean_lambda_values())

        if slow:
            shouldContinue =  (time.time() - beginning < 110)
        else:
            i += 1
            shouldContinue = (i < it)

    return 0


def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepZ relaxation')
    parser.add_argument('--net',
                        type=str,
                        choices=['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5'],
                        required=True,
                        help='Neural network to verify.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    parser.add_argument('--slow', type=int, required=False, default=0, help='Run for almost 2 minutes.')
    parser.add_argument('--it', type=int, required=False, default=100, help='Number of iterations (if not choosing --slow).')
    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

    net = loadNetwork(args, DEVICE)

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, inputs, eps, true_label, args.slow, args.it):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()
