import argparse
import torch
from networks import FullyConnected, Conv, Normalization
from transformers import TransformedNetwork, upper_lower
import torch.nn as nn
from torch import optim

DEVICE = 'cpu'
INPUT_SIZE = 28
VERBOSE = True

# TODO figure out why any image is always certified...


def analyze(net, inputs, eps, true_label):
    transformed_net = TransformedNetwork(net, eps, INPUT_SIZE)
    optimizer = optim.SGD(transformed_net.parameters(), lr=0.001, momentum=0.9)
    for i in range(100):
        torch.autograd.set_detect_anomaly(True)
        output_zonotope = transformed_net.forward(inputs)
        upper, lower = upper_lower(output_zonotope)
        # we want to prove that the lower bound for the true label is smaller to the
        # max upper bound for all the other labels
        lower_bound = lower[0, true_label]
        upper[0, true_label] = -1000000
        upper_bound = torch.max(upper)
        if upper_bound <= lower_bound:
            return 1

        # we want to minimize the max of upper bounds (mean used as max not really
        # differentiable (same issue as L1 norm) and maximize the lower bound of
        # the real class
        loss = torch.mean(upper) - lower_bound
        loss.backward()
        optimizer.step()
        transformed_net.clip_lambdas()

        if VERBOSE:
            print("Failed: " + str((upper_bound - lower_bound).item()))
    return 0


def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepZ relaxation')
    parser.add_argument('--net',
                        type=str,
                        choices=['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5'],
                        required=True,
                        help='Neural network to verify.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

    if args.net == 'fc1':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 10]).to(DEVICE)
    elif args.net == 'fc2':
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 50, 10]).to(DEVICE)
    elif args.net == 'fc3':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net == 'fc4':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'fc5':
        net = FullyConnected(DEVICE, INPUT_SIZE, [400, 200, 100, 100, 10]).to(DEVICE)
    elif args.net == 'conv1':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv2':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv3':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [150, 10], 10).to(DEVICE)
    elif args.net == 'conv4':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
    elif args.net == 'conv5':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)

    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()
