import argparse
import torch
from networks import FullyConnected, Conv, Normalization
from transformers import TransformedNetwork, upper_lower, ZonotopeLoss
import torch.nn as nn
from torch import optim
from common import loadNetwork
import time

DEVICE = 'cpu'
INPUT_SIZE = 28
MODE = "NODEBUG"
VERBOSE = False

torch.set_num_threads(4)


def analyze(net, inputs, eps, true_label,
            slow=False, it=100, learning_rate=0.01, use_adam=False, loss_type='mean', n_relus_to_keep=10):
    beginning = time.time()

    if VERBOSE:
        print("net: ", net)

    transformed_net = TransformedNetwork(net, eps, INPUT_SIZE, n_relus_to_keep=n_relus_to_keep)
    parameters = list(transformed_net.get_params())

    zonotope_loss = ZonotopeLoss(kind=loss_type)

    if use_adam:
        optimizer = optim.Adam(transformed_net.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(transformed_net.parameters(), lr=learning_rate, momentum=0.9)

    should_continue = True
    i = 0
    max_lower, min_upper = -float('inf') * torch.ones([10]), float('inf') * torch.ones([10])

    n_iteration_stuck = 0
    previous_lower = 100000000
    previous_upper = -100000000

    while should_continue:
        if MODE == "DEBUG":
            t0 = time.time()

        # torch.autograd.set_detect_anomaly(True)
        output_zonotope = transformed_net.forward(inputs[0])

        # check if we can verify
        upper, lower = upper_lower(output_zonotope)
        upper_true_label = upper[true_label].item()
        upper[true_label] = -float('inf')
        min_upper = torch.min(upper, min_upper)
        max_lower = torch.max(lower, max_lower)

        verficiation_on_normal_bounds = False
        if verficiation_on_normal_bounds:
            lower_bound = lower[true_label]
            upper_bound = torch.max(upper)
        else:
            lower_bound = max_lower[true_label]
            upper_bound = torch.max(min_upper)

        if upper_bound <= lower_bound:
            return 1

        print("lower_bound ", lower_bound)
        print("previous_lower ", previous_lower)
        #if lower_bound == previous_lower and upper_bound == previous_upper:
        #    n_iteration_stuck += 1

        # otherwise computes loss
        loss = zonotope_loss(upper, lower, output_zonotope, true_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if MODE == "DEBUG" and VERBOSE:
            print("Before clipping")
            for m in transformed_net.layers:
                print(m)
                try:
                    print(m.lambda_.grad)
                    nonzero_grads = (m.lambda_.grad != 0).type(torch.FloatTensor)
                    num_nonzero_grads = int(torch.sum(nonzero_grads).item())
                    print("\tNumber of non zero gradient values: %d" % num_nonzero_grads)
                    print("\tTheir values: %s" % m.lambda_.grad[m.lambda_.grad != 0])
                except:
                    print('\tno weight')

        print()

        transformed_net.clip_lambdas()

        if MODE == "DEBUG":
            print("Iteration %i, time taken %f" % (i, time.time() - t0))
            print("\tBounds:")
            print("\t\tLower bound: %f" % lower_bound)
            print("\t\tUpper bound: %f" % upper_bound)
            upper[true_label] = upper_true_label
            print("\t\tIntervals per class (true class is %d):"%true_label)
            for c in range(10):
                print("\t\t\tClass %d: %f +- %f" % (c, (lower[c]+upper[c]).item()/2, (upper[c]-lower[c]).item()/2))
            # few sanity checks
            parameters = transformed_net.assert_only_relu_params_changed(parameters)
            transformed_net.assert_valid_lambda_values()

            if VERBOSE:
                print("After clipping")
                for m in transformed_net.layers:
                    print(m)
                    try:
                        print(m.lambda_.grad)
                        nonzero_grads = (m.lambda_.grad != 0).type(torch.FloatTensor)
                        num_nonzero_grads = int(torch.sum(nonzero_grads).item())
                        print("Number of non zero gradient values: %d" % num_nonzero_grads)
                        print("Their values: %s" % m.lambda_.grad[m.lambda_.grad != 0])
                    except:
                        print('no weight')

            print("\tFailed: " + str((upper_bound - lower_bound).item()))
            print("\t\tLoss: %f" % loss.item())
            print("\t\tMean lambda values: " + str(transformed_net.get_mean_lambda_values()))

        if slow:
            should_continue = (time.time() - beginning <= 120)
        else:
            i += 1
            should_continue = (i < it)

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
    parser.add_argument('--loss_type', type=str, required=False, default='mean', help='Type of loss used.')
    parser.add_argument('--lr', type=float, required=False, default=0.001, help='Learning rate.')
    parser.add_argument('--n_relus_to_keep', type=int, required=False, default=10, help='Number of relu layers to keep.')
    parser.add_argument('--useAdam', type=int, required=False, default=0, help='Use Adam')
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

    if analyze(net, inputs, eps, true_label, args.slow, args.it, args.lr, args.useAdam, args.loss_type, args.n_relus_to_keep):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()
