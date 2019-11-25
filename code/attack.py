import argparse
from common import loadNetwork
import torch
import torch.nn as nn


# Code taken from the solution of the 4th RIAI lab

def fgsm_(model, x, target, eps, targeted=True, clip_min=None, clip_max=None):
    """Internal process for all FGSM and PGD attacks."""
    # create a copy of the input, remove all previous associations to the compute graph...
    input_ = x.clone().detach_()
    # ... and make sure we are differentiating toward that variable
    input_.requires_grad_()

    # run the model and obtain the loss
    logits = model(input_)
    target = torch.LongTensor([target])
    model.zero_grad()
    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()

    #perform either targeted or untargeted attack
    if targeted:
        out = input_ - eps * input_.grad.sign()
    else:
        out = input_ + eps * input_.grad.sign()

    #if desired clip the ouput back to the image domain
    if (clip_min is not None) or (clip_max is not None):
        out.clamp_(min=clip_min, max=clip_max)
    return out

def pgd_(model, x, target, k, eps, eps_step, targeted=True, clip_min=None, clip_max=None):
    x_min = x - eps
    x_max = x + eps
    for i in range(k):
        # FGSM step
        # We don't clamp here (arguments clip_min=None, clip_max=None)
        # as we want to apply the attack as defined
        x = fgsm_(model, x, target, eps_step, targeted)
        # Projection Step
        x = torch.max(x_min, x)
        x = torch.min(x_max, x)

    # if desired clip the ouput back to the image domain
    if clip_min is not None or clip_max is not None:
        x.clamp_(min=clip_min, max=clip_max)
    return x

def pgd_targeted(model, x, target, k, eps, eps_step, clip_min=None, clip_max=None, **kwargs):
    return pgd_(model, x, target, k, eps, eps_step, targeted=True, **kwargs)

def pgd_untargeted(model, x, label, k, eps, eps_step, clip_min=None, clip_max=None, **kwargs):
    return pgd_(model, x, label, k, eps, eps_step, targeted=False, **kwargs)




DEVICE = 'cpu'
INPUT_SIZE = 28
VERBOSE = True



def main():
    # Example usage
    # python attack.py --net fc1 --spec ../test_cases/fc1/img0_0.06000.txt --k 50 --eps 0.2 --eps_step 0.01

    parser = argparse.ArgumentParser(description='Generate adversial input using Projected Gradient Descent')
    parser.add_argument('--net',
                        type=str,
                        choices=['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5'],
                        required=True,
                        help='Neural network to attack.')
    parser.add_argument('--spec', type=str, required=True, help='Image to perturb.')
    parser.add_argument('--k', type=int, required=True, help='Number of PGD iterations.')
    parser.add_argument('--eps', type=float, required=True, help='Maximum allowed perturbation from input.')
    parser.add_argument('--eps_step', type=float, required=True, help='Size of a perturbation step.')
    parser.add_argument('--saveFile', type=bool, required=False, default=False, help='Save the file at the end')


    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

    net = loadNetwork(args, DEVICE)



    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    perturbedInput = pgd_untargeted(net, inputs, true_label, args.k, args.eps, args.eps_step)


    outs = net(perturbedInput)
    pred_label = outs.max(dim=1)[1].item()

    success = pred_label != true_label

    if success:
        print("Successfully generated an adversarial example")
    else:
        print("Unable to generate an adversarial example with the given parameters")

    if args.saveFile:
        successMsg = 'success' if success else 'failure'
        folder, filename = '/'.join(args.spec.split('/')[:-1]), args.spec.split('/')[-1]
        newFilename = folder + '/perturbed_%s_%s' % (successMsg, args.eps) + filename
        #print("Saving perturbed image to file %s" % newFilename)

        perturbedInput = perturbedInput.view(-1)
        with open(newFilename, 'w') as f:
            f.write('%d\n' % true_label)
            for i in range(perturbedInput.shape[0]):
                f.write('%f\n' % perturbedInput[i].item())







if __name__ == '__main__':
    main()
