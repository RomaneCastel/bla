import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import FullyConnected, Conv, Normalization

"""
The goal is to transform a network into zonotope verifier network.

At each layer, instead of dealing with a single image, it deals with several images:
 - the bias image;
 - one weight image per epsilon_i. The number of weight images can possibly increase by
 1 for each ReLU layer.
Note that those images can be flatten.
Therefore a zonotope will be represent as a tensor of size (1 + n_eps) x image height x image width

Transforming a sequential network of several layers requires to be able to transform every
layer. Also, the following relationship holds for zonotope transformation:
Transformation(Sequential([L_i])) = Sequential([Transformation(L_i)])
We therefore need to transform the following layers:
 - Input (clipping to stay between [0,1]);
 - Normalization;
 - Linear;
 - Conv2D;
 - Flatten;
 - ReLU.
"""

VERBOSE_LOGGING = False


# utils function: compute lower and upper bounds of a zonotope
def upper_lower(zonotope):
    upper = zonotope[0] + torch.sum(torch.abs(zonotope[1:]), dim=0)
    lower = zonotope[0] - torch.sum(torch.abs(zonotope[1:]), dim=0)
    assert upper.shape == zonotope.shape[1:]
    return upper, lower


@torch.jit.script
def new_error_terms(x, condition, receiver, start_index):
    # type: (Tensor, Tensor, Tensor, int) -> int
    # x is the ND tensor, condition a x-size boolean tensor (True if a new tensor has to be created for this value)
    # receiver is the N+1D tensor in which the newly created sparse tensor will be stored
    # start_index is the index from which the new tensors will be added
    i_error = start_index
    # when vector
    if len(x.shape) == 1:
        for i in range(x.shape[0]):
            if condition[i].item():
                receiver[i_error, i] = x[i]
                i_error += 1
    # when image
    else:
        for f in range(x.shape[0]):
            for i in range(x.shape[1]):
                for j in range(x.shape[2]):
                    if condition[f, i, j].item():
                        receiver[i_error, f, i, j] = x[f, i, j]
                        i_error += 1
    return i_error


class Zonotope:
    # object representing zonotopes
    def __init__(self, x=2000, n_error_terms=0):
        if x is not None:
            self.zonotope = torch.zeros(
                [1+n_error_terms,  # 0 for value, all the others will be for error terms, considered as batch size
                 x.shape[0],
                 x.shape[1],
                 x.shape[2]
                 ],
                dtype=x.dtype)
        self.created_terms = []
        self.last_error_term = 1

    def add_space(self, n=None):
        # if the number of new items is not specified, consider that we have to add one error term for each value
        # of the bias tensor
        if n is None:
            if len(self.zonotope.shape) == 4:
                n = self.zonotope.shape[1] * self.zonotope.shape[2] * self.zonotope.shape[3]
            else:
                n = self.zonotope.shape[1]
        # creates a new zonotope
        new_shape = list(self.zonotope.shape)
        new_shape[0] = n
        new_space = torch.zeros(new_shape, dtype=self.zonotope.dtype)
        # adds previous values
        self.zonotope = torch.cat([self.zonotope, new_space], dim=0)

    def fill_bias(self, x):
        self.zonotope[0] = x

    def get_zonotope(self):
        return self.zonotope[:self.last_error_term]

    def new_error_terms(self, error_terms, condition):
        # checks that there is enough space to put the new error terms
        n_new_error_terms = torch.sum(condition).item()
        if self.last_error_term + n_new_error_terms > self.zonotope.shape[0]:
            self.add_space()
        if n_new_error_terms > 0:
            self.created_terms = []
            self.last_error_term = new_error_terms(error_terms,
                                                   condition,
                                                   self.zonotope,
                                                   self.last_error_term,
                                                   self.created_terms)


class TransformedInput(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # creates all the height x width error matrices/vectors

        # initializes zonotope
        zonotope = torch.zeros([
            1 + x.shape[0] * x.shape[1] * x.shape[2],
            x.shape[0],
            x.shape[1],
            x.shape[2]
        ])

        # fills biases
        zonotope[0] = x + nn.functional.relu(self.eps - x)/2 - nn.functional.relu(x-(1-self.eps))/2

        # creates error terms
        error_terms = self.eps - nn.functional.relu(self.eps - x)/2 - nn.functional.relu(x-(1-self.eps))/2
        new_error_terms(error_terms, error_terms >= 0, zonotope, 1)

        return zonotope


class TransformedNormalization(nn.Module):
    def __init__(self, normalization_layer):
        super().__init__()
        self.mean = normalization_layer.mean[0, 0, 0, 0]
        self.sigma = normalization_layer.sigma[0, 0, 0, 0]

    def forward(self, x):
        x[0] -= self.mean
        x /= self.sigma
        return x


# for Linear layers (both linear transformers)
class TransformedLinear(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        # input of size (n_eps + 1) x in_features
        # output of size (n_eps + 1) x out_features
        # bias receives full affine transform, error weights only the linear part
        # x: (1 + h x w x n_channels) x n_channels x (width x height)
        x = F.linear(x, self.layer.weight, None)  # no bias for the moment
        if self.layer.bias is not None:
            x[0] += self.layer.bias
        return x


class TransformedConv2D(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        # input of shape
        # 1+n_errors x in_features x h x w
        # output of shape
        # 1+n_errors x out_features x h' x w'
        x = F.conv2d(x, self.layer.weight, bias=None, stride=self.layer.stride, padding=self.layer.padding)
        x[0] += self.layer.bias.unsqueeze(-1).unsqueeze(-1)

        return x


class TransformedFlatten(nn.Module):
    def __init__(self):
        super().__init__()
        self.start_dim = -3
        self.end_dim = -1

    def forward(self, x):
        # input of shape
        # 1+n_errors x n_features x h x w
        # output of shape
        # 1+n_errors x (n_features * h * w)
        x = x.flatten(self.start_dim, self.end_dim)

        return x


class TransformedReLU(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.lambda_ = nn.Parameter(torch.Tensor(shape), requires_grad=True)
        # if we have already initialized lambda
        self.is_lambda_set = False

    def _set_lambda(self, lower, upper):
        # set as its optimal area value
        #  if u <= 0, l = 0
        #  if l >= 0, l = 1
        #  else l = u / (u-l)
        _lambda = (lower >= 0).type(torch.FloatTensor) \
                  + (lower * upper < 0).type(torch.FloatTensor) \
                  * upper / (upper - lower)
        # set all nans to 1
        _lambda[_lambda != _lambda] = 0.5
        self.lambda_.data = _lambda
        self.is_lambda_set = True

    def forward(self, x):
        # x: (1 + h x w x n_channels) x n_channels x width x height
        # or x: (1 + h x w x n_channels) x n_channels x (width x height)
        # computes minimum and maximum boundaries for every input value
        # upper and lower bound are tensor of size n_features x (h x w) || vector_size
        # creates the lambda values
        upper, lower = upper_lower(x)

        # lambda has a size of ((n_c x h x w) || vector_size)
        if not self.is_lambda_set:
            self._set_lambda(lower, upper)

        # terms that have new error weights
        has_new_error_term = (lower * upper < 0).type(torch.FloatTensor)
        n_new_error_terms = int(torch.sum(has_new_error_term).item())

        shape = list(x.shape)
        shape[0] += n_new_error_terms
        transformed_x = torch.zeros(shape)

        # delta is the difference in height between top and bottom lines
        # how to compute delta ?
        #   - bottom border is lambda * x
        #   - top border has the following lambda * x + delta, with delta to determine
        #     top border is above ReLU curve in u, so lambda * u + delta >= u
        #     top border is above ReLU curve in l, so lambda * l + delta >= 0
        #     delta >= (1 - lambda) * u
        #     delta >= -lambda * l
        #     so delta >= max((1 - lambda) * u, -lambda * l), and we take equality
        #   - difference between the two lines is delta = max((1 - lambda) * u, -lambda * l)
        # the new bias is therefore (in crossing border cases) delta/2
        delta = torch.max(-self.lambda_ * lower, (1 - self.lambda_) * upper)

        # for crossing border cases, we modify bias
        # for negative case, we 0 is the new bias
        # for positive case, we don't change anything
        transformed_x[0] = (delta / 2 + self.lambda_ * x[0]) \
                           * (lower * upper < 0).type(torch.FloatTensor) \
                           + x[0] * (lower >= 0).type(torch.FloatTensor)

        # for crossing border cases, we multiply by lambda error weights
        # for positive cases, we don't change anything
        # for negative cases, it is 0
        # modifying already existing error weights
        transformed_x[1:x.shape[0]] = x[1:] * self.lambda_ \
                                      * (lower * upper < 0).type(torch.FloatTensor) \
                                      + x[1:] * (lower >= 0).type(torch.FloatTensor)

        # filling new error terms
        new_error_terms(delta/2, has_new_error_term, transformed_x, x.shape[0])

        return transformed_x

    def clip_lambda(self):
        # clips lambda into [0, 1] (might happen after gradient descent)
        self.lambda_.data.clamp_(min=0, max=1)


# general layer transformer class that, when given a layer, returns the corresponding transformed layer
class LayerTransformer:
    def __call__(self, layer, shape):
        # shape is for ReLU layers
        # layer type dependent transformation
        if isinstance(layer, Normalization):
            return TransformedNormalization(layer)
        elif isinstance(layer, nn.Linear):
            return TransformedLinear(layer)
        elif isinstance(layer, nn.Conv2d):
            return TransformedConv2D(layer)
        elif isinstance(layer, nn.Flatten):
            return TransformedFlatten()
        elif isinstance(layer, nn.ReLU):
            return TransformedReLU(shape)
        else:
            raise NotImplementedError('Unknown layer')


# transformed network
class TransformedNetwork(nn.Module):
    def __init__(self, network, eps, input_size):
        super().__init__()
        # if conv network
        self.input_size = [1, 1, input_size, input_size]
        self.initial_network_layers = network.layers
        shapes = self.get_shape_after_each_layer()

        layers = [TransformedInput(eps)]
        transformer = LayerTransformer()
        for i, layer in enumerate(self.initial_network_layers):
            layers.append(transformer(layer, shapes[i]))
            # freeze weights if layer is not ReLU layer
            for param in layers[-1].parameters():
                if isinstance(layers[-1], TransformedReLU):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        self.layers = nn.Sequential(*layers)

    def get_shape_after_each_layer(self):
        # precompute sizes of the tensor after each layer
        x = torch.zeros(self.input_size)
        shapes = [x.shape]
        for layer in self.initial_network_layers:
            x = layer.forward(x)
            shapes.append(x.shape)
        return shapes

    def clip_lambdas(self):
        # clips the lambda parameters of all the ReLU layers
        for i, layer in enumerate(self.layers):
            if isinstance(layer, TransformedReLU):
                layer.clip_lambda()

    def assert_only_relu_params_changed(self, old_params):
        # quality assessment method, to verify that only wanted params changed
        i = 0
        new_params = []
        for layer in self.layers:
            for param in layer.parameters():
                if not isinstance(layer, TransformedReLU):
                    assert torch.equal(old_params[i], param), \
                        'Param %d changed, problem with ' % i + type(layer).__name__
                new_params.append(param)
                i += 1
        return new_params

    def assert_valid_lambda_values(self):
        # assert that no lambda gets out of range
        for layer in self.layers:
            for param in layer.parameters():
                if isinstance(layer, TransformedReLU):
                    assert torch.max(param) <= 1, \
                        'Some lambda values over 1'
                    assert torch.min(param) >= 0, \
                        'Some lambda values under 0'

    def get_mean_lambda_values(self):
        # return mean lambda for every relu layer, for debugging purposes
        values = []
        for layer in self.layers:
            if isinstance(layer, TransformedReLU):
                for param in layer.parameters():
                    values.append(torch.mean(param).item())
        return values

    def get_params(self):
        params = []
        for i, layer in enumerate(self.layers):
            for param in layer.parameters():
                params.append(param)
        return params

    def forward(self, x):
        return self.layers(x)


class ZonotopeLoss:
    def __init__(self, kind='mean', **params):
        self.kind = kind

    def __call__(self, output_zonotope, true_label):
        if self.kind == 'mean':
            upper, lower = upper_lower(output_zonotope)
            # we want to prove that the lower bound for the true label is smaller than the
            # max upper bound for all the other labels, because this means the true label value
            # will always be bigger than the other labels, and so the classification will be correct
            lower_bound = lower[true_label]

            # If the lower bound of the true label is higher than the upper bound of
            # any other output, we have verified the input!
            upper[true_label] = -float('inf')  # Ignore upper bound of the true label

            # we want to minimize the max of upper bounds (mean used as max not really
            # differentiable (same issue as L1 norm, improving only one upper bound may
            # come at the cost of worsening other upper bounds, and is potentially quite slow
            # even in the very rare case that it works; using the mean ensure we try to
            # reduce all upper bounds, avoiding that problem)) and maximize the lower bound of
            # the real class
            # Set the upper bound of the true label to 0 because we don't want to take it into
            # account when computing the loss. What we care about is the difference between
            # the true label lower bound and the upper bound of the other labels. We don't care
            # about the upper bound of the true label (because it doesn't matter for verification)
            upper[true_label] = 0
            loss = torch.mean(upper) - lower_bound

        elif self.kind == 'pseudo_exponential':
            # based on the idea that we want upper bounds of classes that are higher than the lower bound of the
            # true class to be highly reduce, and the one that are lower not to be changed we use an exponential
            # penalization of the difference between classes upper bounds and true class lower bound.
            # As a purely exponential loss would lead to skyrocketing loss values, the part after 0 is replaced by
            # a polynomial function
            upper, lower = upper_lower(output_zonotope)

            diff = upper - lower[true_label]
            diff[true_label] = 0  # we don't to lower difference between upper and lower bounds for true class

            poly = 1 + diff
            loss = torch.sum(torch.exp(diff) * (diff < 0) + poly * (diff >= 0)) - lower[true_label]

        else:
            raise NotImplementedError(self.kind + " is not a possible loss type")

        return loss
