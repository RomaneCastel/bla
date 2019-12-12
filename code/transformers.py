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
    upper = zonotope[:, 0] + torch.sum(torch.abs(zonotope[:, 1:]), dim=1)
    lower = zonotope[:, 0] - torch.sum(torch.abs(zonotope[:, 1:]), dim=1)
    return upper, lower

@torch.jit.script
def new_error_terms(x, condition, receiver, start_index, created_terms):
    # type: (Tensor, Tensor, Tensor, int, List[List[int]]) -> int
    # x is the ND tensor, condition a x-size boolean tensor (True if a new tensor has to be created for this value)
    # receiver is the N+1D tensor in which the newly created sparse tensor will be stored
    # start_index is the index from which the new tensors will be added
    i_error = start_index
    # when vector
    if len(x.shape) == 2:
        for i in range(x.shape[1]):
            if condition[0, i].item():
                receiver[:, i_error, i] = x[:, i]
                created_terms.append([i_error, i])
                i_error += 1
    # when image
    else:
        for f in range(x.shape[1]):
            for i in range(x.shape[2]):
                for j in range(x.shape[3]):
                    if condition[0, f, i, j].item():
                        receiver[0, i_error, f, i, j] = x[0, f, i, j]
                        created_terms.append([i_error, f, i, j])
                        i_error += 1
    return i_error


class Zonotope:
    # object representing zonotopes
    def __init__(self, x=None, n_error_terms=0):
        if x is not None:
            self.zonotope = torch.zeros(
                [x.shape[0],
                 1+n_error_terms,  # 0 for value, all the others will be for error terms
                 x.shape[1],
                 x.shape[2],
                 x.shape[3]
                 ],
                dtype=x.dtype)
        self.created_terms = []
        self.last_error_term = 1

    def add_space(self, n=None):
        # if the number of new items is not specified, consider that we have to add one error term for each value
        # of the bias tensor
        if n is None:
            if len(self.zonotope.shape) == 5:
                n = self.zonotope.shape[2] * self.zonotope.shape[3] * self.zonotope.shape[4]
            else:
                n = self.zonotope.shape[2]
        # creates a new zonotope
        new_shape = list(self.zonotope.shape)
        new_shape[1] += n
        new_zonotope = torch.zeros(new_shape, dtype=self.zonotope.dtype)
        # adds previous values
        new_zonotope[:, :self.zonotope.shape[1]] = self.zonotope
        # replaces old zonotope by the new
        self.zonotope = new_zonotope

    def fill_bias(self, x):
        self.zonotope[0, 0] = x

    def get_zonotope(self):
        return self.zonotope[:, :self.last_error_term]

    def new_error_terms(self, error_terms, condition):
        # checks that there is enough space to put the new error terms
        n_new_error_terms = torch.sum(condition).item()
        if self.last_error_term + n_new_error_terms > self.zonotope.shape[1]:
            self.add_space()
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
        zonotope = Zonotope(x)

        # fills biases
        zonotope.fill_bias(x + nn.functional.relu(self.eps - x)/2 - nn.functional.relu(x-(1-self.eps))/2)

        # creates error terms
        error_terms = self.eps - nn.functional.relu(self.eps - x)/2 - nn.functional.relu(x-(1-self.eps))/2
        zonotope.new_error_terms(error_terms, error_terms >= 0)

        return zonotope


class TransformedNormalization(nn.Module):
    def __init__(self, normalization_layer):
        super().__init__()
        self.mean = normalization_layer.mean[0, 0, 0, 0]
        self.sigma = normalization_layer.sigma[0, 0, 0, 0]

    def forward(self, x):
        x.zonotope[:, 0, :, :, :] -= self.mean
        x.zonotope /= self.sigma
        return x


# for Linear layers (both linear transformers)
class TransformedLinear(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        # input of size batch_size x (n_eps + 1) x in_features
        # output of size batch_size x (n_eps + 1) x out_features
        # bias receives full affine transform, error weights only the linear part
        # x: batch_size x (1 + h x w x n_channels) x n_channels x (width x height)
        output = F.linear(x.zonotope, self.layer.weight, None)  # no bias for the moment
        if self.layer.bias is not None:
            output[:, 0, :] += self.layer.bias
        x.zonotope = output
        return x


class TransformedConv2D(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        # input of shape
        # batch_size x 1+n_errors x in_features x h x w
        # output of shape
        # batch_size x 1+n_errors x out_features x h' x w'
        # x: batch_size x (1 + h x w x n_channels) x n_channels x width x height
        in_features = self.layer.weight.shape[1]
        shape_output = self.layer.forward(
            torch.zeros(
                [x.zonotope.shape[2], in_features, x.zonotope.shape[3], x.zonotope.shape[4]]
            )
        ).shape
        out_features = shape_output[1]
        output = torch.zeros(
            [x.zonotope.shape[0], x.zonotope.shape[1], out_features, shape_output[2], shape_output[3]],
            dtype=x.zonotope.dtype
        )
        output[:, 0, :, :, :] = self.layer.forward(x.zonotope[:, 0, :, :, :]).squeeze()
        for i in range(1, x.zonotope.shape[1]):
            # no bias for error weight
            output[:, i, :, :, :] = self.layer.forward(x.zonotope[:, i, :, :, :])
            output[:, i, :, :, :] -= self.layer.bias.unsqueeze(-1).unsqueeze(-1)

        x.zonotope = output

        return x


class TransformedFlatten(nn.Module):
    def __init__(self):
        super().__init__()
        self.start_dim = -3
        self.end_dim = -1

    def forward(self, x):
        # input of shape
        # batch_size x 1+n_errors x n_features x h x w
        # output of shape
        # batch_size x 1+n_errors x (n_features * h * w)
        # x: batch_size x (1 + h x w x n_channels) x n_channels x width x height
        x.zonotope = x.zonotope.flatten(self.start_dim, self.end_dim)

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
        # x: batch_size x (1 + h x w x n_channels) x n_channels x width x height
        # or x: batch_size x (1 + h x w x n_channels) x n_channels x (width x height)
        # computes minimum and maximum boundaries for every input value
        # upper and lower bound are tensor of size batch_size x n_features x ((h x w) || vector_size)
        # creates the n_features x batch_size x vector_size lambda values
        upper, lower = upper_lower(x.zonotope)

        # lambda has a size of batch_size x ((h x w) || vector_size)
        if not self.is_lambda_set:
            self._set_lambda(lower, upper)

        transformed_x = torch.zeros(x.zonotope.shape)
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
        transformed_x[:, 0] = (delta / 2 + self.lambda_ * x.zonotope[:, 0]) \
                              * (lower * upper < 0).type(torch.FloatTensor) \
                              + x.zonotope[:, 0] * (lower >= 0).type(torch.FloatTensor)

        # for crossing border cases, we multiply by lambda error weights
        # for positive cases, we don't change anything
        # for negative cases, it is 0
        # modifying already existing error weights
        transformed_x[:, 1:] = x.zonotope[:, 1:] * self.lambda_.unsqueeze(1) \
                               * (lower * upper < 0).type(torch.FloatTensor) \
                               + x.zonotope[:, 1:] * (lower >= 0).type(torch.FloatTensor)

        x.zonotope = transformed_x

        # adding new error weights
        # correct as batch_size is equal to 1 here
        has_new_error_term = (lower * upper < 0).type(torch.FloatTensor)

        # filling new error terms
        x.new_error_terms(delta/2, has_new_error_term)

        return x

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
            upper, lower = upper_lower(output_zonotope.zonotope)
            # we want to prove that the lower bound for the true label is smaller than the
            # max upper bound for all the other labels, because this means the true label value
            # will always be bigger than the other labels, and so the classification will be correct
            lower_bound = lower[0, true_label]

            # If the lower bound of the true label is higher than the upper bound of
            # any other output, we have verified the input!
            upper[0, true_label] = -float('inf')  # Ignore upper bound of the true label

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
            upper[0, true_label] = 0
            loss = torch.mean(upper) - lower_bound

        elif self.kind == 'pseudo_exponential':
            # based on the idea that we want upper bounds of classes that are higher than the lower bound of the
            # true class to be highly reduce, and the one that are lower not to be changed we use an exponential
            # penalization of the difference between classes upper bounds and true class lower bound.
            # As a purely exponential loss would lead to skyrocketing loss values, the part after 0 is replaced by
            # a polynomial function
            upper, lower = upper_lower(output_zonotope.zonotope)

            diff = upper[0, :] - lower[0, true_label]
            diff[true_label] = 0  # we don't to lower difference between upper and lower bounds for true class

            poly = 1 + diff
            loss = torch.sum(torch.exp(diff) * (diff < 0) + poly * (diff >= 0)) - lower[0, true_label]

        else:
            raise NotImplementedError(self.kind + " is not a possible loss type")

        return loss
