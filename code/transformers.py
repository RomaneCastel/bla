import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import FullyConnected, Conv, Normalization
from torch.distributions import Normal

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
def new_error_terms(x, condition, receiver, start_index, created_terms):
    # type: (Tensor, Tensor, Tensor, int, List[List[int]]) -> int
    # x is the ND tensor, condition a x-size boolean tensor (True if a new tensor has to be created for this value)
    # receiver is the N+1D tensor in which the newly created sparse tensor will be stored
    # start_index is the index from which the new tensors will be added
    i_error = start_index
    # when vector
    if len(x.shape) == 1:
        for i in range(x.shape[0]):
            if condition[i].item():
                receiver[i_error, i] = x[i]
                created_terms.append([i_error, i])
                i_error += 1
    # when image
    else:
        for f in range(x.shape[0]):
            for i in range(x.shape[1]):
                for j in range(x.shape[2]):
                    if condition[f, i, j].item():
                        receiver[i_error, f, i, j] = x[f, i, j]
                        created_terms.append([i_error, f])
                        i_error += 1
    return i_error


@torch.jit.script
def optimized_conv(x, created_terms, i_first_error_term_alone, weight, stride, padding):
    # type: (Tensor, List[List[int]], int, Tensor, List[int], List[int]) -> Tensor
    first_block = F.conv2d(x[:i_first_error_term_alone], weight, None, stride, padding)
    single_layers_processed = [first_block]
    for e in created_terms:
        i_error = e[0]
        feature = e[1]
        single_layers_processed.append(
            F.conv2d(x[i_error, feature].unsqueeze(0).unsqueeze(0),
                     weight[:, feature].unsqueeze(1), None, stride, padding)
        )
    return torch.cat(single_layers_processed, dim=0)


class TransformedInput(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, x, created_terms=[]):
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
        created_terms = []
        new_error_terms(error_terms, error_terms >= 0, zonotope, 1, created_terms)

        # returns zonotope, and an array for newly created terms
        return zonotope, created_terms


class TransformedNormalization(nn.Module):
    def __init__(self, normalization_layer):
        super().__init__()
        self.mean = normalization_layer.mean[0, 0, 0, 0]
        self.sigma = normalization_layer.sigma[0, 0, 0, 0]

    def forward(self, x, created_terms):
        x[0] -= self.mean
        x /= self.sigma
        return x, created_terms


# for Linear layers (both linear transformers)
class TransformedLinear(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, created_terms):
        # input of size (n_eps + 1) x in_features
        # created_terms are the error terms that was lastly created i.e. alone in their layer
        # output of size (n_eps + 1) x out_features
        # bias receives full affine transform, error weights only the linear part
        # x: (1 + h x w x n_channels) x n_channels x (width x height)
        x = F.linear(x, self.layer.weight, None)  # no bias for the moment
        if self.layer.bias is not None:
            x[0] += self.layer.bias
        return x, []  # there is no longer term alone in its error weight map


class TransformedConv2D(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, created_terms, use_created_terms=True):
        # input of shape
        # 1+n_errors x in_features x h x w
        # output of shape
        # 1+n_errors x out_features x h' x w'

        if use_created_terms and len(created_terms) > 0:
            i_first_error_term_alone = created_terms[0][0]
            output_x = optimized_conv(x, created_terms, i_first_error_term_alone,
                                      self.layer.weight, self.layer.stride, self.layer.padding)
        else:
            output_x = F.conv2d(x, self.layer.weight, bias=None, stride=self.layer.stride, padding=self.layer.padding)

        output_x[0] += self.layer.bias.unsqueeze(-1).unsqueeze(-1)

        return output_x, []


class TransformedFlatten(nn.Module):
    def __init__(self):
        super().__init__()
        self.start_dim = -3
        self.end_dim = -1

    def forward(self, x, created_terms):
        # input of shape
        # 1+n_errors x n_features x h x w
        # output of shape
        # 1+n_errors x (n_features * h * w)
        x = x.flatten(self.start_dim, self.end_dim)

        return x, []  # created terms do not matter for vectors


class TransformedReLU(nn.Module):
    def __init__(self, shape, is_last_relu_layer=False):
        super().__init__()
        self.lambda_ = nn.Parameter(torch.Tensor(shape), requires_grad=True)
        # if we have already initialized lambda
        self.is_lambda_set = False

        # WIP
        self.is_last_relu_layer = is_last_relu_layer
        if self.is_last_relu_layer:
            self.init_value = 'auto'  # 0.9999
        else:
            self.init_value = 'auto'

    def _set_lambda(self, lower, upper):
        if self.init_value == 'auto':
            # set as its optimal area value
            #  if u <= 0, l = 0
            #  if l >= 0, l = 1
            #  else l = u / (u-l)
            _lambda = (lower >= 0).type(torch.FloatTensor) \
                      + (lower * upper < 0).type(torch.FloatTensor) \
                      * upper / (upper - lower)
            # set all nans to 1
            _lambda[_lambda != _lambda] = 0.5
        else:
            _lambda = (lower >= 0).type(torch.FloatTensor) \
                      + (lower * upper < 0).type(torch.FloatTensor) \
                      * self.init_value
        self.lambda_.data = _lambda
        self.is_lambda_set = True
        std = 1
        self.lambda_gaussian = Normal(self.lambda_.data, std)

    def shuffle_lambda(self):
        self.lambda_.data = self.lambda_gaussian.sample()
        self.clip_lambda()

    def forward(self, x, created_terms):
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
        created_terms = []
        new_error_terms(delta/2, has_new_error_term, transformed_x, x.shape[0], created_terms)

        return transformed_x, created_terms

    def clip_lambda(self):
        # clips lambda into [0, 1] (might happen after gradient descent)
        self.lambda_.data.clamp_(min=0, max=1)


# general layer transformer class that, when given a layer, returns the corresponding transformed layer
class LayerTransformer:
    def __call__(self, layer, shape, is_last_relu_layer=False):
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
            return TransformedReLU(shape, is_last_relu_layer=is_last_relu_layer)
        else:
            raise NotImplementedError('Unknown layer')


# transformed network
class TransformedNetwork(nn.Module):
    def __init__(self, network, eps, input_size, n_relus_to_keep=10000):
        # n_relus_to_keep is the number of ReLU layers that will have their parameters free to move, starting from
        # the deepest layers
        super().__init__()
        # if conv network
        self.input_size = [1, 1, input_size, input_size]
        self.initial_network_layers = network.layers
        shapes = self.get_shape_after_each_layer()

        layer_names = [type(l).__name__ for l in self.initial_network_layers]
        is_relu_layer = [name == 'ReLU' for i, name in enumerate(layer_names)]
        last_relu_layer = [i for i, b in enumerate(is_relu_layer) if b][-1]

        layers = [TransformedInput(eps)]
        transformer = LayerTransformer()
        for i, layer in enumerate(self.initial_network_layers):
            layers.append(transformer(layer, shapes[i], is_last_relu_layer=(last_relu_layer==i)))

        relu_layer_to_freeze = is_relu_layer
        if n_relus_to_keep >= 1:
            for i in range(len(relu_layer_to_freeze)-1, -1, -1):
                if relu_layer_to_freeze[i]:
                    relu_layer_to_freeze[i] = False
                    n_relus_to_keep -= 1
                if n_relus_to_keep == 0:
                    break
        relu_layer_to_keep = [not e for e in relu_layer_to_freeze]
        # note: other layers than ReLU are marked as True but we will not keep active their gradients

        for i, layer in enumerate(layers):
            # freeze weights if layer is not ReLU layer
            for param in layer.parameters():
                if isinstance(layer, TransformedReLU):
                    param.requires_grad = relu_layer_to_keep[i]
                else:
                    param.requires_grad = False
        self.layers = nn.Sequential(*layers)

    def shuffle_lambda(self, n_relu_to_shuffle):
        i_relu_to_shuffle = 0
        for layer in self.layers:
            if isinstance(layer, TransformedReLU):
                layer.shuffle_lambda()
                i_relu_to_shuffle += 1
                if i_relu_to_shuffle >= n_relu_to_shuffle:
                    break

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

    def get_params(self, only_relu=False):
        params = []
        for i, layer in enumerate(self.layers):
            for param in layer.parameters():
                if not only_relu or type(layer).__name__ == "TransformedReLU":
                    params.append(param)
        return params

    def forward(self, x):
        created_terms = []
        for i, layer in enumerate(self.layers):
            x, created_terms = layer.forward(x, created_terms)
        return x


class ZonotopeLoss:
    def __init__(self, kind='mean'):
        self.kind = kind

    def __call__(self, upper, lower, output_zonotope, true_label):
        if self.kind == 'mean':
            # we want to prove that the lower bound for the true label is smaller than the
            # max upper bound for all the other labels, because this means the true label value
            # will always be bigger than the other labels, and so the classification will be correct
            lower_bound = lower[true_label]

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
            # loss = torch.mean(upper) - lower_bound
            loss = torch.mean(upper)

        elif self.kind == 'pseudo_exponential':
            # based on the idea that we want upper bounds of classes that are higher than the lower bound of the
            # true class to be highly reduce, and the one that are lower not to be changed we use an exponential
            # penalization of the difference between classes upper bounds and true class lower bound.
            # As a purely exponential loss would lead to skyrocketing loss values, the part after 0 is replaced by
            # a polynomial function

            diff = upper - lower[true_label]
            diff[true_label] = 0  # we don't to lower difference between upper and lower bounds for true class

            poly = 1 + diff
            loss = torch.sum(torch.exp(diff) * (diff < 0) + poly * (diff >= 0)) - lower[true_label]

        # WIP
        elif self.kind == 'weighted_L2':
            diff = upper - lower[true_label]
            diff[true_label] = 0
            poly = 1 + diff
            class_weight = torch.exp(diff) * (diff < 0) + poly * (diff >= 0)
            loss = torch.sum(class_weight * torch.sum(output_zonotope[1:] * output_zonotope[1:], dim=0))
            lower_bound = lower[true_label]
            upper[true_label] = 0
            loss += torch.mean(upper) - lower_bound

        # WIP
        elif self.kind == 'weighted_L1':
            diff = upper - lower[true_label]
            diff[true_label] = 0
            poly = 1 + diff
            class_weight = torch.exp(diff) * (diff < 0) + poly * (diff >= 0)
            loss = torch.sum(class_weight * torch.sum(torch.abs(output_zonotope[1:]), dim=0))
            lower_bound = lower[true_label]
            upper[true_label] = 0
            loss += torch.mean(upper) - lower_bound

        else:
            raise NotImplementedError(self.kind + " is not a possible loss type")

        return loss