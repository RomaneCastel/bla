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


# utils function: compute lower and upper bounds of a zonotope
def upper_lower(zonotope):
    upper = zonotope[:, 0] + torch.sum(torch.abs(zonotope[:, 1:]), dim=1)
    lower = zonotope[:, 0] - torch.sum(torch.abs(zonotope[:, 1:]), dim=1)
    return upper, lower


class TransformedInput(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # creates all the height x width error matrices/vectors
        # x is of size batch_size x n_features x width x height
        # output is of size batch_size x (1 + h x w x n_features)
        #                          x n_features x width x height
        zonotope = torch.zeros(
            [x.shape[0],
             1+x.shape[1]*x.shape[2]*x.shape[3],
             x.shape[1],
             x.shape[2],
             x.shape[3]
            ],
            dtype=x.dtype)

        for i_batch in range(x.shape[0]):
            for f in range(x.shape[1]):
                i_error = 1
                for i_h in range(x.shape[2]):
                    for i_w in range(x.shape[3]):
                        pixel_value = x[i_batch, f, i_h, i_w]
                        error_term = self.eps
                        # modifies them if pixel_value is out of [eps, 1-eps]
                        if pixel_value < error_term:
                            new_pixel_value = (pixel_value + error_term) / 2.
                            new_error_term = (error_term + pixel_value) / 2.
                        elif pixel_value > 1 - error_term:
                            new_pixel_value = (pixel_value + 1 - error_term) / 2.
                            new_error_term = (1 - pixel_value + error_term) / 2.
                        else:
                            new_error_term = error_term
                            new_pixel_value = pixel_value
                        # add bias value
                        zonotope[i_batch, 0, f, i_h, i_w] = new_pixel_value
                        # add error value
                        zonotope[i_batch, i_error, f, i_h, i_w] = new_error_term
                        i_error += 1

        return zonotope


class TransformedNormalization(nn.Module):
    def __init__(self, normalization_layer):
        super().__init__()
        self.mean = normalization_layer.mean[0, 0, 0, 0]
        self.sigma = normalization_layer.sigma[0, 0, 0, 0]

    def forward(self, x):
        x[:, 0, :, :, :] -= self.mean
        x /= self.sigma
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
        output = F.linear(x, self.layer.weight, None)  # no bias for the moment
        if self.layer.bias is not None:
            output[:, 0, :] += self.layer.bias

        print(output)
        return output


class TransformedConv2D(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        # input of shape
        # batch_size x 1+n_errors x in_features x h x w
        # output of shape
        # batch_size x 1+n_errors x out_features x h' x w'
        in_features = self.layer.weight.shape[1]
        shape_output = self.layer.forward(torch.zeros([x.shape[2], in_features, x.shape[3], x.shape[4]])).shape
        out_features = shape_output[1]
        output = torch.zeros([x.shape[0], x.shape[1], out_features, shape_output[2], shape_output[3]], dtype=x.dtype)
        output[:, 0, :, :, :] = self.layer.forward(x[:, 0, :, :, :]).squeeze()
        for i in range(1, x.shape[1]):
            # no bias for error weight
            output[:, i, :, :, :] = self.layer.forward(x[:, i, :, :, :])
            output[:, i, :, :, :] -= self.layer.bias.unsqueeze(-1).unsqueeze(-1)

        print(output)
        return output


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
        final_x = x.flatten(self.start_dim, self.end_dim)
        print(final_x)
        return final_x


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
        # computes minimum and maximum boundaries for every input value
        # upper and lower bound are tensor of size batch_size x n_features x ((h x w) || vector_size)
        # creates the n_features x batch_size x vector_size lambda values
        upper, lower = upper_lower(x)

        # lambda has a size of batch_size x ((h x w) || vector_size)
        if not self.is_lambda_set:
            self._set_lambda(lower, upper)

        transformed_x = torch.zeros(x.shape)
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
        transformed_x[:, 0] = (delta / 2 + self.lambda_ * x[:, 0]) * (lower * upper < 0).type(torch.FloatTensor) \
            + x[:, 0] * (lower >= 0).type(torch.FloatTensor)

        # for crossing border cases, we multiply by lambda error weights
        # for positive cases, we don't change anything
        # for negative cases, it is 0
        # modifying already existing error weights
        transformed_x[:, 1:] = x[:, 1:] * self.lambda_.unsqueeze(1) * (lower * upper < 0).type(torch.FloatTensor) \
            + x[:, 1:] * (lower >= 0).type(torch.FloatTensor)

        # adding new error weights
        # correct as batch_size is equal to 1 here
        n_old_error_weights = x.shape[1]
        has_new_error_term = (lower * upper < 0).type(torch.FloatTensor)
        n_new_error_weights = int(torch.sum(has_new_error_term).item())
        # create new tensor that is able to host all the new error terms
        if len(x.shape) == 3:
            final_x = torch.cat([transformed_x, torch.zeros([x.shape[0], n_new_error_weights, x.shape[2]])], dim=1)
        else:
            # TODO take a time to execute, how to make it faster ?
            final_x = torch.zeros(x.shape[0], x.shape[1] + n_new_error_weights, x.shape[2], x.shape[3], x.shape[4])
            final_x[:, :x.shape[1]] = transformed_x

        # filling new error terms
        # when vector
        if len(x.shape) == 3:
            i_error = n_old_error_weights
            for i in range(x.shape[2]):
                if has_new_error_term[0, i] == 1:
                    final_x[:, i_error, i] = delta[:, i] / 2
                    i_error += 1
        # when image
        else:
            i_error = n_old_error_weights
            for f in range(x.shape[2]):
                for i in range(x.shape[3]):
                    for j in range(x.shape[4]):
                        if has_new_error_term[0, f, i, j].item():
                            final_x[0, i_error, f, i, j] = delta[0, f, i, j] / 2
                            i_error += 1
        print(final_x)
        return final_x

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
