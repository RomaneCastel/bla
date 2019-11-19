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
        # x is of size batch_size x 1 x width x height
        # output is of size batch_size x (1 + h x w) x width x height
        zonotope = torch.zeros(
            [x.shape[0], 1+x.shape[2]*x.shape[3], x.shape[2], x.shape[3]],
            dtype=x.dtype)

        for i_batch in range(x.shape[0]):
            i_error = 1
            for i_h in range(x.shape[2]):
                for i_w in range(x.shape[3]):
                    pixel_value = x[i_batch, 0, i_h, i_w]
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
                    zonotope[i_batch, 0, i_h, i_w] = new_pixel_value
                    # add error value
                    zonotope[i_batch, i_error, i_h, i_w] = new_error_term
                    i_error += 1

        return zonotope


class TransformedNormalization(nn.Module):
    def __init__(self, normalization_layer):
        super().__init__()
        self.mean = normalization_layer.mean[0, 0, 0, 0]
        self.sigma = normalization_layer.sigma[0, 0, 0, 0]

    def forward(self, x):
        x[:, 0, :, :] -= self.mean
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
        return output


class TransformedConv2D(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        shape_output = self.layer.forward(torch.zeros([1, 1, x.shape[2], x.shape[3]])).shape
        output = torch.zeros([x.shape[0], x.shape[1], shape_output[2], shape_output[3]], dtype=x.dtype)
        output[:, 0, :, :] = self.layer.forward(x[:, 0, :, :].unsqueeze(1)).squeeze()
        for i in range(1, x.shape[1]):
            # no bias for error weight
            output[:, i, :, :] = self.layer.forward(x[:, i, :, :].unsqueeze(1)).squeeze() - self.layer.bias

        return output


class TransformedFlatten(nn.Module):
    def __init__(self):
        super().__init__()
        self.start_dim = -2
        self.end_dim = -1

    def forward(self, x):
        # you just want to flatten the last 2 dimensions
        return x.flatten(self.start_dim, self.end_dim)


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
                       + (lower < 0).type(torch.FloatTensor) \
                       * (upper > 0).type(torch.FloatTensor) \
                       * upper / (upper - lower)
        # set all nans to 0
        _lambda[self.lambda_ != self.lambda_] = 0
        self.lambda_.data = _lambda
        self.is_lambda_set = True

    def forward(self, x):
        # computes minimum and maximum boundaries for every input value
        # upper and lower bound are tensor of size batch_size x ((h x w) || vector_size)
        # creates the batch_size x vector_size lambda values
        upper, lower = upper_lower(x)

        # lambda has a size of batch_size x ((h x w) || vector_size)
        if not self.is_lambda_set:
            self._set_lambda(lower, upper)

        transformed_x = torch.zeros(x.shape)
        # modifying bias: bias = lambda * bias - lambda * lower / 2
        transformed_x[:, 0] = self.lambda_ * x[:, 0] - self.lambda_ * lower / 2
        # modifying already existing error weights
        transformed_x[:, 1:] = x[:, 1:] * self.lambda_
        # adding new error weights
        # correct as batch_size is equal to 1 here
        n_old_error_weights = x.shape[1]
        has_new_error_term = (lower < 0).type(torch.FloatTensor) \
                             * (upper > 0).type(torch.FloatTensor)
        n_new_error_weights = int(torch.sum(has_new_error_term).item())
        if len(x.shape) == 3:
            final_x = torch.cat([transformed_x, torch.zeros([x.shape[0], n_new_error_weights, x.shape[2]])], dim=1)
        else:
            final_x = torch.cat([transformed_x, torch.zeros([x.shape[0], n_new_error_weights, x.shape[2], x.shape[3]])], dim=1)
        # filling new error terms

        # when vector
        if len(x.shape) == 3:
            i_error = n_old_error_weights
            for i in range(x.shape[2]):
                if has_new_error_term[0, i] == 1:
                    final_x[:, i_error, i] = -self.lambda_[:, i] * lower[:, i] / 2
                    i_error += 1
        # when image
        else:
            i_error = n_old_error_weights
            for i in range(x.shape[2]):
                for j in range(x.shape[3]):
                    if has_new_error_term[0, i, j] == 1:
                        final_x[:, i_error, i, j] = -self.lambda_[:, i, j] * lower[:, i, j] / 2
                        i_error += 1

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
            return TransformedLinear(layer)
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
        if isinstance(network, Conv):
            self.input_size = [1, 1, input_size, input_size]
        else:
            self.input_size = [1, 1, input_size * input_size]
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

    def forward(self, x):
        return self.layers(x)
