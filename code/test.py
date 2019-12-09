import unittest
import torch
import torch.nn as nn
from transformers import TransformedInput, TransformedNetwork, TransformedReLU, TransformedFlatten, \
    TransformedNormalization, TransformedLinear, TransformedConv2D, LayerTransformer, upper_lower
from networks import Normalization, FullyConnected, Conv
from torch.nn import Linear
import numpy as np


"""
Testing suite for transformers
Unit tests to quickly verify that the transformers have no obvious bugs
"""


class TransformedInputTester(unittest.TestCase):
    def setUp(self):
        input = torch.FloatTensor([[[[0., 1.], [0.3, 0.7]]]])
        eps = 0.1
        transformed_input = TransformedInput(eps)
        self.output = transformed_input.forward(input)
        self.expected_output = torch.zeros([1, 5, 1, 2, 2])
        self.expected_output[0, 0, 0, 0, 0] = eps / 2
        self.expected_output[0, 0, 0, 0, 1] = 1 - eps / 2
        self.expected_output[0, 0, 0, 1, 0] = 0.3
        self.expected_output[0, 0, 0, 1, 1] = 0.7
        self.expected_output[0, 1, 0, 0, 0] = eps / 2
        self.expected_output[0, 2, 0, 0, 1] = eps / 2
        self.expected_output[0, 3, 0, 1, 0] = eps
        self.expected_output[0, 4, 0, 1, 1] = eps

    def test_size(self):
        assert self.expected_output.size() == self.output.size(), \
            "Different sizes"

    def test_bias(self):
        assert self.expected_output[:, 0].equal(self.output[:, 0]), \
            "Error for bias term"

    def test_error(self):
        exp = self.expected_output[:, 1:]
        out = self.output[:, 1:]
        diff = torch.sum(exp - out).item()
        assert diff < 1e-4, \
            "Error for error weights"


class TransformedNormalizationTester(unittest.TestCase):
    def setUp(self):
        normalization_layer = Normalization('cpu')
        mean = normalization_layer.mean
        sigma = normalization_layer.sigma

        input = torch.FloatTensor([[[[0., 1.], [0.3, 0.7]]]])
        eps = 0.1
        transformed_input = TransformedInput(eps)
        input = transformed_input.forward(input)
        transformed_normalization = TransformedNormalization(normalization_layer)
        self.output = transformed_normalization.forward(input)

        self.expected_output = torch.zeros([1, 5, 1, 2, 2])
        self.expected_output[0, 0, 0, 0, 0] = (eps / 2 - mean) / sigma
        self.expected_output[0, 0, 0, 0, 1] = (1 - eps / 2 - mean) / sigma
        self.expected_output[0, 0, 0, 1, 0] = (0.3 - mean) / sigma
        self.expected_output[0, 0, 0, 1, 1] = (0.7 - mean) / sigma
        self.expected_output[0, 1, 0, 0, 0] = eps / 2 / sigma
        self.expected_output[0, 2, 0, 0, 1] = eps / 2 / sigma
        self.expected_output[0, 3, 0, 1, 0] = eps / sigma
        self.expected_output[0, 4, 0, 1, 1] = eps / sigma

    def test_size(self):
        assert self.expected_output.size() == self.output.size(), \
            "Different sizes"

    def test_bias(self):
        assert self.expected_output[:, 0].equal(self.output[:, 0]), \
            "Error for bias term"

    def test_error(self):
        assert self.expected_output[:, 1:].equal(self.output[:, 1:]), \
            "Error for error weights"


class TransformedLinearTester(unittest.TestCase):
    def setUp(self):
        # based on example given in Problem 2 of zonotope exercise + added bias
        linear_layer = Linear(2, 2)
        linear_layer.weight[0, 0] = 1
        linear_layer.weight[0, 1] = 2
        linear_layer.weight[1, 0] = -1
        linear_layer.weight[1, 1] = 1
        linear_layer.bias[0] = 0
        linear_layer.bias[1] = 1

        transformed_linear = TransformedLinear(linear_layer)

        input = torch.zeros([1, 3, 2])
        input[0, 0, 0] = 4
        input[0, 0, 1] = 3
        input[0, 1, 0] = 2
        input[0, 1, 1] = 1
        input[0, 2, 0] = 1
        input[0, 2, 1] = 2
        self.output = transformed_linear.forward(input)

        self.expected_output = torch.zeros([1, 3, 2])
        self.expected_output[0, 0, 0] = 10
        self.expected_output[0, 0, 1] = 0
        self.expected_output[0, 1, 0] = 4
        self.expected_output[0, 1, 1] = -1
        self.expected_output[0, 2, 0] = 5
        self.expected_output[0, 2, 1] = 1

    def test_size(self):
        assert self.expected_output.size() == self.output.size(), \
            "Different sizes"

    def test_bias(self):
        assert self.expected_output[:, 0].equal(self.output[:, 0]), \
            "Error for bias term"

    def test_error(self):
        assert self.expected_output[:, 1:].equal(self.output[:, 1:]), \
            "Error for error weights"


class TransformedConv2DTester(unittest.TestCase):
    def setUp(self):
        kernel_filter = nn.Parameter(
            torch.FloatTensor(
                [[[
                    [-1, 0, 1],
                    [1, 1, 0],
                    [0, 1, 1]
                ]]]
            )
        )
        bias = 1
        padding = 1
        stride = 1
        conv_layer = nn.Conv2d(1, 1, 3, stride=stride, padding=padding, bias=True)
        conv_layer.weight = kernel_filter
        conv_layer.bias[0] = bias

        image = [[
            [-2, 1, 3],
            [0, 1, -1],
            [1, -2, 1]
        ]]
        # only one for simpler test
        error_weight = [[
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0]
        ]]
        input = torch.FloatTensor([[image, error_weight]])
        transformed_conv = TransformedConv2D(conv_layer)

        self.expected_output = torch.FloatTensor([
            [
                [[[0, 0, 4],
                 [1, 6, 1],
                 [3, -1, -1]]],
                [[[1, 0, 0],
                 [1, 1, 0],
                 [0, -1, 0]]]
            ],
        ])

        self.output = transformed_conv.forward(input)

    def test_size(self):
        assert self.expected_output.size() == self.output.size(), \
            "Different sizes"

    def test_bias(self):
        assert self.expected_output[:, 0].equal(self.output[:, 0]), \
            "Error for bias term"

    def test_error(self):
        assert self.expected_output[:, 1:].equal(self.output[:, 1:]), \
            "Error for error weights"


class TransformedFlattenTester(unittest.TestCase):
    def setUp(self):
        image = [[
            [-2, 1, 3],
            [0, 1, -1],
            [1, -2, 1]
        ]]
        error_weight = [[
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0]
        ]]
        input = torch.FloatTensor([[image, error_weight]])
        transformed_flatten = TransformedFlatten()

        self.output = transformed_flatten.forward(input)

        self.expected_output = torch.FloatTensor([
            [
                [-2, 1, 3, 0, 1, -1, 1, -2, 1],
                [0, 0, 0, 1, 0, 0, 0, 0, 0]
            ]
        ])

    def test_size(self):
        assert self.expected_output.size() == self.output.size(), \
            "Different sizes"

    def test_bias(self):
        assert self.expected_output[:, 0].equal(self.output[:, 0]), \
            "Error for bias term"

    def test_error(self):
        assert self.expected_output[:, 1:].equal(self.output[:, 1:]), \
            "Error for error weights"


class TransformedReluTester(unittest.TestCase):
    def setUp(self):
        image_input = [[
            [0, 1],
            [-0.5, 0.5]
        ]]
        image_error_weight = [[
            [1, -1],
            [0.5, 0]
        ]]
        image_zonotope = torch.FloatTensor([[image_input, image_error_weight]])

        vector_input = [0, 1, -0.5, 0]
        vector_error_weight = [1, -1, 0.5, 0]
        vector_zonotope = torch.FloatTensor([[vector_input, vector_error_weight]])

        transformed_relu = TransformedReLU(torch.Size([1, 1, 2, 2]))
        self.image_output = transformed_relu.forward(image_zonotope)
        self.expected_image_output = torch.FloatTensor([[
            [[
                [0.25, 1],
                [0, 0.5]
            ]],
            [[
                [0.5, -1],
                [0, 0]
            ]],
            [[
                [0.25, 0],
                [0, 0]
            ]]
        ]])

        transformed_relu = TransformedReLU(torch.Size([1, 4]))
        self.vector_output = transformed_relu.forward(vector_zonotope)
        self.expected_vector_output = torch.FloatTensor([
            [
                [0.25, 1, 0, 0],
                [0.5, -1, 0, 0],
                [0.25, 0, 0, 0]
            ]
        ])

        lambda_vector_input = [0, 0, 1, 1]
        lambda_vector_error_weight = [1, 1, 0, 2]
        lambda_vector_zonotope = torch.FloatTensor([[lambda_vector_input, lambda_vector_error_weight]])
        transformed_relu = TransformedReLU(torch.Size([1, 3]))
        transformed_relu.lambda_.data = torch.FloatTensor([[0, 1, 0.5, 1/3]])
        transformed_relu.is_lambda_set = True
        self.lambda_vector_output = transformed_relu.forward(lambda_vector_zonotope)
        self.expected_lambda_vector_output = torch.FloatTensor([
            [
                [0.5, 0.5, 1, 1],
                [0, 1, 0, 2/3],
                [0.5, 0, 0, 0],
                [0, 0.25, 0, 0],
                [0, 0, 0, 1/3]
            ]
        ])


    def test_size_image(self):
        assert self.expected_image_output.size() == self.image_output.size(), \
            "Different sizes (for image input)"

    def test_bias_image(self):
        assert self.expected_image_output[:, 0].equal(self.image_output[:, 0]), \
            "Error for bias term (for image input)"

    def test_error_image(self):
        assert self.expected_image_output[:, 1:].equal(self.image_output[:, 1:]), \
            "Error for error weights (for image input)"

    def test_size_vector(self):
        assert self.expected_vector_output.size() == self.vector_output.size(), \
            "Different sizes (for vector input)"

    def test_bias_vector(self):
        assert self.expected_vector_output[:, 0].equal(self.vector_output[:, 0]), \
            "Error for bias term (for vector input)"

    def test_error_vector(self):
        assert self.expected_vector_output[:, 1:].equal(self.vector_output[:, 1:]), \
            "Error for error weights (for vector input)"

    def test_bias_lambda_vector(self):
        exp = self.expected_lambda_vector_output[:, 0]
        out = self.lambda_vector_output[:, 0]
        diff = torch.sum(exp - out).item()
        assert diff < 1e-4, \
            "Error for bias term (for specific lambda vector input)"

    def test_error_lambda_vector(self):
        exp = self.expected_lambda_vector_output[:, 1:]
        out = self.lambda_vector_output[:, 1:]
        diff = torch.sum(exp - out).item()
        assert diff < 1e-4, \
            "Error for error weights (for specific lambda vector input)"


class TransformedNetworkTester(unittest.TestCase):
    def setUp(self):
        self.fc_network = FullyConnected('cpu', 28, [100, 10]).to('cpu')
        self.conv_network = Conv('cpu', 28, [(32, 4, 2, 1)], [100, 10], 10).to('cpu')

    def test_no_error_transformation_fc(self):
        TransformedNetwork(self.fc_network, 0.01, 28)
        assert True

    def test_no_error_transformation_fc(self):
        TransformedNetwork(self.conv_network, 0.01, 28)
        assert True


class ToyNetworkTester(unittest.TestCase):
    def setUp(self):
        layers = [nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2)]
        layers[0].weight.data = torch.FloatTensor([[1, 1], [1, -1]])
        layers[0].bias.data = torch.FloatTensor([0, 0])
        layers[2].weight.data = torch.FloatTensor([[1, 1], [1, -1]])
        layers[2].bias.data = torch.FloatTensor([0, 0])
        self.net = nn.Sequential(*layers)
        self.transformed_net = nn.Sequential(*[LayerTransformer()(layer, []) for layer in layers])
        self.input = torch.FloatTensor([
            [
                [0.3, 0.4],
                [0.3, 0],
                [0, 0.3]
            ]
        ])
        self.output = self.transformed_net(self.input)
        self.expected_output = torch.FloatTensor([
            [
                [0.7+0.25*0.5/1.2, 0.7-0.25*0.5/1.2],
                [0.425, 0.175],
                [0.175, 0.425],
                [0.35/2.4, -0.35/2.4]
            ]
        ])

    def test_right_bias_output(self):
        exp = self.expected_output[:, 0]
        out = self.output[:, 0]
        diff = torch.sum(exp - out).item()
        assert diff < 1e-4, \
            "Wrong bias output"

    def test_right_error_weight_output(self):
        exp = self.expected_output[:, 1:]
        out = self.output[:, 1:]
        diff = torch.sum(exp - out).item()
        assert diff < 1e-4, \
            "Wrong error weights output"

    def test_is_verified(self):
        # we want to show lower o1 >= upper o2
        o1 = self.output[:, :, 0]
        l1, _ = upper_lower(o1)
        o2 = self.output[:, :, 1]
        _, o2 = upper_lower(o2)
        assert l1 > o2, "Should have certified"


if __name__ == '__main__':
    unittest.main()
