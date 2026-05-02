import torch

from model.activation_functions import ReLU, ReLU_Squared, GeLU


def test_relu_clips_negatives():
    x = torch.tensor([-1.0, 0.0, 2.0])
    assert torch.equal(ReLU()(x), torch.tensor([0.0, 0.0, 2.0]))


def test_relu_squared_squares_positives():
    x = torch.tensor([-1.0, 0.0, 2.0])
    assert torch.equal(ReLU_Squared()(x), torch.tensor([0.0, 0.0, 4.0]))


def test_gelu_preserves_shape():
    x = torch.randn(4, 8)
    assert GeLU()(x).shape == x.shape
