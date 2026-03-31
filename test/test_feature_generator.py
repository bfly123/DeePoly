import numpy as np
import torch
from torch import nn

from src.abstract_class.features_generator import FeatureGenerator


class DummyFeatureModel(nn.Module):
    def forward(self, x):
        h = torch.cat((x[:, :1], 2.0 * x[:, :1]), dim=1)
        out = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        return h, out


def test_sniper_features_without_derivatives_matches_expected_basis():
    x = np.array([[0.0], [2.0]])
    features = FeatureGenerator.sniper_features(x, degree=[2], derivative=[0])
    expected = np.array([[1.0, 0.0, 0.0], [1.0, 2.0, 4.0]])
    assert np.allclose(features, expected)


def test_sniper_features_with_derivative_returns_polynomial_gradients():
    x = np.array([[1.0], [3.0]])
    features = FeatureGenerator.sniper_features(x, degree=[2], derivative=[1])
    expected = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 6.0]])
    assert np.allclose(features, expected)


def test_spotter_features_returns_hidden_features_and_derivatives_on_cpu():
    x = np.array([[0.25], [0.75]])
    model = DummyFeatureModel().to(dtype=torch.float64)

    features = FeatureGenerator.spotter_features(
        x,
        x_min=np.array([0.0]),
        x_max=np.array([1.0]),
        model=model,
        derivative=[0],
        device="cpu",
    )
    derivatives = FeatureGenerator.spotter_features(
        x,
        x_min=np.array([0.0]),
        x_max=np.array([1.0]),
        model=model,
        derivative=[1],
        device="cpu",
    )

    assert np.allclose(features, np.array([[0.25, 0.5], [0.75, 1.5]]))
    assert np.allclose(derivatives, np.array([[1.0, 2.0], [1.0, 2.0]]))


def test_init_coefficients_returns_last_linear_weight_matrix():
    model = nn.Sequential(
        nn.Linear(1, 3, bias=False, dtype=torch.float64),
        nn.Tanh(),
        nn.Linear(3, 2, bias=False, dtype=torch.float64),
    )
    with torch.no_grad():
        model[2].weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64))

    coeffs = FeatureGenerator.init_coefficients(model, device="cpu")

    assert np.allclose(coeffs, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
