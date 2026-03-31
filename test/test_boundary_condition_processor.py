import numpy as np
import torch

from src.abstract_class.boundary_conditions import TimePDEBoundaryProcessor


class DummyConfig:
    device = "cpu"


class DummyNetwork:
    def __init__(self):
        self.config = DummyConfig()

    def __call__(self, x):
        return None, x.sum(dim=1, keepdim=True)

    def gradients(self, u_pred, x_point):
        del u_pred
        return (torch.ones_like(x_point),)


class DummyTwoVarNetwork(DummyNetwork):
    def __call__(self, x):
        summed = x.sum(dim=1, keepdim=True)
        return None, torch.cat([summed, summed], dim=1)


def test_boundary_processor_prepare_gpu_data_handles_standard_and_periodic():
    processor = TimePDEBoundaryProcessor(DummyConfig())
    data = {
        "global_boundary_dict": {
            0: {
                "dirichlet": {
                    "x": np.array([[0.0], [1.0]]),
                    "values": np.array([[1.0], [2.0]]),
                },
                "periodic": {
                    "pairs": [
                        {
                            "x_1": np.array([[0.0]]),
                            "x_2": np.array([[1.0]]),
                            "segment_1": 0,
                            "segment_2": 1,
                        }
                    ]
                },
            }
        }
    }
    prepared = processor.prepare_boundary_gpu_data(data)["global_boundary_dict"][0]
    assert prepared["dirichlet"]["x"].requires_grad is True
    assert prepared["dirichlet"]["values"].requires_grad is False
    assert prepared["periodic"]["pairs"][0]["x_1"].requires_grad is True


def test_boundary_processor_compute_losses():
    processor = TimePDEBoundaryProcessor(DummyConfig())
    network = DummyNetwork()
    x = torch.tensor([[1.0], [2.0]], dtype=torch.float64, requires_grad=True)
    values = torch.tensor([[1.5], [2.5]], dtype=torch.float64)
    normals = torch.tensor([[1.0], [1.0]], dtype=torch.float64)
    data_gpu = {
        "global_boundary_dict": {
            0: {
                "dirichlet": {"x": x, "values": values},
                "neumann": {"x": x, "values": values, "normals": normals},
                "periodic": {"pairs": [{"x_1": x[:1], "x_2": x[1:2]}]},
            }
        }
    }
    loss = processor.compute_boundary_loss(data_gpu, network)
    assert torch.is_tensor(loss)
    assert loss.item() >= 0.0


def test_boundary_processor_averages_errors_across_boundary_groups():
    processor = TimePDEBoundaryProcessor(DummyConfig())
    network = DummyTwoVarNetwork()
    x0 = torch.tensor([[1.0], [2.0]], dtype=torch.float64, requires_grad=True)
    x1 = torch.tensor([[3.0]], dtype=torch.float64, requires_grad=True)
    data_gpu = {
        "global_boundary_dict": {
            0: {
                "dirichlet": {
                    "x": x0,
                    "values": torch.tensor([[1.5], [2.5]], dtype=torch.float64),
                }
            },
            1: {
                "dirichlet": {
                    "x": x1,
                    "values": torch.tensor([[4.0]], dtype=torch.float64),
                }
            },
        }
    }

    loss = processor.compute_boundary_loss(data_gpu, network)

    # Squared errors are [0.25, 0.25, 1.0], so the pooled mean should be 0.5.
    assert torch.isclose(loss, torch.tensor(0.5, dtype=torch.float64))
