import torch
from torch import nn

from src.abstract_class.base_net_training_mixin import BaseNetTrainingMixin


class RetryTestModel(BaseNetTrainingMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.Linear(1, 1, bias=False, dtype=torch.float64)

    def physics_loss(self, data_GPU, **kwargs):
        _ = (data_GPU, kwargs)
        return (self.layer.weight ** 2).sum()

    @staticmethod
    def model_init(config):
        return RetryTestModel(config)


class DummyConfig:
    def __init__(self):
        self.device = "cpu"
        self.learning_rate = 1e-3
        self.epochs_adam = 0
        self.epochs_lbfgs = 0
        self.DNNtol = 1e-4
        self.max_retries = 1
        self.seed = 7


def test_retry_reinitializes_model_weights_from_fresh_seed():
    config = DummyConfig()
    torch.manual_seed(config.seed + 1)
    expected = RetryTestModel.model_init(config)

    torch.manual_seed(123)
    model = RetryTestModel(config)
    with torch.no_grad():
        model.layer.weight.fill_(9.0)

    model._reset_random_seed_for_retry(1)
    model._reinitialize_model_for_retry(model)

    assert torch.allclose(model.layer.weight, expected.layer.weight)


def test_training_tolerance_uses_absolute_dnntol_not_dt_scaled():
    config = DummyConfig()
    model = RetryTestModel(config)

    assert model._get_training_tolerance(dt=0.1) == config.DNNtol


def test_build_optimizers_skips_lbfgs_when_disabled():
    config = DummyConfig()
    config.epochs_adam = 10
    config.epochs_lbfgs = 0
    model = RetryTestModel(config)

    optimizer_adam, optimizer_lbfgs = model._build_optimizers(model)

    assert optimizer_adam is not None
    assert optimizer_lbfgs is None


def test_lr_adjuster_applies_adam_milestones_without_adaptive_mode():
    config = DummyConfig()
    config.adaptive_learning_rate = False
    config.adam_lr_milestones = [2, 5]
    config.adam_lr_milestone_gamma = 0.5
    model = RetryTestModel(config)
    optimizer_adam, _ = model._build_optimizers(model)

    adjust_learning_rate = model._build_lr_adjuster(optimizer_adam)

    changed0, lr0 = adjust_learning_rate(current_loss=10.0, epoch=0)
    changed1, lr1 = adjust_learning_rate(current_loss=10.0, epoch=2)
    changed2, lr2 = adjust_learning_rate(current_loss=10.0, epoch=5)

    assert changed0 is False
    assert lr0 == config.learning_rate
    assert changed1 is True
    assert abs(lr1 - config.learning_rate * 0.5) < 1e-12
    assert changed2 is True
    assert abs(lr2 - config.learning_rate * 0.25) < 1e-12


def test_lr_adjuster_without_features_is_identity():
    config = DummyConfig()
    config.adaptive_learning_rate = False
    config.adam_lr_milestones = []
    model = RetryTestModel(config)
    optimizer_adam, _ = model._build_optimizers(model)

    adjust_learning_rate = model._build_lr_adjuster(optimizer_adam)
    changed, lr = adjust_learning_rate(current_loss=1.0, epoch=10)

    assert changed is False
    assert lr == config.learning_rate
