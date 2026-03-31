from __future__ import annotations


class NullSpotterModel:
    """Placeholder model used when an external spotter backend owns inference/training."""

    is_null_spotter_model = True

    def to(self, device):
        _ = device
        return self

    def train(self):
        return self

    def eval(self):
        return self

    def parameters(self):
        return iter(())

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        _ = state
        return self

    def __call__(self, *args, **kwargs):
        _ = (args, kwargs)
        raise RuntimeError(
            "NullSpotterModel cannot be evaluated directly. "
            "Use an external spotter feature provider or enable Torch spotter training."
        )
