import json
from dataclasses import dataclass

import numpy as np

from src.abstract_class.config.base_config import BaseConfig
from src.abstract_class.config.base_pde_config import BasePDEConfig


class DummyBaseConfig(BaseConfig):
    def __post_init__(self):
        return None

    def __init__(self, case_dir: str):
        super().__init__(case_dir)
        self.seed = 1
        self.n_segments = [1]
        self.poly_degree = [2]
        self.x_domain = [[0.0, 1.0]]
        self.list_field = []

    def _int_list_fields(self):
        return ["n_segments", "poly_degree"]

    def _list_fields(self):
        return ["list_field"]

    def _process_list_field(self, key, value):
        if key == "list_field":
            return [item.upper() for item in value]
        return value


@dataclass
class DummyPDEConfig(BasePDEConfig):
    def _solver_specific_init(self):
        self.solver_initialized = True


def test_base_config_loads_known_fields_with_type_coercion(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "n_segments": ["3"],
                "poly_degree": ["4"],
                "list_field": ["a", "b"],
                "unknown_field": 99,
            }
        ),
        encoding="utf-8",
    )
    config = DummyBaseConfig(str(tmp_path))
    assert config.load_config_from_json(str(tmp_path)) is True
    assert config.n_segments == [3]
    assert config.poly_degree == [4]
    assert config.list_field == ["A", "B"]
    assert not hasattr(config, "unknown_field")


def test_base_config_exports_public_fields(tmp_path):
    config = DummyBaseConfig(str(tmp_path))
    config.array_field = np.array([[1.0, 2.0]])
    output_path = tmp_path / "export.json"
    config.export_to_json(str(output_path))
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["array_field"] == [[1.0, 2.0]]
    assert payload["device_info"] in {"cpu", "cuda"}
    assert "export_time" in payload


def test_base_pde_config_relaxed_load_normalizes_fields(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "vars_list": ["u"],
                "spatial_vars": ["x"],
                "eq": ["diff(u,x) + u"],
                "const_list": [],
                "x_domain": [0.0, 1.0],
                "n_segments": ["2"],
                "poly_degree": ["3"],
                "hidden_dims": ["8", "4"],
                "points_domain_test": ["5"],
                "new_runtime_flag": "7",
            }
        ),
        encoding="utf-8",
    )
    config = DummyPDEConfig(case_dir=str(tmp_path))
    assert config.eq == {"L1": ["diff(u,x) + u"]}
    assert config.n_eqs == 1
    assert config.n_dim == 1
    assert config.hidden_dims == [8, 4]
    assert config.points_domain_test == [5]
    assert config.new_runtime_flag == 7
    assert config.solver_initialized is True
    assert config.x_domain.shape == (1, 2)
