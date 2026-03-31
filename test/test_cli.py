import json
from pathlib import Path
from types import SimpleNamespace

from src import cli


def test_cli_resolve_case_dir_accepts_directory_and_config_file(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text('{"problem_type": "general_pde"}', encoding="utf-8")

    assert cli.resolve_case_dir(str(tmp_path)) == str(tmp_path.resolve())
    assert cli.resolve_case_dir(str(config_path)) == str(tmp_path.resolve())


def test_cli_parse_args_defaults_to_current_directory():
    args = cli.parse_args([])

    assert args.case_path == "."
    assert args.device == "auto"
    assert args.linear_device is None


def test_cli_run_case_dispatches_solver_with_overrides(monkeypatch, tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"problem_type": "general_pde"}), encoding="utf-8")

    events = {}

    class DummyConfig:
        def __init__(self, case_dir):
            self.case_dir = case_dir
            self.device = "cpu"
            self.linear_device = "cpu"
            self.results_dir = None

    class DummySolver:
        def __init__(self, config=None, case_dir=None):
            events["config"] = config
            events["case_dir"] = case_dir

        def solve(self):
            events["solved"] = True

    monkeypatch.setitem(
        cli.SOLVER_REGISTRY,
        "general_pde",
        {
            "config_class": DummyConfig,
            "solver_class": DummySolver,
        },
    )

    args = SimpleNamespace(
        case_path=str(tmp_path),
        device="cpu",
        linear_device="cuda",
        results_dir=str(tmp_path / "results_cli"),
        skip_autocode=True,
    )

    cli.run_case(args)

    assert events["solved"] is True
    assert events["case_dir"] == str(tmp_path.resolve())
    assert events["config"].device == "cpu"
    assert events["config"].linear_device == "cuda"
    assert Path(events["config"].results_dir) == (tmp_path / "results_cli").resolve()
