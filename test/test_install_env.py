from src import install_env


def test_detect_environment_name_prefers_conda():
    env = {
        "CONDA_DEFAULT_ENV": "ML",
        "VIRTUAL_ENV": "/tmp/venv",
    }

    assert install_env.detect_environment_name(env=env) == "ML"


def test_detect_environment_name_uses_virtualenv_suffix():
    env = {
        "VIRTUAL_ENV": "/tmp/test-env",
    }

    assert install_env.detect_environment_name(env=env) == "test-env"


def test_find_missing_modules_reports_only_missing_entries():
    imported = []

    def fake_importer(name):
        imported.append(name)
        if name == "sympy":
            raise ModuleNotFoundError(name)
        return object()

    missing = install_env.find_missing_modules(
        required_modules=("numpy", "sympy", "torch"),
        importer=fake_importer,
    )

    assert missing == ("sympy",)
    assert imported == ["numpy", "sympy", "torch"]


def test_collect_environment_check_reads_torch_cuda_state():
    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available():
                return True

    def fake_importer(name):
        if name == "torch":
            return FakeTorch()
        return object()

    result = install_env.collect_environment_check(importer=fake_importer, env={})

    assert result.missing_modules == ()
    assert result.torch_cuda_available is True


def test_validate_environment_flags_missing_dependencies():
    result = install_env.EnvironmentCheckResult(
        python_executable="/usr/bin/python",
        python_version="3.11.0",
        python_ok=True,
        env_name="ML",
        env_prefix="/tmp/env",
        scripts_dir="/tmp/env/bin",
        missing_modules=("torch", "sympy"),
        torch_cuda_available=None,
    )

    errors = install_env.validate_environment(result)

    assert len(errors) == 1
    assert "torch, sympy" in errors[0]


def test_console_script_path_uses_current_scripts_dir(monkeypatch):
    monkeypatch.setattr(install_env.sysconfig, "get_path", lambda name: "/tmp/env/bin")

    assert str(install_env.console_script_path()) == "/tmp/env/bin/deepoly"
