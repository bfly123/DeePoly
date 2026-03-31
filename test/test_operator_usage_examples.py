import importlib


def test_examples_module_imports():
    module = importlib.import_module("examples.operator_usage_example")
    assert hasattr(module, "example_multi_var_usage")
    assert hasattr(module, "run_simple_multi_var_demo")
    assert hasattr(module, "run_precompiled_system_demo")


def test_compat_wrapper_resolves_example_symbols():
    wrapper = importlib.import_module("src.abstract_class.operator_usage_example")
    assert callable(wrapper.example_multi_var_usage)
    assert callable(wrapper.test_simple_multi_var)
    assert callable(wrapper.run_simple_multi_var_demo)


def test_simple_example_runs():
    module = importlib.import_module("examples.operator_usage_example")
    result = module.run_simple_multi_var_demo()
    assert isinstance(result, dict)
    assert "L1_func" in result
    assert "N_func" in result
