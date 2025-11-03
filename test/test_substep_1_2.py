import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from visualization.core.data_context import DataContext
from visualization.core.registry import PlotRegistry
from visualization.core.base import PlotComponent, FigureContext


class DummyComponent(PlotComponent):
    def render(self, context: FigureContext, data):
        pass

    def get_required_keys(self):
        return ["x", "y"]


def test_data_context():
    print("Testing DataContext...")

    dc = DataContext()

    dc.register_source("x", np.array([1, 2, 3]))
    dc.register_source("y", np.array([4, 5, 6]))

    assert dc.has_source("x")
    assert dc.has_source("y")
    assert not dc.has_source("z")

    x_data = dc.get("x")
    assert np.array_equal(x_data, np.array([1, 2, 3]))

    dc.register_transform("double", lambda x: x * 2)
    y_doubled = dc.get("y", apply_transform="double")
    assert np.array_equal(y_doubled, np.array([8, 10, 12]))

    sources = dc.list_sources()
    assert set(sources) == {"x", "y"}

    dc.update_source("x", np.array([7, 8, 9]))
    new_x = dc.get("x")
    assert np.array_equal(new_x, np.array([7, 8, 9]))

    dc.register_transform("double", lambda x: x * 3)
    y_tripled = dc.get("y", apply_transform="double")
    assert np.array_equal(y_tripled, np.array([12, 15, 18]))

    print("✓ DataContext tests passed")


def test_plot_registry():
    print("Testing PlotRegistry...")

    PlotRegistry.clear()

    @PlotRegistry.register("dummy")
    class TestComponent(DummyComponent):
        pass

    assert PlotRegistry.has("dummy")
    assert not PlotRegistry.has("nonexistent")

    component_class = PlotRegistry.get("dummy")
    assert component_class == TestComponent

    instance = PlotRegistry.create("dummy", component_name="test_instance", color="red")
    assert isinstance(instance, PlotComponent)
    assert instance.name == "test_instance"
    assert instance.config["color"] == "red"

    components = PlotRegistry.list_components()
    assert "dummy" in components

    print("✓ PlotRegistry tests passed")


def test_integration():
    print("Testing Integration...")

    PlotRegistry.clear()

    @PlotRegistry.register("line_plot")
    class LinePlot(DummyComponent):
        pass

    dc = DataContext()
    dc.register_source("x", np.linspace(0, 10, 100))
    dc.register_source("y", np.sin(np.linspace(0, 10, 100)))

    component = PlotRegistry.create("line_plot")

    data_dict = dc.to_dict()
    assert component.validate_data(data_dict)

    print("✓ Integration tests passed")


if __name__ == "__main__":
    test_data_context()
    test_plot_registry()
    test_integration()
    print("\n✅ All validation tests passed!")
