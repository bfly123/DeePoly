from src.meta_coding.auto_repalce_nonlinear import (
    EquationProcessor,
    NonlinearCodeGenerator,
    update_hybrid_fitter_code,
)


def test_nonlinear_processor_generates_variable_declarations():
    processor = EquationProcessor(["x", "y"], ["u", "v"])
    declarations = processor.generate_variable_declarations(
        ("diff(u,x) + v",),
        ["u", "v"],
        ["x", "y"],
    )
    assert 'U = variables["U"][segment_idx]' in declarations
    assert 'U_x = variables["U_x"][segment_idx]' in declarations
    assert 'V = variables["V"][segment_idx]' in declarations
    assert "V_y" not in declarations


def test_nonlinear_code_generator_replaces_auto_code_block(tmp_path):
    target = tmp_path / "hybrid_fitter.py"
    target.write_text(
        "before\n#-----begin auto code-----\nold\n#-----end auto code-----\nafter\n"
    )

    update_hybrid_fitter_code(
        equations=("diff(u,x) + u",),
        vars_list=["u"],
        dimensions=["x"],
        hybrid_fitter_path=str(target),
    )

    content = target.read_text()
    assert "before" in content
    assert "after" in content
    assert 'U = variables["U"][segment_idx]' in content
    assert "dL[0]" in content


def test_nonlinear_code_generator_raises_when_markers_missing(tmp_path):
    target = tmp_path / "hybrid_fitter.py"
    target.write_text("no markers here\n")

    generator = NonlinearCodeGenerator(["x"], ["u"], str(target))
    try:
        generator.update_code(("u",))
    except ValueError as exc:
        assert "未找到需要替换的代码段" in str(exc)
    else:
        raise AssertionError("expected ValueError")
