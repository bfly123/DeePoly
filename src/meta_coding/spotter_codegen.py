import json
import re
from typing import Dict, Iterable, List, Optional, Tuple, Union


AUTO_CODE_BEGIN = "# auto code begin"
AUTO_CODE_END = "# auto code end"
REQUIRED_OPERATORS = ("L1", "L2", "F", "N", "S")
TIME_OPERATOR_FIELDS = (
    ("eq_L1", "L1", "L1 operators"),
    ("eq_L2", "L2", "L2 operators"),
    ("eq_F", "F", "F operators"),
    ("eq_N", "N", "N operators"),
)
TIME_EQUATION_FIELDS = ("eq_L1", "eq_L2", "eq_F", "eq_N", "f_L2", "N")


def normalize_equation_items(items: Union[str, List[str], Tuple[str, ...], None]) -> List[str]:
    if not items:
        return []
    if isinstance(items, list):
        return list(items)
    if isinstance(items, tuple):
        return list(items)
    return [items]


def collect_equations(config_dict: Dict, problem_type: str) -> List[str]:
    equations: List[str] = []
    eq_dict = config_dict.get("eq", {})
    if eq_dict:
        for eq_items in eq_dict.values():
            equations.extend(normalize_equation_items(eq_items))
    if problem_type == "time_pde":
        for field in TIME_EQUATION_FIELDS:
            equations.extend(normalize_equation_items(config_dict.get(field)))
    return equations


def resolve_operator_signature(config_dict: Dict, problem_type: str) -> Dict[str, List[str]]:
    if problem_type == "time_pde":
        return _resolve_time_pde_signature(config_dict)
    if "eq" in config_dict:
        eq = config_dict["eq"]
        return {
            "L1": normalize_equation_items(eq.get("L1")),
            "L2": normalize_equation_items(eq.get("L2")),
            "F": normalize_equation_items(eq.get("F")),
            "N": normalize_equation_items(eq.get("N")),
            "S": normalize_equation_items(eq.get("S")),
            "const_list": config_dict.get("const_list", []),
        }
    return {
        "L1": normalize_equation_items(config_dict.get("eq_L1")),
        "L2": normalize_equation_items(config_dict.get("eq_L2")),
        "F": normalize_equation_items(config_dict.get("f_L2")),
        "N": normalize_equation_items(config_dict.get("N")),
        "S": normalize_equation_items(config_dict.get("S")),
        "const_list": config_dict.get("const_list", []),
    }


def _resolve_time_pde_signature(config_dict: Dict) -> Dict[str, List[str]]:
    has_separate_fields = any(field in config_dict for field, _, _ in TIME_OPERATOR_FIELDS)
    if has_separate_fields:
        return {
            "L1": normalize_equation_items(config_dict.get("eq_L1")),
            "L2": normalize_equation_items(config_dict.get("eq_L2")),
            "F": normalize_equation_items(config_dict.get("eq_F")),
            "N": normalize_equation_items(config_dict.get("eq_N")),
            "S": normalize_equation_items(config_dict.get("S")),
            "const_list": config_dict.get("const_list", []),
        }

    eq = config_dict.get("eq", {})
    return {
        "L1": normalize_equation_items(eq.get("L1")),
        "L2": normalize_equation_items(eq.get("L2")),
        "F": normalize_equation_items(eq.get("F")),
        "N": normalize_equation_items(eq.get("N")),
        "S": normalize_equation_items(eq.get("S")),
        "const_list": config_dict.get("const_list", []),
    }


def build_config_signature(config_dict: Dict, problem_type: Optional[str] = None) -> str:
    resolved_problem_type = problem_type or config_dict.get("problem_type", "linear_pde")
    signature = resolve_operator_signature(config_dict, resolved_problem_type)
    return json.dumps(signature, sort_keys=True)


def build_unified_operators(processor, config_dict: Dict) -> str:
    problem_type = config_dict.get("problem_type", "linear_pde")
    eq_dict = config_dict.get("eq", {})
    blocks: List[List[str]] = []
    defined_operators = set()

    if problem_type == "time_pde":
        for field, operator_name, comment in TIME_OPERATOR_FIELDS:
            equations = _resolve_time_operator_equations(config_dict, eq_dict, field, operator_name)
            blocks.append(
                render_operator_block(
                    processor,
                    operator_name,
                    equations,
                    comment,
                    empty_value="[]",
                )
            )
            defined_operators.add(operator_name)
        blocks.append(render_scalar_operator_block("S", "S operators (source term)", "[0]"))
        defined_operators.add("S")
    elif problem_type == "linear_pde":
        blocks.extend(_build_linear_pde_blocks(processor, eq_dict))
        defined_operators.update({"L1", "L2", "F", "N"})
    else:
        for op_name, eq_list in eq_dict.items():
            blocks.append(
                render_operator_block(
                    processor,
                    op_name,
                    normalize_equation_items(eq_list),
                    f"{op_name} operators",
                )
            )
            defined_operators.add(op_name)

    blocks.extend(build_missing_required_operator_blocks(defined_operators))
    return join_blocks(blocks)


def _resolve_time_operator_equations(
    config_dict: Dict,
    eq_dict: Dict,
    field: str,
    operator_name: str,
) -> List[str]:
    has_separate_fields = any(config_field in config_dict for config_field, _, _ in TIME_OPERATOR_FIELDS)
    if has_separate_fields:
        return normalize_equation_items(config_dict.get(field))
    return normalize_equation_items(eq_dict.get(operator_name))


def _build_linear_pde_blocks(processor, eq_dict: Dict) -> List[List[str]]:
    blocks = []
    if "L1" in eq_dict:
        blocks.append(
            render_operator_block(
                processor,
                "L1",
                normalize_equation_items(eq_dict["L1"]),
                "L1 operators",
            )
        )
    blocks.append(render_scalar_operator_block("L2", "L2 operators (not used in linear PDEs)", "[]"))
    blocks.append(render_scalar_operator_block("F", "F operators (not used in linear PDEs)", "[]"))
    blocks.append(render_scalar_operator_block("N", "N operators (not used in linear PDEs)", "[]"))
    return blocks


def render_operator_block(
    processor,
    operator_name: str,
    equations: Iterable[str],
    comment: str,
    empty_value: Optional[str] = None,
) -> List[str]:
    eq_terms = [processor._convert_equation_to_pytorch(eq) for eq in equations]
    if not eq_terms:
        return render_scalar_operator_block(operator_name, comment, empty_value or "[0]")

    lines = [f"        # {comment}"]
    if len(eq_terms) == 1:
        lines.append(f"        {operator_name} = [{eq_terms[0]}]")
        return lines

    lines.append(f"        {operator_name} = [")
    for index, term in enumerate(eq_terms):
        suffix = "" if index == len(eq_terms) - 1 else ","
        lines.append(f"            {term}{suffix}")
    lines.append("        ]")
    return lines


def render_scalar_operator_block(operator_name: str, comment: str, value: str) -> List[str]:
    return [f"        # {comment}", f"        {operator_name} = {value}"]


def render_constant_block(config_dict: Dict) -> List[str]:
    const_list = config_dict.get("const_list", []) or []
    if not const_list:
        return []

    lines = ["        # Extract configured constants"]
    for const_dict in const_list:
        if not isinstance(const_dict, dict):
            continue
        for const_name, const_value in const_dict.items():
            rendered_value = const_value if isinstance(const_value, str) else repr(const_value)
            lines.append(f"        {const_name} = {rendered_value}")
    return lines


def build_missing_required_operator_blocks(defined_operators: Iterable[str]) -> List[List[str]]:
    blocks = []
    for operator_name in REQUIRED_OPERATORS:
        if operator_name in defined_operators:
            continue
        blocks.append(
            render_scalar_operator_block(
                operator_name,
                f"{operator_name} operators (default)",
                "[0]",
            )
        )
    return blocks


def join_blocks(blocks: List[List[str]]) -> str:
    rendered = ["\n".join(block) for block in blocks if block]
    return "\n\n".join(rendered)


def compose_pytorch_code(
    processor,
    equations,
    problem_type: str = "linear_pde",
    config_dict: Optional[Dict] = None,
) -> str:
    all_equations = collect_equations(config_dict, problem_type) if config_dict else list(equations)
    derivatives_code = processor.generate_pytorch_derivatives(all_equations)
    operators_code = build_unified_operators(processor, config_dict) if config_dict else ""
    constants_code = ""
    if config_dict:
        constant_lines = render_constant_block(config_dict)
        if constant_lines:
            constants_code = "\n".join(constant_lines) + "\n\n"
    signature_comment = ""
    if config_dict:
        signature_comment = f"# Config signature: {build_config_signature(config_dict, problem_type)}\n"

    return (
        f"{AUTO_CODE_BEGIN}\n"
        f"{signature_comment}{derivatives_code}\n\n"
        f"{constants_code}"
        f"{operators_code}\n\n"
        f"{AUTO_CODE_END}"
    )


def extract_auto_code_block(net_content: str) -> Optional[str]:
    if AUTO_CODE_BEGIN not in net_content or AUTO_CODE_END not in net_content:
        return None
    begin_idx = net_content.find(AUTO_CODE_BEGIN)
    end_idx = net_content.find(AUTO_CODE_END)
    return net_content[begin_idx:end_idx]


def has_generated_code(auto_code_block: str) -> bool:
    lines = auto_code_block.splitlines()[1:]
    return any(line.strip() and not line.strip().startswith("#") for line in lines)


def extract_saved_signature(auto_code_block: str) -> Optional[Dict[str, List[str]]]:
    signature_match = re.search(r"#\s*Config signature:\s*(.+)", auto_code_block)
    if not signature_match:
        return None
    return json.loads(signature_match.group(1).strip())


def find_signature_diff(saved_dict: Dict, current_dict: Dict) -> str:
    diffs = []
    for key in ["L1", "L2", "F", "N", "S", "const_list"]:
        saved = saved_dict.get(key, [])
        current = current_dict.get(key, [])
        if saved == current:
            continue
        if not saved and current:
            diffs.append(f"{key}添加了Operators")
        elif saved and not current:
            diffs.append(f"{key}删ExceptOperators")
        else:
            diffs.append(f"{key}Operators已Modification")
    return ", ".join(diffs) if diffs else "未知差异"


def abstract_consistency_check(auto_code_block: str, config_signature: str) -> Tuple[bool, str]:
    try:
        config_dict = json.loads(config_signature)
    except Exception as exc:
        return True, f"ConsistencyCheckExit错: {exc}"

    for operator_name, operators in config_dict.items():
        if operator_name == "const_list":
            continue
        if not operators:
            continue
        if not _block_contains_operator_definition(auto_code_block, operator_name):
            return True, f"{operator_name}OperatorsAt代yard中未Find"
    return False, "PassAbstractCheck，代yard与Configuration基本一致"


def _block_contains_operator_definition(auto_code_block: str, operator_name: str) -> bool:
    pattern = rf"{operator_name}\s*=\s*\[(.*?)\]"
    match = re.search(pattern, auto_code_block, re.DOTALL)
    if not match:
        return False
    code_content = match.group(1).strip()
    return bool(code_content and not code_content.isspace())


def replace_auto_code_section(file_path: str, generated_code: str) -> None:
    with open(file_path, "r") as file_obj:
        lines = file_obj.readlines()

    new_lines = []
    skip_mode = False
    found_section = False

    for line in lines:
        if AUTO_CODE_BEGIN in line:
            new_lines.append(generated_code + "\n")
            skip_mode = True
            found_section = True
            continue
        if AUTO_CODE_END in line:
            skip_mode = False
            continue
        if not skip_mode:
            new_lines.append(line)

    if not found_section:
        raise ValueError(f"AtFile {file_path} 中未FindNeed替换的代yard段")

    with open(file_path, "w") as file_obj:
        file_obj.writelines(new_lines)
