from .auto_eq_helpers import (
    extract_diff_content,
    generate_all_derivatives,
    parse_coefficient,
    parse_diff_term,
    split_equations,
)
from .auto_eq_parser import parse_equation_to_list_impl


def parse_equation_to_list(
    equation_str,
    equation_nonlinear_str,
    vars_list,
    spatial_vars,
    const_list=[{"nu": 0.1}],
):
    """
    将方程字符串解析为特定格式的列表

    Args:
        equation_str: 线性方程字符串列表，如 ["diff(u,x) + diff(v,y)", "diff(p,x) - diff(u,x,2)/Re"]
        equation_nonlinear_str: 非线性方程字符串列表，如 ["0", "u*diff(u,x) + v*diff(u,y)"]
        vars_list: 变量列表，如 ["u", "v", "p"]
        spatial_vars: 空间变量列表，如 ["x", "y"]
        const_list: 常数列表，如 [{"nu": 0.1}]

    Returns:
        result: 线性方程系数和导数项列表
        deriv_orders: 所有导数阶数组合
        max_deriv_orders: 每个变量在每个维度的最高导数阶数
        result_nonlinear: 非线性项列表
        all_derivatives: 所有可能的导数组合，按变量分组
    """
    return parse_equation_to_list_impl(
        equation_str,
        equation_nonlinear_str,
        vars_list,
        spatial_vars,
        const_list,
    )


def test_parse_equation():
    """测试方程解析函数"""
    # 纳维-斯托克斯方程示例
    equation_str = [
        "diff(u,x) + diff(v,y)",
        "diff(p,x) - diff(u,x,2)/Re - diff(u,y,2)/Re",
        "diff(p,y) - diff(v,x,2)/Re - diff(v,y,2)/Re",
    ]

    equation_nonlinear_str = [
        "0",
        "u*diff(u,x) + v*diff(u,y)",
        "u*diff(v,x) + v*diff(v,y)",
    ]

    vars_list = ["u", "v", "p"]
    spatial_vars = ["x", "y"]
    const_list = [{"nu": 0.1}, {"Re": 100}]

    # 执行解析
    result, deriv_orders, max_deriv_orders, result_nonlinear, all_derivatives = parse_equation_to_list(
        equation_str, equation_nonlinear_str, vars_list, spatial_vars, const_list
    )

    print("线性方程结果:")
    for i, eq in enumerate(result):
        print(f"方程 {i+1}:", eq)

    print("\n导数阶数组合:")
    for i, order in enumerate(deriv_orders):
        print(f"{i}: {order}")

    print("\n最高导数阶数:")
    for i, var_orders in enumerate(max_deriv_orders):
        print(f"{vars_list[i]}: {var_orders}")

    print("\n非线性项:")
    for item in result_nonlinear:
        print(item)

    print("\n所有可能的导数组合:")
    for i, var_derivs in enumerate(all_derivatives):
        print(f"{vars_list[i]}的导数组合:", var_derivs)

    return result, deriv_orders, max_deriv_orders, result_nonlinear, all_derivatives


if __name__ == "__main__":
    test_parse_equation()

## 测试用例
#equation_str = ["diff(u,x)"]
#equation_nonlinear_str = ["u*diff(u,x)"]  # 非线性项
#vars_list = ["u"]
#spatial_vars = ["x", "y"]
#const_list = [{"Re": 100}]
#
#result, _, _, result_nonlinear, _ = parse_equation_to_list(
#    equation_str, 
#    equation_nonlinear_str, 
#    vars_list, 
#    spatial_vars, 
#    const_list
#)
#print("非线性项:", result_nonlinear)
