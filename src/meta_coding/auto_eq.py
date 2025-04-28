def split_equations(equation_str):
    """智能分割方程，处理diff函数中的逗号"""
    equations = []
    current_eq = ""
    paren_count = 0

    for char in equation_str:
        if char == "(":
            paren_count += 1
        elif char == ")":
            paren_count -= 1
        elif char == "," and paren_count == 0:
            equations.append(current_eq.strip())
            current_eq = ""
            continue
        current_eq += char

    if current_eq:
        equations.append(current_eq.strip())

    return equations


def extract_diff_content(term):
    """提取diff函数内容，正确处理嵌套括号"""
    if "diff(" not in term:
        return None

    start_idx = term.index("diff(") + 5
    paren_count = 1
    end_idx = start_idx

    for i in range(start_idx, len(term)):
        if term[i] == "(":
            paren_count += 1
        elif term[i] == ")":
            paren_count -= 1
            if paren_count == 0:
                end_idx = i
                break

    return term[start_idx:end_idx]


def parse_diff_term(content, vars_list, spatial_vars):
    """解析导数项，返回变量名、导数变量和阶数"""
    parts = split_equations(content)
    if len(parts) < 2:
        return None, None, 0

    var_name = parts[0].strip()
    deriv_var = parts[1].strip()

    if var_name not in vars_list or deriv_var not in spatial_vars:
        return None, None, 0

    deriv_order = 1
    if len(parts) > 2:
        try:
            deriv_order = int(parts[2].strip())
        except ValueError:
            return None, None, 0

    return var_name, deriv_var, deriv_order


def parse_coefficient(term_part, const_list):
    """解析项的系数部分"""
    try:
        return float(term_part)
    except ValueError:
        # 尝试解析为常数
        for const_dict in const_list:
            for const_name, const_val in const_dict.items():
                if const_name in term_part:
                    term_part = term_part.replace(const_name, str(const_val))
        try:
            return float(eval(term_part))
        except:
            return 1.0


def generate_all_derivatives(max_deriv_orders):
    """
    根据最高导数阶数，生成所有可能的导数组合
    
    Args:
        max_deriv_orders: 每个变量在每个维度的最高导数阶数，如 [[1,2], [1,2], [0,0]]
    
    Returns:
        all_derivatives: 所有可能的导数组合，按变量分组
    """
    from itertools import product
    
    all_derivatives = []
    
    for var_idx, max_orders in enumerate(max_deriv_orders):
        var_derivatives = []
        # 为每个维度创建可能的值范围（从0到max_order-1）
        ranges = [range(order) for order in max_orders]
        
        # 使用product生成所有组合
        if all(r.stop > 0 for r in ranges):  # 确保每个范围都不为空
            for combo in product(*ranges):
                var_derivatives.append(list(combo))
        else:
            # 如果有任何一个维度的最大导数阶数为0，则只添加全0组合
            var_derivatives.append([0] * len(max_orders))
        
        all_derivatives.append(var_derivatives)
    
    return all_derivatives


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
    # 初始化数据结构
    n_dim = len(spatial_vars)
    n_vars = len(vars_list)
    max_deriv_orders = [[0] * n_dim for _ in range(n_vars)]
    deriv_orders_set = {tuple([0] * n_dim)}  # 初始只包含零阶导数

    # 更新最高导数阶数和导数阶数集合
    def update_max_deriv(var_name, deriv_var, order):
        if var_name in vars_list and deriv_var in spatial_vars:
            var_index = vars_list.index(var_name)
            dim_index = spatial_vars.index(deriv_var)
            max_deriv_orders[var_index][dim_index] = max(
                max_deriv_orders[var_index][dim_index], order
            )

            deriv_tuple = [0] * n_dim
            deriv_tuple[dim_index] = order
            deriv_orders_set.add(tuple(deriv_tuple))

    # 第一次遍历：收集所有需要的导数阶数
    def collect_derivatives(equations, is_nonlinear=False):
        for eq in equations:
            if not eq or eq == "0":
                continue

            eq = eq.replace("-", "+-").replace(" ", "")
            if eq.startswith("+"):
                eq = eq[1:]

            terms = [t for t in eq.split("+") if t]

            for term in terms:
                if is_nonlinear and "*" in term:
                    sub_terms = term.split("*")
                    for sub_term in sub_terms:
                        if "diff(" in sub_term:
                            content = extract_diff_content(sub_term)
                            if content:
                                var_name, deriv_var, order = parse_diff_term(
                                    content, vars_list, spatial_vars
                                )
                                if var_name:
                                    update_max_deriv(var_name, deriv_var, order)
                elif "diff(" in term:
                    content = extract_diff_content(term)
                    if content:
                        var_name, deriv_var, order = parse_diff_term(
                            content, vars_list, spatial_vars
                        )
                        if var_name:
                            update_max_deriv(var_name, deriv_var, order)

    # 收集线性和非线性方程中的导数
    collect_derivatives(equation_str)
    if equation_nonlinear_str:
        collect_derivatives(equation_nonlinear_str, True)

    # 将导数阶数集合转换为列表，排序以确保一致性
    deriv_orders = sorted(list(deriv_orders_set))

    # 创建导数索引映射和名称映射
    deriv_to_index = {deriv: idx for idx, deriv in enumerate(deriv_orders)}

    def get_deriv_name(var, deriv_tuple):
        """根据变量和导数阶数生成导数名称"""
        if all(d == 0 for d in deriv_tuple):
            return var.upper()  # 基本变量名大写
        
        name = var.upper()  # 基本变量名大写
        for dim, count in enumerate(deriv_tuple):
            if count > 0:
                # 空间变量（如x,y）保持小写
                name += "_" + spatial_vars[dim] * count
        return name

    deriv_names = {}
    for deriv_idx, deriv_tuple in enumerate(deriv_orders):
        for var_idx, var in enumerate(vars_list):
            deriv_names[f"{var_idx},{deriv_idx}"] = get_deriv_name(var, deriv_tuple)

    # 修改第二次遍历：解析线性方程项
    result = []
    
    for eq in equation_str:
        if not eq or eq == "0":
            result.append([])
            continue
            
        eq_terms = []
        eq = eq.replace("-", "+-").replace(" ", "")
        if eq.startswith("+"):
            eq = eq[1:]

        terms = [t for t in eq.split("+") if t]

        for term in terms:
            coef = 1.0
            if term.startswith("-"):
                coef = -1.0
                term = term[1:]

            # 处理除法
            term_parts = term.split("/")
            num_part = term_parts[0]
            
            # 处理分母
            if len(term_parts) > 1:
                denom = term_parts[1]
                denom_val = None
                # 检查分母是否为常数
                for const_dict in const_list:
                    if denom in const_dict:
                        denom_val = const_dict[denom]
                        break
                if denom_val is None:
                    try:
                        denom_val = float(denom)
                    except ValueError:
                        print(f"警告: 无法解析分母: {denom}")
                        continue
                coef /= denom_val

            # 处理导数项
            if "diff(" in num_part:
                # 提取前面的系数
                diff_idx = num_part.find("diff(")
                if diff_idx > 0:
                    coef_part = num_part[:diff_idx]
                    coef_val = parse_coefficient(coef_part, const_list)
                    if coef_val is not None:
                        coef *= coef_val

                content = extract_diff_content(num_part[diff_idx:])
                if not content:
                    continue

                var_name, deriv_var, deriv_order = parse_diff_term(content, vars_list, spatial_vars)
                if not var_name:
                    continue

                var_index = vars_list.index(var_name)
                dim_index = spatial_vars.index(deriv_var)

                # 构建导数元组
                deriv_tuple = [0] * n_dim
                deriv_tuple[dim_index] = deriv_order
                deriv_index = deriv_to_index[tuple(deriv_tuple)]
                
                term_name = deriv_names[f"{var_index},{deriv_index}"]
                eq_terms.append([coef, var_index, deriv_index, term_name])

            # 处理普通变量
            elif num_part in vars_list:
                var_index = vars_list.index(num_part)
                deriv_index = deriv_to_index[tuple([0] * n_dim)]
                term_name = deriv_names[f"{var_index},{deriv_index}"]
                eq_terms.append([coef, var_index, deriv_index, term_name])

            # 处理常数项
            else:
                const_val = parse_coefficient(num_part, const_list)
                if const_val is not None:
                    eq_terms.append([coef * const_val, -1, -1, "const"])

        result.append(eq_terms)

    # 修改非线性方程解析部分
    nonlinear_terms_set = set()
    if equation_nonlinear_str:
        for eq in equation_nonlinear_str:
            if not eq or eq == "0":
                continue

            eq = eq.replace("-", "+-").replace(" ", "")
            if eq.startswith("+"):
                eq = eq[1:]

            terms = [t for t in eq.split("+") if t]

            for term in terms:
                if "*" not in term:
                    continue
                    
                # 分解非线性项
                factors = []
                for sub_term in term.split("*"):
                    sub_term = sub_term.strip()
                    
                    # 跳过数值系数
                    try:
                        float(sub_term)
                        continue
                    except ValueError:
                        # 检查是否为常数
                        is_const = any(sub_term in const_dict for const_dict in const_list)
                        if is_const:
                            continue
                    
                    if "diff(" in sub_term:
                        content = extract_diff_content(sub_term)
                        if content:
                            var_name, deriv_var, deriv_order = parse_diff_term(
                                content, vars_list, spatial_vars
                            )
                            if var_name:
                                var_index = vars_list.index(var_name)
                                dim_index = spatial_vars.index(deriv_var)
                                deriv_tuple = [0] * n_dim
                                deriv_tuple[dim_index] = deriv_order
                                deriv_index = deriv_to_index[tuple(deriv_tuple)]
                                factors.append((var_index, deriv_index))
                    elif sub_term in vars_list:
                        var_index = vars_list.index(sub_term)
                        deriv_index = deriv_to_index[tuple([0] * n_dim)]
                        factors.append((var_index, deriv_index))
                
                # 只有当找到有效的非线性因子时才添加
                if len(factors) >= 2:  # 确保至少有两个因子
                    nonlinear_terms_set.update(factors)

    # 输出非线性项列表
    result_nonlinear = []
    for var_idx, deriv_idx in sorted(nonlinear_terms_set):
        term_name = deriv_names[f"{var_idx},{deriv_idx}"]
        result_nonlinear.append([var_idx, deriv_idx, term_name])

    # 添加生成所有导数组合的操作
    all_derivatives = generate_all_derivatives(max_deriv_orders)
    
    return result, deriv_orders, max_deriv_orders, result_nonlinear, all_derivatives


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


#if __name__ == "__main__":
#    test_parse_equation()
#
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
