from src.meta_coding import parse_operators


class PseudotimeConfigOperatorMixin:
    def _normalize_operator_splitting(self):
        """Normalize equation format and ensure all operator lists exist."""
        if isinstance(self.eq, dict):
            self.eq_L1 = self.eq.get("L1", ["0"])
            self.eq_L2 = self.eq.get("L2", ["0"])
            self.eq_F = self.eq.get("F", ["1"])
            self.eq_N = self.eq.get("N", ["0"])
            print("Extracted operator splitting from config (with forced existence):")
            print(f"  L1 (Primary linear operators): {self.eq_L1}")
            print(f"  L2 (Semi-implicit linear operators): {self.eq_L2}")
            print(f"  F (Nonlinear functions): {self.eq_F}")
            print(f"  N (Fully nonlinear terms): {self.eq_N}")
        elif isinstance(self.eq, list):
            self.eq_L1 = self.eq if self.eq else ["0"]
            self.eq_L2 = ["0"]
            self.eq_F = ["1"]
            self.eq_N = ["0"]
            self.f_L2 = self.eq_F
            self.N = self.eq_N
            print(f"Using legacy list format as L1 operators: {self.eq_L1}")
            print(f"Legacy format preserved L1 operators: {self.eq_L1}")
            return
        else:
            raise ValueError(f"Invalid eq format: {type(self.eq)}. Must be list or dict.")

        n_vars = len(self.vars_list)
        if n_vars > 0:
            self.eq_L1 = self._pad_operator_list(self.eq_L1, n_vars, "0")
            self.eq_L2 = self._pad_operator_list(self.eq_L2, n_vars, "0")
            self.eq_F = self._pad_operator_list(self.eq_F, n_vars, "1")
            self.eq_N = self._pad_operator_list(self.eq_N, n_vars, "0")
        self.f_L2 = self.eq_F
        self.N = self.eq_N
        print(f"Normalized operators to {n_vars} variables:")
        print(f"  L1: {self.eq_L1}")
        print(f"  L2: {self.eq_L2}")
        print(f"  F: {self.eq_F}")
        print(f"  N: {self.eq_N}")

    @staticmethod
    def _pad_operator_list(op_list, target_length: int, default_op: str):
        """Pad or truncate operator lists to match the variable count."""
        if len(op_list) == target_length:
            return op_list
        if len(op_list) == 1:
            return op_list * target_length
        if len(op_list) == 0:
            return [default_op] * target_length
        if len(op_list) < target_length:
            return op_list + [default_op] * (target_length - len(op_list))
        return op_list[:target_length]

    def _parse_operator_splitting(self):
        """Parse equations with operator splitting."""
        print("Parsing operator splitting with forced existence...")
        print(f"L1 operators: {self.eq_L1}")
        print(f"L2 operators: {self.eq_L2}")
        print(f"F functions: {self.eq_F}")
        print(f"N operators: {self.eq_N}")

        l2f_terms = []
        for index, (l2, f_val) in enumerate(zip(self.eq_L2, self.eq_F)):
            term = self._build_l2f_term(l2, f_val)
            l2f_terms.append(term)
            print(f"Generated semi-implicit term {index + 1}: {term}")

        unified_eq = {
            "L1": self.eq_L1,
            "L2": self.eq_L2,
            "F": self.eq_F,
            "N": self.eq_N,
            "L2F": l2f_terms,
        }
        try:
            self.operator_parse = parse_operators(
                unified_eq,
                self.vars_list,
                self.spatial_vars,
                self.const_list,
            )
            print("Successfully parsed equations for general PDE solver")
        except Exception as exc:
            print(f"Warning: Failed to parse operator splitting: {exc}")
            self.operator_parse = {}

    @staticmethod
    def _build_l2f_term(l2: str, f_val: str) -> str:
        if l2 != "0" and f_val != "1":
            return f"{l2}*({f_val})"
        if l2 != "0" and f_val == "1":
            return l2
        if l2 == "0":
            return "0"
        return f"0*({f_val})"
