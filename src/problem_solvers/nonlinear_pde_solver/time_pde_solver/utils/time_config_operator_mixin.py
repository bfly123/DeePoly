from src.meta_coding import parse_operators


class TimeConfigOperatorMixin:
    def _normalize_operator_splitting(self):
        if isinstance(self.eq, dict):
            self._load_operator_dict(self.eq)
            self._expand_operator_lists()
            self._sync_operator_aliases()
            self._print_operator_summary("Extracted operator splitting from config")
            return
        if isinstance(self.eq, list):
            self.eq_L1 = self.eq if self.eq else ["0"]
            self.eq_L2 = ["0"]
            self.eq_F = ["1"]
            self.eq_N = ["0"]
            self._sync_operator_aliases()
            print(f"Using legacy list format as L1 operators: {self.eq_L1}")
            return
        raise ValueError(f"Invalid eq format: {type(self.eq)}. Must be list or dict.")

    def _load_operator_dict(self, eq_dict):
        self.eq_L1 = eq_dict.get("L1", ["0"])
        self.eq_L2 = eq_dict.get("L2", ["0"])
        self.eq_F = eq_dict.get("F", ["1"])
        self.eq_N = eq_dict.get("N", ["0"])

    def _expand_operator_lists(self):
        n_vars = len(self.vars_list)
        if n_vars <= 0:
            return
        self.eq_L1 = self._pad_operator_list(self.eq_L1, n_vars, "0")
        self.eq_L2 = self._pad_operator_list(self.eq_L2, n_vars, "0")
        self.eq_F = self._pad_operator_list(self.eq_F, n_vars, "1")
        self.eq_N = self._pad_operator_list(self.eq_N, n_vars, "0")

    def _sync_operator_aliases(self):
        self.f_L2 = self.eq_F
        self.N = self.eq_N

    def _print_operator_summary(self, prefix: str):
        print(f"{prefix} (with forced existence):")
        print(f"  L1: {self.eq_L1}")
        print(f"  L2: {self.eq_L2}")
        print(f"  F: {self.eq_F}")
        print(f"  N: {self.eq_N}")

    def _pad_operator_list(self, op_list, target_length: int, default_op: str):
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
        print("Parsing operator splitting with forced existence...")
        self._print_operator_summary("Current operators")
        unified_eq = {
            "L1": self.eq_L1,
            "L2": self.eq_L2,
            "F": self.eq_F,
            "N": self.eq_N,
            "L2F": self._build_l2f_terms(),
        }
        try:
            self.operator_parse = parse_operators(
                unified_eq,
                self.vars_list,
                self.spatial_vars,
                self.const_list,
            )
            print("Successfully parsed equations for time PDE solver")
        except Exception as exc:
            print(f"Warning: Failed to parse operator splitting: {exc}")
            self.operator_parse = {}

    def _build_l2f_terms(self):
        l2f_terms = []
        for index, (l2, f_val) in enumerate(zip(self.eq_L2, self.eq_F), start=1):
            l2f_terms.append(self._combine_l2f_term(l2, f_val))
            print(f"Generated semi-implicit term {index}: {l2f_terms[-1]}")
        return l2f_terms

    @staticmethod
    def _combine_l2f_term(l2: str, f_val: str) -> str:
        if l2 == "0":
            return "0"
        if f_val == "1":
            return l2
        return f"{l2}*({f_val})"
