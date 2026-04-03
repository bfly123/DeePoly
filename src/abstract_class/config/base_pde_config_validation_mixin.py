from src.meta_coding import parse_operators


class BasePDEConfigValidationMixin:
    def _normalize_eq_format(self) -> None:
        if isinstance(self.eq, list):
            self.eq = {"L1": self.eq}
            return
        if not isinstance(self.eq, dict):
            raise ValueError(f"Invalid eq format: {type(self.eq)}. Must be list or dict.")

    def _calculate_n_eqs(self) -> int:
        return len(self.vars_list)

    def _validate_pde_config(self) -> None:
        self._validate_required_fields()
        self._validate_dimension_lengths()

    def _validate_required_fields(self) -> None:
        required_fields = ["vars_list", "spatial_vars", "x_domain"]
        method = str(getattr(self, "method", "deepoly") or "deepoly").lower()
        if method == "deepoly":
            required_fields.extend(["n_segments", "poly_degree"])
        for field_name in required_fields:
            if not hasattr(self, field_name) or not getattr(self, field_name):
                raise ValueError(f"Required field '{field_name}' is missing or empty")

    def _validate_dimension_lengths(self) -> None:
        field_pairs = {
            "x_domain": len(self.x_domain),
        }
        method = str(getattr(self, "method", "deepoly") or "deepoly").lower()
        if method == "deepoly":
            field_pairs["n_segments"] = len(self.n_segments)
            field_pairs["poly_degree"] = len(self.poly_degree)
        expected = len(self.spatial_vars)
        for field_name, actual in field_pairs.items():
            if actual != expected:
                raise ValueError(
                    f"{field_name} dimension ({actual}) must match spatial_vars dimension ({expected})"
                )

    def _parse_operators(self) -> None:
        try:
            self.operator_parse = parse_operators(
                self.eq,
                self.vars_list,
                self.spatial_vars,
                self.const_list,
            )
        except Exception as exc:
            print(f"Warning: Failed to parse operators: {exc}")
            self.operator_parse = {}
