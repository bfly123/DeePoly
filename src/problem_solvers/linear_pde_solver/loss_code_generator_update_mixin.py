class LossCodeGeneratorUpdateMixin:
    def update_code(self, equations, nonlinear_equations=None) -> None:
        """Update the physics_loss auto-code block in the target model file."""
        derivatives_code = self._generate_derivatives_code(equations + (nonlinear_equations or []))
        equations_code = self._generate_equations_code(equations)
        if nonlinear_equations:
            nonlinear_code = self._generate_equations_code(
                nonlinear_equations,
                is_nonlinear=True,
            )
            equations_code += "\n\n" + nonlinear_code
        loss_code = self._generate_loss_code(len(equations))
        self._backup_model_file()
        lines = self._read_model_lines()
        new_lines = self._replace_auto_code_block(lines, derivatives_code, equations_code, loss_code)
        with open(self.model_path, "w") as f:
            f.writelines(new_lines)

    def _backup_model_file(self) -> None:
        backup_path = self.model_path + ".backup"
        try:
            with open(self.model_path, "r") as f:
                current_content = f.read()
            with open(backup_path, "w") as f:
                f.write(current_content)
        except Exception as exc:
            print(f"Warning: Failed to create backup file: {exc}")

    def _read_model_lines(self):
        with open(self.model_path, "r") as f:
            return f.readlines()

    def _replace_auto_code_block(self, lines, derivatives_code: str, equations_code: str, loss_code: str):
        start_marker = "# auto code begin"
        end_marker = "# auto code end"
        rendered_code = self._render_indented_code(derivatives_code, equations_code, loss_code)
        new_lines = []
        skip_mode = False
        found_section = False

        for line in lines:
            if "pde_loss = torch.mean" in line:
                continue
            if start_marker in line:
                new_lines.append(line)
                new_lines.append(rendered_code + "\n\n")
                skip_mode = True
                found_section = True
                continue
            if end_marker in line:
                skip_mode = False
                new_lines.append(line)
                continue
            if not skip_mode:
                new_lines.append(line)

        if not found_section:
            raise ValueError(f"Code segment to replace not found in file {self.model_path}")
        return new_lines

    @staticmethod
    def _render_indented_code(derivatives_code: str, equations_code: str, loss_code: str) -> str:
        combined = (derivatives_code + "\n\n" + equations_code + "\n\n" + loss_code).strip()
        return "\n".join(
            "        " + code_line if code_line.strip() else ""
            for code_line in combined.split("\n")
        )
