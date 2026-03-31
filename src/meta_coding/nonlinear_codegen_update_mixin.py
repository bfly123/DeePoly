import os
import shutil

from datetime import datetime
from typing import Tuple


class NonlinearCodegenUpdateMixin:
    def _backup_file(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{self.hybrid_fitter_path}.{timestamp}.bak"
        shutil.copy2(self.hybrid_fitter_path, backup_path)
        print(f"已创建备份: {backup_path}")
        return backup_path

    def _generate_and_update_code(
        self,
        equations: Tuple[str, ...],
        backup_path: str,
    ) -> None:
        temp_file = os.path.join(os.path.dirname(__file__), "temp_generated_code.txt")
        self.processor.generate_code_for_hybrid_fitter(equations, temp_file)

        try:
            with open(temp_file, "r") as handle:
                generated_code = handle.read()

            self._update_file_content(generated_code)
            os.remove(temp_file)
            print(f"已成功更新 {self.hybrid_fitter_path}")
        except Exception:
            print("正在恢复备份...")
            shutil.copy2(backup_path, self.hybrid_fitter_path)
            print("已恢复备份")
            raise

    def _update_file_content(self, generated_code: str) -> None:
        start_marker = "#-----begin auto code-----"
        end_marker = "#-----end auto code-----"

        with open(self.hybrid_fitter_path, "r") as handle:
            lines = handle.readlines()

        new_lines = []
        skip_mode = False
        found_section = False

        for line in lines:
            if start_marker in line:
                new_lines.append(generated_code + "\n")
                skip_mode = True
                found_section = True
            elif end_marker in line:
                skip_mode = False
                continue
            elif not skip_mode:
                new_lines.append(line)

        if not found_section:
            raise ValueError(
                f"在文件 {self.hybrid_fitter_path} 中未找到需要替换的代码段"
            )

        with open(self.hybrid_fitter_path, "w") as handle:
            handle.writelines(new_lines)
