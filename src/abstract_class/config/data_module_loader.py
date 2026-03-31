import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Optional


class CustomDataModuleLoader:
    """Load case-local helper modules without mutating sys.path."""

    def load_from_case_dir(self, case_dir: Optional[str]) -> Optional[ModuleType]:
        if not case_dir:
            return None
        module_path = Path(case_dir) / "data_generate.py"
        if not module_path.exists():
            return None
        module_name = self._module_name_for_path(module_path)
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _module_name_for_path(self, module_path: Path) -> str:
        digest = hashlib.sha1(str(module_path.resolve()).encode("utf-8")).hexdigest()[:12]
        return f"deepoly_case_data_generate_{digest}"
