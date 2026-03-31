from typing import Optional

from src.abstract_class.config.data_helpers import CustomDataModuleLoader


class BaseDataCustomLoaderMixin:
    def _load_custom_data_generator(self) -> Optional[object]:
        case_dir = getattr(self.config, "case_dir", None)
        if not case_dir:
            return None
        try:
            return CustomDataModuleLoader().load_from_case_dir(case_dir)
        except ImportError:
            return None
