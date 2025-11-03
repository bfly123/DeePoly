from typing import Dict, Any, Optional, Callable


class DataContext:

    def __init__(self):
        self._data_sources: Dict[str, Any] = {}
        self._transforms: Dict[str, Callable] = {}
        self._cache: Dict[str, Any] = {}

    def register_source(self, name: str, data: Any) -> None:
        self._data_sources[name] = data
        self._invalidate_cache(name)

    def register_transform(self, name: str, transform: Callable) -> None:
        self._transforms[name] = transform
        self._invalidate_transform_cache(name)

    def get(self, key: str, apply_transform: Optional[str] = None) -> Any:
        cache_key = f"{key}:{apply_transform}" if apply_transform else key

        if cache_key in self._cache:
            return self._cache[cache_key]

        if key not in self._data_sources:
            raise KeyError(f"Data source '{key}' not found")

        data = self._data_sources[key]

        if apply_transform:
            if apply_transform not in self._transforms:
                raise KeyError(f"Transform '{apply_transform}' not found")
            data = self._transforms[apply_transform](data)

        self._cache[cache_key] = data
        return data

    def update_source(self, name: str, data: Any) -> None:
        self.register_source(name, data)

    def has_source(self, name: str) -> bool:
        return name in self._data_sources

    def list_sources(self) -> list:
        return list(self._data_sources.keys())

    def clear_cache(self) -> None:
        self._cache.clear()

    def _invalidate_cache(self, source_name: str) -> None:
        keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"{source_name}:") or k == source_name]
        for key in keys_to_remove:
            del self._cache[key]

    def _invalidate_transform_cache(self, transform_name: str) -> None:
        keys_to_remove = [k for k in self._cache.keys() if k.endswith(f":{transform_name}")]
        for key in keys_to_remove:
            del self._cache[key]

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data_sources)
