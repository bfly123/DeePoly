from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SpotterCandidateConfig:
    name: str
    backend: str
    runtime: str = "inprocess"
    enabled: bool = True
    priority: int = 0
    device: Optional[str] = None
    python_executable: Optional[str] = None
    provider_cfg: Dict[str, Any] = field(default_factory=dict)
    allow_torch_training: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpotterRuntimeConfig:
    mode: str = "single"
    selector: Dict[str, Any] = field(default_factory=dict)
    candidates: List[SpotterCandidateConfig] = field(default_factory=list)

    def enabled_candidates(self) -> List[SpotterCandidateConfig]:
        return [candidate for candidate in self.candidates if candidate.enabled]
