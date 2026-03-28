from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class ResultExporter(ABC):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def export(self, *, run_id: str, extracted_data: Dict[str, Any], structured_output: Dict[str, Any]) -> Path:
        raise NotImplementedError
