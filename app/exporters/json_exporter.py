from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from app.exporters.result_exporter import ResultExporter


class JSONExporter(ResultExporter):
    def export(self, *, run_id: str, extracted_data: Dict[str, Any], structured_output: Dict[str, Any]) -> Path:
        payload = {
            "run_id": run_id,
            "extracted_data": extracted_data,
            "structured_output": structured_output,
        }
        export_path = self.output_dir / f"export_{run_id}.json"
        export_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return export_path
