from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict

from app.exporters.result_exporter import ResultExporter


class CSVExporter(ResultExporter):
    """
    CSV export strategy:
    - If extracted_data has at least one list[dict], export the first such list as a table.
    - Otherwise export flattened key/value rows.
    """

    def export(self, *, run_id: str, extracted_data: Dict[str, Any], structured_output: Dict[str, Any]) -> Path:
        export_path = self.output_dir / f"export_{run_id}.csv"

        list_key = None
        list_items = None
        for key, value in extracted_data.items():
            if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
                list_key = key
                list_items = value
                break

        with export_path.open("w", newline="", encoding="utf-8") as f:
            if list_items is not None:
                fieldnames = sorted({field for item in list_items for field in item.keys()})
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(list_items)
            else:
                writer = csv.writer(f)
                writer.writerow(["key", "value"])
                for key, value in extracted_data.items():
                    writer.writerow([key, json.dumps(value, ensure_ascii=False)])

                writer.writerow([])
                writer.writerow(["structured_output", json.dumps(structured_output, ensure_ascii=False)])
                if list_key:
                    writer.writerow(["source_list_key", list_key])

        return export_path
