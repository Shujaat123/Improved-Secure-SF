import csv
import os
import time
from typing import Dict, Any


class CSVLogger:
    def __init__(self, csv_path: str, fieldnames):
        self.csv_path = csv_path
        self.fieldnames = list(fieldnames)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        self._file = open(csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        self._writer.writeheader()
        self._file.flush()

        self._t0 = time.time()

    def log(self, row: Dict[str, Any]):
        out = {k: row.get(k, None) for k in self.fieldnames}
        self._writer.writerow(out)
        self._file.flush()

    def elapsed(self) -> float:
        return time.time() - self._t0

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass

    def __del__(self):
        self.close()
