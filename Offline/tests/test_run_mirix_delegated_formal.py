from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_mirix_delegated_formal import _update_master


class DelegatedMirixStatusTest(unittest.TestCase):
    def test_updates_one_benchmark_without_losing_existing_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            path.write_text(
                json.dumps({"formal:WorldMemArena": {"status": "running"}}),
                encoding="utf-8",
            )
            _update_master(
                path,
                benchmark="MemEye",
                values={"status": "completed", "delegated_status": "running"},
            )
            status = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(status["formal:WorldMemArena"]["status"], "running")
            self.assertEqual(status["formal:MemEye"]["status"], "completed")
            self.assertEqual(status["formal:MemEye"]["delegated_status"], "running")


if __name__ == "__main__":
    unittest.main()
