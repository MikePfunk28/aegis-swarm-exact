import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from mdap_small import history
from mdap_small.models import DEFAULT_MODELS
from mdap_small.red_flags import has_red_flags, parse_step
from mdap_small.server import app
from mdap_small.validation import paper_strict_profile
from mdap_small.validation_gate import load_and_check_report


class HistoryTests(unittest.TestCase):
    def test_append_and_read_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            old_dir = history.RUNS_DIR
            old_file = history.RUNS_FILE
            try:
                history.RUNS_DIR = base / "runs"
                history.RUNS_FILE = history.RUNS_DIR / "history.jsonl"
                history.append_run({"result": {"accuracy": 0.99}})
                rows = history.get_recent_runs(limit=5)
                self.assertEqual(len(rows), 1)
                self.assertIn("timestamp", rows[0])
                self.assertIn("result", rows[0])
            finally:
                history.RUNS_DIR = old_dir
                history.RUNS_FILE = old_file


class ApiTests(unittest.TestCase):
    def test_ui_config_and_health(self):
        client = TestClient(app)

        ui = client.get("/api/ui-config")
        self.assertEqual(ui.status_code, 200)
        ui_payload = ui.json()
        self.assertIn("machine", ui_payload)
        self.assertIn("workflow_modes", ui_payload)

        health = client.get("/api/health/models")
        self.assertEqual(health.status_code, 200)
        health_payload = health.json()
        self.assertIn("models", health_payload)

        gate = client.get("/api/validation/status")
        self.assertEqual(gate.status_code, 200)
        gate_payload = gate.json()
        self.assertIn("ok", gate_payload)
        self.assertIn("requirements", gate_payload)
        self.assertIn("checklist", gate_payload)

    def test_validation_job_dry_run(self):
        client = TestClient(app)
        start = client.post("/api/validation/run", json={"dry_run": True})
        self.assertEqual(start.status_code, 200)
        payload = start.json()
        self.assertTrue(payload.get("ok"))

        status = None
        for _ in range(20):
            resp = client.get("/api/validation/job")
            self.assertEqual(resp.status_code, 200)
            status = resp.json().get("status")
            if status in ("completed", "failed", "cancelled"):
                break
            time.sleep(0.05)
        self.assertIn(status, ("completed", "failed", "cancelled"))


class ParserTests(unittest.TestCase):
    def test_parse_step_two_line_format(self):
        text = "move = [1, 0, 2]\nnext_state = [[4, 3, 2], [1], []]"
        step = parse_step(text, parser_mode="red_flagging")
        self.assertIsNotNone(step)
        assert step is not None
        self.assertEqual(step.move.as_tuple(), (1, 0, 2))
        self.assertEqual(step.next_state, ((4, 3, 2), (1,), ()))

    def test_repairing_parser_allows_json(self):
        text = '{"move": [1, 0, 2], "next_state": [[4, 3, 2], [1], []]}'
        strict = parse_step(text, parser_mode="red_flagging")
        repair = parse_step(text, parser_mode="repairing")
        self.assertIsNone(strict)
        self.assertIsNotNone(repair)

    def test_red_flags_token_cutoff(self):
        text = "move = [1, 0, 2]\nnext_state = [[4, 3, 2], [1], []]"
        self.assertFalse(
            has_red_flags(text, max_tokens_approx=750, parser_mode="red_flagging")
        )


class ProfileTests(unittest.TestCase):
    def test_paper_profile_sets_expected_values(self):
        strict = paper_strict_profile(DEFAULT_MODELS, k_value=3)
        self.assertEqual(strict.parser_mode, "red_flagging")
        self.assertEqual(strict.red_flag_token_cutoff, 750)
        self.assertEqual(strict.ahead_k, 3)


class ValidationGateTests(unittest.TestCase):
    def test_gate_fails_missing_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = load_and_check_report(Path(tmp) / "missing.json")
            self.assertFalse(status.ok)
            self.assertTrue(status.messages)


if __name__ == "__main__":
    unittest.main()
