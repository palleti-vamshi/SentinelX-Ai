
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from response import audit
from response import state_machine


class TestResponseSafety(unittest.TestCase):

    def setUp(self):
        self.temp_file = tempfile.TemporaryDirectory()
        self.log_path = Path(self.temp_file.name) / "audit.jsonl"

        self.patch_log = patch.object(
            audit, "AUDIT_FILE", self.log_path
        )
        self.patch_log.start()

        self.action = {
            "action_id": "ACT-TEST-001",
            "incident_id": "INC-TEST-001",
            "action_type": "isolate_host",
            "target": "WS-014",
            "state": "recommended"
        }

    def tearDown(self):
        self.patch_log.stop()
        self.temp_file.cleanup()

    def test_approval_and_simulated_execution(self):
        state_machine.transition(
            self.action, "pending_approval",
            "analyst", "Needs review"
        )
        state_machine.transition(
            self.action, "approved",
            "analyst", "Evidence verified"
        )

        result = state_machine.execute_simulated(self.action)

        self.assertEqual(self.action["state"], "executed_simulated")
        self.assertIn("simulated", result)
        self.assertEqual(len(audit.read_events()), 3)

    def test_execution_without_approval_is_blocked(self):
        with self.assertRaises(ValueError):
            state_machine.execute_simulated(self.action)

        self.assertEqual(self.action["state"], "recommended")
        self.assertEqual(len(audit.read_events()), 1)

    def test_rejection_prevents_execution(self):
        state_machine.transition(
            self.action, "pending_approval",
            "analyst", "Needs review"
        )
        state_machine.transition(
            self.action, "rejected",
            "analyst", "Evidence insufficient"
        )

        with self.assertRaises(ValueError):
            state_machine.execute_simulated(self.action)

        self.assertEqual(self.action["state"], "rejected")
        self.assertEqual(len(audit.read_events()), 3)

    
    def test_missing_reason_is_rejected(self):
        # First, move the action to pending approval.
        state_machine.transition(
            self.action,
            "pending_approval",
            "analyst",
            "Needs review"
        )

        # Approval without a reason must fail.
        with self.assertRaises(ValueError):
            state_machine.transition(
                self.action,
                "approved",
                "analyst",
                ""
            )

        # Rejection without a reason must also fail.
        with self.assertRaises(ValueError):
            state_machine.transition(
                self.action,
                "rejected",
                "analyst",
                "   "
            )

        # The action must still be pending approval.
        self.assertEqual(
            self.action["state"],
            "pending_approval"
        )

        # Only the valid initial transition should be logged.
        self.assertEqual(len(audit.read_events()), 1)


        with self.assertRaises(ValueError):
            state_machine.transition(
                self.action, "approved", "analyst", ""
            )

        self.assertEqual(len(audit.read_events()), 1)

    def test_missing_actor_is_rejected(self):
        with self.assertRaises(ValueError):
            state_machine.transition(
                self.action, "pending_approval",
                "", "Needs review"
            )

        self.assertEqual(self.action["state"], "recommended")
        self.assertEqual(len(audit.read_events()), 0)

    def test_direct_execution_transition_is_blocked(self):
        with self.assertRaises(ValueError):
            state_machine.transition(
                self.action, "executed_simulated",
                "analyst", "Attempted bypass"
            )

        self.assertEqual(self.action["state"], "recommended")
        self.assertEqual(len(audit.read_events()), 0)

    def test_second_execution_is_blocked(self):
        state_machine.transition(
            self.action, "pending_approval",
            "analyst", "Needs review"
        )
        state_machine.transition(
            self.action, "approved",
            "analyst", "Evidence verified"
        )
        state_machine.execute_simulated(self.action)

        with self.assertRaises(ValueError):
            state_machine.execute_simulated(self.action)

        self.assertEqual(len(audit.read_events()), 4)

    def test_rejected_action_cannot_be_approved(self):
        state_machine.transition(
            self.action, "pending_approval",
            "analyst", "Needs review"
        )
        state_machine.transition(
            self.action, "rejected",
            "analyst", "Unsafe"
        )

        with self.assertRaises(ValueError):
            state_machine.transition(
                self.action, "approved",
                "analyst", "Trying again"
            )

        self.assertEqual(len(audit.read_events()), 2)


if __name__ == "__main__":
    unittest.main()

