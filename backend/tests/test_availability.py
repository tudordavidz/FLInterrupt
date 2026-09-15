import unittest

from app.availability import (
    class_metrics_from_confusion,
    parse_interruption_schedule,
    row_normalize_confusion,
    scheduled_offline,
)
from app.federated import ClientState, FederatedSimulation


class ScheduleTests(unittest.TestCase):
    def test_inclusive_window(self):
        sched = parse_interruption_schedule(
            [{"client_id": 3, "offline_from": 8, "offline_to": 12}]
        )
        self.assertTrue(scheduled_offline(sched, 3, 8))
        self.assertTrue(scheduled_offline(sched, 3, 12))
        self.assertFalse(scheduled_offline(sched, 3, 7))
        self.assertFalse(scheduled_offline(sched, 3, 13))
        self.assertFalse(scheduled_offline(sched, 2, 10))

    def test_rejects_inverted_window(self):
        with self.assertRaises(ValueError):
            parse_interruption_schedule(
                [{"client_id": 0, "offline_from": 5, "offline_to": 4}]
            )

    def test_schedule_applied_at_round_boundary(self):
        sim = FederatedSimulation()
        sim.clients = [ClientState(client_id=i) for i in range(4)]
        sim.interruption_schedule = parse_interruption_schedule(
            [{"client_id": 3, "offline_from": 8, "offline_to": 12}]
        )
        sim._apply_schedule(8)
        self.assertFalse(sim.clients[3].connected)
        self.assertTrue(sim.clients[0].connected)
        sim._apply_schedule(13)
        self.assertTrue(sim.clients[3].connected)


class MetricsTests(unittest.TestCase):
    def test_row_normalize_and_f1(self):
        cm = [[2.0, 2.0], [0.0, 4.0]]
        norm = row_normalize_confusion(cm)
        self.assertAlmostEqual(norm[0][0], 0.5)
        self.assertAlmostEqual(norm[1][1], 1.0)
        rows = class_metrics_from_confusion(cm, ["a", "b"])
        self.assertEqual(rows[0]["class_name"], "a")
        self.assertGreater(rows[1]["f1"], rows[0]["f1"])


if __name__ == "__main__":
    unittest.main()
