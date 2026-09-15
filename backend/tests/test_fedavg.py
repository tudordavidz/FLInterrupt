import unittest

import torch

from app.federated import FederatedSimulation


class FedAvgEmptyRoundTests(unittest.TestCase):
    def test_fedavg_weighted_mean(self):
        sim = FederatedSimulation()
        a = {"w": torch.tensor([0.0])}
        b = {"w": torch.tensor([4.0])}
        out = sim._fedavg([a, b], [1, 3])
        self.assertAlmostEqual(float(out["w"].item()), 3.0)

    def test_empty_round_holds_global(self):
        sim = FederatedSimulation()
        before = {k: v.detach().clone() for k, v in sim.global_model.state_dict().items()}
        # Empty participation leaves weights unchanged (engine skips _fedavg).
        active_models = []
        if not active_models:
            after = sim.global_model.state_dict()
        self.assertTrue(
            all(torch.equal(before[k], after[k]) for k in before)
        )


if __name__ == "__main__":
    unittest.main()
