import unittest

import numpy as np
import tensorflow as tf

from mate.models import ModifiedDavidsonTravelTime


class TestModifiedDavidsonPerformance(unittest.TestCase):
    def test_modified_davidson_parameter_ranges(self):
        capacities = np.array([100.0, 200.0], dtype=np.float32)
        tt_ff = np.array([10.0, 5.0], dtype=np.float32)
        perf = ModifiedDavidsonTravelTime(capacities, tt_ff, initial_JD=0.5, initial_mu=0.8, trainable=False)
        perf.build()

        jd = perf.JD.numpy()
        mu = perf.mu.numpy()

        self.assertTrue(np.all(jd > 0.0))
        self.assertTrue(np.all(mu > 0.0))
        self.assertTrue(np.all(mu < 1.0))

    def test_modified_davidson_piecewise_matches_formulas(self):
        capacities = np.array([100.0], dtype=np.float32)
        tt_ff = np.array([10.0], dtype=np.float32)
        perf = ModifiedDavidsonTravelTime(capacities, tt_ff, initial_JD=0.5, initial_mu=0.8, trainable=False, y_max=2.0)
        perf.build()

        mu = float(perf.mu.numpy()[0])
        jd = float(perf.JD.numpy()[0])

        y1 = 0.5 * mu
        y2 = min(1.2 * mu, 1.5)

        flows = tf.constant([[y1 * capacities[0]], [y2 * capacities[0]]], dtype=tf.float32)
        tt = perf(flows).numpy().reshape(-1)

        factor1 = 1.0 + jd * (y1 / (1.0 - y1))
        factor_mu = 1.0 + jd * (mu / (1.0 - mu))
        factor2 = factor_mu + jd * (y2 - mu) / ((1.0 - mu) ** 2)

        self.assertTrue(np.isclose(tt[0], tt_ff[0] * factor1, rtol=1e-5, atol=1e-5))
        self.assertTrue(np.isclose(tt[1], tt_ff[0] * factor2, rtol=1e-5, atol=1e-5))

    def test_modified_davidson_continuity_at_mu_value(self):
        capacities = np.array([100.0], dtype=np.float32)
        tt_ff = np.array([10.0], dtype=np.float32)
        perf = ModifiedDavidsonTravelTime(capacities, tt_ff, initial_JD=0.5, initial_mu=0.8, trainable=False)
        perf.build()

        mu = float(perf.mu.numpy()[0])
        jd = float(perf.JD.numpy()[0])

        eps = 1e-4
        y1 = max(mu - eps, 1e-3)
        y2 = mu + eps

        flows = tf.constant([[y1 * capacities[0]], [y2 * capacities[0]]], dtype=tf.float32)
        tt = perf(flows).numpy().reshape(-1)

        f1 = 1.0 + jd * (y1 / (1.0 - y1))
        f_mu = 1.0 + jd * (mu / (1.0 - mu))
        f2 = f_mu + jd * (y2 - mu) / ((1.0 - mu) ** 2)

        self.assertLess(abs(tt[0] - tt_ff[0] * f1), 1e-3)
        self.assertLess(abs(tt[1] - tt_ff[0] * f2), 1e-3)

    def test_modified_davidson_clip_y_max_avoids_blowup(self):
        capacities = np.array([100.0], dtype=np.float32)
        tt_ff = np.array([10.0], dtype=np.float32)
        perf = ModifiedDavidsonTravelTime(capacities, tt_ff, initial_JD=1.0, initial_mu=0.9, trainable=False, y_max=2.0)
        perf.build()

        flows = tf.constant([[10.0 * capacities[0]]], dtype=tf.float32)
        tt = float(perf(flows).numpy().reshape(-1)[0])

        mu = float(perf.mu.numpy()[0])
        jd = float(perf.JD.numpy()[0])
        y = 2.0
        f_mu = 1.0 + jd * (mu / (1.0 - mu))
        factor2 = f_mu + jd * (y - mu) / ((1.0 - mu) ** 2)

        self.assertTrue(np.isfinite(tt))
        self.assertTrue(np.isclose(tt, tt_ff[0] * factor2, rtol=1e-5, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
