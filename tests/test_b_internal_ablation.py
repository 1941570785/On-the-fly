import unittest

import torch

from args import get_args
from asr_gs.config import resolve_config
from asr_gs.response_sampling import combine_sampling_response


class BInternalConfigurationTests(unittest.TestCase):
    def test_signal_modes_resolve_to_expected_components(self):
        expected = {
            "base": (False, 0.0, 0.0, False),
            "r": (True, 1.0, 0.0, False),
            "r_e": (True, 0.65, 0.35, False),
            "r_d": (True, 1.0, 0.0, True),
            "r_e_d": (True, 0.65, 0.35, True),
        }
        for mode, values in expected.items():
            with self.subTest(mode=mode):
                config = resolve_config(
                    "asr-gs",
                    {"c"},
                    sampling_mode=mode,
                ).sampling
                self.assertEqual(
                    (
                        config.enabled,
                        config.photometric_weight,
                        config.edge_weight,
                        config.coverage_gate_enabled,
                    ),
                    values,
                )
                self.assertEqual(config.signal_mode, mode)

    def test_default_configuration_remains_the_full_b_module(self):
        config = resolve_config("asr-gs").sampling
        self.assertEqual(config.signal_mode, "r_e_d")
        self.assertEqual(config.photometric_weight, 0.65)
        self.assertEqual(config.edge_weight, 0.35)
        self.assertTrue(config.coverage_gate_enabled)

    def test_cli_exposes_the_internal_signal_mode(self):
        args = get_args(
            [
                "-s",
                "/tmp/source",
                "--method",
                "asr-gs",
                "--ablate-c",
                "--b-signal-mode",
                "r_d",
            ]
        )
        self.assertEqual(args.b_signal_mode, "r_d")
        self.assertEqual(args.asr_gs_config.sampling.signal_mode, "r_d")
        self.assertFalse(args.asr_gs_config.refinement.enabled)


class BInternalResponseTests(unittest.TestCase):
    def test_response_combination_uses_the_configured_weights(self):
        photometric = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
        edge = torch.tensor([[8.0, 6.0], [4.0, 2.0]])

        residual_only = combine_sampling_response(
            photometric,
            edge,
            resolve_config("asr-gs", sampling_mode="r").sampling,
        )
        fused = combine_sampling_response(
            photometric,
            edge,
            resolve_config("asr-gs", sampling_mode="r_e").sampling,
        )

        self.assertTrue(torch.equal(residual_only, photometric))
        self.assertTrue(
            torch.allclose(fused, 0.65 * photometric + 0.35 * edge)
        )

    def test_coverage_gate_can_be_disabled_without_changing_the_guide(self):
        photometric = torch.tensor([[1.0, 2.0]])
        edge = torch.tensor([[3.0, 4.0]])
        without_gate = resolve_config("asr-gs", sampling_mode="r_e").sampling
        with_gate = resolve_config("asr-gs", sampling_mode="r_e_d").sampling

        self.assertTrue(
            torch.equal(
                combine_sampling_response(photometric, edge, without_gate),
                combine_sampling_response(photometric, edge, with_gate),
            )
        )
        self.assertFalse(without_gate.coverage_gate_enabled)
        self.assertTrue(with_gate.coverage_gate_enabled)


if __name__ == "__main__":
    unittest.main()
