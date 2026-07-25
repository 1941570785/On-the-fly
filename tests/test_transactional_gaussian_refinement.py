from __future__ import annotations

import torch

import scene.transactional_gaussian_refinement as transactional_refinement
from scene.transactional_gaussian_refinement import (
    DEFAULT_PARAMETER_GRADIENT_SCALES,
    overwrite_selected_gaussian_snapshot,
    refinement_candidate_acceptance,
    refinement_reference_guard,
    refinement_time_budget_seconds,
    rendering_response_gap,
    restore_selected_gaussian_state,
    scale_gaussian_gradients,
    select_refinement_reference_indices,
    snapshot_selected_gaussian_state,
)


def _parameter_state(rows: int = 4) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "xyz": {
            "val": torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3),
            "exp_avg": torch.ones(rows, 3),
            "exp_avg_sq": torch.ones(rows, 3) * 2,
            "lr": torch.arange(rows, dtype=torch.float32) + 0.1,
        },
        "f_dc": {
            "val": torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 1, 3),
            "exp_avg": torch.ones(rows, 1, 3) * 3,
            "exp_avg_sq": torch.ones(rows, 1, 3) * 4,
            "lr": torch.tensor(0.01),
        },
    }


def test_sparse_snapshot_restore_recovers_values_moments_and_learning_rates():
    selection = torch.tensor([False, True, False, True])
    params = _parameter_state()
    originals = {
        name: {key: value.clone() for key, value in state.items()}
        for name, state in params.items()
    }

    snapshot = snapshot_selected_gaussian_state(params, selection)
    for state in params.values():
        for key, value in state.items():
            if value.ndim > 0 and value.shape[0] == selection.numel():
                value[selection] += 100
            else:
                value.add_(100)

    restore_selected_gaussian_state(params, selection, snapshot)

    for name, state in params.items():
        for key, expected in originals[name].items():
            assert torch.equal(state[key], expected), (name, key)


def test_gradient_scaling_is_conservative_for_geometry_and_updates_all_groups():
    params = {}
    for name in ("xyz", "scaling", "rotation", "opacity", "f_dc", "f_rest"):
        value = torch.ones(4, 3, requires_grad=True)
        value.grad = torch.ones_like(value)
        params[name] = {"val": value}

    scale_gaussian_gradients(params)

    for name, expected in DEFAULT_PARAMETER_GRADIENT_SCALES.items():
        assert params[name]["val"].grad is not None
        assert torch.allclose(
            params[name]["val"].grad,
            torch.full_like(params[name]["val"], expected),
        )


def test_existing_snapshot_can_be_overwritten_with_the_best_candidate_in_place():
    selection = torch.tensor([False, True, False, True])
    params = _parameter_state()
    snapshot = snapshot_selected_gaussian_state(params, selection)

    for state in params.values():
        state["val"][selection] += 10
        state["exp_avg"][selection] += 20
        state["exp_avg_sq"][selection] += 30
        if state["lr"].ndim > 0:
            state["lr"][selection] += 40
    expected = {
        name: {key: value.clone() for key, value in state.items()}
        for name, state in params.items()
    }
    overwrite_selected_gaussian_snapshot(params, selection, snapshot)

    for state in params.values():
        state["val"][selection] += 100
        state["exp_avg"][selection] += 100
        state["exp_avg_sq"][selection] += 100
        if state["lr"].ndim > 0:
            state["lr"][selection] += 100
    restore_selected_gaussian_state(params, selection, snapshot)

    for name, state in params.items():
        for key, value in state.items():
            assert torch.equal(value, expected[name][key]), (name, key)


def test_best_candidate_keeps_values_without_committing_optimizer_state():
    selection = torch.tensor([False, True, False, True])
    params = _parameter_state()
    initial = {
        name: {key: value.clone() for key, value in state.items()}
        for name, state in params.items()
    }
    snapshot = snapshot_selected_gaussian_state(params, selection)

    for state in params.values():
        state["val"][selection] += 10
        state["exp_avg"][selection] += 20
        state["exp_avg_sq"][selection] += 30
        if state["lr"].ndim > 0:
            state["lr"][selection] += 40
    candidate_values = {
        name: state["val"].clone()
        for name, state in params.items()
    }
    transactional_refinement.overwrite_selected_gaussian_value_snapshot(
        params,
        selection,
        snapshot,
    )

    for state in params.values():
        state["val"][selection] += 100
        state["exp_avg"][selection] += 100
        state["exp_avg_sq"][selection] += 100
        if state["lr"].ndim > 0:
            state["lr"][selection] += 100
    restore_selected_gaussian_state(params, selection, snapshot)

    for name, state in params.items():
        assert torch.equal(state["val"], candidate_values[name]), name
        for key in ("exp_avg", "exp_avg_sq", "lr"):
            assert torch.equal(state[key], initial[name][key]), (name, key)


def test_internal_refinement_budget_targets_eight_percent_of_base_runtime():
    assert refinement_time_budget_seconds(100.0, 3.0) == 5.0
    assert refinement_time_budget_seconds(100.0, 9.0) == 0.0


def test_rendering_response_gap_ignores_uniform_error_and_detects_local_underfit():
    uniform = torch.ones(16, 16) * 0.1
    localized = torch.ones(16, 16) * 0.01
    localized[4:12, 4:12] = 0.40

    uniform_gap = rendering_response_gap(uniform)
    localized_gap = rendering_response_gap(localized)

    assert uniform_gap["coverage_deficit"] == 0.0
    assert uniform_gap["selective"] is False
    assert localized_gap["selective"] is True
    assert 0.08 <= localized_gap["coverage_deficit"] <= 0.25


def test_reference_selection_uses_recent_historical_training_views_only():
    indices = select_refinement_reference_indices(
        [True, False, True, False, False, True, False],
        current_index=6,
        max_references=2,
    )

    assert indices == [3, 4]


def test_reference_guard_rejects_cross_view_regression():
    reference = [
        {"total_loss": 1.0, "rgb_mse": 0.10},
        {"total_loss": 2.0, "rgb_mse": 0.20},
    ]
    accepted = refinement_reference_guard(
        reference,
        [
            {"total_loss": 0.999, "rgb_mse": 0.099},
            {"total_loss": 2.001, "rgb_mse": 0.2001},
        ],
    )
    rejected = refinement_reference_guard(
        reference,
        [
            {"total_loss": 0.999, "rgb_mse": 0.099},
            {"total_loss": 2.001, "rgb_mse": 0.203},
        ],
    )

    assert accepted["accepted"] is True
    assert rejected["accepted"] is False
    assert rejected["reason"] == "reference_rgb_mse_guard_failed"


def test_candidate_requires_total_and_rgb_improvement_with_bounded_auxiliary_losses():
    reference = {
        "total_loss": 1.0,
        "rgb_mse": 0.10,
        "dssim": 0.20,
        "depth": 0.30,
    }
    accepted = refinement_candidate_acceptance(
        reference,
        {
            "total_loss": 0.98,
            "rgb_mse": 0.099,
            "dssim": 0.2003,
            "depth": 0.302,
        },
    )
    rgb_rejected = refinement_candidate_acceptance(
        reference,
        {
            "total_loss": 0.98,
            "rgb_mse": 0.101,
            "dssim": 0.19,
            "depth": 0.29,
        },
    )
    structure_rejected = refinement_candidate_acceptance(
        reference,
        {
            "total_loss": 0.98,
            "rgb_mse": 0.099,
            "dssim": 0.201,
            "depth": 0.29,
        },
    )
    depth_rejected = refinement_candidate_acceptance(
        reference,
        {
            "total_loss": 0.98,
            "rgb_mse": 0.099,
            "dssim": 0.19,
            "depth": 0.304,
        },
    )

    assert accepted["accepted"] is True
    assert rgb_rejected["reason"] == "rgb_mse_not_improved"
    assert structure_rejected["reason"] == "dssim_guard_failed"
    assert depth_rejected["reason"] == "depth_guard_failed"


def test_non_finite_candidate_is_rejected():
    decision = refinement_candidate_acceptance(
        {
            "total_loss": 1.0,
            "rgb_mse": 0.10,
            "dssim": 0.20,
            "depth": 0.30,
        },
        {
            "total_loss": float("nan"),
            "rgb_mse": 0.09,
            "dssim": 0.19,
            "depth": 0.29,
        },
    )

    assert decision["accepted"] is False
    assert decision["reason"] == "non_finite_candidate"
