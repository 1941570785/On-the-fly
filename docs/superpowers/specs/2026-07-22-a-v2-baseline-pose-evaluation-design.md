# A-v2 Baseline Pose Evaluation Design

## Goal

Evaluate whether pose module A improves pose accuracy under the exact On-the-fly
NVS Table 5 protocol without changing modules B or C and without changing A's
high-level risk-triggered verify-or-fallback architecture.

## Invariants

- A remains `risk score -> conditional pose verification -> accept or exact fallback`.
- Test images never update Gaussians, map points, or the future reference pool.
- No ground-truth pose is available to A or to any online decision.
- B, C, input ordering, test split, optimization budget, and all non-A arguments
  remain identical across variants.
- The paper evaluator remains unchanged: test views, per-scene Sim(3), RMSE,
  translation times 100, rotation in radians, and macro mean over three scenes.

## Variants

- `a_off`: current final model with pose initialization risk mode disabled.
- `a_current`: current final model with `verify_v1`.
- `a_v2`: the same final model with `verify_v2`.

`verify_v2` differs internally in two ways. It may verify a held-out test frame in
pose-only mode, and it accepts a correction only from correspondence evidence not
used to solve that correction. Existing `verify_v1` behavior is preserved.

## A-v2 Verification

The incremental pose initializer records a reference-keyframe ID for every 2D-3D
correspondence. A deterministic splitter reserves complete reference keyframes for
validation when at least two references are present. With one reference, a stable
spatial split is used and explicitly labeled as weaker evidence. The solve subset
drives PnP and MiniBA. Acceptance uses only the validation subset and retains the
existing motion-bound and support-collapse guards. Insufficient independent support
returns the exact input pose.

## Evaluation

The primary table evaluates final test poses with the existing Table 5 evaluator.
The diagnostic table stores and evaluates `T_initial`, `T_A`, and `T_final`. Direct
stage deltas use one common Sim(3) fitted from frames not accepted by A, avoiding a
separate alignment that could absorb A's correction. It reports all eligible frames,
triggered frames, and accepted frames separately, plus success rate, retention rate,
P95, catastrophic failure rate, and repeat variability.

## Execution

Run all nine scenes sequentially on one physical GPU. Use five repeated runs for
each variant. Each run stores the exact command, repository commit, GPU UUID,
timestamps, metadata, risk trace, final paper-protocol metrics, and stage diagnostics.

## Decision Rules

- A is accuracy-effective only if `a_v2` improves final paper-protocol metrics and
  the paired post-A deltas have the same direction.
- A is stabilization-effective if mean error is neutral while P95, catastrophic
  failure rate, or repeat variance decreases.
- If post-A improves but final does not, downstream pose optimization is erasing A.
- If neither stage nor final metrics improve, the current A claim is not supported.
