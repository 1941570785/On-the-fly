# ASR-GS Method Contract

This file states the executable method contract used by the paper
release. Numeric values are defined once in `asr_gs/config.py` and are
serialized into every result.

## Per-Frame Flow

1. The baseline feature matcher constructs 2D-3D correspondences from
   the current image and reference frames.
2. PnP-RANSAC and local MiniBA estimate an initial pose.
3. Module A reviews geometrically weak or failed estimates through at
   most two additional random hypotheses. A candidate replaces the
   initial estimate only after a supported, residual-aware, and
   motion-bounded acceptance test.
4. The accepted pose initializes the baseline direct Gaussian sampling.
5. Module B combines current photometric residual with its spatial
   variation. When the current view has a projection-coverage gap and
   the normalized response is selective, it reallocates the existing
   per-pixel Bernoulli sampling mass and clips the final probabilities
   to `[0, 1]`. A short scene history disables further reallocations
   after repeated non-improving responses.
6. The baseline joint optimization updates pose and Gaussian
   parameters.
7. Module C evaluates the current-view response and representation
   coverage. A qualifying frame requests at most eight additional
   iterations with pose updates disabled.
8. C snapshots visible Gaussian values and optimizer state before the
   update. It commits the best candidate only when current-view and
   recent-reference guards pass; otherwise it restores the snapshot.
9. The baseline anchor clustering and active/stored set maintenance
   incorporate the resulting representation.

## Component Boundaries

- A changes only pose initialization.
- B changes only where a fixed sampling mass is allocated.
- C changes only visible Gaussian parameters after base joint
  optimization.
- None of A, B, or C changes the baseline test split, keyframe policy,
  or anchor lifecycle.

## Auditable Outputs

`metadata.json` contains:

- the full resolved configuration and fingerprint;
- A registration, review, and accepted-review counts;
- B event, applied-event, clipped-pixel, and sampling-mass statistics;
- C request, attempt, commit, rollback, iteration, and runtime
  statistics.

`pose_reliability_trace.json` contains the per-frame A evidence and every
reviewed hypothesis. These outputs are intended to support direct
component ablations rather than infer activation from final rendering
metrics alone.
