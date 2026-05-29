from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    tau_R_low: float = 0.40
    tau_R_high: float = 0.75
    tau_V: float = 0.55
    tau_V_min: float = 0.25
    tau_Q: float = 0.10
