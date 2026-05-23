"""Phase scheduling for the 3-phase end-to-end trainer (spec §6.2 / §6.3).

Epoch → phase mapping (defaults from spec §7):
    0   .. 19   warmup
    20  .. 59   joint
    60  .. 79   refinement

Boundaries are inclusive on the low end, exclusive on the high end.

The scheduler returns the phase name; the e2e model uses
``InATToE2E.set_phase(name)`` to freeze/unfreeze the appropriate params
and to look up the loss-weight dict at forward time.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class PhaseSchedule:
    warmup_end: int = 20      # exclusive
    joint_end:  int = 60      # exclusive
    total_epochs: int = 80

    def __post_init__(self):
        if not (0 < self.warmup_end <= self.joint_end <= self.total_epochs):
            raise ValueError(
                f"Invalid schedule: warmup_end={self.warmup_end}  "
                f"joint_end={self.joint_end}  total={self.total_epochs}"
            )

    def phase_of(self, epoch: int) -> str:
        if epoch < self.warmup_end:
            return "warmup"
        if epoch < self.joint_end:
            return "joint"
        return "refinement"
