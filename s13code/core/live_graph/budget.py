"""A per-run work budget for the live graph.

Session 12 invariant 8 requires that "every run must have hard limits on time,
tokens, tool calls, and cost". The live graph enforces none: a planner may keep
expanding the frontier for as long as each outcome suggests more work, so a
single request can launch unbounded tasks (each one an LLM or network call).

``RunBudget`` caps how many tasks a run may *launch*. When the cap is reached the
graph takes a **budget-aware branch**: it drops the remaining backlog instead of
paying for it, and the planner finalises an answer from the evidence already
gathered. The budget bounds the work frontier — it never suppresses the
finalisation step, because a run that spends its budget and returns nothing is
strictly worse than one that answers from partial evidence.

The default is ``None`` (unlimited), so an unconfigured store behaves exactly as
before and the cap is opt-in per deployment or per test.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunBudget:
    """Hard limit on task launches per run. ``None`` means unlimited."""

    max_task_launches: int | None = None

    def __post_init__(self) -> None:
        if self.max_task_launches is not None and self.max_task_launches < 1:
            raise ValueError("max_task_launches must be at least one when set")

    @property
    def enabled(self) -> bool:
        return self.max_task_launches is not None

    def remaining(self, launches: int) -> int | None:
        """Launches still permitted, or None when unlimited."""
        if self.max_task_launches is None:
            return None
        return max(0, self.max_task_launches - launches)

    def exhausted(self, launches: int) -> bool:
        if self.max_task_launches is None:
            return False
        return launches >= self.max_task_launches
