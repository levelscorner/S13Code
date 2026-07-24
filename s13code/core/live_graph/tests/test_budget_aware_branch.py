"""Budget-aware branch: a per-run cap on task launches.

Session 12 invariant 8 requires hard per-run limits on time, tokens, tool calls
and cost. The live graph enforced none — a planner could keep expanding the
frontier indefinitely, so one request could launch unbounded LLM/network work.

With a budget the graph takes a *branch* instead of overspending: it drops the
remaining backlog and the planner finalises an answer from the evidence already
gathered. These tests prove the cap is enforced *before* work starts, that
budget-refused work never runs, and that a task cancelled mid-flight by the
budget can never leak a late result into the graph.
"""

import asyncio

import pytest

from s13code.core.live_graph import GraphPatch, GraphStore, LiveGraphExecutor, RunBudget, TaskSpec


class ScriptedPlanner:
    def __init__(self, script):
        self.script = script

    async def plan(self, graph, event):
        return self.script(graph, event)


def _four_items_then_finalise(graph, event):
    """Queue four work items, then finalise once everything is terminal."""
    if event.kind == "run_started":
        return GraphPatch(
            add=tuple(TaskSpec(f"w{i}", "work") for i in range(1, 5)),
            reason="four independent work items",
        )
    terminal = ("succeeded", "failed", "cancelled")
    if graph.nodes and all(node["state"] in terminal for node in graph.nodes.values()):
        if "answer" not in graph.nodes:
            return GraphPatch(add=(TaskSpec("answer", "answer"),), reason="budget-aware finalisation")
        return GraphPatch(finish=True, reason="answered from partial evidence")
    return GraphPatch()


def _events(store, run_id, kind, node_id=None):
    return [e for e in store.events(run_id)
            if e.kind == kind and (node_id is None or e.node_id == node_id)]


# ── unit ──────────────────────────────────────────────────────────────────────

def test_budget_defaults_to_unlimited():
    budget = RunBudget()
    assert not budget.enabled
    assert budget.remaining(10_000) is None
    assert not budget.exhausted(10_000)


def test_budget_counts_and_validates():
    budget = RunBudget(max_task_launches=3)
    assert budget.enabled
    assert budget.remaining(0) == 3 and budget.remaining(2) == 1
    assert budget.remaining(5) == 0  # never negative
    assert not budget.exhausted(2) and budget.exhausted(3) and budget.exhausted(4)
    with pytest.raises(ValueError):
        RunBudget(max_task_launches=0)


# ── behaviour: the cap is enforced before work starts, then the run finalises ──

@pytest.mark.asyncio
async def test_budget_stops_new_work_and_still_answers(tmp_path):
    launched: list[str] = []

    async def work(task):
        launched.append(task.id)
        return {"ok": task.id}

    async def answer(task):
        return {"answer": "partial evidence answer"}

    store = GraphStore(tmp_path / "g.db", budget=RunBudget(max_task_launches=2))
    report = await LiveGraphExecutor(
        store, ScriptedPlanner(_four_items_then_finalise),
        {"work": work, "answer": answer}, max_workers=1,
    ).run("budget")

    # exactly two work items were paid for; the rest never ran
    assert launched == ["w1", "w2"]
    assert [n for n in ("w3", "w4") if _events(store, "budget", "task_started", n)] == []
    snapshot = store.snapshot("budget")
    assert snapshot.nodes["w3"]["state"] == "cancelled"
    assert snapshot.nodes["w4"]["state"] == "cancelled"
    # the budget branch was journalled once, with its accounting
    exhausted = _events(store, "budget", "budget_exhausted")
    assert len(exhausted) == 1
    assert exhausted[0].payload["launches"] == 2
    assert exhausted[0].payload["max_task_launches"] == 2
    assert sorted(exhausted[0].payload["cancelled"]) == ["w3", "w4"]
    # and the run still produced an answer instead of returning nothing
    assert snapshot.nodes["answer"]["state"] == "succeeded"
    assert report.finished


@pytest.mark.asyncio
async def test_unbudgeted_store_is_unchanged(tmp_path):
    """Without a budget every queued item still runs (no behaviour change)."""
    launched: list[str] = []

    async def work(task):
        launched.append(task.id)
        return {"ok": task.id}

    async def answer(task):
        return {"answer": "full"}

    store = GraphStore(tmp_path / "g.db")  # unlimited
    report = await LiveGraphExecutor(
        store, ScriptedPlanner(_four_items_then_finalise),
        {"work": work, "answer": answer}, max_workers=1,
    ).run("nobudget")
    assert sorted(launched) == ["w1", "w2", "w3", "w4"]
    assert report.finished
    assert _events(store, "nobudget", "budget_exhausted") == []


# ── Part 3, attack the claim: a late result from budget-cancelled work ────────

@pytest.mark.asyncio
async def test_adversarial_late_result_from_budget_cancelled_task_never_lands(tmp_path):
    """Attack: two tasks launch together and exhaust the budget. The first one
    finishes and trips the branch while the second is *still running*. That
    in-flight task must be cancelled and its late result must never be recorded
    as an outcome, never appear in the graph, and never reach the answer.

    On the pre-feature store this fails: with no cap all four items run, nothing
    is cancelled, and no budget_exhausted event exists."""

    async def work(task):
        if task.id == "w2":
            await asyncio.sleep(0.25)  # still in flight when w1 trips the cap
            return {"leaked": True}
        return {"ok": task.id}

    async def answer(task):
        snapshot = store.snapshot("attack")
        # the answer worker must not see the cancelled task's payload
        return {"answer": "partial", "saw": [n for n, node in snapshot.nodes.items()
                                             if (node.get("result") or {}).get("leaked")]}

    store = GraphStore(tmp_path / "g.db", budget=RunBudget(max_task_launches=2))
    report = await asyncio.wait_for(
        LiveGraphExecutor(
            store, ScriptedPlanner(_four_items_then_finalise),
            {"work": work, "answer": answer}, max_workers=2,
        ).run("attack"),
        timeout=10,
    )

    snapshot = store.snapshot("attack")
    # w2 was cancelled mid-flight ...
    assert snapshot.nodes["w2"]["state"] == "cancelled"
    cancels = _events(store, "attack", "task_cancelled", "w2")
    assert len(cancels) == 1 and cancels[0].payload["was_running"] is True
    assert cancels[0].payload["budget_cancelled"] is True
    # ... and its late result never became an outcome or graph state
    assert _events(store, "attack", "task_succeeded", "w2") == []
    assert snapshot.nodes["w2"]["result"] is None
    # ... nor reached the answer
    assert snapshot.nodes["answer"]["result"]["saw"] == []
    # the budget still let the run finalise
    assert _events(store, "attack", "budget_exhausted")
    assert report.finished
