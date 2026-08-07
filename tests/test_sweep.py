"""Sweep spec expansion + resumable executor (architecture.md §8, sprints/sprint2.md Phase 2.4)."""

from ladder.storage import connect, list_runs
from ladder.sweep import (
    SweepModel,
    SweepSpec,
    SweepTarget,
    _sweep_run_key,
    expand_sweep,
    load_sweep_spec,
    run_sweep,
)

_SPEC = SweepSpec(
    models=[
        SweepModel(model_id="dummy", revisions=["rev-a", "rev-b"]),
    ],
    targets=[
        SweepTarget(dataset="arc_easy", variant="arc_easy/mc_letter_v1", evaluator="loglik_mc", split="fixture"),
        SweepTarget(dataset="hellaswag", variant="hellaswag/mc_context_v1", evaluator="loglik_mc", split="fixture"),
    ],
    limit=5,
    seed=0,
)


def test_expand_sweep_produces_models_outer_targets_inner_order():
    runs = expand_sweep(_SPEC)
    assert len(runs) == 4  # 2 revisions x 2 targets

    assert [(r.revision, r.dataset) for r in runs] == [
        ("rev-a", "arc_easy"),
        ("rev-a", "hellaswag"),
        ("rev-b", "arc_easy"),
        ("rev-b", "hellaswag"),
    ]
    for r in runs:
        assert r.model_id == "dummy"
        assert r.limit == 5
        assert r.seed == 0


def test_expand_sweep_groups_by_model_and_revision_are_contiguous():
    spec = SweepSpec(
        models=[
            SweepModel(model_id="dummy", revisions=["rev-a", "rev-b"]),
            SweepModel(model_id="dummy2", revisions=["rev-c"]),
        ],
        targets=_SPEC.targets,
        limit=5,
    )
    runs = expand_sweep(spec)
    seen_groups = []
    for r in runs:
        group = (r.model_id, r.revision)
        if not seen_groups or seen_groups[-1] != group:
            seen_groups.append(group)
    # Each distinct (model, revision) group appears exactly once as a contiguous block.
    assert seen_groups == [("dummy", "rev-a"), ("dummy", "rev-b"), ("dummy2", "rev-c")]
    assert len(set(seen_groups)) == len(seen_groups)


def test_load_sweep_spec_reads_yaml(tmp_path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        """
models:
  - model_id: dummy
    revisions: [main]
targets:
  - dataset: arc_easy
    variant: arc_easy/mc_letter_v1
    evaluator: loglik_mc
    split: fixture
limit: 3
seed: 7
""",
        encoding="utf-8",
    )
    spec = load_sweep_spec(spec_path)
    assert spec.models[0].model_id == "dummy"
    assert spec.limit == 3
    assert spec.seed == 7


def test_run_sweep_executes_all_runs_and_marks_them_done(tmp_path):
    conn = connect(tmp_path / "ladder.db")
    executed = run_sweep(conn, _SPEC)

    assert len(executed) == 4
    assert all(r.status == "done" for r in executed)
    assert all(r.n_examples == 5 for r in executed)

    runs = list_runs(conn)
    assert len(runs) == 4


def test_run_sweep_resume_skips_done_runs_with_zero_recomputation(tmp_path):
    db_path = tmp_path / "ladder.db"
    conn = connect(db_path)

    # Simulate an interrupted sweep: only the first two (model, revision, target)
    # combinations have completed.
    partial_spec = SweepSpec(
        models=[SweepModel(model_id="dummy", revisions=["rev-a"])],
        targets=_SPEC.targets,
        limit=5,
        seed=0,
    )
    first_batch = run_sweep(conn, partial_spec)
    assert len(first_batch) == 2
    assert all(r.status == "done" for r in first_batch)

    # Rerun the full spec against the same DB ("kill and resume"): the two
    # already-done runs must be skipped entirely (no new row, no client call),
    # and only the remaining two runs execute.
    second_batch = run_sweep(conn, _SPEC)
    assert len(second_batch) == 2
    assert {(r.revision, r.dataset) for r in second_batch} == {
        ("rev-b", "arc_easy"),
        ("rev-b", "hellaswag"),
    }

    all_runs = list_runs(conn)
    assert len(all_runs) == 4  # no duplicate rows for the resumed (rev-a) runs

    # Every request in this sweep was already issued (and cached) during the
    # first batch's rev-a runs, since DummyClient's output only depends on
    # (seed, model_id, revision, prompt, continuations) not on run_id — so the
    # newly executed rev-b runs are genuinely new work, but nothing from rev-a
    # was recomputed: exactly 4 run rows total, never more.
    done_keys = {_sweep_run_key(r) for r in expand_sweep(_SPEC)}
    stored_keys = {
        (r.model_id, r.revision, r.dataset, r.split, r.prompt_variant_id, r.evaluator, r.config.get("limit"), r.seed)
        for r in all_runs
    }
    assert stored_keys == done_keys


def test_run_sweep_continues_after_a_failed_run(tmp_path):
    conn = connect(tmp_path / "ladder.db")
    spec = SweepSpec(
        models=[SweepModel(model_id="dummy", revisions=["main"])],
        targets=[
            SweepTarget(dataset="not_a_real_dataset", variant="arc_easy/mc_letter_v1", evaluator="loglik_mc", split="fixture"),
            SweepTarget(dataset="arc_easy", variant="arc_easy/mc_letter_v1", evaluator="loglik_mc", split="fixture"),
        ],
        limit=5,
    )
    executed = run_sweep(conn, spec)

    assert len(executed) == 2
    assert executed[0].status == "failed"
    assert executed[1].status == "done"

    runs = list_runs(conn)
    assert len(runs) == 2
