import pytest

from horizon.analyze import bootstrap_fit, load, step_table
from horizon.models import SimulatedModel
from horizon.parse import parse_final, parse_trace
from horizon.prompts import trace_prompt
from horizon.run import main as run_main
from horizon.run import run_trial, score_trace
from horizon.tasks import Op, apply_op, format_state, make_task


def test_ops():
    s = {"A": 7, "B": 2}
    assert apply_op(s, Op("add", "A", 5)) == {"A": 2, "B": 2}
    assert apply_op(s, Op("copy", "A", "B")) == {"A": 2, "B": 2}
    assert apply_op(s, Op("swap", "A", "B")) == {"A": 2, "B": 7}
    assert s == {"A": 7, "B": 2}


def test_task_is_deterministic_and_consistent():
    t1, t2 = make_task(50, 5, seed=3), make_task(50, 5, seed=3)
    assert t1.to_dict() == t2.to_dict()
    assert len(t1.states) == 51
    for i, op in enumerate(t1.ops):
        assert apply_op(t1.states[i], op) == t1.states[i + 1]


def test_parse_tolerates_formatting():
    text = "**Step 1:** A=1 B=2\nStep 2 - A = 3, B = 4\nnoise\nStep 3: A=5"
    p = parse_trace(text, ["A", "B"])
    assert p[1] == {"A": 1, "B": 2}
    assert p[2] == {"A": 3, "B": 4}
    assert p[3] is None
    assert parse_final("thinking...\nFINAL: A=9 B=0", ["A", "B"]) == {"A": 9, "B": 0}


def test_perfect_model_scores_perfectly():
    task = make_task(20, 4, seed=1)
    text = "\n".join(f"Step {i}: {format_state(s)}" for i, s in enumerate(task.states[1:], 1))
    r = score_trace(task, parse_trace(text, task.registers))
    assert r["final_ok"] and r["first_error"] is None
    assert all(r["local_ok"]) and all(r["global_ok"])


def test_local_score_isolates_single_error():
    task = make_task(10, 4, seed=2)
    states = {i: dict(s) for i, s in enumerate(task.states[1:], 1)}
    wrong = dict(states[5])
    wrong["A"] = (wrong["A"] + 1) % 10
    states[5] = wrong
    for i in range(6, 11):
        states[i] = apply_op(states[i - 1], task.ops[i - 1])
    r = score_trace(task, states)
    assert r["local_ok"].count(False) == 1 and r["local_ok"][4] is False
    assert r["first_error"] == 5


def test_simulator_reads_our_own_prompt():
    task = make_task(12, 5, seed=4)
    out = SimulatedModel(0.0, 0.0).complete("", trace_prompt(task.registers, task.init, task.ops), 1000)
    assert parse_trace(out, task.registers)[12] == task.final


@pytest.mark.parametrize("cond", ["trace", "reset", "direct"])
def test_conditions_run(cond):
    r = run_trial(SimulatedModel(0.0, 0.0), make_task(17, 5, seed=5), cond, chunk=4)
    assert r["final_ok"] is True


def test_analysis_recovers_known_slope(tmp_path):
    out = tmp_path / "t.jsonl"
    run_main(["--model", "sim:0.01,0.0015", "--lengths", "64", "--trials", "60",
              "--conditions", "trace", "--workers", "1", "--out", str(out)])
    fit = bootstrap_fit(step_table(load(out)), reps=100)
    assert fit["slope_ci"][0] < 0.0015 < fit["slope_ci"][1]
    assert fit["slope_ci"][0] > 0
