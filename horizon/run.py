"""Run the experiment and write one JSON line per trial.

Example:
    python -m horizon.run --model anthropic:<model-id> --lengths 8 16 32 64 128 --trials 30
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .models import Model, load_model
from .parse import parse_final, parse_trace
from .prompts import SYSTEM, direct_prompt, trace_prompt
from .tasks import Task, apply_op, make_task

CONDITIONS = ("trace", "reset", "direct")


def _budget(n_steps: int, n_regs: int) -> int:
    return n_steps * (6 * n_regs + 12) + 256


def score_trace(task: Task, model_states: dict[int, dict | None]) -> dict:
    """Score every step two ways.

    global_ok[i]  model's state after step i equals the true state.
    local_ok[i]   model's state after step i equals the op applied to the
                  model's OWN state after step i-1. This isolates the error
                  made at step i from errors inherited from earlier steps,
                  so it measures the per-step hazard at every depth.
    """
    n = task.n_steps
    prev = task.init
    local_ok: list[bool | None] = []
    global_ok: list[bool | None] = []
    for i in range(1, n + 1):
        cur = model_states.get(i)
        global_ok.append(None if cur is None else cur == task.states[i])
        if cur is None or prev is None:
            local_ok.append(None)
        else:
            local_ok.append(cur == apply_op(prev, task.ops[i - 1]))
        prev = cur
    first_error = next((i + 1 for i, g in enumerate(global_ok) if g is not True), None)
    return {
        "local_ok": local_ok,
        "global_ok": global_ok,
        "final_ok": global_ok[-1] is True,
        "first_error": first_error,
        "n_unparsed": sum(s is None for s in global_ok),
    }


def run_trial(model: Model, task: Task, condition: str, chunk: int) -> dict:
    regs = task.registers
    if condition == "trace":
        text = model.complete(SYSTEM, trace_prompt(regs, task.init, task.ops), _budget(task.n_steps, len(regs)))
        result = score_trace(task, parse_trace(text, regs))
    elif condition == "reset":
        states: dict[int, dict | None] = {}
        texts = []
        cur = task.init
        for start in range(0, task.n_steps, chunk):
            ops = task.ops[start : start + chunk]
            if cur is None:
                break
            t = model.complete(SYSTEM, trace_prompt(regs, cur, ops, start=start + 1), _budget(len(ops), len(regs)))
            texts.append(t)
            parsed = parse_trace(t, regs)
            for i in range(start + 1, start + len(ops) + 1):
                states[i] = parsed.get(i)
            cur = states[start + len(ops)]
        text = "\n---\n".join(texts)
        result = score_trace(task, states)
    elif condition == "direct":
        text = model.complete(SYSTEM, direct_prompt(regs, task.init, task.ops), 256)
        final = parse_final(text, regs)
        result = {"final_ok": final == task.final, "n_unparsed": int(final is None)}
    else:
        raise ValueError(condition)
    return {
        "model": model.name,
        "condition": condition,
        "chunk": chunk if condition == "reset" else None,
        "task_id": task.task_id,
        "n_steps": task.n_steps,
        "n_registers": len(regs),
        **result,
        "raw": text,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="anthropic:<id> | openai:<id> | sim:<base>,<slope>")
    ap.add_argument("--lengths", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--trials", type=int, default=30, help="tasks per length")
    ap.add_argument("--registers", type=int, default=5)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    ap.add_argument("--chunk", type=int, default=8, help="steps per call in the reset condition")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/trials.jsonl")
    args = ap.parse_args(argv)

    model = load_model(args.model)
    try:
        if not args.model.startswith("sim:"):
            model.complete(SYSTEM, "Reply with the word OK.", 16)
    except Exception as e:
        raise SystemExit(
            f"Preflight call to {args.model} failed, so nothing was run.\n"
            f"  {type(e).__name__}: {e}\n"
            "Check that the API key is set (Colab: Secrets panel, notebook access on) "
            "and that the model ID is correct."
        )
    jobs = [
        (make_task(n, args.registers, seed=1000 * n + t), c)
        for n in args.lengths
        for t in range(args.trials)
        for c in args.conditions
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = failed = 0
    with out.open("a") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_trial, model, task, c, args.chunk): (task, c) for task, c in jobs}
        for fut in as_completed(futures):
            done += 1
            try:
                f.write(json.dumps(fut.result()) + "\n")
                f.flush()
            except Exception as e:
                failed += 1
                task, c = futures[fut]
                print(f"FAILED {task.task_id} [{c}]: {type(e).__name__}: {e}", flush=True)
                if failed >= 10 and failed == done:
                    pool.shutdown(cancel_futures=True)
                    raise SystemExit("First 10 trials all failed; stopping. See errors above.")
            if done % max(25, len(jobs) // 10) == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} trials ({failed} failed)", flush=True)


if __name__ == "__main__":
    main()
