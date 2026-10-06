"""Prompts for the three experimental conditions.

trace   one call; the full op list is shown and the model writes the state
        after every step. Context grows with depth.
reset   the same ops in chunks of k. Each call sees only the current state
        (the model's own previous answer) and the next k ops, so context
        stays short no matter how deep we are.
direct  one call; the model gives only the final state, no written steps.
        Baseline for how much the scratchpad helps.
"""

from __future__ import annotations

from .tasks import Op, State, format_state

SYSTEM = (
    "You are a careful state tracker. Follow the instructions exactly and "
    "use the required output format. Do not add commentary."
)


def _op_lines(ops: list[Op], start: int) -> str:
    return "\n".join(f"Step {start + i}: {op.render()}" for i, op in enumerate(ops))


def trace_prompt(registers: list[str], init: State, ops: list[Op], start: int = 1) -> str:
    example = " ".join(f"{r}=<digit>" for r in registers)
    return (
        f"Registers {', '.join(registers)} each hold a digit 0-9.\n"
        f"Initial state: {format_state(init)}\n\n"
        f"Apply these operations in order:\n{_op_lines(ops, start)}\n\n"
        "After EVERY step, write one line with the full state, exactly like:\n"
        f"Step <number>: {example}\n"
        f"Write one line for each step from Step {start} to Step {start + len(ops) - 1}, "
        "and nothing else."
    )


def direct_prompt(registers: list[str], init: State, ops: list[Op]) -> str:
    example = " ".join(f"{r}=<digit>" for r in registers)
    return (
        f"Registers {', '.join(registers)} each hold a digit 0-9.\n"
        f"Initial state: {format_state(init)}\n\n"
        f"Apply these operations in order:\n{_op_lines(ops, 1)}\n\n"
        "Do not show your work. Reply with exactly one line giving the final state:\n"
        f"FINAL: {example}"
    )
