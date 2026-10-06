"""Synthetic multi-step state-tracking tasks with constant per-step difficulty.

A task is a set of registers (A, B, C, ...) holding digits 0-9 and a sequence
of N operations. Every operation has the same form and difficulty at every
depth, so any change in error rate with depth comes from depth itself, not
from the task getting harder.

Operations (all arithmetic is mod 10, so values never grow):
    add   X c      X = (X + c) mod 10
    copy  X Y      X = Y
    swap  X Y      exchange X and Y
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field, asdict

OP_KINDS = ("add", "copy", "swap")

State = dict[str, int]


@dataclass(frozen=True)
class Op:
    kind: str
    target: str
    arg: str | int

    def render(self) -> str:
        if self.kind == "add":
            return f"{self.target} = ({self.target} + {self.arg}) mod 10"
        if self.kind == "copy":
            return f"{self.target} = {self.arg}"
        if self.kind == "swap":
            return f"swap {self.target} and {self.arg}"
        raise ValueError(self.kind)


def apply_op(state: State, op: Op) -> State:
    s = dict(state)
    if op.kind == "add":
        s[op.target] = (s[op.target] + int(op.arg)) % 10
    elif op.kind == "copy":
        s[op.target] = s[str(op.arg)]
    elif op.kind == "swap":
        s[op.target], s[str(op.arg)] = s[str(op.arg)], s[op.target]
    else:
        raise ValueError(op.kind)
    return s


@dataclass
class Task:
    task_id: str
    registers: list[str]
    init: State
    ops: list[Op]
    states: list[State] = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        return len(self.ops)

    @property
    def final(self) -> State:
        return self.states[-1]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ops"] = [asdict(o) for o in self.ops]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        return cls(
            task_id=d["task_id"],
            registers=d["registers"],
            init=d["init"],
            ops=[Op(**o) for o in d["ops"]],
            states=d["states"],
        )


def make_task(n_steps: int, n_registers: int = 5, seed: int = 0) -> Task:
    if not 2 <= n_registers <= 26:
        raise ValueError("n_registers must be between 2 and 26")
    rng = random.Random(seed)
    regs = list(string.ascii_uppercase[:n_registers])
    init = {r: rng.randrange(10) for r in regs}
    ops: list[Op] = []
    for _ in range(n_steps):
        kind = rng.choice(OP_KINDS)
        target = rng.choice(regs)
        if kind == "add":
            arg: str | int = rng.randrange(1, 10)
        else:
            arg = rng.choice([r for r in regs if r != target])
        ops.append(Op(kind, target, arg))
    states = [init]
    for op in ops:
        states.append(apply_op(states[-1], op))
    return Task(f"n{n_steps}_r{n_registers}_s{seed}", regs, init, ops, states)


def format_state(state: State) -> str:
    return " ".join(f"{k}={v}" for k, v in state.items())
