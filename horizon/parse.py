"""Parse model output into per-step states."""

from __future__ import annotations

import re

from .tasks import State

_STEP = re.compile(r"^\s*\**\s*Step\s+(\d+)\s*\**\s*[:.\-]\s*(.*)$", re.IGNORECASE | re.MULTILINE)
_FINAL = re.compile(r"FINAL\s*:\s*(.*)", re.IGNORECASE)
_ASSIGN = re.compile(r"\b([A-Z])\s*=\s*(\d)\b")


def parse_state(text: str, registers: list[str]) -> State | None:
    found = {k: int(v) for k, v in _ASSIGN.findall(text)}
    if not all(r in found for r in registers):
        return None
    return {r: found[r] for r in registers}


def parse_trace(text: str, registers: list[str]) -> dict[int, State | None]:
    """Map step number -> parsed state. Later duplicates of a step win."""
    out: dict[int, State | None] = {}
    for m in _STEP.finditer(text):
        out[int(m.group(1))] = parse_state(m.group(2), registers)
    return out


def parse_final(text: str, registers: list[str]) -> State | None:
    matches = _FINAL.findall(text)
    if matches:
        return parse_state(matches[-1], registers)
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return parse_state(lines[-1], registers) if lines else None
