"""Model clients. Each exposes complete(system, prompt, max_tokens) -> str."""

from __future__ import annotations

import os
import random
import re
import time
from typing import Protocol

from .tasks import Op, apply_op, format_state


class Model(Protocol):
    name: str

    def complete(self, system: str, prompt: str, max_tokens: int) -> str: ...


def _retry(fn, attempts: int = 5):
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(2 ** i + random.random())


class AnthropicModel:
    def __init__(self, model: str, temperature: float = 0.0):
        import anthropic

        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.name = model
        self.temperature = temperature

    def complete(self, system: str, prompt: str, max_tokens: int) -> str:
        def call():
            r = self.client.messages.create(
                model=self.name,
                system=system,
                max_tokens=max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")

        return _retry(call)


class OpenAIModel:
    def __init__(self, model: str, temperature: float = 0.0):
        import openai

        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.name = model
        self.temperature = temperature

    def complete(self, system: str, prompt: str, max_tokens: int) -> str:
        def call():
            r = self.client.chat.completions.create(
                model=self.name,
                temperature=self.temperature,
                max_completion_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            return r.choices[0].message.content or ""

        return _retry(call)


_INIT = re.compile(r"Initial state:\s*(.*)")
_OPLINE = re.compile(r"^Step (\d+): (.*)$", re.MULTILINE)
_ADD = re.compile(r"^([A-Z]) = \(\1 \+ (\d)\) mod 10$")
_COPY = re.compile(r"^([A-Z]) = ([A-Z])$")
_SWAP = re.compile(r"^swap ([A-Z]) and ([A-Z])$")


def _parse_op(text: str) -> Op:
    if m := _ADD.match(text):
        return Op("add", m.group(1), int(m.group(2)))
    if m := _COPY.match(text):
        return Op("copy", m.group(1), m.group(2))
    if m := _SWAP.match(text):
        return Op("swap", m.group(1), m.group(2))
    raise ValueError(f"unrecognized op: {text}")


class SimulatedModel:
    """A fake model with a known per-step error rate, for validating the pipeline.

    Error probability at the d-th step *within one call* is
        p(d) = base + slope * (d - 1)
    slope = 0 is pure compounding (H1); slope > 0 is context degradation (H2).
    Because depth is counted within a call, the reset condition genuinely
    resets it, exactly as H2 predicts for a real model. If the analysis
    cannot recover base and slope from this model, it cannot be trusted on
    real ones.
    """

    def __init__(self, base: float = 0.01, slope: float = 0.0, seed: int = 0, name: str | None = None):
        self.base, self.slope = base, slope
        self.rng = random.Random(seed)
        self.name = name or f"sim(base={base},slope={slope})"

    def complete(self, system: str, prompt: str, max_tokens: int) -> str:
        init_text = _INIT.search(prompt).group(1)
        state = {k: int(v) for k, v in re.findall(r"([A-Z])=(\d)", init_text)}
        regs = list(state)
        steps = [(int(n), _parse_op(t.strip())) for n, t in _OPLINE.findall(prompt)]
        lines = []
        for d, (n, op) in enumerate(steps, start=1):
            state = apply_op(state, op)
            if self.rng.random() < min(1.0, self.base + self.slope * (d - 1)):
                r = self.rng.choice(regs)
                state[r] = self.rng.choice([v for v in range(10) if v != state[r]])
            lines.append(f"Step {n}: {format_state(state)}")
        if "FINAL:" in prompt:
            return f"FINAL: {format_state(state)}"
        return "\n".join(lines)


def load_model(spec: str) -> Model:
    """'anthropic:<model-id>', 'openai:<model-id>', or 'sim:<base>,<slope>'."""
    provider, _, arg = spec.partition(":")
    if provider == "anthropic":
        return AnthropicModel(arg)
    if provider == "openai":
        return OpenAIModel(arg)
    if provider == "sim":
        base, slope = (float(x) for x in arg.split(","))
        return SimulatedModel(base, slope, name=spec)
    raise ValueError(f"unknown model spec: {spec}")
