"""Analyze trial results: does per-step error rise with depth?

    python -m horizon.analyze results/trials.jsonl --out results/

Outputs summary.md plus three figures:
    accuracy_vs_length.png   final-state accuracy by task length and condition
    hazard_vs_depth.png      per-step (local) error rate by depth, with fitted line
    compounding_check.png    observed error-free rate vs what a constant
                             per-step error (H1) would predict
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"trace": "#2a6fdb", "reset": "#d9822b", "direct": "#7a7a7a"}
SHALLOW = 8  # depths 1..SHALLOW estimate the baseline per-step error for H1


def load(path: str | Path) -> pd.DataFrame:
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(rows)
    df["trial"] = df["model"] + "|" + df["condition"] + "|" + df["task_id"]
    return df


def step_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per scorable step: depth and whether that step was an error."""
    recs = []
    for r in df[df.condition.isin(["trace", "reset"])].itertuples():
        for d, ok in enumerate(r.local_ok, start=1):
            if ok is not None:
                recs.append((r.model, r.condition, r.trial, r.n_steps, d, 0 if ok else 1))
    return pd.DataFrame(recs, columns=["model", "condition", "trial", "n_steps", "depth", "err"])


def fit_line(steps: pd.DataFrame) -> tuple[float, float]:
    """Linear probability model err = base + slope * (depth - 1)."""
    x = steps["depth"].to_numpy(float) - 1
    y = steps["err"].to_numpy(float)
    if len(x) < 2 or np.ptp(x) == 0:
        return float(y.mean()) if len(y) else float("nan"), float("nan")
    slope, base = np.polyfit(x, y, 1)
    return float(base), float(slope)


def bootstrap_fit(steps: pd.DataFrame, reps: int = 300, seed: int = 0):
    """Resample whole trials (steps within a trial are not independent)."""
    rng = np.random.default_rng(seed)
    groups = {t: g for t, g in steps.groupby("trial")}
    keys = list(groups)
    base, slope = fit_line(steps)
    samples = []
    for _ in range(reps):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        samples.append(fit_line(pd.concat([groups[keys[i]] for i in pick])))
    s = np.array(samples)
    lo, hi = np.nanpercentile(s, [2.5, 97.5], axis=0)
    return {"base": base, "slope": slope, "base_ci": (lo[0], hi[0]), "slope_ci": (lo[1], hi[1])}


def summarize(df: pd.DataFrame, steps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    acc = df.groupby(["model", "condition", "n_steps"]).agg(
        final_acc=("final_ok", "mean"), n=("final_ok", "size"), unparsed=("n_unparsed", "mean")
    ).reset_index()

    fits = []
    for (m, c), g in steps.groupby(["model", "condition"]):
        b = bootstrap_fit(g)
        fits.append({"model": m, "condition": c, **b, "steps_scored": len(g)})
    fits = pd.DataFrame(fits)

    # H1 check: if per-step error were constant at its shallow value h,
    # P(no error in N steps) = (1 - h)^N. Compare with what we observe.
    checks = []
    for (m, c), g in df[df.condition.isin(["trace", "reset"])].groupby(["model", "condition"]):
        sg = steps[(steps.model == m) & (steps.condition == c)]
        h = sg[sg.depth <= SHALLOW]["err"].mean()
        for n, gn in g.groupby("n_steps"):
            clean = gn["local_ok"].apply(lambda xs: all(x is True for x in xs)).mean()
            checks.append({"model": m, "condition": c, "n_steps": n, "observed_error_free": clean,
                           "h1_predicted": (1 - h) ** n, "shallow_h": h})
    return acc, fits, pd.DataFrame(checks)


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def plot(acc, steps, fits, checks, out: Path) -> None:
    for model in acc.model.unique():
        tag = "" if acc.model.nunique() == 1 else f"_{model.replace(':', '-').replace('/', '-')}"

        fig, ax = plt.subplots(figsize=(6, 4))
        for c, g in acc[acc.model == model].groupby("condition"):
            ax.plot(g.n_steps, g.final_acc, "o-", color=COLORS.get(c), label=c)
        ax.set_xscale("log", base=2)
        ax.set_ylim(-0.02, 1.02)
        _style(ax, f"Final-state accuracy vs task length\n{model}", "task length (steps)", "accuracy")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"accuracy_vs_length{tag}.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        sm = steps[steps.model == model]
        edges = [0, 8, 16, 32, 64, 128, 256, 512, 1024]
        for c, g in sm.groupby("condition"):
            g = g.assign(bin=pd.cut(g.depth, edges))
            b = g.groupby("bin", observed=True).agg(d=("depth", "mean"), e=("err", "mean"), n=("err", "size"))
            se = np.sqrt(b.e * (1 - b.e) / b.n)
            ax.errorbar(b.d, b.e, yerr=1.96 * se, fmt="o", color=COLORS.get(c), label=c, capsize=3)
            f = fits[(fits.model == model) & (fits.condition == c)].iloc[0]
            xs = np.linspace(1, g.depth.max(), 100)
            ax.plot(xs, f.base + f.slope * (xs - 1), "-", color=COLORS.get(c), alpha=0.6)
        _style(ax, f"Per-step error rate by depth\n{model}", "step depth", "P(error at this step)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"hazard_vs_depth{tag}.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        for c, g in checks[checks.model == model].groupby("condition"):
            ax.plot(g.n_steps, g.observed_error_free, "o-", color=COLORS.get(c), label=f"{c}: observed")
            ax.plot(g.n_steps, g.h1_predicted, "--", color=COLORS.get(c), alpha=0.6, label=f"{c}: constant-error prediction")
        ax.set_xscale("log", base=2)
        ax.set_ylim(-0.02, 1.02)
        _style(ax, f"Error-free runs: observed vs pure compounding\n{model}", "task length (steps)", "P(no step errors)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"compounding_check{tag}.png", dpi=150)
        plt.close(fig)


def write_summary(acc, fits, checks, out: Path) -> None:
    lines = ["# Results summary", ""]
    lines += ["## Per-step error vs depth (linear fit, 95% bootstrap CI over trials)", "",
              "| model | condition | base error (step 1) | slope per step | slope per 100 steps | steps scored |",
              "|---|---|---|---|---|---|"]
    for f in fits.itertuples():
        lines.append(
            f"| {f.model} | {f.condition} | {f.base:.4f} [{f.base_ci[0]:.4f}, {f.base_ci[1]:.4f}] "
            f"| {f.slope:.5f} [{f.slope_ci[0]:.5f}, {f.slope_ci[1]:.5f}] "
            f"| {100 * f.slope:+.3f} | {f.steps_scored} |"
        )
    lines += ["", "A slope CI that excludes 0 in `trace` but not in `reset` is the signature of context "
              "degradation (H2). Slopes near 0 in both point to pure compounding (H1).", ""]
    lines += ["## Final-state accuracy", "", acc.to_markdown(index=False, floatfmt=".3f"), ""]
    lines += ["## Compounding check", "", checks.to_markdown(index=False, floatfmt=".3f"), ""]
    (out / "summary.md").write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trials")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load(args.trials)
    steps = step_table(df)
    acc, fits, checks = summarize(df, steps)
    plot(acc, steps, fits, checks, out)
    write_summary(acc, fits, checks, out)
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
