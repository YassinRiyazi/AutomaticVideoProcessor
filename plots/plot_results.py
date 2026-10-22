#!/usr/bin/env python
"""Build an interactive HTML dashboard from every ``tilt-XX-rep-YYY/result.csv``.

The dashboard shows how droplet measurements (friction force, velocity, contact
angles, ...) evolve over time for each tilt angle and repetition, so that any
degradation from one repetition to the next becomes visible.

Usage
-----
    conda activate torch3.11
    python plot_results.py                       # scan the folder of this script
    python plot_results.py /path/to/output -o dashboard.html --cdn
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.colors import sample_colorscale
from plotly.offline import get_plotlyjs

RUN_DIR_RE = re.compile(r"^tilt-(?P<tilt>\d+(?:\.\d+)?)-rep-(?P<rep>\d+)$")
COLUMN_RE = re.compile(r"^(?:(?P<cam>side|front)\s+)?(?P<name>.+?)\s*(?:\((?P<unit>[^)]*)\))?$")

SKIP_COLUMNS = {"file number", "front threshold", "front touches_border"}
UNIT_PRETTY = {"degree": "°", "uN": "µN", "m/s^2": "m/s²", "cm^2": "cm²", "px^2": "px²"}
NAME_PRETTY = {
    "time": "Time",
    "x_center": "Position (x centre)",
    "y_center": "Position (y centre)",
    "adv": "Advancing contact angle",
    "rec": "Receding contact angle",
    "contact_line_length": "Contact-line length",
    "middle_angle_degree": "Middle angle",
    "velocity": "Velocity",
    "a": "Acceleration",
    "friction force": "Friction force",
    "baseline_y": "Baseline y",
    "base_width": "Base width",
    "centroid_x": "Centroid x",
    "centroid_y": "Centroid y",
    "centroid_height": "Centroid height",
}
# Variables shown first in the selector, in this order.
PREFERRED_ORDER = [
    "friction force (uN)",
    "side velocity (cm/s)",
    "a (m/s^2)",
    "side adv (degree)",
    "side rec (degree)",
    "hysteresis",
    "side x_center (cm)",
]
TILT_COLORS = ["#2E6FBA", "#E07A1F", "#2A9D6F", "#C8414B", "#7B5EA7", "#8C6D46", "#D35FB7",
               "#17A2B8", "#B8A100", "#4A4F5C", "#E5739A", "#6C8EBF","#9B1C1C"]


def describe_column(col: str) -> dict:
    m = COLUMN_RE.match(col)
    cam, name, unit = m.group("cam"), m.group("name").strip(), (m.group("unit") or "")
    label = NAME_PRETTY.get(name, name.replace("_", " ").capitalize())
    group = {"side": "Side camera", "front": "Front camera"}.get(cam, "Dynamics")
    return {"label": label, "unit": UNIT_PRETTY.get(unit, unit), "group": group}


def discover_runs(root: Path, require_done: bool) -> tuple[list[tuple[float, int, Path]], list[str]]:
    runs, skipped = [], []
    for d in sorted(root.iterdir()):
        m = RUN_DIR_RE.match(d.name)
        if not (d.is_dir() and m):
            continue
        if not (d / "result.csv").is_file():
            skipped.append(f"{d.name}: no result.csv yet")
        elif require_done and not (d / ".done").exists():
            skipped.append(f"{d.name}: still processing (no .done marker)")
        else:
            runs.append((float(m["tilt"]), int(m["rep"]), d))
    return sorted(runs), skipped


def read_tilt(run_dir: Path, fallback: float) -> float:
    params = run_dir / "experiment_parameters.csv"
    if params.is_file():
        p = pd.read_csv(params)
        row = p.loc[p["parameter"] == "tilt", "value"]
        if not row.empty:
            return float(row.iloc[0])
    return fallback


def compact(values: np.ndarray) -> list:
    """Round to 6 significant digits and turn NaN into None to keep the HTML small."""
    return [None if not np.isfinite(v) else float(f"{v:.6g}") for v in values]


def load_run(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c in SKIP_COLUMNS or "(px" in c], errors="ignore")
    df = df.select_dtypes("number")
    if "side adv (degree)" in df and "side rec (degree)" in df:
        df["hysteresis"] = df["side adv (degree)"] - df["side rec (degree)"]
    if "side time (s)" in df:
        df["time_ms"] = (df["side time (s)"] - df["side time (s)"].iloc[0]) * 1e3
        df = df.drop(columns="side time (s)")
    return df


def track_coverage(v: np.ndarray, lo: float, hi: float, bins: int = 60) -> float:
    """Fraction of the interval [lo, hi] that a run actually visits (share of equal-width bins hit by a sample)."""
    v = v[np.isfinite(v) & (v >= lo) & (v <= hi)]
    if not len(v) or hi <= lo:
        return 0.0
    return len(np.unique(np.clip(((v - lo) / (hi - lo) * bins).astype(int), 0, bins - 1))) / bins


def flag_truncated(runs: list[dict], key: str, frac: float) -> None:
    """Flag runs that cover much less track than a typical run (e.g. the tracker lost the droplet).

    Every run images the same stretch of surface, so for position the reference is the common track (median of
    the per-run 2nd/98th percentiles) and a run is flagged when it visits less than ``frac`` of it; this is robust
    to single glitched coordinates and also catches runs with large gaps. For time (which depends on the tilt)
    the reference is the median robust duration of runs with the same tilt.
    """
    unit = describe_column(key)["unit"]
    vals = {r["id"]: np.array(r["data"][key], dtype=float) for r in runs}
    pct = {k: np.nanpercentile(v, [2, 98]) if np.isfinite(v).any() else np.array([np.nan, np.nan])
           for k, v in vals.items()}
    if key != "time_ms":
        lo = float(np.nanmedian([p[0] for p in pct.values()]))
        hi = float(np.nanmedian([p[1] for p in pct.values()]))
        for r in runs:
            cov = track_coverage(vals[r["id"]], lo, hi)
            if cov < frac:
                r["flag"] = f"covers only {cov:.0%} of the common track ({lo:.2f}–{hi:.2f} {unit})"
        return
    spans = {k: float(p[1] - p[0]) if np.isfinite(p).all() else 0.0 for k, p in pct.items()}
    for r in runs:
        typical = np.median([spans[o["id"]] for o in runs if o["tilt"] == r["tilt"]])
        if typical > 0 and spans[r["id"]] < frac * typical:
            r["flag"] = (f"lasts only {spans[r['id']] / typical:.0%} of a typical run at this tilt "
                         f"({spans[r['id']]:.1f} of {typical:.1f} {unit})")


def build_payload(root: Path, require_done: bool = True, min_coverage: float = 0.8) -> tuple[dict, list[str]]:
    found, skipped = discover_runs(root, require_done)
    frames, mtimes = {}, []
    for tilt_from_name, rep, run_dir in found:
        try:
            frames[(read_tilt(run_dir, tilt_from_name), rep, run_dir.name)] = load_run(run_dir / "result.csv")
            marker = run_dir / ".done" if (run_dir / ".done").exists() else run_dir / "result.csv"
            mtimes.append(marker.stat().st_mtime)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, KeyError, IndexError) as exc:
            skipped.append(f"{run_dir.name}: unreadable ({exc.__class__.__name__})")
    if not frames:
        raise SystemExit(f"No usable tilt-*-rep-*/result.csv files found in {root}")

    # Keep a column if it varies in at least one run, so every run exposes the same set of columns.
    all_cols = list(dict.fromkeys(c for df in frames.values() for c in df.columns))
    columns = {c: describe_column(c) for c in all_cols
               if any(c in df and df[c].nunique() > 1 for df in frames.values())}
    columns["hysteresis"] = {"label": "Contact-angle hysteresis", "unit": "°", "group": "Side camera"}
    columns["time_ms"] = {"label": "Time", "unit": "ms", "group": "Time"}

    runs = []
    for (tilt, rep, name), df in sorted(frames.items()):
        runs.append({
            "id": name, "tilt": tilt, "rep": rep, "flag": None,
            "data": {c: compact(df[c].to_numpy(dtype=float)) if c in df else [None] * len(df) for c in columns},
        })
    flag_truncated(runs, "side x_center (cm)" if "side x_center (cm)" in columns else "time_ms", min_coverage)

    order = [c for c in PREFERRED_ORDER if c in columns]
    order += sorted((c for c in columns if c not in order and c != "time_ms"),
                    key=lambda c: (columns[c]["group"], columns[c]["label"]))
    tilts = sorted({r["tilt"] for r in runs})
    dt_ms = float(np.nanmedian(np.concatenate([np.diff(np.array(r["data"]["time_ms"], dtype=float)) for r in runs])))
    max_rep = max(r["rep"] for r in runs)
    rep_colors = sample_colorscale("Plasma", [0.9 * i / max(max_rep - 1, 1) for i in range(max_rep)])
    payload = {
        "source": str(root),
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "latestRun": datetime.fromtimestamp(max(mtimes)).strftime("%Y-%m-%d %H:%M"),
        "runs": runs,
        "skipped": skipped,
        "columns": columns,
        "yOrder": order,
        "xOptions": [c for c in ["time_ms", "side x_center (cm)"] if c in columns],
        "tilts": tilts,
        "reps": list(range(1, max_rep + 1)),
        "repColors": {str(i + 1): c for i, c in enumerate(rep_colors)},
        "tiltColors": dict(zip((f"{t:g}" for t in tilts), TILT_COLORS if len(tilts) <= len(TILT_COLORS)
                                else sample_colorscale("Turbo", list(np.linspace(0.05, 0.95, len(tilts)))))),
        "dtMs": dt_ms,
    }
    return payload, skipped

with open("plots/base.html") as f:
    HTML_TEMPLATE = f.read()

def render_html(payload: dict, title: str, cdn: bool) -> str:
    plotly_tag = ('<script src="https://cdn.plot.ly/plotly-3.1.0.min.js" charset="utf-8"></script>' if cdn
                  else f"<script>{get_plotlyjs()}</script>")
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return (HTML_TEMPLATE.replace("__PLOTLY__", plotly_tag)
            .replace("__TITLE__", title)
            .replace("__DATA__", data))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = "/run/media/roboprocessing/Data/Samba-Share/RoboprocessingOutput/2026-10-16_18-12-14-317689"
    # ap.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parent,
    #                 help="folder containing the tilt-XX-rep-YYY run folders")
    
    ap.add_argument("-o", "--output", type=Path, help="output HTML file (default: <root>/results_dashboard.html)")
    ap.add_argument("--title", default="Droplet sliding experiments — degradation dashboard")
    ap.add_argument("--cdn", action="store_true", help="load plotly.js from a CDN instead of embedding it")
    ap.add_argument("--include-unfinished", action="store_true",
                    help="also read run folders without a .done marker (may still be being written)")
    ap.add_argument("--min-coverage", type=float, default=0.8,
                    help="flag runs that visit less than this fraction of the common track (default 0.8)")
    args = ap.parse_args()

    payload, skipped = build_payload(Path(root).resolve(), not args.include_unfinished, args.min_coverage)
    out = args.output or Path(root) / "results_dashboard.html"
    
    out.write_text(render_html(payload, args.title, args.cdn), encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB) from {len(payload['runs'])} runs")
    for msg in skipped:
        print(f"  skipped  {msg}")
    for r in payload["runs"]:
        if r["flag"]:
            print(f"  flagged  {r['id']}: {r['flag']}")


if __name__ == "__main__":
    main()
