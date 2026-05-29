#!/usr/bin/env python3
"""Plot actual Vast.ai large-scale 20-sample surface run performance.

Uses S3 bundle manifests as the source of actual completed-surface timing. Cost
metadata comes from the recovered Vast monitor budget table in Codex history plus
any current entries in logs/vast_monitor_state.json.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "surface_computation"))

N_SURFACES = 32_800
STORAGE_RATE_PER_HR = 0.0306

EXPECTED_SPH = {
    "RTX_3090": 756,
    "RTX_4090": 1141,
    "RTX_5060_Ti": 498,
    "RTX_5090": 1292,
    "RTX_PRO_6000_WS": 1426,
    "H100_NVL": 1456,
    "A100_SXM4": 884,
}

# Recovered from the 2026-05-28 Codex budget table generated from S3 manifest
# intervals during the actual n_samples=20 Vast run. The S3 manifests are the
# source for final throughput; this table supplies instance GPU type and price.
RECOVERED_WORKER_META = {
    "38254781": {"gpu": "RTX_3090", "price_per_hour": 0.3111},
    "38247514": {"gpu": "RTX_3090", "price_per_hour": 0.4167},
    "38249884": {"gpu": "RTX_3090", "price_per_hour": 0.3233},
    "38249565": {"gpu": "RTX_3090", "price_per_hour": 0.3500},
    "38247801": {"gpu": "RTX_3090", "price_per_hour": 0.5556},
    "38245048": {"gpu": "RTX_3090", "price_per_hour": 0.4756},
    "38253918": {"gpu": "RTX_3090", "price_per_hour": 0.4389},
    "38245033": {"gpu": "RTX_4090", "price_per_hour": 0.6833},
    "38256607": {"gpu": "RTX_3090", "price_per_hour": 0.4089},
    "38240079": {"gpu": "RTX_3090", "price_per_hour": 0.4867},
    "38239804": {"gpu": "RTX_3090", "price_per_hour": 0.5500},
    "38257362": {"gpu": "RTX_3090", "price_per_hour": 0.5550},
    "38246931": {"gpu": "RTX_4090", "price_per_hour": 0.9611},
    "38239940": {"gpu": "RTX_4090", "price_per_hour": 0.8167},
    "38256251": {"gpu": "RTX_3090", "price_per_hour": 0.4511},
    "38248885": {"gpu": "RTX_3090", "price_per_hour": 0.5444},
    "38256602": {"gpu": "RTX_4090", "price_per_hour": 1.0222},
    "38254366": {"gpu": "RTX_4090", "price_per_hour": 1.0222},
    "38250545": {"gpu": "RTX_4090", "price_per_hour": 0.9444},
    "38261635": {"gpu": "RTX_3090", "price_per_hour": 0.5500},
    "38241241": {"gpu": "RTX_4090", "price_per_hour": 1.0278},
    "38239547": {"gpu": "A100_SXM4", "price_per_hour": 0.7444},
    "38247630": {"gpu": "RTX_3090", "price_per_hour": 0.4378},
    "38253020": {"gpu": "RTX_4090", "price_per_hour": 1.0222},
}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        line = line.removeprefix("export ")
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_state_meta(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    meta: dict[str, dict[str, Any]] = {}
    billed: dict[str, float] = {}
    if not path.exists():
        return meta, billed
    state = json.loads(path.read_text())
    for iid, item in state.get("spawned", {}).items():
        meta[iid] = {
            "gpu": item.get("gpu_name"),
            "price_per_hour": item.get("price_per_hour"),
        }
    billed = {str(k): float(v) for k, v in state.get("_billed", {}).items()}
    return meta, billed


def list_manifest_rows(prefix: str) -> list[dict[str, Any]]:
    from surface_computation.object_store import ObjectStoreConfig, SurfaceObjectStore

    os.environ["S3_PREFIX"] = prefix
    cfg = ObjectStoreConfig.from_env()
    if cfg is None:
        raise RuntimeError("Missing S3 bucket configuration")
    store = SurfaceObjectStore(cfg)
    rows: list[dict[str, Any]] = []
    for obj in store.list_bundle_objects():
        if not obj["name"].endswith(".manifest.json"):
            continue
        body = store._client.get_object(Bucket=cfg.bucket, Key=obj["key"])["Body"].read()
        manifest = json.loads(body)
        created = manifest.get("created_at") or obj.get("last_modified")
        surface_count = int(manifest.get("surface_count") or len(manifest.get("surface_ids", [])))
        rows.append({
            "manifest": obj["name"],
            "key": obj["key"],
            "machine_id": manifest.get("machine_id", "unknown"),
            "instance_id": str(manifest.get("machine_id", "unknown")).split("-", 1)[0],
            "chunk_id": manifest.get("chunk_id"),
            "surface_count": surface_count,
            "created_at": parse_dt(created),
            "last_modified": parse_dt(obj["last_modified"]),
        })
    return rows


def build_worker_stats(rows: list[dict[str, Any]], meta: dict[str, dict[str, Any]], billed: dict[str, float]) -> pd.DataFrame:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["machine_id"]].append(row)

    out = []
    for machine_id, items in grouped.items():
        items = sorted(items, key=lambda x: x["created_at"])
        instance_id = str(machine_id).split("-", 1)[0]
        total_surfaces = sum(item["surface_count"] for item in items)
        manifest_count = len(items)
        first = items[0]["created_at"]
        last = items[-1]["created_at"]
        span_hours = (last - first).total_seconds() / 3600.0
        first_count = items[0]["surface_count"]
        measured_surfaces = max(0, total_surfaces - first_count)
        real_sph = measured_surfaces / span_hours if span_hours > 0 else np.nan

        item_meta = dict(RECOVERED_WORKER_META.get(instance_id, {}))
        item_meta.update({k: v for k, v in meta.get(instance_id, {}).items() if v is not None})
        gpu = item_meta.get("gpu")
        price = item_meta.get("price_per_hour")
        expected_sph = EXPECTED_SPH.get(str(gpu).replace(" ", "_")) if gpu else np.nan
        hourly = float(price) + STORAGE_RATE_PER_HR if price is not None else np.nan

        out.append({
            "machine_id": machine_id,
            "instance_id": instance_id,
            "gpu": str(gpu).replace(" ", "_") if gpu else "unknown",
            "price_per_hour": float(price) if price is not None else np.nan,
            "storage_rate_per_hour": STORAGE_RATE_PER_HR,
            "all_in_hourly": hourly,
            "expected_sph": expected_sph,
            "real_sph": real_sph,
            "manifest_count": manifest_count,
            "surface_count": total_surfaces,
            "timed_surface_count": measured_surfaces,
            "span_hours": span_hours,
            "first_manifest_at": first.isoformat(),
            "last_manifest_at": last.isoformat(),
            "cost_per_1000_projected": 1000 * hourly / real_sph if real_sph and not np.isnan(hourly) else np.nan,
            "projected_full_cost": hourly * N_SURFACES / real_sph if real_sph and not np.isnan(hourly) else np.nan,
            "expected_full_cost": hourly * N_SURFACES / expected_sph if expected_sph and not np.isnan(hourly) else np.nan,
            "real_over_expected_cost": (expected_sph / real_sph) if real_sph and expected_sph else np.nan,
            "surfaces_per_dollar": real_sph / hourly if real_sph and not np.isnan(hourly) else np.nan,
            "billed_cost": billed.get(instance_id, np.nan),
            "billed_cost_per_1000_completed": 1000 * billed[instance_id] / total_surfaces if instance_id in billed and total_surfaces else np.nan,
        })
    return pd.DataFrame(out).sort_values(["gpu", "machine_id"])


def save_manifest_timeline(rows: list[dict[str, Any]], path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows).sort_values("created_at")
    df["cumulative_surfaces"] = df["surface_count"].cumsum()
    df.to_csv(path, index=False)
    return df


def style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 180,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })


def plot_speed_cost(df: pd.DataFrame, out: Path) -> None:
    plot_df = df.dropna(subset=["real_sph", "projected_full_cost"]).copy()
    plot_df = plot_df[plot_df["gpu"] != "unknown"].sort_values("projected_full_cost")
    fig, axes = plt.subplots(1, 2, figsize=(15, max(7, 0.32 * len(plot_df))))

    sns.barplot(data=plot_df, y="machine_id", x="real_sph", hue="gpu", dodge=False, ax=axes[0])
    axes[0].set_title("Actual throughput")
    axes[0].set_xlabel("surfaces per hour")
    axes[0].set_ylabel("worker")
    for i, row in enumerate(plot_df.itertuples()):
        axes[0].text(row.real_sph + 10, i, f"{row.real_sph:.0f}", va="center", fontsize=8)

    sns.barplot(data=plot_df, y="machine_id", x="projected_full_cost", hue="gpu", dodge=False, ax=axes[1])
    axes[1].axvline(30, color="crimson", linestyle="--", linewidth=1.5, label="$30 threshold")
    axes[1].set_title("Projected full-set cost from actual speed")
    axes[1].set_xlabel("USD for 32,800 surfaces")
    axes[1].set_ylabel("")
    for i, row in enumerate(plot_df.itertuples()):
        axes[1].text(row.projected_full_cost + 0.5, i, f"${row.projected_full_cost:.1f}", va="center", fontsize=8)
    axes[1].legend(loc="lower right")
    axes[0].legend_.remove()
    fig.suptitle("Actual Vast large-scale run performance, n_samples=20", y=0.995, fontsize=15)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_est_vs_actual(df: pd.DataFrame, out: Path) -> None:
    plot_df = df.dropna(subset=["real_sph", "expected_sph", "projected_full_cost", "expected_full_cost"]).copy()
    plot_df = plot_df[plot_df["gpu"] != "unknown"].sort_values("real_over_expected_cost")
    fig, axes = plt.subplots(1, 2, figsize=(15, max(7, 0.32 * len(plot_df))))

    y = np.arange(len(plot_df))
    axes[0].barh(y - 0.18, plot_df["expected_sph"], height=0.35, color="#9ecae1", label="expected")
    axes[0].barh(y + 0.18, plot_df["real_sph"], height=0.35, color="#3182bd", label="actual")
    axes[0].set_yticks(y, plot_df["machine_id"])
    axes[0].set_xlabel("surfaces per hour")
    axes[0].set_title("Expected vs actual speed")
    axes[0].legend()

    axes[1].barh(y - 0.18, plot_df["expected_full_cost"], height=0.35, color="#a1d99b", label="expected")
    axes[1].barh(y + 0.18, plot_df["projected_full_cost"], height=0.35, color="#31a354", label="actual-speed projection")
    axes[1].axvline(30, color="crimson", linestyle="--", linewidth=1.5)
    axes[1].set_yticks(y, [])
    axes[1].set_xlabel("USD for 32,800 surfaces")
    axes[1].set_title("Expected vs actual projected cost")
    axes[1].legend()

    fig.suptitle("Budget model error from real S3 manifest timing", y=0.995, fontsize=15)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_value_scatter(df: pd.DataFrame, out: Path) -> None:
    plot_df = df.dropna(subset=["real_sph", "price_per_hour", "surfaces_per_dollar"]).copy()
    plot_df = plot_df[plot_df["gpu"] != "unknown"]
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=plot_df,
        x="all_in_hourly",
        y="real_sph",
        hue="gpu",
        size="manifest_count",
        sizes=(50, 350),
        alpha=0.85,
        ax=ax,
    )
    for row in plot_df.itertuples():
        ax.annotate(row.instance_id, (row.all_in_hourly, row.real_sph), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set_title("Throughput vs hourly cost")
    ax.set_xlabel("all-in hourly cost, USD/hr (compute + storage)")
    ax.set_ylabel("actual surfaces per hour")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_billed_efficiency(df: pd.DataFrame, out: Path) -> None:
    plot_df = df.dropna(subset=["billed_cost", "billed_cost_per_1000_completed"]).copy()
    plot_df = plot_df[(plot_df["gpu"] != "unknown") & (plot_df["surface_count"] > 0)]
    plot_df = plot_df.sort_values("billed_cost_per_1000_completed")
    fig, ax = plt.subplots(figsize=(11, max(6, 0.32 * len(plot_df))))
    sns.barplot(data=plot_df, y="machine_id", x="billed_cost_per_1000_completed", hue="gpu", dodge=False, ax=ax)
    ax.set_title("Actual billed spend per completed 1,000 surfaces")
    ax.set_xlabel("USD / 1,000 completed surfaces")
    ax.set_ylabel("worker")
    for i, row in enumerate(plot_df.itertuples()):
        ax.text(row.billed_cost_per_1000_completed + 0.04, i, f"${row.billed_cost_per_1000_completed:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_timeline(timeline: pd.DataFrame, out: Path) -> None:
    if timeline.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(pd.to_datetime(timeline["created_at"]), timeline["cumulative_surfaces"], color="#2b8cbe", linewidth=2)
    ax.set_title("S3 manifest completion timeline")
    ax.set_xlabel("UTC time")
    ax.set_ylabel("cumulative completed surfaces")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s3-prefix", default="demixing/surfaces_full")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "results" / "benchmarks")
    ap.add_argument("--state-file", type=Path, default=ROOT / "logs" / "vast_monitor_state.json")
    args = ap.parse_args()

    load_env_file(ROOT / ".env.docker")
    load_env_file(ROOT / ".env")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()

    state_meta, billed = load_state_meta(args.state_file)
    rows = list_manifest_rows(args.s3_prefix)
    timeline = save_manifest_timeline(rows, args.output_dir / "vast_actual_20samples_manifest_timeline.csv")
    df = build_worker_stats(rows, state_meta, billed)
    df.to_csv(args.output_dir / "vast_actual_20samples_worker_stats.csv", index=False)

    known = df[(df["gpu"] != "unknown") & df["real_sph"].notna()].copy()
    summary = {
        "s3_prefix": args.s3_prefix,
        "manifest_count": len(rows),
        "worker_count": int(df.shape[0]),
        "known_cost_worker_count": int(known.shape[0]),
        "unknown_cost_worker_count": int((df["gpu"] == "unknown").sum()),
        "surface_count": int(df["surface_count"].sum()),
        "known_cost_surface_count": int(known["surface_count"].sum()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "actual speed from S3 manifest created_at intervals; projected cost uses dph_total + storage hourly rate; billed efficiency uses logs/vast_monitor_state.json _billed values when available",
    }
    (args.output_dir / "vast_actual_20samples_summary.json").write_text(json.dumps(summary, indent=2))

    plot_speed_cost(df, args.output_dir / "vast_actual_20samples_speed_cost.png")
    plot_est_vs_actual(df, args.output_dir / "vast_actual_20samples_est_vs_actual.png")
    plot_value_scatter(df, args.output_dir / "vast_actual_20samples_value_scatter.png")
    plot_billed_efficiency(df, args.output_dir / "vast_actual_20samples_billed_efficiency.png")
    plot_timeline(timeline, args.output_dir / "vast_actual_20samples_timeline.png")

    print(json.dumps(summary, indent=2))
    print("Top known workers by projected full cost:")
    cols = ["machine_id", "gpu", "price_per_hour", "real_sph", "projected_full_cost", "cost_per_1000_projected", "manifest_count", "surface_count"]
    print(known.sort_values("projected_full_cost")[cols].head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
