#!/usr/bin/env python3
"""Export simple done/pending/missing status matrices by category and source."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from export_detector_stage_tables import (
    CATEGORY_CONFIGS,
    METRIC_ORDER,
    parse_stage_and_detector,
)


SCRIPT_DIR = Path(__file__).resolve().parent
TABLES_DIR = SCRIPT_DIR / "detector-stage-metric-tables"
OUTPUT_DIR = SCRIPT_DIR / "detector-status-matrices"
README_OUTPUT_PATH = OUTPUT_DIR / "README.md"
STAGE_ORDER = [
    "S1",
    "S2",
    "S4",
    "S5",
    "S6-MPG",
    "S6-UTA",
    "S6-fuzzer",
    "S8-deepseek",
    "S8-llama",
    "S8-ministral",
]
CATEGORY_TITLES = {"academic": "Academic", "industry": "Industry"}
SOURCE_TITLES = {"hw": "HW / GD", "llm": "LLM"}


def read_columns(csv_path: Path) -> set[str]:
    return set(pd.read_csv(csv_path, nrows=0).columns.tolist())


def collect_artifact_presence() -> dict[str, dict[str, dict[str, bool]]]:
    artifact_present: dict[str, dict[str, dict[str, bool]]] = defaultdict(lambda: defaultdict(dict))

    for category_name, config in CATEGORY_CONFIGS.items():
        detectors = config["detectors"]
        for source_name, source_dir in (("llm", config["llm_dir"]), ("hw", config["hw_dir"])):
            source_path = Path(source_dir)
            if not source_path.exists():
                continue

            for csv_path in sorted(source_path.glob("*.csv")):
                parsed = parse_stage_and_detector(csv_path, source_name, category_name)
                if parsed is None:
                    continue

                stage_name, standalone_detector = parsed
                if stage_name not in STAGE_ORDER:
                    continue

                columns = read_columns(csv_path)
                if standalone_detector is not None:
                    detector_config = detectors.get(standalone_detector)
                    if detector_config is None:
                        continue
                    standalone_column = detector_config.get("standalone_column")
                    if standalone_column and standalone_column in columns:
                        artifact_present[f"{category_name}:{source_name}"][standalone_detector][stage_name] = True
                    continue

                for detector_name, detector_config in detectors.items():
                    combined_column = detector_config.get("combined_column")
                    if combined_column and combined_column in columns:
                        artifact_present[f"{category_name}:{source_name}"][detector_name][stage_name] = True

    return artifact_present


def build_status_frame(
    category_name: str,
    source_name: str,
    artifact_present: dict[str, dict[str, dict[str, bool]]],
) -> pd.DataFrame:
    category_dir = TABLES_DIR / category_name
    rows: list[dict[str, object]] = []
    source_key = f"{category_name}:{source_name}"

    detector_names = [path.stem for path in sorted(category_dir.glob("*.csv"))]
    detector_frames = {
        path.stem: pd.read_csv(path) for path in sorted(category_dir.glob("*.csv"))
    }

    for stage_name in STAGE_ORDER:
        row: dict[str, object] = {"stage": stage_name}
        for detector_name in detector_names:
            df = detector_frames[detector_name]
            stage_row = df.loc[df["stage"].astype(str) == stage_name]
            metric_column = f"{source_name}_f1"
            has_metric = False
            if not stage_row.empty:
                has_metric = pd.to_numeric(stage_row.iloc[0][metric_column], errors="coerce") == pd.to_numeric(
                    stage_row.iloc[0][metric_column], errors="coerce"
                )

            if has_metric:
                status = "done"
            elif artifact_present[source_key].get(detector_name, {}).get(stage_name, False):
                status = "pending"
            else:
                status = "missing"
            row[detector_name] = status
        rows.append(row)

    return pd.DataFrame(rows, columns=["stage", *detector_names])


def render_status_table(title: str, df: pd.DataFrame) -> list[str]:
    lines = [f"## {title}", ""]
    lines.append("<table>")
    lines.append("  <thead>")
    lines.append("    <tr>")
    for column in df.columns:
        lines.append(f"      <th>{column}</th>")
    lines.append("    </tr>")
    lines.append("  </thead>")
    lines.append("  <tbody>")
    for _, row in df.iterrows():
        lines.append("    <tr>")
        for column in df.columns:
            lines.append(f"      <td>{row[column]}</td>")
        lines.append("    </tr>")
    lines.append("  </tbody>")
    lines.append("</table>")
    lines.append("")
    return lines


def write_readme(frames: dict[tuple[str, str], pd.DataFrame]) -> None:
    lines = [
        "# Detector Status Matrices",
        "",
        "Simple detector coverage matrices grouped by category and source.",
        "",
        "- `done`: usable stage-level metrics already exist.",
        "- `pending`: a matching result artifact exists, but the stage does not yet have usable metrics in evaluation.",
        "- `missing`: no matching result artifact was found for that detector-stage-source.",
        "",
    ]

    for category_name in ["academic", "industry"]:
        lines.append(f"# {CATEGORY_TITLES[category_name]}")
        lines.append("")
        for source_name in ["hw", "llm"]:
            lines.extend(
                render_status_table(
                    f"{CATEGORY_TITLES[category_name]} {SOURCE_TITLES[source_name]}",
                    frames[(category_name, source_name)],
                )
            )

    README_OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_present = collect_artifact_presence()
    frames: dict[tuple[str, str], pd.DataFrame] = {}

    for category_name in ["academic", "industry"]:
        for source_name in ["hw", "llm"]:
            df = build_status_frame(category_name, source_name, artifact_present)
            frames[(category_name, source_name)] = df
            output_path = OUTPUT_DIR / f"{category_name}_{source_name}_status.csv"
            df.to_csv(output_path, index=False)
            print(f"[OK] {output_path}")

    write_readme(frames)
    print(f"[OK] {README_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
