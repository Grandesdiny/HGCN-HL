#!/usr/bin/env python

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


METRIC_NAMES = ("OA", "AA", "Kappa")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run train_original.py repeatedly and aggregate outer x inner "
            "runs into CSV/Markdown/HTML tables."
        )
    )
    parser.add_argument("--dataset", default="muufl")
    parser.add_argument("--train-samples-per-class", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--outer-runs", type=int, default=10)
    parser.add_argument("--inner-runs", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Directory for all repeated-run outputs. Default: "
            "model_original/<dataset>_<train px>px_<outer>x<inner>."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip a batch when its result JSON already exists.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only read existing batch JSON files and rebuild tables.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch train_original.py.",
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=Path("train_original.py"),
    )
    args, extra_train_args = parser.parse_known_args()
    return args, extra_train_args


def result_json_path(output_dir, dataset, train_samples_per_class):
    return (
        output_dir
        / f"{dataset}_{train_samples_per_class}px_original_hgcn_results.json"
    )


def run_batch(args, extra_train_args, batch_index, batch_dir, result_path):
    seed = args.base_seed + batch_index * args.inner_runs
    command = [
        args.python,
        str(args.train_script),
        "--dataset",
        args.dataset,
        "--train-samples-per-class",
        str(args.train_samples_per_class),
        "--runs",
        str(args.inner_runs),
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--output-dir",
        str(batch_dir),
        *extra_train_args,
    ]
    print(
        f"[batch {batch_index + 1:02d}/{args.outer_runs:02d}] "
        f"seed={seed} output={batch_dir}",
        flush=True,
    )
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)
    if not result_path.exists():
        raise FileNotFoundError(
            f"Expected result JSON was not created: {result_path}"
        )


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric_values_from_runs(runs, metric_name):
    if metric_name in METRIC_NAMES:
        return np.asarray(
            [run[metric_name] for run in runs],
            dtype=np.float64,
        )
    class_index = int(metric_name[1:]) - 1
    return np.asarray(
        [run["class_accuracy"][class_index] for run in runs],
        dtype=np.float64,
    )


def summarize_runs(runs, metric_names):
    summary = {}
    for metric_name in metric_names:
        values = metric_values_from_runs(runs, metric_name)
        summary[metric_name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    return summary


def format_percent(summary_item):
    return (
        f"{summary_item['mean'] * 100:.2f}"
        f"±{summary_item['std'] * 100:.2f}"
    )


def write_csv(path, metric_names, columns, table):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Metric", *columns])
        for metric_name in metric_names:
            writer.writerow(
                [
                    metric_name.upper() if metric_name == "Kappa" else metric_name,
                    *[
                        format_percent(table[column][metric_name])
                        for column in columns
                    ],
                ]
            )


def write_markdown(path, metric_names, columns, table):
    lines = []
    lines.append("| Metric | " + " | ".join(columns) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(columns)) + "|")
    for metric_name in metric_names:
        display_name = metric_name.upper() if metric_name == "Kappa" else metric_name
        values = [
            format_percent(table[column][metric_name])
            for column in columns
        ]
        lines.append(f"| {display_name} | " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html(path, dataset_name, metric_names, columns, table, inner_runs):
    def row_class(metric_name):
        return "summary-row" if metric_name in METRIC_NAMES else ""

    rows = []
    for metric_name in metric_names:
        display_name = metric_name.upper() if metric_name == "Kappa" else metric_name
        cells = "".join(
            f"<td>{format_percent(table[column][metric_name])}</td>"
            for column in columns
        )
        rows.append(
            f"<tr class=\"{row_class(metric_name)}\">"
            f"<th>{display_name}</th>{cells}</tr>"
        )

    headers = "".join(
        f"<th>{column}<br><span>{dataset_name} | {inner_runs} runs</span></th>"
        for column in columns
    )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{dataset_name} original HGCN-HL 10x10 summary</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      background: #f7fafc;
      color: #001b3f;
    }}
    table {{
      border-collapse: collapse;
      background: white;
      box-shadow: 0 1px 8px rgba(0, 0, 0, 0.08);
      width: max-content;
      min-width: 100%;
    }}
    th, td {{
      border: 1px solid #e3eaf2;
      padding: 12px 16px;
      text-align: center;
      font-weight: 700;
      white-space: nowrap;
    }}
    thead th {{
      background: #f3f6fa;
      font-size: 16px;
    }}
    thead span {{
      display: inline-block;
      margin-top: 6px;
      color: #5f6f89;
      font-size: 12px;
      font-weight: 600;
    }}
    tbody th {{
      background: #f6f9fd;
      text-align: left;
      color: #0f3a8b;
    }}
    tr.summary-row th {{
      background: #eaf2ff;
      border-left: 5px solid #2d67ff;
    }}
    tr.summary-row td {{
      background: #dff2ff;
      color: #005f91;
    }}
  </style>
</head>
<body>
  <h2>{dataset_name} original HGCN-HL repeated summary</h2>
  <p>Each batch column is mean±std over one train_original.py call with {inner_runs} runs. The final column aggregates all runs.</p>
  <table>
    <thead><tr><th>Metric</th>{headers}</tr></thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def main():
    args, extra_train_args = parse_args()
    if args.outer_runs <= 0 or args.inner_runs <= 0:
        raise ValueError("--outer-runs and --inner-runs must be positive.")

    output_root = args.output_root
    if output_root is None:
        output_root = Path("model_original") / (
            f"{args.dataset}_{args.train_samples_per_class}px_"
            f"{args.outer_runs}x{args.inner_runs}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    batch_payloads = []
    for batch_index in range(args.outer_runs):
        batch_dir = output_root / f"batch_{batch_index + 1:02d}"
        result_path = result_json_path(
            batch_dir,
            args.dataset,
            args.train_samples_per_class,
        )
        if args.aggregate_only:
            if not result_path.exists():
                raise FileNotFoundError(
                    f"Missing result JSON for aggregate-only mode: {result_path}"
                )
        elif args.resume and result_path.exists():
            print(
                f"[batch {batch_index + 1:02d}/{args.outer_runs:02d}] "
                f"resume: using existing {result_path}",
                flush=True,
            )
        else:
            batch_dir.mkdir(parents=True, exist_ok=True)
            run_batch(
                args,
                extra_train_args,
                batch_index,
                batch_dir,
                result_path,
            )
        batch_payloads.append(load_json(result_path))

    all_runs = [
        run
        for payload in batch_payloads
        for run in payload["runs"]
    ]
    if not all_runs:
        raise RuntimeError("No runs were found in result JSON files.")

    class_count = len(all_runs[0]["class_accuracy"])
    metric_names = [
        "OA",
        "AA",
        "Kappa",
        *[f"C{index + 1}" for index in range(class_count)],
    ]
    table = {}
    columns = []
    dataset_display = args.dataset.upper()
    for batch_index, payload in enumerate(batch_payloads):
        column = f"{dataset_display} #{batch_index + 1}"
        columns.append(column)
        table[column] = summarize_runs(payload["runs"], metric_names)
    total_column = f"{dataset_display} all {len(all_runs)}"
    columns.append(total_column)
    table[total_column] = summarize_runs(all_runs, metric_names)

    summary_json = {
        "config": {
            "dataset": args.dataset,
            "train_samples_per_class": args.train_samples_per_class,
            "outer_runs": args.outer_runs,
            "inner_runs": args.inner_runs,
            "total_runs": len(all_runs),
            "base_seed": args.base_seed,
            "device": args.device,
            "output_root": str(output_root),
            "extra_train_args": extra_train_args,
        },
        "columns": columns,
        "metrics": metric_names,
        "table": table,
    }
    summary_stem = f"original_hgcn_{args.outer_runs}x{args.inner_runs}_summary"
    summary_json_path = output_root / f"{summary_stem}.json"
    summary_csv_path = output_root / f"{summary_stem}.csv"
    summary_md_path = output_root / f"{summary_stem}.md"
    summary_html_path = output_root / f"{summary_stem}.html"

    summary_json_path.write_text(
        json.dumps(summary_json, indent=2),
        encoding="utf-8",
    )
    write_csv(summary_csv_path, metric_names, columns, table)
    write_markdown(summary_md_path, metric_names, columns, table)
    write_html(
        summary_html_path,
        dataset_display,
        metric_names,
        columns,
        table,
        args.inner_runs,
    )

    print("Repeated original HGCN-HL summary written to:")
    print(f"  JSON: {summary_json_path.resolve()}")
    print(f"  CSV : {summary_csv_path.resolve()}")
    print(f"  MD  : {summary_md_path.resolve()}")
    print(f"  HTML: {summary_html_path.resolve()}")


if __name__ == "__main__":
    main()
