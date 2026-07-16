#!/usr/bin/env python

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


SUMMARY_METRICS = ("OA", "AA", "Kappa")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep DuRM-style dummy output-logit dimensions for a fixed "
            "demo_train.py C-mediator baseline and write HTML/CSV/JSON "
            "summaries."
        )
    )
    parser.add_argument("--dataset", default="muufl")
    parser.add_argument("--train-samples-per-class", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--dummy-min", type=int, default=12)
    parser.add_argument("--dummy-max", type=int, default=20)
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated dummy dimensions, e.g. 12,16,20.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: "
            "model_demo/dummy_logit_ablation_<dataset>_<train px>px_<runs>runs."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--train-script", type=Path, default=Path("demo_train.py"))
    args, extra_train_args = parser.parse_known_args()
    return args, extra_train_args


def dummy_dims(args):
    if args.only is not None:
        dims = [
            int(item.strip())
            for item in args.only.split(",")
            if item.strip()
        ]
    else:
        dims = list(range(args.dummy_min, args.dummy_max + 1))
    if not dims:
        raise ValueError("No dummy-logit dimensions selected.")
    if any(dim <= 0 for dim in dims):
        raise ValueError("Dummy-logit dimensions must be positive.")
    return dims


def base_demo_args(dummy_dim):
    return [
        "--graph-layout",
        "separate",
        "--lidar-segmentation",
        "slic",
        "--lidar-graph-prior",
        "rag-height-knn",
        "--lidar-rag-hops",
        "1",
        "--lidar-height-knn-k",
        "5",
        "--lidar-modulation",
        "rag-lowhigh",
        "--fdsm-scope",
        "hsi",
        "--post-gat-consensus-graph",
        "intersection-mediator",
        "--consensus-graph-transport",
        "bidirectional",
        "--consensus-graph-transport-message",
        "qk-prior",
        "--consensus-graph-transport-prior-weight",
        "1.0",
        "--consensus-graph-transport-fusion",
        "bilinear",
        "--consensus-graph-transport-lambda",
        "0.1",
        "--consensus-graph-transport-gamma-init",
        "0.1",
        "--dummy-logit-dim",
        str(dummy_dim),
    ]


def shell_join(command):
    return " ".join(shlex.quote(str(item)) for item in command)


def newest_result_json(output_dir, since_time=None):
    candidates = list(output_dir.glob("*.json"))
    if since_time is not None:
        candidates = [
            path
            for path in candidates
            if path.stat().st_mtime >= since_time - 1e-6
        ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_dim(args, extra_train_args, dummy_dim, output_root):
    dim_dir = output_root / f"dummy{dummy_dim}"
    dim_dir.mkdir(parents=True, exist_ok=True)
    existing_result = newest_result_json(dim_dir)
    if args.aggregate_only:
        if existing_result is None:
            raise FileNotFoundError(
                f"Missing result JSON for aggregate-only mode: {dim_dir}"
            )
        return existing_result, None
    if args.resume and existing_result is not None:
        print(
            f"[resume] dummy-logit-dim={dummy_dim}: using {existing_result}",
            flush=True,
        )
        return existing_result, None

    command = [
        args.python,
        str(args.train_script),
        "--dataset",
        args.dataset,
        "--train-samples-per-class",
        str(args.train_samples_per_class),
        "--runs",
        str(args.runs),
        "--seed",
        str(args.base_seed),
        "--device",
        args.device,
        "--output-dir",
        str(dim_dir),
        *base_demo_args(dummy_dim),
        *extra_train_args,
    ]
    print(
        f"[run] dummy-logit-dim={dummy_dim}",
        flush=True,
    )
    print(shell_join(command), flush=True)
    if args.dry_run:
        return None, command

    start_time = time.time()
    subprocess.run(command, check=True)
    result_path = newest_result_json(dim_dir, since_time=start_time)
    if result_path is None:
        result_path = newest_result_json(dim_dir)
    if result_path is None:
        raise FileNotFoundError(f"No result JSON was created in {dim_dir}")
    return result_path, command


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric_values_from_runs(runs, metric_name):
    if metric_name in SUMMARY_METRICS:
        return np.asarray([run[metric_name] for run in runs], dtype=np.float64)
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


def display_metric_name(metric_name):
    return metric_name.upper() if metric_name == "Kappa" else metric_name


def write_csv(path, metric_names, dims, table):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Metric", *[f"dummy-{dim}" for dim in dims]])
        for metric_name in metric_names:
            writer.writerow(
                [
                    display_metric_name(metric_name),
                    *[
                        format_percent(table[str(dim)][metric_name])
                        for dim in dims
                    ],
                ]
            )


def write_markdown(path, metric_names, dims, table):
    columns = [f"dummy-{dim}" for dim in dims]
    lines = []
    lines.append("| Metric | " + " | ".join(columns) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(columns)) + "|")
    for metric_name in metric_names:
        values = [
            format_percent(table[str(dim)][metric_name])
            for dim in dims
        ]
        lines.append(
            f"| {display_metric_name(metric_name)} | "
            + " | ".join(values)
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html(path, args, extra_train_args, dims, metric_names, table, commands):
    def row_class(metric_name):
        return "summary-row" if metric_name in SUMMARY_METRICS else ""

    headers = "".join(
        f"<th>dummy-{dim}<br><span>{args.runs} runs</span></th>"
        for dim in dims
    )
    rows = []
    for metric_name in metric_names:
        cells = "".join(
            f"<td>{format_percent(table[str(dim)][metric_name])}</td>"
            for dim in dims
        )
        rows.append(
            f"<tr class=\"{row_class(metric_name)}\">"
            f"<th>{display_metric_name(metric_name)}</th>{cells}</tr>"
        )

    command_blocks = []
    for dim in dims:
        command = commands.get(str(dim))
        if command is None:
            command_text = "Existing result JSON reused via --resume/--aggregate-only."
        else:
            command_text = shell_join(command)
        command_blocks.append(
            "<details>"
            f"<summary>dummy-logit-dim={dim}</summary>"
            f"<code>{command_text}</code>"
            "</details>"
        )

    extra_text = shell_join(extra_train_args) if extra_train_args else "(none)"
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{args.dataset.upper()} dummy-logit ablation</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      background: #f7fafc;
      color: #001b3f;
    }}
    .card {{
      background: white;
      border: 1px solid #e3eaf2;
      border-radius: 8px;
      box-shadow: 0 1px 8px rgba(0, 0, 0, 0.08);
      padding: 18px 20px;
      max-width: 1500px;
    }}
    table {{
      border-collapse: collapse;
      margin-top: 16px;
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
    details {{
      margin: 10px 0;
      background: #fbfdff;
      border: 1px solid #e3eaf2;
      border-radius: 6px;
      padding: 10px 12px;
    }}
    summary {{
      cursor: pointer;
      font-weight: 700;
      color: #0f3a8b;
    }}
    code {{
      display: block;
      white-space: pre-wrap;
      margin-top: 10px;
      background: #f3f6fa;
      border: 1px solid #e3eaf2;
      border-radius: 6px;
      padding: 10px;
      color: #0b2447;
      font-size: 13px;
      line-height: 1.45;
    }}
  </style>
</head>
<body>
  <div class="card">
    <h2>{args.dataset.upper()} dummy-logit ablation ({args.runs} runs each)</h2>
    <p>Fixed baseline: separate graph, LiDAR RAG-height-KNN with 1 hop, HSI FDSM, intersection mediator, qk-prior bidirectional bilinear transport. Only <code style="display:inline; padding:2px 4px;">--dummy-logit-dim</code> changes.</p>
    <p>Train samples per class: {args.train_samples_per_class}; device: {args.device}; base seed: {args.base_seed}; extra forwarded args: {extra_text}.</p>
    <table>
      <thead><tr><th>Metric</th>{headers}</tr></thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
    <h3>Commands</h3>
    {''.join(command_blocks)}
  </div>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def main():
    args, extra_train_args = parse_args()
    if args.runs <= 0:
        raise ValueError("--runs must be positive.")
    dims = dummy_dims(args)

    output_root = args.output_root
    if output_root is None:
        output_root = Path("model_demo") / (
            f"dummy_logit_ablation_{args.dataset}_"
            f"{args.train_samples_per_class}px_{args.runs}runs"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    result_paths = {}
    commands = {}
    for dim in dims:
        result_path, command = run_dim(
            args,
            extra_train_args,
            dim,
            output_root,
        )
        if command is not None:
            commands[str(dim)] = command
        if result_path is not None:
            result_paths[str(dim)] = result_path

    if args.dry_run:
        print("Dry run only; no HTML summary was written.")
        return

    payloads = {
        str(dim): load_json(result_paths[str(dim)])
        for dim in dims
    }
    first_runs = next(iter(payloads.values()))["runs"]
    class_count = len(first_runs[0]["class_accuracy"])
    metric_names = [
        "OA",
        "AA",
        "Kappa",
        *[f"C{index + 1}" for index in range(class_count)],
    ]
    table = {
        str(dim): summarize_runs(payloads[str(dim)]["runs"], metric_names)
        for dim in dims
    }

    summary = {
        "config": {
            "dataset": args.dataset,
            "train_samples_per_class": args.train_samples_per_class,
            "runs": args.runs,
            "base_seed": args.base_seed,
            "device": args.device,
            "dummy_dims": dims,
            "output_root": str(output_root),
            "extra_train_args": extra_train_args,
            "fixed_base_args": base_demo_args("<DIM>"),
        },
        "result_paths": {
            dim: str(path)
            for dim, path in result_paths.items()
        },
        "metrics": metric_names,
        "table": table,
    }
    dims_tag = f"{min(dims)}to{max(dims)}" if len(dims) > 1 else str(dims[0])
    summary_stem = (
        f"dummy_logit_ablation_{args.dataset}_"
        f"{args.train_samples_per_class}px_{args.runs}runs_{dims_tag}"
    )
    json_path = output_root / f"{summary_stem}.json"
    csv_path = output_root / f"{summary_stem}.csv"
    md_path = output_root / f"{summary_stem}.md"
    html_path = output_root / f"{summary_stem}.html"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(csv_path, metric_names, dims, table)
    write_markdown(md_path, metric_names, dims, table)
    write_html(
        html_path,
        args,
        extra_train_args,
        dims,
        metric_names,
        table,
        commands,
    )

    print("Dummy-logit ablation summary written to:")
    print(f"  JSON: {json_path.resolve()}")
    print(f"  CSV : {csv_path.resolve()}")
    print(f"  MD  : {md_path.resolve()}")
    print(f"  HTML: {html_path.resolve()}")


if __name__ == "__main__":
    main()
