"""Run the VLCS Gaussian pseudo-label noise robustness experiment."""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class _Tee(io.TextIOBase):
    def __init__(self, stream: io.TextIOBase, buffer: io.StringIO) -> None:
        self.stream = stream
        self.buffer = buffer

    def write(self, text: str) -> int:
        self.stream.write(text)
        self.stream.flush()
        self.buffer.write(text)
        return len(text)

    def flush(self) -> None:
        self.stream.flush()
        self.buffer.flush()


class _Trace:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []
        self.correct = 0
        self.total = 0

    @property
    def accuracy(self) -> Optional[float]:
        if self.total == 0:
            return None
        return self.correct / self.total

    @property
    def sample_order(self) -> List[Tuple[str, int, int]]:
        return [
            (str(row["client_id"]), int(row["sample_index"]), int(row["original_index"]))
            for row in self.rows
        ]

    @property
    def predictions(self) -> List[int]:
        return [int(row["prediction"]) for row in self.rows]


def _install_trace_hooks(runner_module: Any, trace: _Trace) -> Callable[[], None]:
    original_evaluate_one = runner_module.BaseClient.evaluate_one
    original_base_predict = runner_module.BaseClient.predict
    original_mcc_predict = runner_module.MccTtaClient.predict
    context: Dict[str, Any] = {}

    def traced_evaluate_one(self: Any) -> Tuple[int, int]:
        if self.curr_idx >= self.num_samples:
            return original_evaluate_one(self)

        sample_index = int(self.curr_idx)
        original_index = sample_index
        if hasattr(self.dataset, "indices"):
            original_index = int(self.dataset.indices[sample_index])

        context["current"] = {
            "client_id": "" if getattr(self, "client_id", None) is None else str(self.client_id),
            "sample_index": sample_index,
            "original_index": original_index,
            "prediction": None,
        }
        try:
            correct, num_samples = original_evaluate_one(self)
            current = context.get("current")
            if num_samples and current is not None:
                current["correct"] = int(correct)
                trace.correct += int(correct)
                trace.total += int(num_samples)
                trace.rows.append(dict(current))
            return correct, num_samples
        finally:
            context.pop("current", None)

    def traced_base_predict(self: Any, image_feature: Any) -> int:
        pred = original_base_predict(self, image_feature)
        current = context.get("current")
        if current is not None:
            current["prediction"] = int(pred)
        return pred

    def traced_mcc_predict(self: Any, image_feature: Any) -> int:
        pred = original_mcc_predict(self, image_feature)
        current = context.get("current")
        if current is not None:
            current["prediction"] = int(pred)
        return pred

    runner_module.BaseClient.evaluate_one = traced_evaluate_one
    runner_module.BaseClient.predict = traced_base_predict
    runner_module.MccTtaClient.predict = traced_mcc_predict

    def restore() -> None:
        runner_module.BaseClient.evaluate_one = original_evaluate_one
        runner_module.BaseClient.predict = original_base_predict
        runner_module.MccTtaClient.predict = original_mcc_predict

    return restore


def _parse_accuracy(output: str, trace: _Trace) -> Optional[float]:
    matches = re.findall(r"Total Accuracy:\s*([0-9.]+)%\s*\((\d+)\s*/\s*(\d+)\)", output)
    if matches:
        _, correct, total = matches[-1]
        if int(total) > 0:
            return int(correct) / int(total)
    return trace.accuracy


def _parse_noise_stats(output: str) -> Dict[str, Optional[float]]:
    def int_match(pattern: str) -> Optional[int]:
        match = re.search(pattern, output)
        return int(match.group(1)) if match else None

    def float_match(pattern: str) -> Optional[float]:
        match = re.search(pattern, output)
        return float(match.group(1)) if match else None

    return {
        "num_updates": int_match(r"Gaussian label-noise updates:\s*(\d+)"),
        "num_corrupted": int_match(r"Artificially corrupted Gaussian updates:\s*(\d+)"),
        "realized_noise": float_match(r"Realized Gaussian label-noise ratio:\s*([0-9.]+)"),
    }


def _run_runner(argv: Sequence[str], repo_root: Path, trace_enabled: bool = True) -> Dict[str, Any]:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import mcc_tta_runner

    trace = _Trace()
    restore = _install_trace_hooks(mcc_tta_runner, trace) if trace_enabled else (lambda: None)
    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    try:
        os.chdir(repo_root)
        sys.argv = [str(repo_root / "mcc_tta_runner.py"), *argv]
        with redirect_stdout(_Tee(sys.stdout, stdout_buffer)), redirect_stderr(_Tee(sys.stderr, stderr_buffer)):
            mcc_tta_runner.main()
    finally:
        restore()
        sys.argv = old_argv
        os.chdir(old_cwd)

    output = stdout_buffer.getvalue() + stderr_buffer.getvalue()
    return {
        "trace": trace,
        "accuracy": _parse_accuracy(output, trace),
        "noise_stats": _parse_noise_stats(output),
        "output": output,
    }


def _base_argv(args: argparse.Namespace) -> List[str]:
    argv = [
        "--config", args.config,
        "--datasets", args.datasets,
        "--data-root", args.data_root,
        "--backbone", args.backbone,
        "--num-clients", str(args.num_clients),
        "--part-rate", str(args.part_rate),
        "--sync-freq", str(args.sync_freq),
        "--seed", str(args.seed),
    ]
    if args.cache_features:
        argv.append("--cache-features")
    if args.topology != "similarity":
        argv.extend(["--topology", args.topology])
    return argv


def _write_tsv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "noise_ratio",
                "accuracy",
                "delta_from_clean",
                "num_updates",
                "num_corrupted",
                "realized_noise",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Optional[float], scale: float = 1.0, decimals: int = 2) -> str:
    if value is None:
        return "nan"
    return f"{value * scale:.{decimals}f}"


def _print_table(rows: Sequence[Dict[str, Any]]) -> None:
    print("=" * 56)
    print("Gaussian Pseudo-Label Robustness | VLCS | ViT-B/16")
    print("=" * 56)
    print(f"{'Additional noise (%)':24s}{'Accuracy (%)':>16s}{'Delta (pp)':>14s}")
    for row in rows:
        print(
            f"{int(round(float(row['noise_ratio']) * 100)):24d}"
            f"{_fmt(float(row['accuracy'])):>16s}"
            f"{_fmt(float(row['delta_from_clean'])):>14s}"
        )
    print("=" * 56)
    for row in rows:
        print(
            "noise={:.2f}: Gaussian updates={}, corrupted={}, realized={:.6f}".format(
                float(row["noise_ratio"]),
                row["num_updates"],
                row["num_corrupted"],
                float(row["realized_noise"]),
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled Gaussian pseudo-label noise robustness.")
    parser.add_argument("--config", default="configs/mcc_tta_vlcs.yaml")
    parser.add_argument("--datasets", default="VLCS")
    parser.add_argument("--data-root", default="./dataset/")
    parser.add_argument("--backbone", default="ViT-B/16")
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--part-rate", type=float, default=1.0)
    parser.add_argument("--sync-freq", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--topology", default="similarity", choices=["similarity", "ring", "random", "full"])
    parser.add_argument("--gaussian-noise-seed", type=int, default=2026)
    parser.add_argument("--noise-levels", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3])
    parser.add_argument("--cache-features", dest="cache_features", action="store_true", default=True)
    parser.add_argument("--no-cache-features", dest="cache_features", action="store_false")
    parser.add_argument("--output", default="results/rebuttal/vlcs_vitb16_gaussian_noise_robustness.tsv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    base = _base_argv(args)

    print("Running original MCC-TTA sanity check...")
    original = _run_runner(base, repo_root=repo_root, trace_enabled=True)

    print("Running noise=0.0 MCC-TTA sanity check...")
    clean_argv = [
        *base,
        "--gaussian_label_noise", "0.0",
        "--gaussian_noise_seed", str(args.gaussian_noise_seed),
        "--gaussian-noise-report",
    ]
    clean = _run_runner(clean_argv, repo_root=repo_root, trace_enabled=True)

    if original["accuracy"] is None or clean["accuracy"] is None:
        raise SystemExit("Could not parse accuracy for sanity check.")

    acc_match = abs(float(original["accuracy"]) - float(clean["accuracy"])) <= 1e-12
    order_match = original["trace"].sample_order == clean["trace"].sample_order
    pred_match = original["trace"].predictions == clean["trace"].predictions
    update_match = clean["noise_stats"]["num_updates"] == clean["trace"].total

    print("\nSanity checks:")
    print(f"  original accuracy: {original['accuracy'] * 100.0:.4f}%")
    print(f"  noise=0 accuracy : {clean['accuracy'] * 100.0:.4f}%")
    print(f"  final accuracy identical: {'YES' if acc_match else 'NO'}")
    print(f"  sample order identical  : {'YES' if order_match else 'NO'}")
    print(f"  prediction identical    : {'YES' if pred_match else 'NO'}")
    print(f"  Gaussian update count OK: {'YES' if update_match else 'NO'}")

    if not (acc_match and order_match and pred_match and update_match):
        raise SystemExit("Sanity check failed; stopping before noisy runs.")

    rows: List[Dict[str, Any]] = []
    clean_acc = float(clean["accuracy"]) * 100.0

    for noise_ratio in args.noise_levels:
        if abs(float(noise_ratio)) <= 1e-12:
            result = clean
        else:
            print(f"\nRunning Gaussian label noise={noise_ratio:.2f}...")
            noisy_argv = [
                *base,
                "--gaussian_label_noise", str(float(noise_ratio)),
                "--gaussian_noise_seed", str(args.gaussian_noise_seed),
                "--gaussian-noise-report",
            ]
            result = _run_runner(noisy_argv, repo_root=repo_root, trace_enabled=False)

        accuracy = float(result["accuracy"]) * 100.0
        stats = result["noise_stats"]
        rows.append({
            "noise_ratio": f"{float(noise_ratio):.2f}",
            "accuracy": f"{accuracy:.4f}",
            "delta_from_clean": f"{accuracy - clean_acc:.4f}",
            "num_updates": "" if stats["num_updates"] is None else str(stats["num_updates"]),
            "num_corrupted": "" if stats["num_corrupted"] is None else str(stats["num_corrupted"]),
            "realized_noise": "" if stats["realized_noise"] is None else f"{stats['realized_noise']:.6f}",
        })

    output_path = repo_root / args.output
    _write_tsv(output_path, rows)
    _print_table(rows)
    print(f"\nSaved TSV: {output_path}")


if __name__ == "__main__":
    main()
