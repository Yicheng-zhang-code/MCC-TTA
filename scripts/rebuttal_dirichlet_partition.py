"""Fixed Dirichlet client partitions for rebuttal heterogeneity experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FORMAT_VERSION = 1
SUPPORTED_DATASETS = {
    "CIFAR10CFull": ("CIFAR-10-C-Full", "cifar10c"),
    "CIFAR100CFull": ("CIFAR-100-C-Full", "cifar100c"),
}


def alpha_tag(alpha: float) -> str:
    text = f"{float(alpha):g}".replace(".", "")
    return f"a{text}"


def default_partition_path(dataset_name: str, alpha: float, seed: int) -> Path:
    _, short_name = SUPPORTED_DATASETS[dataset_name]
    return Path("results") / "rebuttal" / "partitions" / f"{short_name}_dirichlet_{alpha_tag(alpha)}_seed{seed}.npz"


def default_stats_path(alpha: float) -> Path:
    return Path("results") / "rebuttal" / f"partition_stats_{alpha_tag(alpha)}.tsv"


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cifar_labels(data_root: Path | str, dataset_name: str, severity: int = 5) -> np.ndarray:
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset for Dirichlet partition: {dataset_name}")
    if not 1 <= int(severity) <= 5:
        raise ValueError("severity must be in [1, 5].")

    folder_name, _ = SUPPORTED_DATASETS[dataset_name]
    label_path = Path(data_root) / "corruption" / folder_name / "labels.npy"
    labels = np.load(label_path)
    num_data = len(labels) // 5
    start = num_data * (int(severity) - 1)
    end = num_data * int(severity)
    return np.asarray(labels[start:end], dtype=np.int64)


def cifar_corruption_names() -> List[str]:
    from datasets.corruption import all_corruptions

    return list(all_corruptions)


def generate_dirichlet_partition(
    labels: np.ndarray,
    domain_names: Sequence[str],
    num_clients: int,
    alpha: float,
    seed: int,
    min_client_size: int = 1,
    max_retries: int = 100,
) -> Dict[str, List[np.ndarray]]:
    if int(num_clients) <= 0:
        raise ValueError("num_clients must be positive.")
    if float(alpha) <= 0.0:
        raise ValueError("dirichlet_alpha must be positive.")

    labels = np.asarray(labels, dtype=np.int64)
    classes = np.unique(labels)
    last_partition: Optional[Dict[str, List[np.ndarray]]] = None

    for retry in range(max(1, int(max_retries))):
        rng = np.random.RandomState(int(seed) + retry)
        partition: Dict[str, List[np.ndarray]] = {}

        for domain_name in domain_names:
            client_indices = [[] for _ in range(int(num_clients))]
            for class_id in classes:
                class_indices = np.flatnonzero(labels == class_id).astype(np.int64)
                rng.shuffle(class_indices)
                proportions = rng.dirichlet(np.full(int(num_clients), float(alpha), dtype=np.float64))
                counts = rng.multinomial(len(class_indices), proportions)

                offset = 0
                for client_idx, count in enumerate(counts):
                    if count:
                        selected = class_indices[offset:offset + count]
                        client_indices[client_idx].extend(selected.tolist())
                    offset += int(count)

            finalized = []
            for values in client_indices:
                arr = np.asarray(values, dtype=np.int64)
                rng.shuffle(arr)
                finalized.append(arr)
            partition[domain_name] = finalized

        last_partition = partition
        if min(len(indices) for clients in partition.values() for indices in clients) >= int(min_client_size):
            return partition

    assert last_partition is not None
    return last_partition


def save_partition_npz(
    output_path: Path | str,
    partition: Mapping[str, Sequence[np.ndarray]],
    dataset_name: str,
    num_clients: int,
    alpha: float,
    seed: int,
    severity: int = 5,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    domain_names = list(partition.keys())
    metadata = {
        "format_version": FORMAT_VERSION,
        "dataset_name": dataset_name,
        "num_clients": int(num_clients),
        "dirichlet_alpha": float(alpha),
        "partition_seed": int(seed),
        "severity": int(severity),
        "domain_names": domain_names,
        "samples_per_domain": {
            domain: int(sum(len(indices) for indices in partition[domain]))
            for domain in domain_names
        },
    }

    arrays: Dict[str, Any] = {
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    for domain_idx, domain_name in enumerate(domain_names):
        clients = partition[domain_name]
        if len(clients) != int(num_clients):
            raise ValueError(f"{domain_name} has {len(clients)} clients, expected {num_clients}.")
        for client_idx, indices in enumerate(clients):
            arrays[f"indices_{domain_idx:03d}_{client_idx:03d}"] = np.asarray(indices, dtype=np.int64)

    np.savez_compressed(path, **arrays)
    return path


def load_partition_npz(path: Path | str) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        domain_names = list(metadata["domain_names"])
        num_clients = int(metadata["num_clients"])
        partition = {}
        for domain_idx, domain_name in enumerate(domain_names):
            partition[domain_name] = [
                np.asarray(data[f"indices_{domain_idx:03d}_{client_idx:03d}"], dtype=np.int64)
                for client_idx in range(num_clients)
            ]
    return partition, metadata


def validate_partition(
    partition: Mapping[str, Sequence[np.ndarray]],
    labels: np.ndarray,
    num_clients: int,
) -> Dict[str, Any]:
    total_before = len(labels) * len(partition)
    total_after = 0
    duplicate_count = 0
    missing_count = 0
    domain_summaries = []
    client_rows = []

    expected = set(range(len(labels)))
    for domain_name, clients in partition.items():
        if len(clients) != int(num_clients):
            raise ValueError(f"{domain_name} has {len(clients)} clients, expected {num_clients}.")

        flattened: List[int] = []
        for client_idx, indices in enumerate(clients):
            arr = np.asarray(indices, dtype=np.int64)
            total_after += len(arr)
            flattened.extend(arr.tolist())

            counts = np.bincount(labels[arr], minlength=int(labels.max()) + 1) if len(arr) else np.asarray([])
            present = np.flatnonzero(counts > 0)
            top_classes = sorted(
                [(int(class_id), int(counts[class_id])) for class_id in present],
                key=lambda item: item[1],
                reverse=True,
            )[:5]
            entropy = _label_entropy(counts)
            client_rows.append({
                "domain": domain_name,
                "client_id": f"{domain_name}_client_{client_idx}",
                "num_samples": int(len(arr)),
                "num_classes_present": int(len(present)),
                "label_entropy": entropy,
                "top5_class_counts": ";".join(f"{class_id}:{count}" for class_id, count in top_classes),
            })

        unique = set(flattened)
        domain_duplicate = len(flattened) - len(unique)
        domain_missing = len(expected - unique)
        duplicate_count += domain_duplicate
        missing_count += domain_missing
        domain_summaries.append({
            "domain": domain_name,
            "total_before": len(labels),
            "total_after": len(flattened),
            "duplicate_count": domain_duplicate,
            "missing_count": domain_missing,
        })

    return {
        "total_before": int(total_before),
        "total_after": int(total_after),
        "duplicate_count": int(duplicate_count),
        "missing_count": int(missing_count),
        "domain_summaries": domain_summaries,
        "client_rows": client_rows,
    }


def apply_partition_file(
    partition_file: Path | str,
    prepared_domains: Sequence[Tuple[str, Any]],
    dataset_name: str,
    num_clients: int,
) -> Tuple["OrderedDict[str, Subset]", Dict[str, Any]]:
    partition, metadata = load_partition_npz(partition_file)
    if metadata.get("dataset_name") != dataset_name:
        raise ValueError(f"Partition dataset {metadata.get('dataset_name')} does not match {dataset_name}.")
    if int(metadata.get("num_clients")) != int(num_clients):
        raise ValueError(f"Partition num_clients {metadata.get('num_clients')} does not match {num_clients}.")

    domain_names = [domain_name for domain_name, _ in prepared_domains]
    if list(metadata.get("domain_names", [])) != domain_names:
        raise ValueError(
            "Partition domain order does not match current dataset: "
            f"{metadata.get('domain_names')} vs {domain_names}"
        )

    all_client_datasets: "OrderedDict[str, Subset]" = OrderedDict()
    for domain_name, dataset in prepared_domains:
        clients = partition[domain_name]
        for client_idx, indices in enumerate(clients):
            max_index = int(np.max(indices)) if len(indices) else -1
            if max_index >= len(dataset):
                raise ValueError(f"{domain_name}_client_{client_idx} has index {max_index}, dataset size {len(dataset)}.")
            all_client_datasets[f"{domain_name}_client_{client_idx}"] = Subset(
                dataset,
                np.asarray(indices, dtype=np.int64).tolist(),
            )

    metadata = dict(metadata)
    metadata["partition_hash"] = sha256_file(partition_file)
    metadata["partition_file"] = _portable_path(partition_file)
    return all_client_datasets, metadata


def write_partition_stats_tsv(
    path: Path | str,
    stats: Mapping[str, Any],
    dataset_name: str,
    alpha: float,
    seed: int,
    partition_file: Path | str,
    partition_hash: str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_type",
        "dataset",
        "alpha",
        "partition_seed",
        "partition_file",
        "partition_hash",
        "domain",
        "client_id",
        "num_samples",
        "num_classes_present",
        "label_entropy",
        "top5_class_counts",
        "total_before",
        "total_after",
        "duplicate_count",
        "missing_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerow({
            "row_type": "overall",
            "dataset": dataset_name,
            "alpha": f"{float(alpha):.6g}",
            "partition_seed": int(seed),
            "partition_file": _portable_path(partition_file),
            "partition_hash": partition_hash,
            "total_before": stats["total_before"],
            "total_after": stats["total_after"],
            "duplicate_count": stats["duplicate_count"],
            "missing_count": stats["missing_count"],
        })
        for row in stats["domain_summaries"]:
            writer.writerow({
                "row_type": "domain",
                "dataset": dataset_name,
                "alpha": f"{float(alpha):.6g}",
                "partition_seed": int(seed),
                "partition_file": _portable_path(partition_file),
                "partition_hash": partition_hash,
                **row,
            })
        for row in stats["client_rows"]:
            out = {
                "row_type": "client",
                "dataset": dataset_name,
                "alpha": f"{float(alpha):.6g}",
                "partition_seed": int(seed),
                "partition_file": _portable_path(partition_file),
                "partition_hash": partition_hash,
                **row,
            }
            out["label_entropy"] = f"{float(out['label_entropy']):.6f}"
            writer.writerow(out)
    return path


def record_dirichlet_result(
    method: str,
    dataset_name: str,
    partition_file: Path | str,
    accuracy: float,
    num_clients: int,
    num_samples: int,
    output_dir: Path | str = Path("results") / "rebuttal",
) -> Tuple[Path, Path]:
    _, metadata = load_partition_npz(partition_file)
    partition_hash = sha256_file(partition_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, short_name = SUPPORTED_DATASETS.get(dataset_name, (dataset_name, dataset_name.lower()))
    runs_path = output_dir / f"{short_name}_dirichlet_runs.tsv"
    summary_path = output_dir / f"{short_name}_dirichlet_summary.tsv"

    fieldnames = [
        "method",
        "dataset",
        "alpha",
        "partition_seed",
        "partition_file",
        "partition_hash",
        "accuracy",
        "num_clients",
        "num_samples",
    ]
    rows = _read_tsv_rows(runs_path)
    rows.append({
        "method": method,
        "dataset": dataset_name,
        "alpha": f"{float(metadata['dirichlet_alpha']):.6g}",
        "partition_seed": str(int(metadata["partition_seed"])),
        "partition_file": _portable_path(partition_file),
        "partition_hash": partition_hash,
        "accuracy": f"{float(accuracy) * 100.0:.4f}",
        "num_clients": str(int(num_clients)),
        "num_samples": str(int(num_samples)),
    })

    with runs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    _write_summary(summary_path, rows)
    return runs_path, summary_path


def _write_summary(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    latest: Dict[Tuple[str, str], Mapping[str, str]] = {}
    for row in rows:
        latest[(row["alpha"], row["method"])] = row

    alphas = sorted({alpha for alpha, _ in latest}, key=lambda value: float(value))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["setting", "Latte", "MCC-TTA", "Gain"], delimiter="\t")
        writer.writeheader()
        for alpha in alphas:
            latte = latest.get((alpha, "Latte"))
            mcc = latest.get((alpha, "MCC-TTA"))
            latte_acc = float(latte["accuracy"]) if latte else None
            mcc_acc = float(mcc["accuracy"]) if mcc else None
            gain = None if latte_acc is None or mcc_acc is None else mcc_acc - latte_acc
            writer.writerow({
                "setting": f"alpha={float(alpha):g}",
                "Latte": "" if latte_acc is None else f"{latte_acc:.2f}",
                "MCC-TTA": "" if mcc_acc is None else f"{mcc_acc:.2f}",
                "Gain": "" if gain is None else f"{gain:+.2f}",
            })


def _read_tsv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _label_entropy(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    p = counts[counts > 0].astype(np.float64) / total
    return float(-(p * np.log(p)).sum())


def _portable_path(path: Path | str) -> str:
    return Path(path).as_posix()


def _print_sanity(stats: Mapping[str, Any], alpha: float, seed: int, partition_file: Path, partition_hash: str) -> None:
    print("=" * 72)
    print("Dirichlet Partition Sanity Check")
    print("=" * 72)
    print(f"alpha: {float(alpha):g}")
    print(f"partition seed: {int(seed)}")
    print(f"partition file: {_portable_path(partition_file)}")
    print(f"partition sha256: {partition_hash}")
    print(f"total samples before: {stats['total_before']}")
    print(f"total samples after : {stats['total_after']}")
    print(f"duplicate sample count: {stats['duplicate_count']}")
    print(f"missing sample count  : {stats['missing_count']}")
    print("=" * 72)
    print(f"{'client':36s}{'samples':>10s}{'classes':>10s}{'entropy':>12s}  top-5")
    for row in stats["client_rows"]:
        print(
            f"{row['client_id'][:36]:36s}"
            f"{int(row['num_samples']):10d}"
            f"{int(row['num_classes_present']):10d}"
            f"{float(row['label_entropy']):12.4f}  "
            f"{row['top5_class_counts']}"
        )
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed CIFAR-C Dirichlet client partitions.")
    parser.add_argument("--dataset", default="CIFAR100CFull", choices=sorted(SUPPORTED_DATASETS))
    parser.add_argument("--data-root", default="./dataset/")
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.5)
    parser.add_argument("--partition_seed", type=int, default=2026)
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--partition_file", default=None)
    parser.add_argument("--stats-output", default=None)
    parser.add_argument("--partition_only", action="store_true")
    parser.add_argument("--min-client-size", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = load_cifar_labels(args.data_root, args.dataset, severity=args.severity)
    domain_names = cifar_corruption_names()
    partition_path = Path(args.partition_file) if args.partition_file else default_partition_path(
        args.dataset, args.dirichlet_alpha, args.partition_seed)
    stats_path = Path(args.stats_output) if args.stats_output else default_stats_path(args.dirichlet_alpha)

    partition = generate_dirichlet_partition(
        labels=labels,
        domain_names=domain_names,
        num_clients=args.num_clients,
        alpha=args.dirichlet_alpha,
        seed=args.partition_seed,
        min_client_size=args.min_client_size,
        max_retries=args.max_retries,
    )
    save_partition_npz(
        partition_path,
        partition,
        dataset_name=args.dataset,
        num_clients=args.num_clients,
        alpha=args.dirichlet_alpha,
        seed=args.partition_seed,
        severity=args.severity,
    )
    partition_hash = sha256_file(partition_path)
    stats = validate_partition(partition, labels, num_clients=args.num_clients)
    write_partition_stats_tsv(
        stats_path,
        stats,
        dataset_name=args.dataset,
        alpha=args.dirichlet_alpha,
        seed=args.partition_seed,
        partition_file=partition_path,
        partition_hash=partition_hash,
    )
    _print_sanity(stats, args.dirichlet_alpha, args.partition_seed, partition_path, partition_hash)

    if stats["total_before"] != stats["total_after"] or stats["duplicate_count"] != 0 or stats["missing_count"] != 0:
        raise SystemExit("Partition sanity check failed.")

    print(f"Saved partition: {partition_path}")
    print(f"Saved stats TSV: {stats_path}")


if __name__ == "__main__":
    main()
