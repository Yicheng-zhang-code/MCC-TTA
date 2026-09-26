"""Direct mechanism diagnostics for MCC-TTA rebuttal experiments.

This file has two roles:
1. A read-only logger imported by mcc_tta_runner.py when --rebuttal_diagnostic is set.
2. A small experiment script that runs TerraIncognita + ViT-B/16 OFF/ON sanity checks.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import sys
from collections import OrderedDict
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


class DirectMechanismDiagnosticLogger:
    """Collect read-only MCC-TTA mechanism diagnostics.

    The logger stores copied CPU values only. Ground-truth labels are attached
    after a query prediction has already been produced by the original pipeline.
    """

    def __init__(self, output_dir: str = "results/rebuttal") -> None:
        self.output_dir = Path(output_dir)
        self.query_rows: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()
        self.memory_rows: List[Dict[str, Any]] = []
        self._sample_to_memory_event_ids: Dict[int, List[int]] = {}
        self._local_slots: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
        self._next_query_uid = 0
        self.last_summary: Optional[Dict[str, Any]] = None
        self.last_output_paths: Dict[str, str] = {}

    def start_query(self, client_id: Any, sample_index: int, original_index: int) -> int:
        sample_uid = self._next_query_uid
        self._next_query_uid += 1
        self.query_rows[sample_uid] = {
            "sample_uid": sample_uid,
            "query_order": sample_uid,
            "client_id": "" if client_id is None else str(client_id),
            "sample_index": int(sample_index),
            "original_index": int(original_index),
        }
        return sample_uid

    def finish_query(
        self,
        sample_uid: Optional[int],
        ground_truth: int,
        final_pred: int,
        correct: int,
    ) -> None:
        if sample_uid is None or sample_uid not in self.query_rows:
            return

        row = self.query_rows[sample_uid]
        row["ground_truth"] = int(ground_truth)
        row["final_pred"] = int(final_pred)
        row["correct"] = int(correct)

        for event_id in self._sample_to_memory_event_ids.get(sample_uid, []):
            mem_row = self.memory_rows[event_id]
            mem_row["ground_truth"] = int(ground_truth)
            mem_row["pseudo_correct"] = int(mem_row["pseudo_label"] == int(ground_truth))

    def record_memory_event(
        self,
        sample_uid: Optional[int],
        client_id: Any,
        pseudo_label: int,
        candidate: bool,
        retained: bool,
        class_index: int,
        slot_index: Optional[int],
        entropy: float,
        normalized_entropy: float,
        reconstruction_error: float,
        regulation_score: float,
        dual_criteria: bool,
    ) -> None:
        row = {
            "memory_event_id": len(self.memory_rows),
            "sample_uid": sample_uid,
            "client_id": "" if client_id is None else str(client_id),
            "pseudo_label": int(pseudo_label),
            "candidate": int(bool(candidate)),
            "retained": int(bool(retained)),
            "class_index": int(class_index),
            "slot_index": "" if slot_index is None else int(slot_index),
            "entropy": float(entropy),
            "normalized_entropy": float(normalized_entropy),
            "reconstruction_error": float(reconstruction_error),
            "regulation_score": float(regulation_score),
            "dual_criteria": int(bool(dual_criteria)),
        }
        self.memory_rows.append(row)

        if sample_uid is not None:
            self._sample_to_memory_event_ids.setdefault(sample_uid, []).append(row["memory_event_id"])

        if retained and slot_index is not None:
            slot_key = (row["client_id"], int(class_index), int(slot_index))
            previous = self._local_slots.get(slot_key)
            if previous is not None and previous is not row:
                previous["retained"] = 0
            self._local_slots[slot_key] = row

    def get_local_coverage_counts(self, client_id: Any, num_class: int) -> List[int]:
        cid = "" if client_id is None else str(client_id)
        counts = [0 for _ in range(int(num_class))]
        for slot_cid, class_index, _ in self._local_slots.keys():
            if slot_cid == cid and 0 <= class_index < num_class:
                counts[class_index] += 1
        return counts

    def record_query_scores(
        self,
        sample_uid: Optional[int],
        client_id: Any,
        clip_pred: int,
        clip_entropy: float,
        local_memory_counts: Optional[Sequence[int]],
        gaussian_applied: bool,
        score_before_gaussian: Any,
        score_after_gaussian: Any,
        score_before_pairwise: Any,
        score_after_pairwise: Any,
        activated_pairs: Sequence[Tuple[int, int]],
    ) -> None:
        if sample_uid is None or sample_uid not in self.query_rows:
            return

        row = self.query_rows[sample_uid]
        row["client_id"] = "" if client_id is None else str(client_id)
        row["clip_pred"] = int(clip_pred)
        row["clip_entropy"] = float(clip_entropy)
        row["local_memory_counts"] = list(map(int, local_memory_counts or []))
        row["gaussian_applied"] = int(bool(gaussian_applied))
        row["score_before_gaussian"] = self._score_to_list(score_before_gaussian)
        row["score_after_gaussian"] = self._score_to_list(score_after_gaussian)
        row["score_before_pairwise"] = self._score_to_list(score_before_pairwise)
        row["score_after_pairwise"] = self._score_to_list(score_after_pairwise)
        row["activated_pairs"] = [(int(a), int(b)) for a, b in activated_pairs]
        row["num_activated_pairs"] = len(row["activated_pairs"])

    def build_summary(
        self,
        final_accuracy: Optional[float] = None,
        total_correct: Optional[int] = None,
        total_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        self._annotate_query_metrics()

        candidate_rows = [
            row for row in self.memory_rows
            if int(row.get("candidate", 0)) == 1 and "ground_truth" in row
        ]
        retained_rows = [row for row in candidate_rows if int(row.get("retained", 0)) == 1]

        num_candidates = len(candidate_rows)
        num_retained = len(retained_rows)
        purity_before = self._ratio(
            sum(int(row["pseudo_label"] == row["ground_truth"]) for row in candidate_rows),
            num_candidates,
        )
        purity_after = self._ratio(
            sum(int(row["pseudo_label"] == row["ground_truth"]) for row in retained_rows),
            num_retained,
        )

        gaussian_rows = [
            row for row in self.query_rows.values()
            if row.get("margin_before_gaussian") is not None
            and row.get("margin_after_gaussian") is not None
        ]
        all_margin_before = self._mean(row["margin_before_gaussian"] for row in gaussian_rows)
        all_margin_after = self._mean(row["margin_after_gaussian"] for row in gaussian_rows)

        poor_rows, coverage_threshold = self._bottom_coverage_rows(gaussian_rows, fraction=0.25)
        poor_margin_before = self._mean(row["margin_before_gaussian"] for row in poor_rows)
        poor_margin_after = self._mean(row["margin_after_gaussian"] for row in poor_rows)

        pair_rows = [
            row for row in self.query_rows.values()
            if row.get("pair_margin_before") is not None
            and row.get("pair_margin_after") is not None
        ]
        pair_margin_before = self._mean(row["pair_margin_before"] for row in pair_rows)
        pair_margin_after = self._mean(row["pair_margin_after"] for row in pair_rows)
        pair_error_before = self._mean(row["pair_error_before"] for row in pair_rows)
        pair_error_after = self._mean(row["pair_error_after"] for row in pair_rows)

        pair_changed_rows = [
            row for row in pair_rows
            if row.get("score_before_pairwise") is not None
            and row.get("score_after_pairwise") is not None
            and row.get("ground_truth") is not None
            and self._score_to_top1(row["score_before_pairwise"]) != self._score_to_top1(row["score_after_pairwise"])
        ]
        pair_changed_top1_before = self._mean(
            int(self._score_to_top1(row["score_before_pairwise"]) == int(row["ground_truth"]))
            for row in pair_changed_rows
        )
        pair_changed_top1_after = self._mean(
            int(self._score_to_top1(row["score_after_pairwise"]) == int(row["ground_truth"]))
            for row in pair_changed_rows
        )

        summary = {
            "final_accuracy": final_accuracy,
            "total_correct": total_correct,
            "total_samples": total_samples,
            "num_candidates": num_candidates,
            "num_retained": num_retained,
            "retention_ratio": self._ratio(num_retained, num_candidates),
            "memory_purity_before": purity_before,
            "memory_purity_after": purity_after,
            "num_gaussian_queries": len(gaussian_rows),
            "all_margin_before": all_margin_before,
            "all_margin_after": all_margin_after,
            "poor_coverage_threshold": coverage_threshold,
            "num_poor_coverage_queries": len(poor_rows),
            "poor_margin_before": poor_margin_before,
            "poor_margin_after": poor_margin_after,
            "num_activated_gt_pairs": len(pair_rows),
            "pair_margin_before": pair_margin_before,
            "pair_margin_after": pair_margin_after,
            "pair_error_before": pair_error_before,
            "pair_error_after": pair_error_after,
            "num_pairwise_changed_queries": len(pair_changed_rows),
            "pair_changed_top1_before": pair_changed_top1_before,
            "pair_changed_top1_after": pair_changed_top1_after,
        }
        self.last_summary = summary
        return summary

    def write_outputs(
        self,
        dataset_name: str,
        backbone: str,
        final_accuracy: Optional[float] = None,
        total_correct: Optional[int] = None,
        total_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        summary = self.build_summary(final_accuracy, total_correct, total_samples)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        prefix = self._output_prefix(dataset_name, backbone)
        summary_path = self.output_dir / f"{prefix}_direct_diagnostics_summary.tsv"
        raw_path = self.output_dir / f"{prefix}_direct_diagnostics_raw.csv"
        query_path = self.output_dir / f"{prefix}_query_diagnostics.csv"
        memory_path = self.output_dir / f"{prefix}_memory_diagnostics.csv"

        self._write_summary_tsv(summary_path, summary)
        self._write_query_csv(query_path)
        self._write_memory_csv(memory_path)
        self._write_raw_csv(raw_path)

        self.last_output_paths = {
            "summary": str(summary_path),
            "raw": str(raw_path),
            "query": str(query_path),
            "memory": str(memory_path),
        }
        return summary

    def print_summary_table(self, dataset_name: str, backbone: str) -> None:
        if self.last_summary is None:
            self.build_summary()
        assert self.last_summary is not None
        print(format_terminal_table(self.last_summary, dataset_name, backbone))

    @staticmethod
    def _score_to_list(score: Any) -> Optional[List[float]]:
        if score is None:
            return None
        if hasattr(score, "detach"):
            score = score.detach().float().cpu()
        if hasattr(score, "tolist"):
            score = score.tolist()
        if isinstance(score, list) and score and isinstance(score[0], list):
            score = score[0]
        if not isinstance(score, list):
            return None
        return [float(value) for value in score]

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> Optional[float]:
        if denominator == 0:
            return None
        return float(numerator) / float(denominator)

    @staticmethod
    def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
        clean = [float(value) for value in values if value is not None]
        if not clean:
            return None
        return sum(clean) / len(clean)

    @staticmethod
    def _margin(scores: Optional[Sequence[float]], y: Optional[int]) -> Optional[float]:
        if scores is None or y is None or y < 0 or y >= len(scores) or len(scores) < 2:
            return None
        competitors = [float(score) for idx, score in enumerate(scores) if idx != y]
        if not competitors:
            return None
        return float(scores[y]) - max(competitors)

    @staticmethod
    def _score_to_top1(scores: Optional[Sequence[float]]) -> Optional[int]:
        if scores is None:
            return None
        if len(scores) == 0:
            return None
        return max(range(len(scores)), key=lambda idx: scores[idx])

    @staticmethod
    def _output_prefix(dataset_name: str, backbone: str) -> str:
        if dataset_name == "TerraIncognita" and backbone == "ViT-B/16":
            return "terra_vitb16"
        safe_dataset = re.sub(r"[^a-z0-9]+", "_", dataset_name.lower()).strip("_")
        safe_backbone = re.sub(r"[^a-z0-9]+", "", backbone.lower())
        return f"{safe_dataset}_{safe_backbone}"

    def _annotate_query_metrics(self) -> None:
        for row in self.query_rows.values():
            y = row.get("ground_truth")
            if y is None:
                continue
            y = int(y)

            counts = row.get("local_memory_counts") or []
            row["correct_class_memory_count"] = counts[y] if 0 <= y < len(counts) else None
            row["margin_before_gaussian"] = self._margin(row.get("score_before_gaussian"), y)
            row["margin_after_gaussian"] = self._margin(row.get("score_after_gaussian"), y)

            before_pair = row.get("score_before_pairwise")
            after_pair = row.get("score_after_pairwise")
            pairs = row.get("activated_pairs") or []
            competitors: List[int] = []
            for a, b in pairs:
                if int(a) == y:
                    competitors.append(int(b))
                elif int(b) == y:
                    competitors.append(int(a))

            if not competitors or before_pair is None or after_pair is None:
                row["pair_competitor"] = None
                row["pair_margin_before"] = None
                row["pair_margin_after"] = None
                row["pair_error_before"] = None
                row["pair_error_after"] = None
                continue

            valid_competitors = [c for c in competitors if 0 <= c < len(before_pair)]
            if not valid_competitors:
                row["pair_competitor"] = None
                row["pair_margin_before"] = None
                row["pair_margin_after"] = None
                row["pair_error_before"] = None
                row["pair_error_after"] = None
                continue

            competitor = max(valid_competitors, key=lambda c: before_pair[c])
            row["pair_competitor"] = int(competitor)
            row["pair_margin_before"] = float(before_pair[y]) - float(before_pair[competitor])
            row["pair_margin_after"] = float(after_pair[y]) - float(after_pair[competitor])
            row["pair_error_before"] = int(float(before_pair[competitor]) >= float(before_pair[y]))
            row["pair_error_after"] = int(float(after_pair[competitor]) >= float(after_pair[y]))

    @staticmethod
    def _bottom_coverage_rows(rows: Sequence[Dict[str, Any]], fraction: float) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        valid = [row for row in rows if row.get("correct_class_memory_count") is not None]
        if not valid:
            return [], None
        target = max(1, int(math.ceil(len(valid) * fraction)))
        sorted_counts = sorted(int(row["correct_class_memory_count"]) for row in valid)
        threshold = sorted_counts[target - 1]
        selected = [row for row in valid if int(row["correct_class_memory_count"]) <= threshold]
        return selected, threshold

    @staticmethod
    def _json_or_empty(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (list, tuple, dict)):
            return json.dumps(value, separators=(",", ":"))
        return str(value)

    @staticmethod
    def _fmt_optional(value: Optional[float], scale: float = 1.0, decimals: int = 6) -> str:
        if value is None:
            return ""
        return f"{float(value) * scale:.{decimals}f}"

    def _write_summary_tsv(self, path: Path, summary: Dict[str, Any]) -> None:
        rows = [
            {
                "metric": "memory_purity",
                "before": self._fmt_optional(summary["memory_purity_before"], scale=100.0, decimals=4),
                "after": self._fmt_optional(summary["memory_purity_after"], scale=100.0, decimals=4),
                "delta": self._fmt_optional(self._delta(summary["memory_purity_after"], summary["memory_purity_before"]), scale=100.0, decimals=4),
                "n": str(summary["num_candidates"]),
            },
            {
                "metric": "all_correct_class_margin",
                "before": self._fmt_optional(summary["all_margin_before"]),
                "after": self._fmt_optional(summary["all_margin_after"]),
                "delta": self._fmt_optional(self._delta(summary["all_margin_after"], summary["all_margin_before"])),
                "n": str(summary["num_gaussian_queries"]),
            },
            {
                "metric": "poor_coverage_correct_class_margin",
                "before": self._fmt_optional(summary["poor_margin_before"]),
                "after": self._fmt_optional(summary["poor_margin_after"]),
                "delta": self._fmt_optional(self._delta(summary["poor_margin_after"], summary["poor_margin_before"])),
                "n": str(summary["num_poor_coverage_queries"]),
            },
            {
                "metric": "confusing_pair_margin",
                "before": self._fmt_optional(summary["pair_margin_before"]),
                "after": self._fmt_optional(summary["pair_margin_after"]),
                "delta": self._fmt_optional(self._delta(summary["pair_margin_after"], summary["pair_margin_before"])),
                "n": str(summary["num_activated_gt_pairs"]),
            },
            {
                "metric": "confusing_pair_error",
                "before": self._fmt_optional(summary["pair_error_before"], scale=100.0, decimals=4),
                "after": self._fmt_optional(summary["pair_error_after"], scale=100.0, decimals=4),
                "delta": self._fmt_optional(self._delta(summary["pair_error_after"], summary["pair_error_before"]), scale=100.0, decimals=4),
                "n": str(summary["num_activated_gt_pairs"]),
            },
            {
                "metric": "confusing_pair_top1_changed",
                "before": self._fmt_optional(summary["pair_changed_top1_before"], scale=100.0, decimals=4),
                "after": self._fmt_optional(summary["pair_changed_top1_after"], scale=100.0, decimals=4),
                "delta": self._fmt_optional(self._delta(summary["pair_changed_top1_after"], summary["pair_changed_top1_before"]), scale=100.0, decimals=4),
                "n": str(summary["num_pairwise_changed_queries"]),
            },
            {"metric": "num_candidates", "before": "", "after": "", "delta": "", "n": str(summary["num_candidates"])},
            {"metric": "num_retained", "before": "", "after": "", "delta": "", "n": str(summary["num_retained"])},
            {
                "metric": "retention_ratio",
                "before": "",
                "after": "",
                "delta": "",
                "n": self._fmt_optional(summary["retention_ratio"], decimals=6),
            },
            {
                "metric": "poor_coverage_threshold",
                "before": "",
                "after": "",
                "delta": "",
                "n": "" if summary["poor_coverage_threshold"] is None else str(summary["poor_coverage_threshold"]),
            },
            {
                "metric": "num_poor_coverage_queries",
                "before": "",
                "after": "",
                "delta": "",
                "n": str(summary["num_poor_coverage_queries"]),
            },
            {
                "metric": "num_activated_gt_pairs",
                "before": "",
                "after": "",
                "delta": "",
                "n": str(summary["num_activated_gt_pairs"]),
            },
        ]
        if summary.get("final_accuracy") is not None:
            rows.append({
                "metric": "final_accuracy",
                "before": "",
                "after": "",
                "delta": "",
                "n": self._fmt_optional(summary["final_accuracy"], scale=100.0, decimals=4),
            })
        if summary.get("total_correct") is not None and summary.get("total_samples") is not None:
            rows.append({
                "metric": "total_correct_over_total",
                "before": "",
                "after": "",
                "delta": "",
                "n": f"{summary['total_correct']}/{summary['total_samples']}",
            })

        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["metric", "before", "after", "delta", "n"], delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _delta(after: Optional[float], before: Optional[float]) -> Optional[float]:
        if after is None or before is None:
            return None
        return float(after) - float(before)

    def _write_query_csv(self, path: Path) -> None:
        fieldnames = [
            "sample_uid",
            "query_order",
            "client_id",
            "sample_index",
            "original_index",
            "ground_truth",
            "final_pred",
            "correct",
            "clip_pred",
            "clip_entropy",
            "local_memory_counts",
            "correct_class_memory_count",
            "gaussian_applied",
            "score_before_gaussian",
            "score_after_gaussian",
            "margin_before_gaussian",
            "margin_after_gaussian",
            "score_before_pairwise",
            "score_after_pairwise",
            "activated_pairs",
            "num_activated_pairs",
            "pair_competitor",
            "pair_margin_before",
            "pair_margin_after",
            "pair_error_before",
            "pair_error_after",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.query_rows.values():
                writer.writerow({key: self._json_or_empty(row.get(key)) for key in fieldnames})

    def _write_memory_csv(self, path: Path) -> None:
        fieldnames = [
            "memory_event_id",
            "sample_uid",
            "client_id",
            "pseudo_label",
            "ground_truth",
            "pseudo_correct",
            "candidate",
            "retained",
            "class_index",
            "slot_index",
            "entropy",
            "normalized_entropy",
            "reconstruction_error",
            "regulation_score",
            "dual_criteria",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.memory_rows:
                writer.writerow({key: self._json_or_empty(row.get(key)) for key in fieldnames})

    def _write_raw_csv(self, path: Path) -> None:
        fieldnames = [
            "record_type",
            "sample_uid",
            "query_order",
            "memory_event_id",
            "client_id",
            "sample_index",
            "original_index",
            "ground_truth",
            "final_pred",
            "correct",
            "clip_pred",
            "clip_entropy",
            "pseudo_label",
            "pseudo_correct",
            "candidate",
            "retained",
            "class_index",
            "slot_index",
            "entropy",
            "normalized_entropy",
            "reconstruction_error",
            "regulation_score",
            "dual_criteria",
            "local_memory_counts",
            "correct_class_memory_count",
            "gaussian_applied",
            "score_before_gaussian",
            "score_after_gaussian",
            "margin_before_gaussian",
            "margin_after_gaussian",
            "score_before_pairwise",
            "score_after_pairwise",
            "activated_pairs",
            "num_activated_pairs",
            "pair_competitor",
            "pair_margin_before",
            "pair_margin_after",
            "pair_error_before",
            "pair_error_after",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.query_rows.values():
                out = {"record_type": "query"}
                out.update(row)
                writer.writerow({key: self._json_or_empty(out.get(key)) for key in fieldnames})
            for row in self.memory_rows:
                out = {"record_type": "memory"}
                out.update(row)
                writer.writerow({key: self._json_or_empty(out.get(key)) for key in fieldnames})


def format_terminal_table(summary: Dict[str, Any], dataset_name: str, backbone: str) -> str:
    def pct(value: Optional[float]) -> str:
        return "nan" if value is None else f"{value * 100.0:.2f}"

    def val(value: Optional[float]) -> str:
        return "nan" if value is None else f"{value:.4f}"

    def signed_pct(value: Optional[float]) -> str:
        return "nan" if value is None else f"{value * 100.0:+.2f}"

    def signed_val(value: Optional[float]) -> str:
        return "nan" if value is None else f"{value:+.4f}"

    memory_delta = DirectMechanismDiagnosticLogger._delta(
        summary.get("memory_purity_after"),
        summary.get("memory_purity_before"),
    )
    poor_delta = DirectMechanismDiagnosticLogger._delta(
        summary.get("poor_margin_after"),
        summary.get("poor_margin_before"),
    )
    pair_margin_delta = DirectMechanismDiagnosticLogger._delta(
        summary.get("pair_margin_after"),
        summary.get("pair_margin_before"),
    )
    pair_error_delta = DirectMechanismDiagnosticLogger._delta(
        summary.get("pair_error_after"),
        summary.get("pair_error_before"),
    )
    pair_changed_delta = DirectMechanismDiagnosticLogger._delta(
        summary.get("pair_changed_top1_after"),
        summary.get("pair_changed_top1_before"),
    )

    title_dataset = "TerraIncognita" if dataset_name == "TerraIncognita" else dataset_name
    lines = [
        "=" * 60,
        f"Direct Mechanism Diagnostics | {title_dataset} | {backbone}",
        "=" * 60,
        f"{'Metric':34s}{'Before':>12s}{'After':>12s}{'Delta':>12s}{'N':>10s}",
        f"{'Memory purity (%)':34s}{pct(summary.get('memory_purity_before')):>12s}{pct(summary.get('memory_purity_after')):>12s}{signed_pct(memory_delta):>12s}{summary.get('num_candidates', 0):>10}",
        f"{'Correct-class margin (poor 25%)':34s}{val(summary.get('poor_margin_before')):>12s}{val(summary.get('poor_margin_after')):>12s}{signed_val(poor_delta):>12s}{summary.get('num_poor_coverage_queries', 0):>10}",
        f"{'Confusing-pair margin':34s}{val(summary.get('pair_margin_before')):>12s}{val(summary.get('pair_margin_after')):>12s}{signed_val(pair_margin_delta):>12s}{summary.get('num_activated_gt_pairs', 0):>10}",
        f"{'Confusing-pair error (%)':34s}{pct(summary.get('pair_error_before')):>12s}{pct(summary.get('pair_error_after')):>12s}{signed_pct(pair_error_delta):>12s}{summary.get('num_activated_gt_pairs', 0):>10}",
        f"{'Pairwise-changed top-1 (%)':34s}{pct(summary.get('pair_changed_top1_before')):>12s}{pct(summary.get('pair_changed_top1_after')):>12s}{signed_pct(pair_changed_delta):>12s}{summary.get('num_pairwise_changed_queries', 0):>10}",
        "=" * 60,
    ]
    return "\n".join(lines)


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


class _RunTrace:
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


def _install_trace_hooks(runner_module: Any, trace: _RunTrace) -> Callable[[], None]:
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


def _parse_accuracy(output: str, trace: _RunTrace) -> Optional[float]:
    matches = re.findall(r"Total Accuracy:\s*([0-9.]+)%\s*\((\d+)\s*/\s*(\d+)\)", output)
    if matches:
        _, correct, total = matches[-1]
        total_int = int(total)
        if total_int:
            return int(correct) / total_int
    return trace.accuracy


def _run_runner(argv: Sequence[str], repo_root: Path, trace_enabled: bool = True) -> Dict[str, Any]:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import mcc_tta_runner

    trace = _RunTrace()
    restore = _install_trace_hooks(mcc_tta_runner, trace) if trace_enabled else (lambda: None)
    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    stdout_tee = _Tee(sys.stdout, stdout_buffer)
    stderr_tee = _Tee(sys.stderr, stderr_buffer)

    try:
        os.chdir(repo_root)
        sys.argv = [str(repo_root / "mcc_tta_runner.py"), *argv]
        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            mcc_tta_runner.main()
    finally:
        restore()
        sys.argv = old_argv
        os.chdir(old_cwd)

    output = stdout_buffer.getvalue() + stderr_buffer.getvalue()
    return {
        "trace": trace,
        "accuracy": _parse_accuracy(output, trace),
        "output": output,
    }


def _default_runner_argv(args: argparse.Namespace) -> List[str]:
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
    if args.separate_domains:
        argv.append("--separate-domains")
    if args.topology != "similarity":
        argv.extend(["--topology", args.topology])
    return argv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCC-TTA direct mechanism diagnostics.")
    parser.add_argument("--config", default="configs/mcc_tta_terra.yaml")
    parser.add_argument("--datasets", default="TerraIncognita")
    parser.add_argument("--data-root", default="./dataset/")
    parser.add_argument("--backbone", default="ViT-B/16")
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--part-rate", type=float, default=1.0)
    parser.add_argument("--sync-freq", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--topology", default="similarity", choices=["similarity", "ring", "random", "full"])
    parser.add_argument("--cache-features", dest="cache_features", action="store_true", default=True)
    parser.add_argument("--no-cache-features", dest="cache_features", action="store_false")
    parser.add_argument("--separate-domains", action="store_true")
    parser.add_argument("--diagnostic-output-dir", default="results/rebuttal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    base_argv = _default_runner_argv(args)

    print("Running MCC-TTA sanity check with diagnostics OFF...")
    off_result = _run_runner(base_argv, repo_root=repo_root, trace_enabled=True)

    print("Running MCC-TTA with --rebuttal_diagnostic ON...")
    on_argv = [
        *base_argv,
        "--rebuttal_diagnostic",
        "--diagnostic-output-dir",
        args.diagnostic_output_dir,
    ]
    on_result = _run_runner(on_argv, repo_root=repo_root, trace_enabled=True)

    off_acc = off_result["accuracy"]
    on_acc = on_result["accuracy"]
    acc_match = (
        off_acc is not None
        and on_acc is not None
        and abs(float(off_acc) - float(on_acc)) <= 1e-12
    )
    order_match = off_result["trace"].sample_order == on_result["trace"].sample_order
    pred_match = off_result["trace"].predictions == on_result["trace"].predictions

    print("\nSanity checks:")
    print(f"  diagnostic OFF accuracy: {off_acc * 100.0:.4f}%" if off_acc is not None else "  diagnostic OFF accuracy: unavailable")
    print(f"  diagnostic ON accuracy : {on_acc * 100.0:.4f}%" if on_acc is not None else "  diagnostic ON accuracy : unavailable")
    print(f"  final accuracy identical: {'YES' if acc_match else 'NO'}")
    print(f"  sample order identical  : {'YES' if order_match else 'NO'}")
    print(f"  prediction identical    : {'YES' if pred_match else 'NO'}")

    if not (acc_match and order_match and pred_match):
        raise SystemExit("Sanity check failed; diagnostic output should not be used until the mismatch is fixed.")


if __name__ == "__main__":
    main()
