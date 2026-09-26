"""Utilities for rebuttal-only pseudo-label noise experiments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass
class GaussianLabelNoiseStats:
    total_updates: int = 0
    corrupted_updates: int = 0

    @property
    def realized_noise(self) -> float:
        if self.total_updates == 0:
            return 0.0
        return self.corrupted_updates / self.total_updates


class GaussianLabelNoiseInjector:
    """Deterministically corrupt only the label used by Gaussian statistics."""

    def __init__(self, num_classes: int, noise_ratio: float = 0.0, seed: int = 2026) -> None:
        if not 0.0 <= float(noise_ratio) <= 1.0:
            raise ValueError("--gaussian_label_noise must be in [0, 1].")
        if int(num_classes) <= 1 and float(noise_ratio) > 0.0:
            raise ValueError("Gaussian label noise requires at least two classes.")

        self.num_classes = int(num_classes)
        self.noise_ratio = float(noise_ratio)
        self.seed = int(seed)
        self.stats = GaussianLabelNoiseStats()

    def maybe_corrupt(self, pseudo_label: int, client_id: object = "") -> int:
        """Return a possibly corrupted Gaussian-update label.

        The original pseudo-label is never mutated. The corruption mask is
        deterministic per Gaussian update, so higher ratios nest lower ratios.
        """

        original = int(pseudo_label)
        update_id = self.stats.total_updates
        self.stats.total_updates += 1

        if self.noise_ratio <= 0.0:
            return original

        key = f"{self.seed}|{client_id}|{update_id}|{original}"
        if self._unit_float(key + "|mask") >= self.noise_ratio:
            return original

        wrong_base = self._uint(key + "|wrong") % (self.num_classes - 1)
        wrong_label = wrong_base + int(wrong_base >= original)
        self.stats.corrupted_updates += 1
        return int(wrong_label)

    @staticmethod
    def _uint(key: str) -> int:
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False)

    @classmethod
    def _unit_float(cls, key: str) -> float:
        return cls._uint(key) / float(1 << 64)
