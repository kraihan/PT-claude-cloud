from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


class ArrayMemoryBank:
    """Class-wise ring buffer for feature/image samples used by generator training."""

    def __init__(self, num_classes: int = 1000, max_size: int = 64, dtype=np.float32):
        self.num_classes = int(num_classes)
        self.max_size = int(max_size)
        self.dtype = dtype
        self.bank: Optional[np.ndarray] = None
        self.feature_shape: Optional[Tuple[int, ...]] = None
        self.ptr = np.zeros(self.num_classes, dtype=np.int32)
        self.count = np.zeros(self.num_classes, dtype=np.int32)
        self.ids = np.full((self.num_classes, self.max_size), -1, dtype=np.int64)
        self.next_id = 0

    def _init_bank(self, sample_shape: Tuple[int, ...]) -> None:
        self.feature_shape = tuple(sample_shape)
        self.bank = np.zeros((self.num_classes, self.max_size, *self.feature_shape), dtype=self.dtype)

    def add(self, samples, labels, ids=None):
        if torch.is_tensor(samples):
            samples = samples.detach().cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy()
        samples = np.asarray(samples)
        labels = np.asarray(labels)
        if self.bank is None:
            self._init_bank(samples.shape[1:])
        if ids is None:
            ids = np.arange(self.next_id, self.next_id + len(labels), dtype=np.int64)
        ids = np.asarray(ids, dtype=np.int64)
        if len(ids) != len(labels):
            raise ValueError("One cache identity is required for each sample")
        if len(ids):
            self.next_id = max(self.next_id, int(ids.max()) + 1)

        for i in range(labels.shape[0]):
            lbl = int(labels[i])
            idx = self.ptr[lbl]
            self.bank[lbl, idx] = samples[i]
            self.ids[lbl, idx] = ids[i]
            self.ptr[lbl] = (idx + 1) % self.max_size
            if self.count[lbl] < self.max_size:
                self.count[lbl] += 1
        return ids

    def sample(self, labels, n_samples: int, *, return_ids=False):
        if self.bank is None or self.feature_shape is None:
            raise RuntimeError("MemoryBank is empty. Call add() before sample().")

        labels = np.asarray(labels)
        bsz = labels.shape[0]
        sample_indices = np.empty((bsz, n_samples), dtype=np.int32)
        for i in range(bsz):
            lbl = int(labels[i])
            valid = int(self.count[lbl])
            if valid <= 0:
                raise ValueError(f"Class {lbl} has no real samples in the memory bank.")
            else:
                sample_indices[i] = np.random.choice(valid, n_samples, replace=(valid < n_samples))

        out = self.bank[labels[:, None], sample_indices]
        tensor = torch.from_numpy(np.asarray(out))
        return (tensor, self.ids[labels[:, None], sample_indices].copy()) if return_ids else tensor
