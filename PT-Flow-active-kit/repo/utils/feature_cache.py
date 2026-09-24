"""Bounded, rank-local cache of deterministic frozen real-image features.

Memory-bank insertion IDs identify the actual augmented tensor, not a dataset
index or ring slot. Overwritten slots can therefore never return stale features.
The cache retains the extractor's original dtype. It is never used on generated
images and must be recreated if extractor weights or activation settings change.
"""
from collections import OrderedDict
import numpy as np
import torch


class FrozenFeatureCache:
    def __init__(self, max_bytes):
        self.max_bytes = int(max_bytes)
        if self.max_bytes <= 0:
            raise ValueError("Feature cache needs a positive byte budget")
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = 0

    @torch.no_grad()
    def get(self, params, images, ids, apply, **kwargs):
        keys, first, inverse = np.unique(np.asarray(ids).reshape(-1), return_index=True, return_inverse=True)
        if np.any(keys < 0) or len(inverse) != len(images):
            raise ValueError("Feature identities must match the real-image batch")
        rows, missing, missing_positions = {}, [], []
        for key, pos in zip(keys.tolist(), first.tolist()):
            if key in self.entries:
                rows[key] = self.entries[key][0]
                self.entries.move_to_end(key)
                self.hits += 1
            else:
                missing.append(key)
                missing_positions.append(pos)
                self.misses += 1
        if missing:
            indices = torch.tensor(missing_positions, device=images.device)
            features = apply(params, images.index_select(0, indices), **kwargs)
            for i, key in enumerate(missing):
                # clone prevents a tiny cached slice retaining a whole batch.
                row = {name: value[i].detach().clone() for name, value in features.items()}
                rows[key] = row
                size = sum(v.numel() * v.element_size() for v in row.values())
                if size <= self.max_bytes:
                    while self.entries and self.bytes + size > self.max_bytes:
                        _, (_, evicted_size) = self.entries.popitem(last=False)
                        self.bytes -= evicted_size
                    self.entries[key] = (row, size)
                    self.bytes += size
        indices = torch.tensor(inverse, device=images.device)
        return {name: torch.stack([rows[int(key)][name] for key in keys]).index_select(0, indices)
                for name in rows[int(keys[0])]}

    def metrics(self):
        result = {"feature_cache/hit_fraction": self.hits / max(1, self.hits + self.misses),
                  "feature_cache/gib": self.bytes / 2**30}
        self.hits = self.misses = 0
        return result
