"""Float64 sufficient statistics for FID, without retaining images/features."""
import numpy as np


class FeatureMoments:
    def __init__(self, dimension):
        self.count = 0
        self.sum = np.zeros(dimension, dtype=np.float64)
        self.cross = np.zeros((dimension, dimension), dtype=np.float64)

    def update(self, features):
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.sum) or not np.isfinite(x).all():
            raise ValueError("Invalid Inception features")
        self.count += len(x)
        self.sum += x.sum(0)
        self.cross += x.T @ x

    def statistics(self):
        if self.count < 2:
            raise ValueError("FID requires at least two features")
        mean = self.sum / self.count
        covariance = (self.cross - np.outer(self.sum, self.sum) / self.count) / (self.count - 1)
        return mean, (covariance + covariance.T) * 0.5

    def reduce(self, device):
        """One collective for small sufficient statistics, not all samples."""
        import torch
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized():
            return
        packed = np.concatenate(([float(self.count)], self.sum, self.cross.reshape(-1)))
        tensor = torch.from_numpy(packed).to(device)
        dist.all_reduce(tensor)
        packed = tensor.cpu().numpy()
        d = len(self.sum)
        self.count = int(round(packed[0]))
        self.sum = packed[1:d + 1].copy()
        self.cross = packed[d + 1:].reshape(d, d).copy()
