"""Sparse semantic occupancy mapping over registered scans."""
from __future__ import annotations
import numpy as np


class SparseOccupancyMap:
    """Voxel hash carrying an occupancy count and a semantic label histogram.

    Dense grids over a kilometre of driving run to tens of millions of cells
    while the occupied set is orders of magnitude smaller, so the map is stored
    as a dict keyed by integer voxel coordinate.
    """

    def __init__(self, voxel_size: float = 0.2, n_classes: int = 5):
        self.voxel_size = float(voxel_size)
        self.n_classes = int(n_classes)
        self._counts: dict[tuple[int, int, int], np.ndarray] = {}

    def _key(self, pts: np.ndarray) -> np.ndarray:
        return np.floor(pts / self.voxel_size).astype(np.int32)

    def integrate(self, points_world: np.ndarray, labels: np.ndarray | None = None):
        keys = self._key(points_world)
        if labels is None:
            labels = np.zeros(len(points_world), dtype=np.int32)
        for k, lab in zip(map(tuple, keys), labels):
            hist = self._counts.get(k)
            if hist is None:
                hist = np.zeros(self.n_classes, dtype=np.int32)
                self._counts[k] = hist
            hist[int(lab)] += 1

    def __len__(self) -> int:
        return len(self._counts)

    def occupied_centres(self) -> np.ndarray:
        if not self._counts:
            return np.zeros((0, 3), dtype=np.float32)
        keys = np.array(list(self._counts.keys()), dtype=np.float32)
        return (keys + 0.5) * self.voxel_size

    def labels(self) -> np.ndarray:
        return np.array([h.argmax() for h in self._counts.values()], dtype=np.int32)

    def to_cubes(self):
        """Yield (centre, size) per occupied voxel, for rendering as mesh cubes."""
        for centre in self.occupied_centres():
            yield centre, self.voxel_size
