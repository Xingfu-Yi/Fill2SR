from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

TileBox = Tuple[int, int, int, int]


def _effective_kernel_size(length: int, kernel: int) -> int:
    if length <= 0:
        raise ValueError("length must be positive")
    if kernel <= 0:
        raise ValueError("kernel must be positive")
    return min(length, kernel)


def compute_tile_origins(length: int, kernel: int, overlap: float) -> List[int]:
    """Return tile start positions that match LocalAttentionPlus stepping."""
    if overlap <= 0:
        raise ValueError("overlap must be > 0")

    kernel = _effective_kernel_size(length, kernel)
    num_tiles = (length - 1) // kernel + 1
    step = kernel if num_tiles == 1 else math.ceil(kernel * overlap)

    origins: List[int] = []
    pos = 0
    last = False
    while pos < length and not last:
        if pos + kernel >= length:
            pos = length - kernel
            last = True
        origins.append(pos)
        pos += step

    return origins


def compute_tile_boxes(
    height: int,
    width: int,
    kernel_height: int,
    kernel_width: int,
    overlap: float,
) -> List[TileBox]:
    """Return half-open tile boxes as (top, left, bottom, right)."""
    kernel_height = _effective_kernel_size(height, kernel_height)
    kernel_width = _effective_kernel_size(width, kernel_width)

    rows = compute_tile_origins(height, kernel_height, overlap)
    cols = compute_tile_origins(width, kernel_width, overlap)

    return [
        (top, left, top + kernel_height, left + kernel_width)
        for top in rows
        for left in cols
    ]


def build_visit_count_map(
    height: int,
    width: int,
    kernel_height: int,
    kernel_width: int,
    overlap: float,
) -> np.ndarray:
    """Count how many tiles cover each pixel."""
    visit_count = np.zeros((height, width), dtype=np.uint16)
    for top, left, bottom, right in compute_tile_boxes(
        height=height,
        width=width,
        kernel_height=kernel_height,
        kernel_width=kernel_width,
        overlap=overlap,
    ):
        visit_count[top:bottom, left:right] += 1
    return visit_count
