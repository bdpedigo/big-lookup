"""Point -> supervoxel lookup over parquet / delta lake sources.

v0: single-process, sequential per-block loop. The only fan-out is CloudVolume's
own threaded / green IO concurrency inside each ``scattered_points`` call. Output
is a folder of one parquet file per block, so reruns are resumable (a block is
skipped if its output file already exists). Works against local or cloud paths.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import polars as pl
from cloudpathlib import AnyPath

__all__ = ["lookup_supervoxels"]

# Default block size, in voxels, per axis. Blocks are the unit of work: all points
# whose voxel coords floor into the same block key are looked up together so that
# CloudVolume's LRU can dedup overlapping chunk downloads within the block.
DEFAULT_BLOCK_SIZE = (512, 512, 512)


def lookup_supervoxels(
    input_path: str,
    cv,
    output_path: str,
    *,
    resolution: Sequence[float] = (1.0, 1.0, 1.0),
    block_size: Sequence[int] = DEFAULT_BLOCK_SIZE,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    extra_cols: Sequence[str] | None = None,
) -> None:
    """Look up the supervoxel id for every point in ``input_path``.

    Parameters
    ----------
    input_path :
        Path (local or cloud) to a parquet file or delta table of points. Must
        contain ``x_col``, ``y_col``, ``z_col`` columns.
    cv :
        Either a ready ``CloudVolume`` instance, or a segmentation source path
        (e.g. ``"graphene://..."`` / ``"precomputed://gs://..."``) from which a
        ``CloudVolume`` is constructed. When passing an instance, the caller is
        responsible for its configuration (mip, lru, caching, https, ...).
    output_path :
        Path (local or cloud) to a folder. One parquet file is written per
        occupied block. Created if it does not exist.
    resolution :
        The coordinate resolution (nm per unit) that the input ``x, y, z`` are
        expressed in. Passed through to ``cv.scattered_points`` as
        ``coord_resolution`` and used to convert the input coordinates into
        integer voxels for partitioning and rejoining. For nanometer input, use
        ``(1, 1, 1)``.
    block_size :
        Per-axis block dimensions, in *voxel* coordinates.
    x_col, y_col, z_col :
        Names of the coordinate columns in the input.
    extra_cols :
        Optional list of additional input column names to carry through, untouched,
        to the output. These are passthrough only and are never used in any
        computation.
    """
    resolution = np.asarray(resolution, dtype=np.float64)
    block_size = np.asarray(block_size, dtype=np.int64)
    if resolution.shape != (3,):
        raise ValueError("resolution must have 3 elements")
    if block_size.shape != (3,) or np.any(block_size <= 0):
        raise ValueError("block_size must have 3 positive elements")

    out_dir = AnyPath(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    cv = _resolve_cloudvolume(cv)

    # Segmentation voxel resolution (nm) at the CloudVolume's active mip, and the
    # segmentation chunk shape (voxels) used for the intra-block chunk-coherent sort.
    seg_resolution = np.asarray(cv.resolution, dtype=np.float64)
    seg_chunk_size = np.asarray(cv.chunk_size, dtype=np.int64)

    # Convert input coords (in `resolution` units) -> integer voxel coords at the
    # segmentation's active mip. This mirrors what scattered_points does internally
    # (pt / (seg_resolution / coord_resolution)) so the returned dict keys line up.
    voxel_factor = seg_resolution / resolution

    extra_cols = list(extra_cols) if extra_cols is not None else []

    # Raw-coordinate width of one block per axis: block_size (voxels) * voxel_factor
    # (raw units per voxel). Because block/voxel are affine maps of the raw coords,
    # the block key b maps to the exact half-open raw interval [b*step, (b+1)*step);
    # filtering on the *raw* columns (not derived keys) lets polars push the
    # predicate into the parquet scan and prune row groups / files.
    step = block_size.astype(np.float64) * voxel_factor

    # Enumerate occupied blocks in one scan. bx = floor(vx / block_size) reduces to
    # floor(x / step) by the nested-floor identity floor(floor(a)/n) == floor(a/n).
    blocks = (
        _scan_points(input_path, x_col, y_col, z_col, [])
        .select(
            (pl.col(x_col) / step[0]).floor().cast(pl.Int64).alias("bx"),
            (pl.col(y_col) / step[1]).floor().cast(pl.Int64).alias("by"),
            (pl.col(z_col) / step[2]).floor().cast(pl.Int64).alias("bz"),
        )
        .unique()
        .sort("bx", "by", "bz")
        .collect()
    )

    n_blocks = blocks.height
    for i, (bx, by, bz) in enumerate(blocks.iter_rows()):
        out_file = out_dir / f"block_{bx}_{by}_{bz}.parquet"
        if out_file.exists():
            print(f"[{i + 1}/{n_blocks}] block ({bx},{by},{bz}) exists, skipping")
            continue

        lo = np.array([bx, by, bz], dtype=np.float64) * step
        hi = lo + step

        block = (
            # Fresh filtered scan on RAW columns -> predicate pushdown / pruning.
            _scan_points(input_path, x_col, y_col, z_col, extra_cols)
            .filter(
                (pl.col(x_col) >= lo[0])
                & (pl.col(x_col) < hi[0])
                & (pl.col(y_col) >= lo[1])
                & (pl.col(y_col) < hi[1])
                & (pl.col(z_col) >= lo[2])
                & (pl.col(z_col) < hi[2])
            )
            .with_columns(
                (pl.col(x_col) / voxel_factor[0]).floor().cast(pl.Int64).alias("vx"),
                (pl.col(y_col) / voxel_factor[1]).floor().cast(pl.Int64).alias("vy"),
                (pl.col(z_col) / voxel_factor[2]).floor().cast(pl.Int64).alias("vz"),
            )
            # chunk-coherent sort: keep each worker's live chunk set to a local band
            .with_columns(
                (pl.col("vx") // seg_chunk_size[0]).alias("_cx"),
                (pl.col("vy") // seg_chunk_size[1]).alias("_cy"),
                (pl.col("vz") // seg_chunk_size[2]).alias("_cz"),
            )
            .sort("_cx", "_cy", "_cz")
            .drop("_cx", "_cy", "_cz")
            .collect()
        )

        result = _lookup_block(block, cv, resolution, x_col, y_col, z_col, extra_cols)
        _write_block(result, out_file)
        print(
            f"[{i + 1}/{n_blocks}] block ({bx},{by},{bz}): "
            f"{result.height} points -> {out_file.name}"
        )


def _resolve_cloudvolume(cv):
    """Return ``cv`` unchanged if it is already a CloudVolume, else build one.

    A string (or path-like) is interpreted as a segmentation source and passed to
    ``CloudVolume(...)``. Construction kwargs are fixed internally for now (they may
    depend on the parallelization backend); safe ones can be re-exposed later.
    Anything else is assumed to already be a usable volume.
    """
    if isinstance(cv, (str, bytes)) or hasattr(cv, "__fspath__"):
        from cloudvolume import CloudVolume

        return CloudVolume(str(cv), use_https=True)
    return cv


def _scan_points(
    input_path: str,
    x_col: str,
    y_col: str,
    z_col: str,
    extra_cols: list[str],
) -> pl.LazyFrame:
    """Lazily scan the point source, selecting coordinate + passthrough columns."""
    path = AnyPath(input_path)
    if path.suffix == ".parquet" or path.is_file():
        lf = pl.scan_parquet(str(input_path))
    else:
        lf = pl.scan_delta(str(input_path))

    return lf.select(x_col, y_col, z_col, *extra_cols)


def _lookup_block(
    block: pl.DataFrame,
    cv,
    resolution: np.ndarray,
    x_col: str,
    y_col: str,
    z_col: str,
    extra_cols: list[str],
) -> pl.DataFrame:
    """Run the supervoxel lookup for one block and rejoin onto input rows.

    ``scattered_points`` returns an unordered ``{(vx, vy, vz): label}`` dict keyed
    by integer voxel coords at the active mip, deduped across points. We rejoin it
    onto the carried voxel coords so every input row (including multiple rows that
    collapse to the same voxel) gets its label.
    """
    pts = block.select(x_col, y_col, z_col).to_numpy()
    labels = cv.scattered_points(pts, coord_resolution=tuple(resolution))

    keys = list(labels.keys())
    lookup = pl.DataFrame(
        {
            "vx": [k[0] for k in keys],
            "vy": [k[1] for k in keys],
            "vz": [k[2] for k in keys],
            "supervoxel": [int(labels[k]) for k in keys],
        },
        schema={
            "vx": pl.Int64,
            "vy": pl.Int64,
            "vz": pl.Int64,
            "supervoxel": pl.UInt64,
        },
    )

    return block.join(lookup, on=["vx", "vy", "vz"], how="left").select(
        *extra_cols, x_col, y_col, z_col, "vx", "vy", "vz", "supervoxel"
    )


def _write_block(result: pl.DataFrame, out_file) -> None:
    """Write one block's result parquet, atomically where possible.

    For local paths we write to a temp file and rename (atomic). For cloud paths
    the per-object write is already atomic, so we write directly.
    """
    if _is_local(out_file):
        tmp = out_file.with_name(out_file.name + ".tmp")
        result.write_parquet(str(tmp))
        tmp.rename(out_file)
    else:
        result.write_parquet(str(out_file))


def _is_local(path) -> bool:
    return type(path).__name__ == "PosixPath" or type(path).__name__ == "WindowsPath"
