"""Smoke test for lookup_supervoxels using a fake CloudVolume."""

import numpy as np
import polars as pl
import pytest
from big_lookup import lookup_supervoxels


class FakeCloudVolume:
    """Minimal stand-in for a CloudVolume returning deterministic labels."""

    resolution = (8, 8, 40)
    chunk_size = (128, 128, 16)

    def __init__(self):
        self.n_calls = 0

    def scattered_points(self, pts, coord_resolution=None):
        self.n_calls += 1
        pts = np.asarray(pts, dtype=np.float64)
        factor = np.asarray(self.resolution, dtype=np.float64) / np.asarray(
            coord_resolution, dtype=np.float64
        )
        result = {}
        for pt in pts:
            vx, vy, vz = np.floor(pt / factor).astype(np.int64)
            # deterministic fake label from the voxel coordinate
            result[(int(vx), int(vy), int(vz))] = int(vx + vy + vz + 1)
        return result


@pytest.fixture
def input_parquet(tmp_path):
    # nanometer coordinates spread across multiple blocks
    df = pl.DataFrame(
        {
            "x": [8000, 8016, 4_000_000, 4_000_016],
            "y": [8000, 8016, 4_000_000, 4_000_016],
            "z": [40000, 40040, 4_000_000, 4_000_040],
            "label": ["a", "b", "c", "d"],
        }
    )
    path = tmp_path / "points.parquet"
    df.write_parquet(path)
    return path


def test_lookup_writes_per_block_files(input_parquet, tmp_path):
    cv = FakeCloudVolume()
    out_dir = tmp_path / "out"

    lookup_supervoxels(
        str(input_parquet),
        cv,
        str(out_dir),
        resolution=(1, 1, 1),
        block_size=(256, 256, 256),
    )

    files = sorted(out_dir.glob("block_*.parquet"))
    assert len(files) == 2  # two spatially separated clusters -> two blocks

    combined = pl.concat([pl.read_parquet(f) for f in files])
    assert combined.height == 4
    assert set(combined.columns) == {
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "supervoxel",
    }
    # every point got a label
    assert combined["supervoxel"].null_count() == 0
    # label matches the fake rule vx+vy+vz+1
    check = combined.with_columns(
        (pl.col("vx") + pl.col("vy") + pl.col("vz") + 1).alias("expected")
    )
    assert (check["supervoxel"] == check["expected"]).all()


def test_extra_cols_carried_through(input_parquet, tmp_path):
    cv = FakeCloudVolume()
    out_dir = tmp_path / "out"

    lookup_supervoxels(
        str(input_parquet),
        cv,
        str(out_dir),
        resolution=(1, 1, 1),
        block_size=(256, 256, 256),
        extra_cols=["label"],
    )

    files = sorted(out_dir.glob("block_*.parquet"))
    combined = pl.concat([pl.read_parquet(f) for f in files])
    assert "label" in combined.columns
    assert set(combined["label"].to_list()) == {"a", "b", "c", "d"}


def test_rerun_skips_existing_blocks(input_parquet, tmp_path):
    out_dir = tmp_path / "out"

    cv1 = FakeCloudVolume()
    lookup_supervoxels(
        str(input_parquet),
        cv1,
        str(out_dir),
        resolution=(1, 1, 1),
        block_size=(256, 256, 256),
    )
    assert cv1.n_calls == 2

    # second run should skip both existing blocks -> no lookups
    cv2 = FakeCloudVolume()
    lookup_supervoxels(
        str(input_parquet),
        cv2,
        str(out_dir),
        resolution=(1, 1, 1),
        block_size=(256, 256, 256),
    )
    assert cv2.n_calls == 0
