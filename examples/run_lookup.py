"""Example: look up supervoxels for a parquet of nanometer points.

Run from the package root with:

    uv run --extra examples python examples/run_lookup.py

Points in ``data/test_input_points.parquet`` are in nanometers, so the input
resolution is ``(1, 1, 1)``. Output is written as one parquet per block under
``data/test_output_points/``.
"""

import shutil
from pathlib import Path

from big_lookup import lookup_supervoxels

INPUT_PATH = "data/test_input_points.parquet"
SEGMENTATION_PATH = "s3://bossdb-open-data/iarpa_microns/minnie/minnie65/ws"
OUTPUT_PATH = "data/test_output_points"
INPUT_RESOLUTION = (1, 1, 1)  # input coordinates are in nanometers


def main() -> None:
    # For this example only: start fresh so resumability doesn't hide changes.
    out_dir = Path(OUTPUT_PATH)
    if out_dir.exists():
        shutil.rmtree(out_dir)
        print("Removed existing output folder:", out_dir)

    lookup_supervoxels(
        INPUT_PATH,
        SEGMENTATION_PATH,
        OUTPUT_PATH,
        resolution=INPUT_RESOLUTION,
        block_size=(2**16, 2**16, 2**10),
        cloudvolume_kwargs=dict(parallel=20, lru_bytes=1e9, use_https=True),
    )


if __name__ == "__main__":
    main()
