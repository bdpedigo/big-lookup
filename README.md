# Distributed Point → Supervoxel Lookup — Design Spec

This is a working document, feedback is welcome.

- [Distributed Point → Supervoxel Lookup — Design Spec](#distributed-point--supervoxel-lookup--design-spec)
  - [Motivation](#motivation)
  - [Basic shape](#basic-shape)
  - [Key findings from existing code](#key-findings-from-existing-code)
    - [`cloudvolume.scattered_points` is already smart](#cloudvolumescattered_points-is-already-smart)
    - [What CloudVolume already owns (do not reinvent)](#what-cloudvolume-already-owns-do-not-reinvent)
    - [MaterializationEngine already has partial chunking logic](#materializationengine-already-has-partial-chunking-logic)
  - [Design](#design)
    - [Coordinate handling](#coordinate-handling)
    - [Necessity: a partition function](#necessity-a-partition-function)
    - [Ideal (feature addition): seg-aligned partitioning](#ideal-feature-addition-seg-aligned-partitioning)
    - [Intra-block memory: LRU reliance and point ordering](#intra-block-memory-lru-reliance-and-point-ordering)
    - [v0 execution: assume spatially-clustered (or small) input](#v0-execution-assume-spatially-clustered-or-small-input)
    - [Dense vs. sparse decode flag](#dense-vs-sparse-decode-flag)
    - [Parallelism: two nested levels](#parallelism-two-nested-levels)
    - [Output \& resumability](#output--resumability)
  - [CloudVolume config baseline (from ME's gateway, as reference)](#cloudvolume-config-baseline-from-mes-gateway-as-reference)
  - [Open decision points](#open-decision-points)
    - [Decide on choice of parallelization backend (or support multiple)](#decide-on-choice-of-parallelization-backend-or-support-multiple)
  - [Possible feature additions](#possible-feature-additions)

## Motivation

I am finding myself needing to look up supervoxels for large numbers of points to link
to the chunkedgraph. I often use Cloudvolume's `scattered_points`, but that does not
handle distribution over many workers and as far as I know relies on the LRU cache for
smart batching of nearby-in-space lookups. The code in materialization for ingesting
annotations is conceptually similar but is locked in to that system. And because I am
working in the parquet/delta lake world a lot, I would prefer a system that at least is
capable of IO from those formats. I also don't want to reinvent the wheel and want to
reuse as much existing tooling as possible.

## Basic shape

The tool is a four-stage pipeline over a payload of coordinate points:

```
  ┌──────────┐   ┌───────────┐   ┌─────────────────┐   ┌──────────┐
  │  SCAN    │──▶│ PARTITION │──▶│  LOOKUP (scaled  │──▶│  WRITE   │
  │ points   │   │  points   │   │  horizontally)   │   │ results  │
  └──────────┘   └───────────┘   └─────────────────┘   └──────────┘
   parquet /      decide how to    per-partition        parquet /
   delta lake     group points     supervoxel lookup    delta lake
   of x,y,z       into blocks       via CloudVolume      (distributed
                                                         write)
```

1. **Scan** — read coordinate points from a parquet / delta lake source (`x, y, z` plus an
   id/index), interpreted in a user-specified **coordinate system / resolution** (see
   [Coordinate handling](#coordinate-handling)).
2. **Partition** — decide how to group those points into **blocks** (our unit of work; the
   interesting design decision; see
   [seg-aligned partitioning](#ideal-feature-addition-seg-aligned-partitioning)).
3. **Lookup, scaled horizontally** — run the per-partition supervoxel lookup across a scaling
   mechanism. The requirement is that it can **scale out on Kubernetes** or **run on a
   laptop**. The concrete backend is an open question — threads, `joblib`, Ray, and Celery
   are examples of the space we'll explore. We may support one or more backends as this evolves.
4. **Write** — emit the per-point lookup results as parquet / delta lake, itself written in a
   distributed / per-partition fashion.

Everything else in this document elaborates or extends these four stages. The core scope is:

- **Input:** parquet / delta lake of points (`x, y, z` + id/index), plus a user-specified
  **coordinate system / resolution** describing how to interpret those `x, y, z`.
- **Output:** parquet / delta lake mapping each point to its supervoxel id. When the
  integer-voxel coordinates used internally differ from the input coordinates, the output also
  carries those integer-voxel coordinates to remove any floating-point / conversion ambiguity.
- **Scope:** supervoxels only. Root-id / agglomeration (chunkedgraph) is explicitly deferred —
  getting supervoxels is the essential operation.

## Key findings from existing code

Citations are pinned permalinks:
[cloud-volume @ `d1d704c`](https://github.com/seung-lab/cloud-volume/tree/d1d704c9b78484fa49cea8204744f0d18b273624)
(`master`) and
[MaterializationEngine @ `146c62a`](https://github.com/CAVEconnectome/MaterializationEngine/tree/146c62a0e3b29b8788c6f9c899ae1548de18a5f1)
(`master`).

### `cloudvolume.scattered_points` is already smart

Entry point:
[`CloudVolumePrecomputed.scattered_points`](https://github.com/seung-lab/cloud-volume/blob/d1d704c9b78484fa49cea8204744f0d18b273624/cloudvolume/frontends/precomputed.py#L873)
(graphene variant with agglomeration:
[`CloudVolumeGraphene.scattered_points`](https://github.com/seung-lab/cloud-volume/blob/d1d704c9b78484fa49cea8204744f0d18b273624/cloudvolume/frontends/graphene.py#L93)).
It calls
[`PrecomputedImageSource.download_points`](https://github.com/seung-lab/cloud-volume/blob/d1d704c9b78484fa49cea8204744f0d18b273624/cloudvolume/datasource/precomputed/image/__init__.py#L120).
For each point, CloudVolume:
- turns the point into a 1×1×1 bbox and issues a download that expands to the chunk boundary
  ([`download_points`](https://github.com/seung-lab/cloud-volume/blob/d1d704c9b78484fa49cea8204744f0d18b273624/cloudvolume/datasource/precomputed/image/__init__.py#L120)
  →
  [`download_single_voxel_unsharded`](https://github.com/seung-lab/cloud-volume/blob/d1d704c9b78484fa49cea8204744f0d18b273624/cloudvolume/datasource/precomputed/image/rx.py#L385));
- **partial-decodes a single voxel** out of the compressed chunk
  ([`decode_single_voxel`](https://github.com/seung-lab/cloud-volume/blob/d1d704c9b78484fa49cea8204744f0d18b273624/cloudvolume/datasource/precomputed/image/rx.py#L827)
  / `chunks.read_voxel`) rather than decoding the whole chunk;
- **dedups downloads by chunk filename** via an in-memory LRU and optional disk cache
  ([`download_chunk`](https://github.com/seung-lab/cloud-volume/blob/d1d704c9b78484fa49cea8204744f0d18b273624/cloudvolume/datasource/precomputed/image/rx.py#L575)
  looks up `lru[filename]` before issuing the GET).

So on the **network axis**, `scattered_points` fetches only *occupied* chunks — the same set
an explicit grouping approach would fetch. Its weaknesses:

1. **Cache-eviction × point ordering.** Points are processed as an unordered `set()`. If the
   working set of touched chunks exceeds the (bounded) LRU, a chunk can be evicted and
   **re-fetched** because points in the same chunk are not processed together.
2. **Decode CPU scales with points, not chunks** (one partial decode per point). Fine when
   sparse; may be wasteful when many points share a chunk.
3. **No work partitioning.** The LRU is per-process; there is no notion of assigning a chunk
   to exactly one worker. Nothing keeps two workers from each downloading the same chunk. (The
   LRU-hit ordering optimization in
   [`download_chunks_threaded`](https://github.com/seung-lab/cloud-volume/blob/d1d704c9b78484fa49cea8204744f0d18b273624/cloudvolume/datasource/precomputed/image/rx.py#L657)
   only helps *within* a process.)

### What CloudVolume already owns (do not reinvent)

The entire download/decode/cache path is CV's:

| Concern | CV primitive |
| --- | --- |
| point → chunk coords | `point_to_mip`, chunk grid math |
| chunk filename | `chunknames()` |
| expand bbox to chunk grid | [`expand_to_chunk_size()`](https://github.com/seung-lab/cloud-volume/blob/d1d704c9b78484fa49cea8204744f0d18b273624/cloudvolume/datasource/precomputed/image/rx.py#L249) |
| GET chunk (threaded / green concurrency) | `CloudFiles` + scheduler |
| dedup by filename | LRU + disk cache ([`download_chunk`](https://github.com/seung-lab/cloud-volume/blob/d1d704c9b78484fa49cea8204744f0d18b273624/cloudvolume/datasource/precomputed/image/rx.py#L575)) |
| single-voxel partial decode | [`decode_single_voxel`](https://github.com/seung-lab/cloud-volume/blob/d1d704c9b78484fa49cea8204744f0d18b273624/cloudvolume/datasource/precomputed/image/rx.py#L827) / `read_voxel` |
| full cutout decode | `download(bbox)` → ndarray |

Underlying chunk shape (for seg-aligned partitioning) comes from
[`PrecomputedMetadata.chunk_size`](https://github.com/seung-lab/cloud-volume/blob/d1d704c9b78484fa49cea8204744f0d18b273624/cloudvolume/datasource/precomputed/metadata.py#L651)
(exposed as
[`CloudVolumePrecomputed.chunk_size`](https://github.com/seung-lab/cloud-volume/blob/d1d704c9b78484fa49cea8204744f0d18b273624/cloudvolume/frontends/precomputed.py#L430)).

Both branches of the dense/sparse flag are CV calls:
- **sparse branch:** `cv.scattered_points(pts)` / `download_points`
- **dense branch:** `cv.download(bbox)` once, then vectorized numpy indexing of the points

### MaterializationEngine already has partial chunking logic

[`ChunkingStrategy`](https://github.com/CAVEconnectome/MaterializationEngine/blob/146c62a0e3b29b8788c6f9c899ae1548de18a5f1/materializationengine/workflows/chunking.py#L14)
in `workflows/chunking.py`, driving
[`process_chunk`](https://github.com/CAVEconnectome/MaterializationEngine/blob/146c62a0e3b29b8788c6f9c899ae1548de18a5f1/materializationengine/workflows/spatial_lookup.py#L672)
→
[`process_and_insert_sub_batch`](https://github.com/CAVEconnectome/MaterializationEngine/blob/146c62a0e3b29b8788c6f9c899ae1548de18a5f1/materializationengine/workflows/spatial_lookup.py#L1435)
in `workflows/spatial_lookup.py`:

- [`select_strategy`](https://github.com/CAVEconnectome/MaterializationEngine/blob/146c62a0e3b29b8788c6f9c899ae1548de18a5f1/materializationengine/workflows/chunking.py#L204)
  picks between a **`grid`** strategy
  ([`_create_grid_chunking`](https://github.com/CAVEconnectome/MaterializationEngine/blob/146c62a0e3b29b8788c6f9c899ae1548de18a5f1/materializationengine/workflows/chunking.py#L268),
  enumerate every cell of the bbox, incl. empty) and a **`data_chunks`** strategy
  ([`_create_data_specific_chunks`](https://github.com/CAVEconnectome/MaterializationEngine/blob/146c62a0e3b29b8788c6f9c899ae1548de18a5f1/materializationengine/workflows/chunking.py#L322))
  that queries Postgres for *occupied* grid cells only:
  ```sql
  SELECT FLOOR(ST_X(pt)/chunk)*chunk AS x_min, ... COUNT(*)
  FROM table GROUP BY x_min, y_min, z_min
  ```
  That `FLOOR(coord/chunk)*chunk GROUP BY` **is** the point→chunk-key partition function.
- The actual supervoxel lookup is a single
  [`cv.scattered_points(...)`](https://github.com/CAVEconnectome/MaterializationEngine/blob/146c62a0e3b29b8788c6f9c899ae1548de18a5f1/materializationengine/workflows/spatial_lookup.py#L1483)
  call per sub-batch.
- Confirms the approach (occupancy-based spatial grouping) is sound and battle-tested.

**Gaps vs. this design:**
1. Its grid is an arbitrary `chunk_size = 1024` cube
   ([`base_chunk_size=1024` default](https://github.com/CAVEconnectome/MaterializationEngine/blob/146c62a0e3b29b8788c6f9c899ae1548de18a5f1/materializationengine/workflows/chunking.py#L14))
   with **no relationship to the segmentation chunk shape / voxel offset**. A cell straddles
   many seg chunks; seg boundaries are not respected. Aligning to the seg grid — the whole
   point here — is exactly what ME does *not* do.
   - **Sub-note: the misalignment penalty is a boundary effect, bounded by one seg chunk.**
     A seg chunk is only double-fetched if it straddles a block boundary — i.e. the waste is a
     shell one-seg-chunk thick around each block, not a volume effect. The redundant fraction
     scales roughly as `(seg_chunk_size / block_size)` **per axis** (surface-to-volume). So
     when the seg chunk is small relative to the 1024 block, the overhead is minimal, and ME's
     unaligned grid is basically fine. The caveat is **anisotropy**: seg chunks are often
     small in x/y but the ratio can be non-trivial in z (or wherever chunk size approaches the
     block size), so the penalty is only negligible when the chunk ≪ block in *every*
     dimension. Aligning to the seg grid drives this residual to exactly zero, but it is a
     small win, not a large one, whenever the ratio is already small.
2. It is PostGIS-native (points already in a spatial DB); our input is parquet.
3. The chunk unit is only a DB-query + Celery-task boundary. The actual download is still
   per-point via `scattered_points`; ME never does "download cutout once, index locally."

**Reusable = one concept, not the code:** partition by floored chunk key, emit work only for
occupied cells. Two possible upgrades: (a) align the grid to the **segmentation** grid, and
(b) do the group-by in a dataframe instead of PostGIS. Neither is required for a first version
(see [v0 execution](#v0-execution-assume-spatially-clustered-or-small-input)); (a) in
particular is a feature addition, not a necessity.

## Design

### Coordinate handling

Alongside the data location, the user specifies the **coordinate system** (resolution, in
nanometers) that the input `x, y, z` should be interpreted in. The code converts these once, up
front, into **integer voxels at the resolution CloudVolume expects** (the segmentation's mip
level) and carries those integer-voxel coordinates through the rest of the pipeline — they are
the join key for reattaching results (see the
[`scattered_points` return contract](#necessity-a-partition-function) below) and the
space every partition/chunk operation lives in. When the converted voxel coordinates differ
from the input, they are also emitted in the output so there is no ambiguity from
floating-point input coordinates or the conversion.

### Necessity: a partition function

The one thing the tool must own is a **partition function**: a rule that maps each point to a
**block** key, so points can be grouped and each group handed to a worker. Any reasonable
spatial grouping works — even ME's arbitrary fixed-size grid — as long as it (a) groups nearby
points together so a worker's downloads overlap, and (b) emits work only for occupied blocks.

> **Vocabulary.** Two grids are in play and must not be conflated: a **segmentation chunk** is
> how the segmentation is stored on disk (CV's grid, `chunk_size` / `voxel_offset`); a **block**
> is *our* partition-grid cell and our unit of work — one block = one task = one output file.
> A block spans one or more segmentation chunks.

CV owns everything downstream of that grouping (download, decode, cache, chunk-name math), so
the partition function plus a thin per-partition runner is essentially the whole tool.

```
                 ┌───────────────────── WE OWN ─────────────────────┐
 parquet points →│ partition points into blocks → group by key        │
                 └───────────────────────┬───────────────────────────┘
                                         │ each block = 1 task
                                         ▼
                 ┌───────────────────── CV OWNS ────────────────────┐
                 │ flag=sparse: cv.scattered_points(pts_in_block)    │
                 │ flag=dense : cv.download(block_bbox) then index   │
                 │ (+ CV's own threaded/green GET concurrency + LRU) │
                 └───────────────────────┬───────────────────────────┘
                                         ▼
                          parquet out (one file / block)
```

> **`scattered_points` return contract.** It returns an **unordered dict keyed by
> integer-voxel `xyz`**, not an array aligned to input order — so the result must be rejoined
> on the **carried integer-voxel coordinate** (this is the concrete reason those coords are
> computed once up front and carried through; see
> [Coordinate handling](#coordinate-handling)). Points that round to the same voxel
> collapse to one key and fan back out to every input row sharing it, so original ids travel
> alongside separately.

A simple, sufficient default is the ME-style fixed grid:

```
key = floor(point_voxel / block_size)
```

grouping points into fixed rectangular **blocks**. Here `block_size` is a **3-element array**
`(bx, by, bz)` giving the block dimensions per axis, not a single scalar — the floor/divide is
elementwise. This matters because segmentation resolution is typically **anisotropic**, so a
single scalar would produce blocks that are cubic in voxels but very non-cubic in world space
(and vice versa). An explicit per-axis array lets the grid be shaped deliberately (e.g. cubic
in world nm, or aligned to the seg chunk shape).

`block_size` is expressed in **voxel coordinates** — the same integer-voxel space every
downstream operation (`point_voxel`, `chunk_size`, `voxel_offset`, chunk-name math) lives in.
Keeping it in voxels avoids a per-cell world→voxel reconversion inside the hot partition loop
and makes seg-grid alignment (which is defined in voxels) a direct comparison. A user-facing
convenience of specifying `block_size` in the input resolution and converting once up front is
reasonable, but the value carried through the guts of the code should be integer voxels.

### Ideal (feature addition): seg-aligned partitioning

The *ideal* partition function aligns the grid to the **segmentation** storage grid rather
than an arbitrary constant:

```
key = floor((point_voxel - voxel_offset) / (chunk_size * multiple))
```

where `chunk_size` and `voxel_offset` come from `cv.info` for the chosen mip, and `multiple`
groups K×K×K segmentation chunks into one block.

Why this is nicer (but not required): grouping already makes single-fetch likely; seg-alignment
makes it single-fetch **by construction**. It turns "single fetch *if* the cache happens to
hold it" into "single fetch guaranteed," and if a block's bytes fit the worker LRU, CV's
eviction-reorder leak disappears entirely — CV's optimal behavior without writing a decoder,
cache, or chunk-name function. As noted above, the gain over an unaligned grid is a bounded
boundary effect, so this is a refinement to reach for once the core works, not a prerequisite.

> **Sharding caveat (why this is deferred).** The target volumes are **sharded**: the physical
> fetch unit is a *shard* (many chunks behind one hash-mapped file), and the chunk→shard map is
> a hash, so spatially-adjacent chunks are **not** guaranteed to share a shard. Aligning a block
> to the *chunk* grid therefore does not align it to *shard* boundaries, so the clean
> by-construction single-fetch guarantee really wants **shard-alignment**, which means depending
> on CV's internal sharding spec. That extra fragility is a large part of why seg-aligned
> partitioning stays a future feature rather than v0.

### Intra-block memory: LRU reliance and point ordering

v0 leans on `scattered_points` intra-block, and that lean rests on one explicit assumption:

> **v0's intra-block single-fetch holds iff the worker LRU can simultaneously hold the block's
> occupied chunk set.** `scattered_points` processes a block's points as an *unordered* set and
> dedups downloads by chunk filename against the LRU; if the occupied-chunk working set exceeds
> the LRU, a chunk can be evicted before the last point needing it is processed and then
> re-fetched. So block size and LRU budget are a **jointly-tuned pair**, not independent knobs.

Two facts soften the requirement:

- It is the **occupied** chunk set (chunks that actually contain a queried point), not the
  block volume — a large block over sparse points can still fit a small LRU; the worst case is
  dense fill.
- The LRU stores chunks **compressed** (ME uses `lru_encoding='crackle'`), so the budget is in
  *compressed* bytes and far more chunks fit per MB than a decoded-array cache would suggest.

**Point ordering shrinks the requirement from "whole block" to "working window."** If the
points fed to `scattered_points` are ordered chunk-coherently (all points in one chunk, then
the next), a chunk is never revisited once finished, so it can evict harmlessly and the LRU
need only hold the *live* window rather than the entire block — letting a block safely exceed
memory. A space-filling-curve sort (Z-order / Hilbert) is the cheap approximation: it doesn't
guarantee strict chunk-contiguity but keeps the active set to a spatially-local band. Perfect
chunk-sort is the exact version; SFC is the ~90% version.

Crucially, this needs **no knowledge of CV's cache internals**: the lever is purely the *order
of the point array we pass in*, and the sort key is the **same** `floor(point_voxel /
chunk_size)` we already compute to split blocks — used here as a *sort* instead of a *split*.
So it is one partition function at two granularities:

- **coarse split → blocks** (assign work to workers; bound network per worker),
- **fine sort → intra-block point order** (bound the *live* chunk set; shrink the LRU
  requirement).

This is not CV-specific — chunk-coherent input helps any chunked/LRU backend, so it is not
"bending to `scattered_points`' internals." (On sharded volumes it still helps the decode/LRU
layer as described; it does not by itself control *shard* fetch order — the same asterisk as
[seg-alignment](#ideal-feature-addition-seg-aligned-partitioning) above.)

### v0 execution: assume spatially-clustered (or small) input


The partition *scheme* (which blocks exist, their bounds) is decided from **file metadata
alone** — `pl.scan_delta()` / `pl.scan_parquet()` expose per-column min/max footer stats, so
global `xyz` bounds and a first cut at occupancy come essentially for free, without scanning a
row. That much is cheap regardless of how the data is laid out.

The *execution* of the partition — actually getting each block's points to a worker — is where
data layout matters, and **v0 deliberately assumes the easy case**: the input is either

- **spatially clustered on disk** (written with spatial locality — e.g. sorted along a
  space-filling curve, or delta Z-ordered / liquid-clustered), so that a block's bounding box
  maps to just a few files / row-groups; or
- **small enough to bulldoze** — it all fits on one node's memory (or SSD), so locality is
  irrelevant and we just load it and group.

Under that assumption the execution is trivial and needs no shuffle:

```
for each occupied block:
    lf = pl.scan_delta(source).select(id, x, y, z)
           .filter((x >= bx0) & (x < bx1) & (y >= by0) & (y < by1) & (z >= bz0) & (z < bz1))
    pts = lf.collect()          # touches only the few files covering this block
    → hand pts to CV (sparse/dense) → write one output file for this block
```

Each block is an **independent lazy filtered scan**. If the source is clustered, predicate
pushdown + footer stats prune the scan to a handful of files, so each worker downloads and
parses almost exactly its own block and nothing else — embarrassingly parallel, no cross-worker
coordination, pure polars, and resumable (block key → output file, skip if present). If the
source instead just fits on a node, the same code runs against an in-memory / SSD-backed frame
and the filter is a cheap local mask.

### Dense vs. sparse decode flag

- Both branches fetch the **same chunks** → same network cost. Since the workload is expected
  to be **network-bound**, the flag is a second-order (CPU) knob, not the throughput driver.
- `sparse`: one partial decode per point (good when ~1 point/chunk).
- `dense`: one full decode per chunk, then vectorized index (good when many points/chunk).
- Expose as a flag; do not assume a regime.

### Parallelism: two nested levels

The horizontal-scaling mechanism (stage 3) is pluggable — threads, `joblib`, Ray, Celery.
Whatever the backend, there are **two nested levels** of concurrency that multiply:

```
total in-flight GETs ≈ N_workers (backend tasks) × cv_concurrency (green/threads per worker)
```

- **One block = one backend task** → disjoint chunk sets → no cross-worker refetch.
  This holds regardless of whether "worker" is a thread, a `joblib` job, a Ray task, or a
  Celery task.
- Inside a task, let **CV** do IO concurrency (green threads compose well). **Do not** nest
  CV's `parallel=` multiprocessing under a process-based backend (oversubscription / fork
  issues).
- Scale workers to add **NICs / nodes**, not threads. On a single node CV's own threading may
  already saturate the NIC; a distributed backend's real payoff is multi-node bandwidth +
  orchestration + resumability.

  (NIC = Network Interface Card — the machine's network hardware; its bandwidth caps how fast
  a single node can pull segmentation chunks from cloud storage, so adding more threads on one
  node stops helping once the NIC is saturated.)

### Output & resumability

- One output partition (parquet / delta lake file) per block. This gives resumability
  nearly for free (skip partitions whose output already exists) and keeps the write itself
  distributed. **Caveat:** this only holds if the per-block write is idempotent and block keys
  are stable across reruns; how that is enforced depends on the parallelization backend (see
  Open decision points).

## CloudVolume config baseline (from ME's gateway, as reference)

From
[`CloudVolumeGateway.get_cv`](https://github.com/CAVEconnectome/MaterializationEngine/blob/146c62a0e3b29b8788c6f9c899ae1548de18a5f1/materializationengine/cloudvolume_gateway.py#L57):
`mip=0, bounded=False, fill_missing=True`, `lru_bytes` per-instance,
`lru_encoding='crackle'`, and
[`parallel=_CV_PARALLEL`](https://github.com/CAVEconnectome/MaterializationEngine/blob/146c62a0e3b29b8788c6f9c899ae1548de18a5f1/materializationengine/cloudvolume_gateway.py#L65)
(default 10, `CLOUDVOLUME_PARALLEL` env). A process-wide LRU budget of ~100 MB is set via
[`CELERY_CLOUDVOLUME_CACHE_BYTES`](https://github.com/CAVEconnectome/MaterializationEngine/blob/146c62a0e3b29b8788c6f9c899ae1548de18a5f1/materializationengine/cloudvolume_gateway.py#L77).
Reuse CV instances keyed by segmentation source.

## Open decision points

Unresolved choices that shape the implementation. These are deliberately punted for now, but
the design should avoid foreclosing any of them.

### Decide on choice of parallelization backend (or support multiple)

Stage 3 (the horizontal-scaling mechanism) is described throughout as "pluggable — threads,
`joblib`, Ray, Celery." That flexibility hides the single hardest piece of real code in the
tool: a task-execution interface that stays honest across backends whose operational
assumptions differ enormously. This needs an explicit decision — pick one, or commit to a
narrow interface that genuinely spans several.

Points to weigh:

- **The backends span a huge operational range.** A thread pool needs nothing (in-process, one
  machine). `joblib` adds process pools / simple clusters. Ray needs a running cluster (head +
  workers). Celery needs a broker (Redis/RabbitMQ) plus deployed, pre-provisioned workers. A
  **managed cloud task queue + Kubernetes workers** (e.g. AWS SQS or Google Cloud Tasks /
  Pub/Sub feeding a pool of k8s pods) trades the self-hosted broker for a managed queue and
  leans on the cluster autoscaler to add/remove worker pods with the backlog. "Runs on a
  laptop" and "scales out on k8s" are near-opposite ends of this range, and an abstraction that
  pretends they are the same tends to leak.
- **What the interface must actually cover.** Beyond "submit a function over a list of blocks":
  result collection, failure/retry semantics, and resumability. These are exactly where the
  backends diverge, so the interface is only as honest as its weakest guarantee.
- **Resumability is not automatically free.** The "skip blocks whose output already exists"
  claim (see [Output & resumability](#output--resumability)) requires (a) the per-block write
  to be **idempotent** and
  (b) block keys to be **stable across reruns**. A partial/failed task must not leave a
  half-written output that looks complete. The simplest robust pattern — one immutable file per
  block, written atomically (temp name + rename, or write-then-commit) — sidesteps most of
  this, but only if the chosen backend gives at-least-once execution with a clean way to detect
  completion.
- **Concurrent writes to a shared table differ per backend.** Independent per-block files in a
  prefix are safe under any backend. But if the sink is a single **delta table**, concurrent
  writers must handle commit conflicts / optimistic-concurrency retries, and the mechanism for
  that is not backend-agnostic. Leaning on one-file-per-block avoids the problem; a delta sink
  reintroduces it.
- **CV concurrency composes differently under each backend.** The "let CV do IO concurrency
  with green threads, don't nest CV's `parallel=` multiprocessing under a process-based
  backend" guidance (see [Parallelism](#parallelism-two-nested-levels)) interacts with the
  backend choice: a thread backend and a
  process/Celery backend impose different constraints on how many in-flight GETs and how much
  LRU each worker should get.
- **CV-instance reuse assumes worker lifetime.** Reusing CV instances keyed by segmentation
  source (and the warm per-worker LRU that makes single-fetch pay off) only helps if workers
  are **long-lived** and process many blocks. Short-lived / one-shot task workers (some Celery
  or serverless setups) start cold each time and lose that benefit — a factor in the backend
  choice, not just a tuning knob.
- **Possible resolution: a thin submit/collect seam.** One option is a minimal interface
  (`map(fn, blocks) -> results`, plus "has this block already completed?") with a local
  thread/process implementation as the reference, and Ray/Celery as adapters. Whether that seam
  can stay thin without lying about retry/resumability semantics is the crux — and the reason
  this is an open decision rather than a settled one.

## Possible feature additions

These extend the four-stage core and are to be refactored into discrete feature additions:

- **Input validation** — optional up-front sanity checks, run cheaply from footer metadata
  and/or a small sample before a large distributed run:
  - *Resolution check.* Convert all (or a random sample of) input points to voxels under the
    given `resolution` and check them against the dataset bounds from `cv.info`; if all/most
    fall outside the volume, the resolution is probably wrong (e.g. points given in nm
    interpreted as voxels, or the wrong mip). Catches the most common silent-wrong-answer
    failure.
  - *Clustering check (validates the v0 assumption).* v0 assumes the source is spatially
    clustered or small. Test it from footer stats alone: if the per-file `xyz` min/max ranges
    all span most of the volume (files overlap heavily rather than tiling space), the data is
    **not** clustered, and v0's per-block filtered scans will each degrade toward a full scan
    (`N_blocks × full_scan` blowup). Fail fast / warn, and point the user at the
    repartition/shuffle feature (or the single-fat-node path).
- **Seg-aligned partitioning** — replace the simple fixed grid with a partition function keyed
  to the segmentation grid (`chunk_size` / `voxel_offset` from `cv.info`) for guaranteed
  single-fetch. See
  [seg-aligned partitioning](#ideal-feature-addition-seg-aligned-partitioning).
- **Root ids / agglomeration** — add a chunkedgraph `get_roots` stage after supervoxel lookup
  (different bottleneck: REST API, batched, timestamped).
- **Repartition / shuffle for unclustered-at-scale input** — lift v0's "clustered or small"
  assumption for sources that are **both** large **and** not spatially co-located on disk. Here
  the per-block filtered scan of v0 breaks down (each block scan can't prune, so total read
  amplifies to ~`N_blocks × full_scan`), and the data doesn't fit one node.

  The key realization: the **occupancy group-by *is* the shuffle**. You must read every `xyz`
  at least once to know which blocks are occupied; v0 on unclustered data then effectively
  re-reads the source once per block. Instead, pay a single distributed pass that both computes
  occupancy and *emits the block assignment*, rather than computing block keys and then
  re-deriving them by filtering. This is the natural home for a distributed dataframe engine
  (**Daft** on Ray, or Dask/Spark) rather than single-node polars: native object-store IO,
  a real shuffle, and SSD spill when the working set exceeds memory. Concretely: `scan → add
  block key = floor(point_voxel / block_size) → repartition/sort by block key →` hand each
  resulting partition to the same CV block runner v0 already uses. The **partition function is
  identical to v0**; only the execution (a shuffle instead of independent filtered scans)
  changes — no lock-in.

  Once you've paid the shuffle, the clustered result has **reuse value**: it is effectively
  "promote this source to clustered," after which every future lookup is a v0-style filtered
  scan. That reframes the intermediate as a durable asset, not scratch.

  - *Open point — where the shuffled/clustered intermediate lives.* Two options, not yet
    decided:
    - **Durable cloud write** (same delta/parquet store): survives spot preemption, visible to
      all workers, and reusable across reruns (converts a one-time shuffle into a permanent
      regime-1 asset). Costs the cloud write + storage.
    - **SSD spill** (local / attached): cheaper and faster within a single run (no egress), but
      ephemeral and **node-local** — preemption vaporizes it and peers can't see it, so it's
      hostile to resumability on spot nodes. Really only fits the single-fat-node case as a
      RAM overflow valve, not as a cross-node handoff.
    - *Note in favor of a cloud temp write:* the pipeline has to write results to a cloud
      bucket **anyway**. The temporary clustered intermediate could be written to the eventual
      output location, and the final per-block supervoxel results could **overwrite exactly
      those temp files** — so the temp data is deleted precisely when it becomes obsolete,
      with no separate cleanup step. Worth exploring.
- **Lance format IO** — add [Lance](https://lancedb.github.io/lance/) as a third read/write
  format alongside parquet / delta lake, plugging into the same scan (`_scan_points`) and write
  (`_write_block`) seams. Lance is attractive here for two reasons that line up with the design:
  its **fast random access + zone-map / pushdown filtering** suit v0's per-block filtered scans
  (a block's bounding-box predicate prunes to a few fragments rather than a full scan, the same
  property v0 relies on for clustered parquet / delta — see
  [v0 execution](#v0-execution-assume-spatially-clustered-or-small-input)), and its
  **fragment-level writes + dataset versioning** fit the one-file(fragment)-per-block,
  idempotent, resumable output model (see [Output & resumability](#output--resumability)).
  polars can scan a Lance dataset (via `pl.scan_pyarrow_dataset` over `lance.dataset(...)`),
  so the change is localized to format detection in the scan/write helpers, not the partition
  or lookup core. Deferred, not required for v0 — parquet / delta already cover the input and
  output contracts.
- **KV-store sink** — replace/augment the parquet/delta write stage with a cloud KV sink.
- **Long-lived listener** — a persistent cluster that listens for lookup requests instead of a
  one-off script run over a fixed payload.
- **Dense vs. sparse decode flag** — already sketched
  [above](#dense-vs-sparse-decode-flag); formalize as a per-run option.
- **Disk cache** — optional per-worker disk cache as an eviction hedge. Turns eviction from
  "re-GET cloud" into "re-read local disk" — a cheap hedge that makes block sizing less
  critical. May not persist on ephemeral workers; strong on a single fat node.
- **Tunable block sizing** — expose block size (in segmentation-chunk multiples) as a
  parameter with a sensible default, jointly tuned with the worker LRU budget (see
  [Intra-block memory](#intra-block-memory-lru-reliance-and-point-ordering)). The crux
  tradeoff: too small → task overhead + poor GET batching; too large →
  the block's occupied (compressed) chunk set overflows the LRU and the eviction leak returns —
  unless intra-block chunk-coherent / SFC point ordering keeps the live working set small.
- **Intra-block point ordering** — sort each block's points chunk-coherently (or by Z-order /
  Hilbert as a cheap approximation) before handing them to `scattered_points`, shrinking the
  LRU requirement from the whole block to a working window (see
  [Intra-block memory](#intra-block-memory-lru-reliance-and-point-ordering)). Pure
  input-ordering; no dependence on CV cache internals.
- **Pluggable multi-node executor** — start from CV threading + chunk-coherent point ordering
  on a single node (no distributed backend), then add a distributed backend purely as the
  multi-node partition executor. The partition function is identical either way, so no
  lock-in. **The backend choice itself is unresolved — see Open decision points.**
