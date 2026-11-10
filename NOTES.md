# NMSLIB HNSW behaviors

Read from nmslib `master` at commit `799c8c0` (2026-01-12):
`similarity_search/src/method/hnsw.cc`, `similarity_search/include/method/hnsw.h`,
`similarity_search/src/method/hnsw_distfunc_opt.cc`. The baseline uses the PyPI
wheel nmslib 2.1.2, whose HNSW code is the same in the parts below.

## Parameters and defaults (`CreateIndex`)

| param | default | meaning |
|---|---|---|
| `M` | 16 | neighbors selected per insert, at every level |
| `maxM` | `M` | cap on an upper-level list |
| `maxM0` | `2 * M` | cap on a level-0 list |
| `efConstruction` | 200 | beam width during insert |
| `mult` (mL) | `1 / ln(M)` | level multiplier |
| `delaunay_type` | 2 | neighbor selection heuristic (see below) |
| `post` | 0 (baseline uses 2) | post-processing, see below |
| `ef` / `efSearch` | 20 | query beam width |
| `algoType` | `hybrid` | query uses `V1Merge` when `ef < 1000`, `old` otherwise |

Compile-time: `USE_BITSET_FOR_INDEXING = 1` (generation-stamped visited array),
`EXTEND_USE_EXTENDED_NEIGHB_AT_CONSTR = 0` (insert selects from the `ef` best
only, not from every node evaluated).

## Level generation

`level = (int)(-ln(u) * mult)`, `u` from `uniform_real_distribution<float>(0, 1)`
(a thread-local RNG). The first data point (id 0) is created before the
parallel loop and becomes the initial entry point.

## Insert (`add`)

1. Draw `curlevel`. Read `maxlevel` and the entry point.
2. If `curlevel < maxlevel`: greedy descent from the entry point through
   levels `maxlevel .. curlevel+1`. At each level, repeatedly scan the current
   node's neighbors and move to any strictly closer one (`d < curdist`) until
   no move happens.
3. For `level = min(curlevel, maxlevel) .. 0`:
   - `search_layer` with `efConstruction` from a single entry point.
   - Select **`M` neighbors at every level, including level 0** (not `maxM0`)
     with heuristic 2.
   - Link each selected neighbor bidirectionally (`link` = `addFriendlevel` on
     both sides).
   - Entry point for the next level down = the **closest** selected neighbor
     only (not the whole candidate set).
4. If `curlevel > entry point's level`: the new node becomes the entry point.

## `search_layer` at construction (`kSearchElementsWithAttemptsLevel`)

Standard: candidate min-heap, result max-heap capped at `ef`. Stop when the
nearest candidate is strictly farther than the worst result (`>`). A neighbor
enters both heaps if `results < ef` or it is strictly closer than the worst
result.

## Neighbor selection heuristics

Candidates are processed nearest first. A candidate is **kept** if it is not
strictly closer to any already kept neighbor than to the base point
(`dist(kept, cand) < dist(cand, base)` rejects it). Stop once `NN` are kept.

- If the input has **fewer than `NN`** candidates, both heuristics return it
  unchanged (no pruning at all).
- **Heuristic 2** (`delaunay_type = 2`, used during insert and overflow
  pruning): rejected candidates are discarded, so the result can have far
  fewer than `NN`.
- **Heuristic 1** (used by `post`): after the pass, refill from rejected
  candidates, nearest first, up to `NN` (the paper's `keepPrunedConnections`).

## Overflow pruning (`addFriendlevel`)

Append the new neighbor (duplicates are refused). If the list now exceeds its
cap (`maxM` above level 0, `maxM0` at level 0), rerun **heuristic 2** on the
whole list (distances to the owner) with `NN = cap`. Because heuristic 2 does
not refill, an overflowing list can shrink well below the cap.

## `post = 2` (what the baseline uses)

1. Build the whole index as above (graph A, insertion order 0..n-1).
2. Build it again from scratch (graph B) in reverse order: id 0 first, then
   ids n-1 down to 1. New random levels, new entry point.
3. For every id except 0: level-0 list = union of its A and B lists, pruned by
   **heuristic 1** with `NN = maxM0` (keeps all if the union has fewer than
   `maxM0` entries).
4. Upper levels, levels and the entry point all come from graph B. Node 0
   keeps its graph-B level-0 list.

So a `post = 2` build costs about two plain builds plus the merge.

## Query (`V1Merge`, `ef < 1000`)

1. Greedy descent from the entry point through levels `maxlevel .. 1`, moving
   to strictly closer neighbors, as in insert.
2. Level 0: a sorted array of capacity `max(ef, k)` seeded with the descent
   result. Repeatedly expand the nearest unexpanded entry among the first
   `ef`; a neighbor is added if not visited and `d < worst` or the array
   holds fewer than `ef` items. Stops when the first `ef` are all expanded.
   This is the same result as the two-heap `search_layer` with
   `ef = max(ef_search, k)`.
3. Return the first `k`.

## Distances

The optimized index (built after `post`) uses squared L2 at query time.
Construction uses the space's L2. Both orderings are the same, so rhnsw uses
squared L2 throughout. Cosine normalizes stored vectors once and queries per
call.

## Consequences for rhnsw

- Select `M` (not `2M`) at level 0 during insert; cap level 0 at `2M`.
- Overflow pruning reruns heuristic 2 without refill.
- Implement `post = 2`: second reverse-order build plus heuristic-1 merge of
  level 0. Without it, recall at a given `ef` will sit below the baseline.
- Strict `<` / `>` comparisons as above, entry point = closest selected.
