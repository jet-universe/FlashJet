"""Pipeline I/O: ship ragged CPU events to the GPU for clustering.

Real datasets are ragged (variable particles per event) and live on the CPU
(awkward arrays from uproot/parquet).  flashjet.cluster() wants padded
(B, N, 4) tensors + mask on the GPU.  This module closes that gap without
per-event Python loops or synchronous copies:

  * collate(): vectorized ragged -> padded+mask (one fancy-indexing scatter),
    optionally truncating to the hardest-pt particles, optionally writing
    into preallocated pinned buffers.
  * to_gpu_batches(): a generator that double-buffers pinned staging memory
    and issues async H2D copies on a side stream, so collation of batch k+1
    and the copy overlap with GPU compute on batch k.

The transfer itself is cheap (a 6000-particle event is ~100 kB ~ 8 us over
PCIe vs ~3 ms to cluster); what this module avoids is the Python-loop padding
and blocking copies that would otherwise dominate.
"""

import numpy as np
import torch

_FIELD_ALIASES = (("px", "py", "pz", "E"), ("px", "py", "pz", "e"), ("px", "py", "pz", "energy"))


def _ragged_to_flat(events):
    """Return (flat (total, 4) float32, counts (B,) int64) from an awkward
    array with px/py/pz/E fields or a sequence of (n_i, 4) arrays."""
    try:
        import awkward as ak

        is_ak = isinstance(events, ak.Array)
    except ImportError:
        is_ak = False

    if is_ak:
        import awkward as ak

        fields = next((f for f in _FIELD_ALIASES if set(f) <= set(events.fields)), None)
        if fields is None:
            raise ValueError(f"awkward input needs px/py/pz/E fields, got {events.fields}")
        counts = np.asarray(ak.num(events), dtype=np.int64)
        flat = np.stack(
            [np.asarray(ak.flatten(events[f]), dtype=np.float32) for f in fields], axis=1
        )
        return flat, counts

    counts = np.fromiter((len(e) for e in events), dtype=np.int64, count=len(events))
    flat = (
        np.concatenate([np.asarray(e, dtype=np.float32).reshape(-1, 4) for e in events], axis=0)
        if len(events)
        else np.zeros((0, 4), np.float32)
    )
    return flat, counts


def _truncate_by_pt(flat, counts, n_max):
    """Keep the n_max hardest-pt particles of oversized events (vectorized
    per oversized event only)."""
    over = np.flatnonzero(counts > n_max)
    if len(over) == 0:
        return flat, counts
    offsets = np.concatenate([[0], np.cumsum(counts)])
    keep = np.ones(len(flat), dtype=bool)
    for b in over:
        seg = slice(offsets[b], offsets[b + 1])
        pt2 = flat[seg, 0] ** 2 + flat[seg, 1] ** 2
        drop = np.argsort(-pt2, kind="stable")[n_max:]
        keep[offsets[b] + drop] = False
    counts = counts.copy()
    counts[over] = n_max
    return flat[keep], counts


def collate(events, n_max=None, truncate="pt", out=None):
    """Vectorized ragged -> (p4 (B, N, 4) float32, mask (B, N) bool) tensors.

    Args:
        events: awkward Array (px/py/pz/E fields) or sequence of (n_i, 4)
            arrays (px, py, pz, E columns).
        n_max: pad/truncate width (default: longest event in the batch).
        truncate: 'pt' keeps the hardest particles of oversized events,
            'first' keeps the leading slice, 'error' raises.
        out: optional (p4, mask) preallocated tensors (e.g. pinned) to fill;
            must be at least (B, n_max, 4) / (B, n_max).

    Returns (p4, mask) torch tensors (views into `out` when given).
    """
    flat, counts = _ragged_to_flat(events)
    B = len(counts)
    if n_max is None:
        n_max = int(counts.max()) if B else 1
    if (counts > n_max).any():
        if truncate == "error":
            raise ValueError(f"event with {int(counts.max())} particles exceeds n_max={n_max}")
        if truncate == "pt":
            flat, counts = _truncate_by_pt(flat, counts, n_max)
        else:  # 'first'
            offsets = np.concatenate([[0], np.cumsum(counts)])
            keep = (np.arange(len(flat)) - offsets[:-1].repeat(counts)) < n_max
            flat, counts = flat[keep], np.minimum(counts, n_max)

    if out is None:
        p4 = torch.zeros(B, n_max, 4, dtype=torch.float32)
        mask = torch.zeros(B, n_max, dtype=torch.bool)
    else:
        p4, mask = out[0][:B, :n_max], out[1][:B, :n_max]
        p4.zero_()
        mask.zero_()

    rows = np.repeat(np.arange(B), counts)
    cols = np.arange(len(flat)) - (np.cumsum(counts) - counts).repeat(counts)
    p4_np = p4.numpy()
    mask_np = mask.numpy()
    p4_np[rows, cols] = flat
    mask_np[rows, cols] = True
    return p4, mask


def _scatter_batch(p4_t, mask_t, seg, cnt):
    """Fill padded tensors from a contiguous flat segment (pure numpy)."""
    bs = len(cnt)
    p4_np, mask_np = p4_t[:bs].numpy(), mask_t[:bs].numpy()
    p4_np[:] = 0.0
    mask_np[:] = False
    rows = np.repeat(np.arange(bs), cnt)
    cols = np.arange(len(seg)) - (np.cumsum(cnt) - cnt).repeat(cnt)
    p4_np[rows, cols] = seg
    mask_np[rows, cols] = True
    return p4_t[:bs], mask_t[:bs]


def _scatter_gpu(seg, cnt, bs, n_max):
    """Build padded (bs, n_max, 4) p4 + mask on the GPU from a flat segment and
    per-event counts -- the device twin of _scatter_batch.  Moving the scatter
    here keeps the CPU off the critical path: the host only stages the (small,
    unpadded) segment, and the (rows, cols) index math + the scatter run on the
    GPU, where they are ~10x cheaper than the numpy fancy-index at the batch
    sizes the pipeline feeds.  rows/cols are unique per particle, so the
    scatter is deterministic and bitwise-matches the numpy collation."""
    dev = seg.device
    S = seg.shape[0]
    off = cnt.cumsum(0) - cnt                                  # start of each event
    rows = torch.repeat_interleave(torch.arange(bs, device=dev), cnt)
    cols = torch.arange(S, device=dev) - torch.repeat_interleave(off, cnt)
    p4 = torch.zeros(bs, n_max, 4, dtype=torch.float32, device=dev)
    mask = torch.zeros(bs, n_max, dtype=torch.bool, device=dev)
    p4[rows, cols] = seg
    mask[rows, cols] = True
    return p4, mask


def to_gpu_batches(events, batch_size, n_max=None, device="cuda", truncate="pt"):
    """Yield (p4, mask) GPU batches from a ragged dataset, overlapping
    collation and H2D copies with downstream GPU compute.

    The ragged dataset is flattened ONCE up front (the only awkward-array
    work); each batch is then a pure numpy scatter from precomputed offsets
    into a ring of two pinned staging buffers, copied on a dedicated stream.
    While the caller clusters batch k on the default stream, batch k+1 is
    collated and copied asynchronously.
    """
    flat, counts = _ragged_to_flat(events)
    if n_max is None:
        n_max = int(counts.max()) if len(counts) else 1
    if (counts > n_max).any():
        if truncate == "error":
            raise ValueError(f"event with {int(counts.max())} particles exceeds n_max={n_max}")
        if truncate == "pt":
            flat, counts = _truncate_by_pt(flat, counts, n_max)
        else:
            offs = np.concatenate([[0], np.cumsum(counts)])
            keep = (np.arange(len(flat)) - offs[:-1].repeat(counts)) < n_max
            flat, counts = flat[keep], np.minimum(counts, n_max)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    n_events = len(counts)

    dev = torch.device(device)
    use_cuda = dev.type == "cuda" and torch.cuda.is_available()
    if not use_cuda:  # CPU fallback: plain synchronous batches
        for s in range(0, n_events, batch_size):
            e = min(s + batch_size, n_events)
            p4 = torch.zeros(e - s, n_max, 4, dtype=torch.float32)
            mask = torch.zeros(e - s, n_max, dtype=torch.bool)
            _scatter_batch(p4, mask, flat[offsets[s] : offsets[e]], counts[s:e])
            yield (p4, mask, None)
        return

    copy_stream = torch.cuda.Stream(dev)
    # ring of 2 pinned staging buffers holding the UNPADDED flat segment + its
    # per-event counts; the padded (B, n_max, 4) is built on the GPU
    # (_scatter_gpu), so per-batch host work is just the staging memcpy -- the
    # CPU scatter+index that otherwise dominates small-N/large-B pipelines is
    # gone from the critical path.  seg_cap bounds pinned use like before.
    seg_cap = batch_size * n_max  # >= particles in any one batch's segment
    ring = []
    for _ in range(2):
        ring.append(
            dict(
                seg=torch.empty(seg_cap, 4, dtype=torch.float32, pin_memory=True),
                cnt=torch.empty(batch_size, dtype=torch.int64, pin_memory=True),
                free=torch.cuda.Event(),  # signaled when the H2D copy + scatter are done
                armed=False,
            )
        )

    pending = None  # (gpu_p4, gpu_mask, copy_done_event)
    slot = 0
    for s in range(0, n_events, batch_size):
        e = min(s + batch_size, n_events)
        bs = e - s
        o0, o1 = int(offsets[s]), int(offsets[e])
        S = o1 - o0
        buf = ring[slot]
        slot ^= 1
        if buf["armed"]:
            buf["free"].synchronize()  # don't overwrite a buffer still in flight
        # stage into pinned (cheap host memcpy) so the H2D copy is truly async
        buf["seg"][:S].copy_(torch.from_numpy(flat[o0:o1]))
        buf["cnt"][:bs].copy_(torch.from_numpy(counts[s:e]))
        with torch.cuda.stream(copy_stream):
            g_seg = buf["seg"][:S].to(dev, non_blocking=True)
            g_cnt = buf["cnt"][:bs].to(dev, non_blocking=True)
            gpu_p4, gpu_mask = _scatter_gpu(g_seg, g_cnt, bs, n_max)
            buf["free"].record(copy_stream)
        buf["armed"] = True

        if pending is not None:
            yield pending
        pending = (gpu_p4, gpu_mask, buf["free"])

    if pending is not None:
        yield pending


def gpu_batch_ready(batch):
    """Make the current stream wait for a batch yielded by to_gpu_batches and
    return (p4, mask).  Call this right before using the tensors.

    Holding several un-readied batches (or dropping one) is data-safe — the
    yielded event object is re-recorded two batches later, but only ever to a
    LATER point on the same copy stream, so the cost is over-synchronization,
    never stale tensors."""
    p4, mask, done = batch
    if done is not None:
        cur = torch.cuda.current_stream()
        cur.wait_event(done)
        # the tensors were allocated on the copy stream; tell the caching
        # allocator they are consumed on this stream before it recycles them
        p4.record_stream(cur)
        mask.record_stream(cur)
    return p4, mask
