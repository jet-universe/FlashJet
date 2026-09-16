"""Batched NumPy CPU backend: the NN strategy, vectorized over the batch.

The torch backend rebuilds the full (B, N, N) d_ij matrix every step, which
is O(N^3) per event and unusable on CPU past a few dozen particles.  This
module runs the same FastJet N2Plain nearest-neighbour strategy as
nn_reference.py / triton_large.py -- O(N^2) per event -- but in lock-step
across the batch, so every step is a handful of (B, N) NumPy passes instead
of a Python loop per event.  That is the regime flashjet targets on CPU:
many small-to-medium events at once (jet reclustering), where FastJet's
per-event PseudoJet loop pays its Python/C++ boundary cost B times over.

Same slot model as everywhere else: a merge overwrites slot i, slot j dies;
initial pseudojet ids are 0..n-1 in mask order.  As in the Triton path the
particle->jet map is decoded afterwards from the merge history, so the hot
loop keeps no per-particle bookkeeping.  Output is bit-identical to
nn_reference.cluster_event_nn on the same event.

Being NumPy, the cost model is *passes over (rows, N)*, not scalar ops, so
the loop is written to minimize them:

  * the per-slot candidate distance `cand` is maintained incrementally.
    A merge only changes it for slot i, for slots whose geometric NN
    improved to i, and for slots whose NN went stale -- everyone else keeps
    a value that is still exact, so the step costs one argmin instead of
    rebuilding min(w_i, w_gnn) * dR2 / R^2 over the whole batch.
  * the stale rows of the whole batch (O(1) per event, but a different
    number in each) are gathered into one flat (M, N) block and rescanned in
    a single pass, so cost does not depend on how they are distributed.
  * finished events are physically dropped from the working set once enough
    of them have accumulated; without that, the tail of the loop keeps
    paying full width for events that stopped clustering long ago.
  * events are clustered in length-sorted chunks sized to trade NumPy
    per-call overhead against cache residency, and independent chunks run on
    a thread pool (NumPy ufuncs release the GIL).

Honest positioning: this is 1-2 orders of magnitude faster than routing CPU
tensors through the O(N^3) torch backend, and at batch sizes typical of a
training loop it beats FastJet's per-event Python interface.  It does NOT
beat FastJet's C++ inner loop per event -- a NumPy step is ~20 passes over
memory where FastJet has one fused register loop -- so the gap grows with N.
For single large events on CPU, use FastJet; for large N use the GPU path.
"""

import math

import numpy as np

from . import _native
from .kinematics import rap_phi_kt2

BEAM = -1
PAD = -2
TWO_PI = 2.0 * math.pi
_BUDGET = 1 << 22  # max temporary elements per vectorized block (~32 MB f64)
_COMPACT_FRAC = 0.6  # drop finished events once this fraction remains
_COMPACT_MIN = 8


def _slot_bytes(itemsize):
    """Working-set estimate per live slot: 9 floats + 2 int64 + 1 bool."""
    return 9 * itemsize + 2 * 8 + 1


def _dr2_into(rap_k, phi_k, rap, phi, out, tmp):
    """out <- dR^2 = dy^2 + dphi^2 (dphi folded into [0, pi]), no allocation."""
    np.subtract(phi_k, phi, out=out)
    np.abs(out, out=out)
    np.subtract(TWO_PI, out, out=tmp)
    np.minimum(out, tmp, out=out)
    np.multiply(out, out, out=out)
    np.subtract(rap_k, rap, out=tmp)
    np.multiply(tmp, tmp, out=tmp)
    np.add(out, tmp, out=out)
    return out


def _cand_from(w_k, w_nn, dr2, inv_R2):
    """Per-slot candidate distance: min(pair distance via the NN, beam).

    Padding slots carry w = tiny**p, which overflows to inf for p < 0; that is
    the intended value (never selected), so the overflow is silenced here
    rather than clamped, which would change the finite values too.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        return np.minimum(np.minimum(w_k, w_nn) * dr2 * inv_R2, w_k)


def _init_nn(rap, phi, act, nnd, nni):
    """Full geometric-NN scan for every slot, in row chunks bounded by
    _BUDGET so a large-N single event never allocates (N, N) at once."""
    B, N = rap.shape
    chunk = max(1, min(N, _BUDGET // max(B * N, 1)))
    for c0 in range(0, N, chunk):
        c1 = min(c0 + chunk, N)
        c = c1 - c0
        d = np.empty((B, c, N), dtype=rap.dtype)
        _dr2_into(rap[:, c0:c1, None], phi[:, c0:c1, None], rap[:, None, :],
                  phi[:, None, :], d, np.empty_like(d))
        d[~np.broadcast_to(act[:, None, :], d.shape)] = np.inf
        d[:, np.arange(c), np.arange(c0, c1)] = np.inf  # self
        j = d.argmin(2)
        dv = np.take_along_axis(d, j[..., None], 2)[..., 0]
        nnd[:, c0:c1] = dv
        nni[:, c0:c1] = np.where(np.isfinite(dv), j, -1)


def _rescan(rap, phi, act, w, nnd, nni, cand, bs, ks, inv_R2):
    """Recompute the geometric NN (and candidate distance) of the (event,
    slot) pairs (bs, ks), as one gathered block."""
    M = len(bs)
    if M == 0:
        return
    N = rap.shape[1]
    chunk = max(1, _BUDGET // max(N, 1))
    for s in range(0, M, chunk):
        b, k = bs[s : s + chunk], ks[s : s + chunk]
        m = len(b)
        rows = np.arange(m)
        ok = act[b]  # fancy indexing already returns a copy
        ok[rows, k] = False
        d = np.empty((m, N), dtype=rap.dtype)
        _dr2_into(rap[b, k][:, None], phi[b, k][:, None], rap[b], phi[b], d,
                  np.empty_like(d))
        np.copyto(d, np.inf, where=~ok)
        j = d.argmin(1)
        dv = d[rows, j]
        w_k = w[b, k]
        nnd[b, k] = dv
        nni[b, k] = np.where(np.isfinite(dv), j, -1)
        cand[b, k] = _cand_from(w_k, w[b, j], dv, inv_R2)


def _n_threads(threads, n_chunks):
    if threads is not None:
        return max(1, min(int(threads), n_chunks))
    import os

    try:
        avail = len(os.sched_getaffinity(0))
    except AttributeError:
        avail = os.cpu_count() or 1
    return max(1, min(avail, n_chunks))


def _cluster_chunk(px, py, pz, pe, act, orig, hp1, hp2, hch, hd, inv_R2, p):
    """Cluster one chunk of events in lock-step, writing into the batch-wide
    history arrays at rows `orig`.  Inputs carry the working dtype."""
    B, N = act.shape
    ft = px.dtype
    tiny = 1e-300 if ft == np.float64 else 1e-30
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        rap, phi, kt2 = rap_phi_kt2(px, py, pz, pe)
        w = np.maximum(kt2, tiny) ** ft.type(p)
    rap = np.where(act, rap, 0.0)  # keep padding out of the geometry
    phi = np.where(act, phi, 0.0)

    nnd = np.empty((B, N), dtype=ft)
    nni = np.empty((B, N), dtype=np.int64)
    _init_nn(rap, phi, act, nnd, nni)
    cand = np.where(act, _cand_from(w, np.take(w, nni.clip(0) + np.arange(B)[:, None] * N),
                                    nnd, inv_R2), np.inf)

    ids = np.cumsum(act, axis=1) - 1
    ids[~act] = PAD
    next_id = act.sum(1).astype(np.int64)
    n_act = next_id.copy()          # live slots per row; -1 per merge step

    col = np.arange(N)[None, :]
    dnew = np.empty((B, N), dtype=ft)
    scratch = np.empty((B, N), dtype=ft)

    for step in range(N):
        alive = n_act > 0
        n_live = int(alive.sum())
        if n_live == 0:
            break

        # ---- drop finished events from the working set
        rows_n = len(orig)
        if n_live <= _COMPACT_FRAC * rows_n and rows_n - n_live >= _COMPACT_MIN:
            keep = np.flatnonzero(alive)
            px, py, pz, pe = px[keep], py[keep], pz[keep], pe[keep]
            rap, phi, w = rap[keep], phi[keep], w[keep]
            act, nnd, nni, cand, ids = act[keep], nnd[keep], nni[keep], cand[keep], ids[keep]
            next_id, n_act, orig = next_id[keep], n_act[keep], orig[keep]
            dnew, scratch = np.empty_like(cand), np.empty_like(cand)
            alive = n_act > 0
            rows_n = len(orig)
        rows = np.arange(rows_n)

        # ---- global argmin over the maintained candidates
        ii = cand.argmin(1)
        gmin = cand[rows, ii]
        wi = w[rows, ii]
        ni = nni[rows, ii]
        jj = np.maximum(ni, 0)
        with np.errstate(over="ignore", invalid="ignore"):
            d_pair_i = np.where(
                ni >= 0, np.minimum(wi, w[rows, jj]) * nnd[rows, ii] * inv_R2, np.inf
            )
        is_pair = d_pair_i < wi  # ties go to the beam, as in the reference
        do_pair = is_pair & alive
        do_beam = (~is_pair) & alive
        any_pair = bool(do_pair.any())

        al = orig[alive]
        hp1[al, step] = ids[rows, ii][alive]
        hp2[al, step] = np.where(do_pair, ids[rows, jj], BEAM)[alive]
        hch[al, step] = np.where(do_pair, next_id, BEAM)[alive]
        hd[al, step] = gmin[alive]
        n_act -= alive

        # ---- apply the merge (slot ii absorbs slot jj, or dies to the beam)
        if any_pair:
            pb, pi, pj = rows[do_pair], ii[do_pair], jj[do_pair]
            for arr in (px, py, pz, pe):
                arr[pb, pi] += arr[pb, pj]
            act[pb, pj] = False
            cand[pb, pj] = np.inf
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                r, f, k2 = rap_phi_kt2(px[pb, pi], py[pb, pi], pz[pb, pi], pe[pb, pi])
                w[pb, pi] = np.maximum(k2, tiny) ** ft.type(p)
            rap[pb, pi], phi[pb, pi] = r, f
            ids[pb, pi] = next_id[do_pair]
            next_id = next_id + do_pair
        if do_beam.any():
            bb, bi = rows[do_beam], ii[do_beam]
            act[bb, bi] = False
            cand[bb, bi] = np.inf

        # ---- NN maintenance.  Slot ii moved and slot jj is gone, so rows
        # pointing at either are stale; every other row can only have
        # IMPROVED, and the improvement is exactly the distance to the new ii.
        notself = col != ii[:, None]
        base = act & notself
        hit = nni == ii[:, None]
        if any_pair:
            hit |= do_pair[:, None] & (nni == jj[:, None])
        stale = base & hit
        if any_pair:
            ok = base & do_pair[:, None]
            _dr2_into(rap[rows, ii][:, None], phi[rows, ii][:, None], rap, phi, dnew, scratch)
            np.copyto(dnew, np.inf, where=~ok)

            # improvements are sparse (only slots the new pseudojet moved
            # closer to), so update them as a gathered list rather than
            # computing a full-width candidate array to write a few entries
            improve = ok & ~stale
            improve &= dnew < nnd
            gb, gk = np.nonzero(improve)
            if len(gb):
                g_ii = ii[gb]
                g_d = dnew[gb, gk]
                nnd[gb, gk] = g_d
                nni[gb, gk] = g_ii
                cand[gb, gk] = _cand_from(w[gb, gk], w[gb, g_ii], g_d, inv_R2)

            # the new pseudojet's own NN falls out of the same row
            jb = dnew.argmin(1)
            dv = dnew[rows, jb]
            fin = np.isfinite(dv)
            nnd[pb, pi] = dv[do_pair]
            nni[pb, pi] = np.where(fin, jb, -1)[do_pair]
            cand[pb, pi] = _cand_from(w[pb, pi], w[pb, jb[do_pair]], dv[do_pair], inv_R2)

        sb, sk = np.nonzero(stale)
        _rescan(rap, phi, act, w, nnd, nni, cand, sb, sk, inv_R2)


def cluster_batch_cpu(p4, mask, R, p, chunk_bytes=1 << 22, threads=None, native=None):
    """Cluster a padded CPU batch; same outputs as cluster_batch_torch().

    Args:
        p4:   (B, N, 4) CPU torch tensor, columns px, py, pz, E.
        mask: (B, N) bool tensor, True for real particles.
        R, p: jet radius and generalized-kt exponent.
        chunk_bytes: target working-set size for one chunk of events.
            Chunking trades NumPy per-call overhead (which favours one big
            chunk) against cache residency (which favours small ones); 4 MB
            is the measured optimum on a Cascade Lake Xeon.
        threads: worker threads (default: one per core).  Pass 1 to stay
            single-threaded.
        native: use the compiled C++ kernel (default: whenever it is built).
            Pass False to force the NumPy path -- the two agree step for step,
            so this is a performance switch only.
    """
    import torch

    from .history import jet_idx_from_history

    if p4.is_cuda:
        raise RuntimeError("cpu backend requires CPU tensors")
    dt = p4.dtype if p4.dtype in (torch.float32, torch.float64) else torch.float32
    B, N, _ = p4.shape

    hp1 = np.full((B, N), PAD, dtype=np.int64)
    hp2 = np.full((B, N), PAD, dtype=np.int64)
    hch = np.full((B, N), PAD, dtype=np.int64)
    hd = np.zeros((B, N), dtype=np.float64 if dt == torch.float64 else np.float32)

    # follow the torch backend: compute in the input dtype.  float64 in
    # reproduces the reference merge order exactly; float32 in halves the
    # memory traffic and may order near-degenerate merges differently.
    use_native = _native.HAS_NATIVE if native is None else bool(native)
    if use_native and not _native.HAS_NATIVE:
        raise RuntimeError("flashjet was built without the C++ CPU kernel")

    if N and B and use_native:
        npdt = np.float64 if dt == torch.float64 else np.float32
        arr = np.ascontiguousarray(p4.detach().cpu().numpy(), dtype=npdt)
        m8 = np.ascontiguousarray(mask.detach().cpu().numpy().view(np.uint8))
        nt = _n_threads(threads, B)
        hp1, hp2, hch, hd = _native.cluster_native(arr, m8, R, p, nt)
    elif N and B:
        work = p4.detach().to(torch.float64 if dt == torch.float64 else torch.float32).numpy()
        act_all = mask.detach().cpu().numpy()
        counts = act_all.sum(1)
        # events are clustered in length-sorted chunks: a chunk of short
        # events then finishes its step loop early instead of being dragged
        # to N steps by the longest event in the batch
        order = np.argsort(counts, kind="stable")
        per_event = _slot_bytes(work.dtype.itemsize) * N
        rows = int(np.clip(chunk_bytes // max(per_event, 1), 1, B))
        chunks = [order[c0 : c0 + rows] for c0 in range(0, B, rows)]

        def run(sel):
            width = int(counts[sel].max()) if len(sel) else 0
            if width == 0:
                return
            px, py, pz, pe = (np.ascontiguousarray(work[sel, :width, k]) for k in range(4))
            _cluster_chunk(px, py, pz, pe, act_all[sel, :width].copy(), sel,
                           hp1, hp2, hch, hd, 1.0 / (R * R), p)

        # chunks touch disjoint rows of the history arrays, and the loop is
        # NumPy ufuncs (GIL released), so threads are safe and do overlap
        nt = _n_threads(threads, len(chunks))
        if nt > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(nt) as ex:
                list(ex.map(run, chunks))
        else:
            for sel in chunks:
                run(sel)

    t1, t2, tc = (torch.from_numpy(a) for a in (hp1, hp2, hch))
    jet_idx, n_jets = jet_idx_from_history(t1, t2, tc, mask.cpu())
    return {"jet_idx": jet_idx, "n_jets": n_jets, "hist_p1": t1, "hist_p2": t2,
            "hist_child": tc, "hist_d": torch.from_numpy(hd).to(dt)}
