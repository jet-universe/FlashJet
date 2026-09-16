"""Large-N (up to ~16k constituents) Triton backend.

One program still clusters one whole event, but particle state lives in
global-memory scratch arrays processed in chunked vector passes instead of
registers, and the FastJet N2Plain strategy replaces the dense pairwise tile:

  * each slot carries its current GEOMETRIC nearest neighbour (min dR^2);
    by the Cacciari-Salam lemma the global d_ij minimum is always realized
    at some slot's geometric NN, so the per-slot candidate is
    min(w_i, w_gnn) * dR2_gnn / R^2 vs the beam distance w_i;
  * geometric NNs only go stale when a slot's position changes or dies
    (O(1) rows per merge on average -- maintaining NNs in the d_ij measure
    instead cascades through every soft particle pointing at the hard core).
    Total work is O(N^2) per event instead of O(N^3).

The kernel is memory-bandwidth bound, so the hot loop is kept to one fused
pass per step: NN updates and the argmin for the NEXT step are computed in
the same sweep (a stale row whose distance to the merged pseudojet is
strictly below its old NND resolves inline, without a rescan -- see the
shortcut comment in the fused pass), and no per-particle bookkeeping is touched
in-kernel -- the particle->jet assignment is decoded afterwards from the
merge history (history.jet_idx_from_history, O(log N) batched gathers).

The exact algorithm is mirrored (and CPU-validated against the brute-force
reference) in nn_reference.py.  The kernel is branchless w.r.t. pair-vs-beam:
both code paths run every step with masked stores, which keeps control flow
uniform and portable across Triton versions.

Cross-thread coherence: scratch state in global memory is written by some
threads of the program and read by others; every store->load handoff is
fenced with tl.debug_barrier() (without it we observed phantom merges with
already-dead slots).

Limitations: float32 only; merge ordering can differ from the float64
reference for near-degenerate distances (physics-level outputs agree).
"""

import os

import torch

from .triton_backend import HAS_TRITON
from .history import jet_idx_from_history

MAX_LARGE_N = 16384

# N-banded launch defaults (block, num_warps, tile), tuned on an A100 via
# scripts/tune_large.py; the top band is the original T4 tuning
_BAND_DEFAULTS = ((1024, 2, (32, 64)), (512, 4, (32, 64)), (1024, 8, (64, 128)))


def _band(N):
    return 0 if N <= 128 else (1 if N <= 2048 else 2)


def _defaults(N, device):
    # H100-measured override (sweep 2026-06): (1024, 8, (64, 64)) wins the
    # upper half of band 1 (+8.4% at N=1024, +22% at N=2048) but is 46%
    # SLOWER at N=512 (BLOCK clamps to 512 there), and band 2 matched the
    # A100 default within noise -- so only 512 < N <= 2048 is overridden.
    if 512 < N <= 2048 and _is_h100(str(device)):
        return (1024, 8, (64, 64))
    return _BAND_DEFAULTS[_band(N)]


def _is_h100(device_str, _cache={}):
    if device_str not in _cache:
        _cache[device_str] = "H100" in torch.cuda.get_device_name(torch.device(device_str))
    return _cache[device_str]

if HAS_TRITON:
    import triton
    import triton.language as tl

    try:
        from triton.language.extra import libdevice as _tld
    except ImportError:  # triton < 3.0
        from triton.language import math as _tld

    @triton.jit
    def _rap_w(px, py, pz, pe, p_exp):
        kt2 = px * px + py * py
        m2 = tl.maximum(pe * pe - pz * pz - kt2, 0.0)
        apz = tl.abs(pz)
        beamlike = (kt2 + m2) <= 0.0
        ratio = tl.where(beamlike, 1.0, (kt2 + m2) / ((pe + apz) * (pe + apz)))
        half_log = 0.5 * tl.log(ratio)
        rap = tl.where(pz >= 0, -half_log, half_log)
        rap = tl.where(beamlike, tl.where(pz >= 0, 1e5 + apz, -(1e5 + apz)), rap)
        w = tl.exp(p_exp * tl.log(tl.maximum(kt2, 1e-30)))
        return rap, w

    @triton.jit
    def _row_rescan(RAP, PHI, ACT, base, k, rap_k, phi_k, N, BLOCK: tl.constexpr):
        """Geometric NN (min dR^2) of slot k over active slots != k."""
        best = rap_k * 0.0 + float("inf")
        bestj = k * 0 - 1
        for c in range(0, N, BLOCK):
            offs = c + tl.arange(0, BLOCK)
            m = offs < N
            rap_j = tl.load(RAP + base + offs, mask=m, other=0.0)
            phi_j = tl.load(PHI + base + offs, mask=m, other=0.0)
            act_j = tl.load(ACT + base + offs, mask=m, other=0) != 0
            dphi = tl.abs(phi_k - phi_j)
            dphi = tl.minimum(dphi, 6.283185307179586 - dphi)
            drap = rap_k - rap_j
            d = tl.where(act_j & (offs != k) & m, tl.fma(drap, drap, dphi * dphi), float("inf"))
            cmin = tl.min(d, axis=0)
            cidx = tl.min(tl.where(d == cmin, offs, 2147483647), axis=0)
            better = cmin < best
            bestj = tl.where(better, cidx.to(tl.int32), bestj)
            best = tl.where(better, cmin, best)
        return best, bestj

    @triton.jit
    def _cluster_large_kernel(
        P4, MASK,                       # (B, N, 4) f32 + (B, N) u8 inputs (read-only)
        PX, PY, PZ, PE,                 # (B, N) f32 scratch copies (mutated)
        RAP, PHI, W, NND,               # (B, N) f32 scratch (NND = geometric dR^2)
        ACT, NNI, IDS, MAP,             # (B, N) i8 / i32 / i32 / i32 scratch
        HP1, HP2, HCH, HD,              # outputs (B, N) i64 / i64 / i64 / f32
        N, inv_R2, p_exp,
        BLOCK: tl.constexpr, BI: tl.constexpr, BJ: tl.constexpr,
    ):
        b = tl.program_id(0)
        base = b * N

        # ---- phase 0: stage inputs into scratch (the merge loop mutates
        # PX..PE/ACT in place, so the caller's tensors are copied here and
        # never written), kinematics + compact pseudojet ids, history
        # prefilled to the PAD contract (-2 / 0.0) so steps >= n_init need
        # no host-side fill ----
        cnt = N * 0
        for c in range(0, N, BLOCK):
            offs = c + tl.arange(0, BLOCK)
            m = offs < N
            px = tl.load(P4 + (base + offs) * 4 + 0, mask=m, other=0.0)
            py = tl.load(P4 + (base + offs) * 4 + 1, mask=m, other=0.0)
            pz = tl.load(P4 + (base + offs) * 4 + 2, mask=m, other=0.0)
            pe = tl.load(P4 + (base + offs) * 4 + 3, mask=m, other=0.0)
            act = tl.load(MASK + base + offs, mask=m, other=0) != 0
            tl.store(PX + base + offs, px, mask=m)
            tl.store(PY + base + offs, py, mask=m)
            tl.store(PZ + base + offs, pz, mask=m)
            tl.store(PE + base + offs, pe, mask=m)
            tl.store(ACT + base + offs, act.to(tl.int8), mask=m)
            rap, w = _rap_w(px, py, pz, pe, p_exp)
            phi = _tld.atan2(py, px)
            tl.store(RAP + base + offs, rap, mask=m)
            tl.store(PHI + base + offs, phi, mask=m)
            tl.store(W + base + offs, w, mask=m)
            ids = cnt + tl.cumsum(act.to(tl.int32), axis=0) - 1
            ids = tl.where(act, ids, -2)
            tl.store(IDS + base + offs, ids, mask=m)
            cnt += tl.sum(act.to(tl.int32), axis=0)
            pad = (offs * 0 - 2).to(tl.int64)
            tl.store(HP1 + base + offs, pad, mask=m)
            tl.store(HP2 + base + offs, pad, mask=m)
            tl.store(HCH + base + offs, pad, mask=m)
            tl.store(HD + base + offs, offs * 0.0, mask=m)
        # the masked popcount IS the live count -- integer-exact, replaces a
        # host-side mask.sum(1) kernel and the per-program NINIT load
        n_init = cnt

        tl.debug_barrier()

        # ---- phase 1: initial geometric nearest neighbours, (BI, BJ)-tiled ----
        for ci in range(0, N, BI):
            ioffs = ci + tl.arange(0, BI)
            mi = ioffs < N
            rap_i = tl.load(RAP + base + ioffs, mask=mi, other=0.0)
            phi_i = tl.load(PHI + base + ioffs, mask=mi, other=0.0)
            act_i = tl.load(ACT + base + ioffs, mask=mi, other=0) != 0
            best = tl.zeros((BI,), dtype=tl.float32) + float("inf")
            bestj = tl.zeros((BI,), dtype=tl.int32) - 1
            for cj in range(0, N, BJ):
                joffs = cj + tl.arange(0, BJ)
                mj = joffs < N
                rap_j = tl.load(RAP + base + joffs, mask=mj, other=0.0)
                phi_j = tl.load(PHI + base + joffs, mask=mj, other=0.0)
                act_j = tl.load(ACT + base + joffs, mask=mj, other=0) != 0
                dphi = tl.abs(phi_i[:, None] - phi_j[None, :])
                dphi = tl.minimum(dphi, 6.283185307179586 - dphi)
                drap = rap_i[:, None] - rap_j[None, :]
                # explicit fma: every dr2 site must produce identical bits
                # (see the shortcut comment in the fused pass)
                d = tl.fma(drap, drap, dphi * dphi)
                ok = act_i[:, None] & act_j[None, :] & (ioffs[:, None] != joffs[None, :])
                d = tl.where(ok & mj[None, :], d, float("inf"))
                tmin = tl.min(d, axis=1)
                tidx = tl.min(tl.where(d == tmin[:, None], joffs[None, :], 2147483647), axis=1)
                better = tmin < best
                bestj = tl.where(better, tidx.to(tl.int32), bestj)
                best = tl.where(better, tmin, best)
            tl.store(NND + base + ioffs, best, mask=mi)
            tl.store(NNI + base + ioffs, bestj, mask=mi)

        tl.debug_barrier()

        # ---- initial argmin of min(d_pair, d_iB) (Cacciari-Salam candidate) ----
        gmin = n_init * 0.0 + float("inf")
        i_sel = n_init * 0 - 1
        for c in range(0, N, BLOCK):
            offs = c + tl.arange(0, BLOCK)
            m = offs < N
            gdr2 = tl.load(NND + base + offs, mask=m, other=0.0)
            nni = tl.load(NNI + base + offs, mask=m, other=-1)
            w = tl.load(W + base + offs, mask=m, other=0.0)
            act = tl.load(ACT + base + offs, mask=m, other=0) != 0
            w_gnn = tl.load(W + base + tl.maximum(nni, 0), mask=m, other=0.0)
            d_pair = tl.where(nni >= 0, tl.minimum(w, w_gnn) * gdr2 * inv_R2, float("inf"))
            cand = tl.where(act & m, tl.minimum(d_pair, w), float("inf"))
            cmin = tl.min(cand, axis=0)
            cidx = tl.min(tl.where(cand == cmin, offs, 2147483647), axis=0)
            better = cmin < gmin
            i_sel = tl.where(better, cidx.to(tl.int32), i_sel)
            gmin = tl.where(better, cmin, gmin)

        # ---- main loop: one merge per iteration, fused update+scan pass ----
        # `bound` is the scan window: live slots are periodically compacted
        # into [0, bound) so per-step cost tracks the shrinking event.
        next_id = n_init
        n_act = n_init
        bound = n_init * 0 + N
        for step in range(0, n_init):
            gdr2_i = tl.load(NND + base + i_sel)
            w_i = tl.load(W + base + i_sel)
            j_sel = tl.load(NNI + base + i_sel)
            w_gnn_i = tl.load(W + base + tl.maximum(j_sel, 0))
            d_pair_i = tl.where(j_sel >= 0, tl.minimum(w_i, w_gnn_i) * gdr2_i * inv_R2, float("inf"))
            is_pair = d_pair_i < w_i
            j_sel = tl.where(is_pair, j_sel, i_sel)

            # history (read parent ids before overwriting); ids stay i32 in
            # scratch and are widened only at the i64 history store
            id_i = tl.load(IDS + base + i_sel)
            id_j = tl.load(IDS + base + j_sel)
            tl.store(HP1 + base + step, id_i.to(tl.int64))
            tl.store(HP2 + base + step, tl.where(is_pair, id_j, -1).to(tl.int64))
            tl.store(HCH + base + step, tl.where(is_pair, next_id, -1).to(tl.int64))
            tl.store(HD + base + step, gmin)

            # merge: slot i gets the E-scheme sum (pair) or keeps its value
            pxi = tl.load(PX + base + i_sel)
            pyi = tl.load(PY + base + i_sel)
            pzi = tl.load(PZ + base + i_sel)
            pei = tl.load(PE + base + i_sel)
            addj = is_pair.to(tl.float32)
            pxi += addj * tl.load(PX + base + j_sel)
            pyi += addj * tl.load(PY + base + j_sel)
            pzi += addj * tl.load(PZ + base + j_sel)
            pei += addj * tl.load(PE + base + j_sel)
            # every warp executes this scalar phase redundantly, and unlike
            # the vector phases it contains no tl.min/tl.sum reductions whose
            # lowering would fence it: without this barrier a lagging warp can
            # re-read slot state after a faster warp already stored the merged
            # values (WAR race; nondeterministic at low-occupancy configs)
            tl.debug_barrier()
            tl.store(PX + base + i_sel, pxi)
            tl.store(PY + base + i_sel, pyi)
            tl.store(PZ + base + i_sel, pzi)
            tl.store(PE + base + i_sel, pei)
            rap_i, w_new = _rap_w(pxi, pyi, pzi, pei, p_exp)
            phi_i = _tld.atan2(pyi, pxi)
            tl.store(RAP + base + i_sel, rap_i)
            tl.store(PHI + base + i_sel, phi_i)
            tl.store(W + base + i_sel, w_new)
            tl.store(IDS + base + i_sel, tl.where(is_pair, next_id, id_i))
            next_id += is_pair.to(tl.int32)

            # kill slot j (pair) or slot i (beam) BEFORE the fused pass
            kill = tl.where(is_pair, j_sel, i_sel)
            tl.store(ACT + base + kill, (next_id * 0).to(tl.int8))
            # kill flag + slot i state must be visible to all threads
            tl.debug_barrier()

            # fused pass: NN maintenance AND the argmin for the next step
            a_best = rap_i * 0.0 + float("inf")
            a_bestj = i_sel * 0 - 1
            g2 = rap_i * 0.0 + float("inf")
            i2 = i_sel * 0 - 1
            for c in range(0, bound, BLOCK):
                offs = c + tl.arange(0, BLOCK)
                m = offs < bound
                act = tl.load(ACT + base + offs, mask=m, other=0) != 0
                rap_j = tl.load(RAP + base + offs, mask=m, other=0.0)
                phi_j = tl.load(PHI + base + offs, mask=m, other=0.0)
                gdr2 = tl.load(NND + base + offs, mask=m, other=0.0)
                nni = tl.load(NNI + base + offs, mask=m, other=-1)
                w = tl.load(W + base + offs, mask=m, other=0.0)
                dphi = tl.abs(phi_i - phi_j)
                dphi = tl.minimum(dphi, 6.283185307179586 - dphi)
                drap = rap_i - rap_j
                # explicit fma: every dr2 site must produce identical bits
                # (see the shortcut comment below)
                dr2_new = tl.fma(drap, drap, dphi * dphi)
                ok = act & (offs != i_sel) & (offs != kill) & m
                dr2_new = tl.where(ok, dr2_new, float("inf"))

                stale = ok & ((nni == i_sel) | (nni == j_sel))
                # Cacciari-Salam shortcut: a stale row with dr2_new STRICTLY
                # below its old NND needs no rescan -- every other active
                # slot is >= the old NND, so the merged pseudojet is the
                # unique new NN.  Strict only: on equality a rescan can find
                # a lower-index slot at the same distance.  Bitwise-identical
                # to the rescan it replaces because dr2 is an explicit
                # tl.fma at every site (phase 1 / _row_rescan / here): a
                # symmetric pair's NND[a] and NND[b] come from different
                # sites, and a 1-ulp contraction asymmetry flips which slot
                # hosts the merge on the lowest-index tie-break (observed:
                # C/A jet numbering diverged in 27% of events).
                upd = ok & is_pair & (dr2_new < gdr2)
                stale = stale & ~upd
                gdr2_u = tl.where(upd, dr2_new, gdr2)
                nni_u = tl.where(upd, i_sel, nni)
                # store only updated lanes: the rest would write back the
                # value they just loaded (stale lanes are rewritten by their
                # rescan below), so skipping them cannot change memory
                tl.store(NND + base + offs, gdr2_u, mask=upd)
                tl.store(NNI + base + offs, nni_u, mask=upd)

                # the new pseudojet's geometric NN
                cmin = tl.min(dr2_new, axis=0)
                cidx = tl.min(tl.where(dr2_new == cmin, offs, 2147483647), axis=0)
                better = cmin < a_best
                a_bestj = tl.where(better, cidx.to(tl.int32), a_bestj)
                a_best = tl.where(better, cmin, a_best)

                # next-step candidates from updated values (stale rows are
                # folded in after their rescan; slot i after the loop)
                w_gnn = tl.load(W + base + tl.maximum(nni_u, 0), mask=m, other=0.0)
                d_pair = tl.where(nni_u >= 0, tl.minimum(w, w_gnn) * gdr2_u * inv_R2, float("inf"))
                # ties break toward the lowest slot index (matches the torch
                # backend's flat argmin; matters for C/A where all d_iB == 1)
                cand = tl.where(ok & ~stale, tl.minimum(d_pair, w), float("inf"))
                ccmin = tl.min(cand, axis=0)
                ccidx = tl.min(tl.where(cand == ccmin, offs, 2147483647), axis=0).to(tl.int32)
                cbetter = (ccmin < g2) | ((ccmin == g2) & (ccidx < i2))
                i2 = tl.where(cbetter, ccidx, i2)
                g2 = tl.where(cbetter, ccmin, g2)

                # rescan the remaining stale rows one by one, folding their
                # candidates
                ns = tl.sum(stale.to(tl.int32), axis=0)
                cum = tl.cumsum(stale.to(tl.int32), axis=0)
                for t in range(0, ns):
                    k = tl.min(tl.where(stale & (cum == t + 1), offs, 2147483647), axis=0).to(tl.int32)
                    rap_k = tl.load(RAP + base + k)
                    phi_k = tl.load(PHI + base + k)
                    kb, kj = _row_rescan(RAP, PHI, ACT, base, k, rap_k, phi_k, bound, BLOCK)
                    tl.store(NND + base + k, kb)
                    tl.store(NNI + base + k, kj)
                    w_k = tl.load(W + base + k)
                    w_kj = tl.load(W + base + tl.maximum(kj, 0))
                    dpk = tl.where(kj >= 0, tl.minimum(w_k, w_kj) * kb * inv_R2, float("inf"))
                    candk = tl.minimum(dpk, w_k)
                    kbetter = (candk < g2) | ((candk == g2) & (k < i2))
                    i2 = tl.where(kbetter, k, i2)
                    g2 = tl.where(kbetter, candk, g2)

            # slot i's own NN and candidate (pair steps only; dead for beam)
            tl.store(NND + base + i_sel, tl.where(is_pair, a_best, gdr2_i))
            tl.store(NNI + base + i_sel, tl.where(is_pair, a_bestj, j_sel))
            w_abj = tl.load(W + base + tl.maximum(a_bestj, 0))
            dpi = tl.where(a_bestj >= 0, tl.minimum(w_new, w_abj) * a_best * inv_R2, float("inf"))
            candi = tl.where(is_pair, tl.minimum(dpi, w_new), float("inf"))
            ibetter = (candi < g2) | ((candi == g2) & (i_sel < i2))
            i2 = tl.where(ibetter, i_sel, i2)
            g2 = tl.where(ibetter, candi, g2)

            # fused-pass stores must be visible before the next scalar phase
            tl.debug_barrier()

            # ---- dead-slot compaction: when half the window is corpses,
            # move survivors to a prefix so scans track the live count ----
            n_act -= 1
            if (n_act + n_act) <= bound:
                if bound > BLOCK:
                    if n_act > 0:
                        # pass 1: prefix-sum new positions into MAP
                        cnt = n_act * 0
                        for c in range(0, bound, BLOCK):
                            offs = c + tl.arange(0, BLOCK)
                            m = offs < bound
                            act = tl.load(ACT + base + offs, mask=m, other=0) != 0
                            pos = cnt + tl.cumsum(act.to(tl.int32), axis=0) - 1
                            tl.store(MAP + base + offs, tl.where(act, pos, -1), mask=m)
                            cnt += tl.sum(act.to(tl.int32), axis=0)
                        tl.debug_barrier()
                        # pass 2: scatter-move survivors left (targets are a
                        # disjoint prefix and never overlap unread sources;
                        # the per-chunk barrier orders loads before stores
                        # across threads)
                        for c in range(0, bound, BLOCK):
                            offs = c + tl.arange(0, BLOCK)
                            m = offs < bound
                            act = tl.load(ACT + base + offs, mask=m, other=0) != 0
                            mapv = tl.load(MAP + base + offs, mask=m, other=-1)
                            vpx = tl.load(PX + base + offs, mask=m, other=0.0)
                            vpy = tl.load(PY + base + offs, mask=m, other=0.0)
                            vpz = tl.load(PZ + base + offs, mask=m, other=0.0)
                            vpe = tl.load(PE + base + offs, mask=m, other=0.0)
                            vra = tl.load(RAP + base + offs, mask=m, other=0.0)
                            vph = tl.load(PHI + base + offs, mask=m, other=0.0)
                            vw = tl.load(W + base + offs, mask=m, other=0.0)
                            vnd = tl.load(NND + base + offs, mask=m, other=0.0)
                            vni = tl.load(NNI + base + offs, mask=m, other=-1)
                            vid = tl.load(IDS + base + offs, mask=m, other=-2)
                            tl.debug_barrier()
                            tgt = tl.maximum(mapv, 0)
                            am = act & m
                            tl.store(PX + base + tgt, vpx, mask=am)
                            tl.store(PY + base + tgt, vpy, mask=am)
                            tl.store(PZ + base + tgt, vpz, mask=am)
                            tl.store(PE + base + tgt, vpe, mask=am)
                            tl.store(RAP + base + tgt, vra, mask=am)
                            tl.store(PHI + base + tgt, vph, mask=am)
                            tl.store(W + base + tgt, vw, mask=am)
                            tl.store(NND + base + tgt, vnd, mask=am)
                            tl.store(NNI + base + tgt, vni, mask=am)
                            tl.store(IDS + base + tgt, vid, mask=am)
                            tl.store(ACT + base + tgt, (vni * 0 + 1).to(tl.int8), mask=am)
                        tl.debug_barrier()
                        # pass 3: remap NN pointers (old -> new indices)
                        for c in range(0, n_act, BLOCK):
                            offs = c + tl.arange(0, BLOCK)
                            m = offs < n_act
                            nni = tl.load(NNI + base + offs, mask=m, other=-1)
                            mapped = tl.load(MAP + base + tl.maximum(nni, 0), mask=m, other=-1)
                            tl.store(NNI + base + offs, tl.where(nni >= 0, mapped, nni), mask=m)
                        i2m = tl.load(MAP + base + tl.maximum(i2, 0))
                        i2 = tl.where(i2 >= 0, i2m, i2)
                        bound = n_act * 0 + n_act
                        tl.debug_barrier()

            gmin = g2
            i_sel = i2


def cluster_batch_triton_large(
    p4: torch.Tensor,
    mask: torch.Tensor,
    R: float,
    p: float,
    block: int = None,
    num_warps: int = None,
    tile: "tuple[int, int]" = None,
    tune: bool = None,
    decode: bool = True,
):
    """Large-N Triton counterpart of cluster_batch_torch (same outputs).

    block/num_warps/tile are tuning knobs; unset ones take N-banded defaults
    tuned on an A100 with scripts/tune_large.py (the previous T4 tuning was
    block=1024, num_warps=8, tile=(64, 128) at every N — still the default
    band above N=2048).  tune=True (or FLASHJET_TUNE=1) benchmarks the
    curated config list on this batch once per (GPU model, N-band) and
    reuses the persisted winner afterwards — see flashjet/tune.py; explicit
    knobs disable it.
    """
    if not HAS_TRITON:
        raise RuntimeError("triton is not installed")
    if not p4.is_cuda:
        raise RuntimeError("triton backend requires CUDA tensors")
    B, N, _ = p4.shape
    if N > MAX_LARGE_N:
        raise ValueError(f"triton-large backend supports N <= {MAX_LARGE_N}, got {N}")
    dev = p4.device

    if tune is None:
        tune = os.environ.get("FLASHJET_TUNE", "") not in ("", "0")
    if tune and block is None and num_warps is None and tile is None:
        from .tune import tuned_config

        block, num_warps, tile = tuned_config(
            _band(N),
            lambda cfg: cluster_batch_triton_large(
                p4, mask, R, p, block=cfg[0], num_warps=cfg[1], tile=cfg[2]
            ),
            dev,
        )
    f32 = dict(dtype=torch.float32, device=dev)
    i64 = dict(dtype=torch.int64, device=dev)

    # kernel inputs: no-ops when p4 is already f32-contiguous; bool mask is
    # 1 byte, so the uint8 view is a zero-copy reinterpret
    p4f = p4.to(torch.float32).contiguous()
    masku = mask.contiguous().view(torch.uint8)

    # scratch: two flat allocations carved into contiguous (B, N) views; no
    # fills -- every cell is kernel-written before it is read (PX..PE / ACT /
    # RAP/PHI/W/IDS in phase 0, NND/NNI in phase 1, MAP in compaction pass 1)
    sf = torch.empty(8, B, N, **f32)
    px, py, pz, pe, rap, phi, w, nnd = sf.unbind(0)
    si = torch.empty(3, B, N, dtype=torch.int32, device=dev)
    nni, ids, cmap = si.unbind(0)
    act = torch.empty(B, N, dtype=torch.int8, device=dev)

    # history: prefilled to PAD (-2) / 0.0 in-kernel (phase 0); allocated
    # int64 so the decode consumes them without a cast
    hp1 = torch.empty(B, N, **i64)
    hp2 = torch.empty(B, N, **i64)
    hch = torch.empty(B, N, **i64)
    hd = torch.empty(B, N, **f32)

    import triton

    band = _defaults(N, dev)
    block = band[0] if block is None else block
    num_warps = band[1] if num_warps is None else num_warps
    tile = band[2] if tile is None else tile

    # block > 1024 produced corrupt history on T4/triton 3.3 (device-side
    # assert in the decode); re-measured on A100/triton 3.6: BLOCK 2048/4096
    # are partly correct now but all slower, and some warp counts still
    # corrupt, so the clamp stays.
    BLOCK = min(max(triton.next_power_of_2(N), 64), min(block, 1024))
    _cluster_large_kernel[(B,)](
        p4f, masku, px, py, pz, pe, rap, phi, w, nnd, act, nni, ids, cmap,
        hp1, hp2, hch, hd,
        N, 1.0 / (R * R), float(p),
        BLOCK=BLOCK, BI=tile[0], BJ=tile[1],
        num_warps=num_warps,
    )
    # the per-particle jet_idx is a launch-bound pointer-jump decode the hot
    # loop deliberately skips; substructure-feature callers that only read the
    # merge history (e.g. ClusterOutput.splitting_scales) pass decode=False to
    # skip it (n_jets is the cheap beam-merge count either way)
    if decode:
        jet_idx, n_jets = jet_idx_from_history(hp1, hp2, hch, mask)
    else:
        jet_idx = None
        n_jets = (hp2 == -1).sum(1).long()
    return {
        "jet_idx": jet_idx,
        "n_jets": n_jets,
        "hist_p1": hp1,
        "hist_p2": hp2,
        "hist_child": hch,
        "hist_d": hd,
    }
