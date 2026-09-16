"""Fused Triton kernel for batched generalized-kt clustering.

One Triton program clusters one event end-to-end: all particle state lives in
registers, the sequential recombination loop (which is inherently serial per
event) runs inside the kernel, and the batch dimension supplies the
parallelism.  This avoids the N_max kernel launches and the (B, N, N)
global-memory traffic per step that the pure-torch backend pays.

Limitations of this v1 kernel:
  * N (padded particles per event) must be <= 128 (the pairwise distance tile
    is materialized in registers).  Larger events fall back to the torch
    backend automatically via flashjet.cluster().
  * float32 only.  Merge ordering can differ from the float64 reference for
    near-degenerate distances; physics-level outputs agree.
"""

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False

BEAM = -1
PAD = -2
MAX_RAP = 1e5
MAX_BLOCK = 128

if HAS_TRITON:
    # atan2 lives in libdevice; its import path moved across triton versions.
    try:
        from triton.language.extra import libdevice as _tld
    except ImportError:  # triton < 3.0
        from triton.language import math as _tld

    @triton.jit
    def _cluster_kernel(
        PX, PY, PZ, PE,            # (B, N) float32, contiguous
        ACT, NINIT,                # (B, N) int8 mask, (B,) int32 particle counts
        JETIDX, NJETS,             # (B, N) int32 out, (B,) int32 out
        HP1, HP2, HCH, HD,         # (B, N) int32/int32/int32/float32 out
        N, inv_R2, p_exp,          # runtime scalars
        BLOCK: tl.constexpr,
    ):
        b = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        lane = offs < N
        base = b * N

        px = tl.load(PX + base + offs, mask=lane, other=0.0)
        py = tl.load(PY + base + offs, mask=lane, other=0.0)
        pz = tl.load(PZ + base + offs, mask=lane, other=0.0)
        pe = tl.load(PE + base + offs, mask=lane, other=0.0)
        act = tl.load(ACT + base + offs, mask=lane, other=0) != 0
        init_mask = act
        n = tl.load(NINIT + b)

        # initial pseudojet ids, compact in mask order
        ids = tl.cumsum(act.to(tl.int32), axis=0) - 1
        ids = tl.where(act, ids, -2)  # PAD (literals: jit code can't see module globals)
        next_id = n
        jcount = n * 0  # 0-d int32 zero (version-portable, unlike tl.zeros(()))
        dest = offs.to(tl.int32)
        jet_idx = tl.full((BLOCK,), -1, dtype=tl.int32)  # BEAM

        INF = float("inf")
        TWO_PI = 6.283185307179586

        for step in range(0, n):
            # kinematics of the current pseudojets (cheap, fully vectorized)
            kt2 = px * px + py * py
            phi = _tld.atan2(py, px)
            m2 = tl.maximum(pe * pe - pz * pz - kt2, 0.0)
            apz = tl.abs(pz)
            beamlike = (kt2 + m2) <= 0.0
            ratio = tl.where(beamlike, 1.0, (kt2 + m2) / ((pe + apz) * (pe + apz)))
            half_log = 0.5 * tl.log(ratio)
            rap = tl.where(pz >= 0, -half_log, half_log)
            rap = tl.where(beamlike, tl.where(pz >= 0, 1e5 + apz, -(1e5 + apz)), rap)
            w = tl.exp(p_exp * tl.log(tl.maximum(kt2, 1e-30)))  # kt^(2p)

            diB = tl.where(act, w, INF)
            drap = rap[:, None] - rap[None, :]
            dphi = tl.abs(phi[:, None] - phi[None, :])
            dphi = tl.minimum(dphi, TWO_PI - dphi)
            ok = act[:, None] & act[None, :] & (offs[:, None] != offs[None, :])
            dij = tl.minimum(w[:, None], w[None, :]) * (drap * drap + dphi * dphi) * inv_R2
            dij = tl.where(ok, dij, INF)

            row_min = tl.min(dij, axis=1)
            best = tl.minimum(row_min, diB)
            gmin = tl.min(best, axis=0)
            i_sel = tl.min(tl.where(best == gmin, offs, BLOCK), axis=0)
            sel_i = offs == i_sel

            diB_i = tl.min(tl.where(sel_i, diB, INF), axis=0)
            row_i = tl.min(tl.where(sel_i[:, None], dij, INF), axis=0)
            row_i_min = tl.min(row_i, axis=0)
            is_pair = row_i_min < diB_i
            j_sel = tl.min(tl.where(row_i == row_i_min, offs, BLOCK), axis=0)
            j_sel = tl.where(is_pair, j_sel, i_sel)
            sel_j = offs == j_sel

            # history
            id_i = tl.min(tl.where(sel_i, ids, 2147483647), axis=0)
            id_j = tl.min(tl.where(sel_j, ids, 2147483647), axis=0)
            tl.store(HP1 + base + step, id_i)
            tl.store(HP2 + base + step, tl.where(is_pair, id_j, -1))
            tl.store(HCH + base + step, tl.where(is_pair, next_id, -1))
            tl.store(HD + base + step, gmin)

            # pair merge: E-scheme sum written into slot i, slot j dies
            both = sel_i | sel_j
            spx = tl.sum(tl.where(both, px, 0.0), axis=0)
            spy = tl.sum(tl.where(both, py, 0.0), axis=0)
            spz = tl.sum(tl.where(both, pz, 0.0), axis=0)
            spe = tl.sum(tl.where(both, pe, 0.0), axis=0)
            write_i = sel_i & is_pair
            px = tl.where(write_i, spx, px)
            py = tl.where(write_i, spy, py)
            pz = tl.where(write_i, spz, pz)
            pe = tl.where(write_i, spe, pe)
            ids = tl.where(write_i, next_id, ids)
            next_id += is_pair.to(tl.int32)
            dest = tl.where(is_pair & (dest == j_sel), i_sel.to(tl.int32), dest)

            # beam merge: slot i becomes a jet owning everything routed to it
            owned = (dest == i_sel) & init_mask
            jet_idx = tl.where((~is_pair) & owned, jcount, jet_idx)
            jcount += (~is_pair).to(tl.int32)

            act = act & ~tl.where(is_pair, sel_j, sel_i)

        tl.store(JETIDX + base + offs, jet_idx, mask=lane)
        tl.store(NJETS + b, jcount)


def cluster_batch_triton(p4: torch.Tensor, mask: torch.Tensor, R: float, p: float):
    """Triton counterpart of torch_backend.cluster_batch_torch (same outputs)."""
    if not HAS_TRITON:
        raise RuntimeError("triton is not installed")
    if not p4.is_cuda:
        raise RuntimeError("triton backend requires CUDA tensors")
    B, N, _ = p4.shape
    if N > MAX_BLOCK:
        raise ValueError(f"triton backend supports N <= {MAX_BLOCK}, got {N}")

    BLOCK = max(triton.next_power_of_2(N), 16)
    p4f = p4.to(torch.float32).contiguous()
    px, py, pz, pe = (p4f[..., i].contiguous() for i in range(4))
    act = mask.to(torch.int8).contiguous()
    ninit = mask.sum(1).to(torch.int32).contiguous()
    dev = p4.device

    jet_idx = torch.full((B, N), -1, dtype=torch.int32, device=dev)
    n_jets = torch.zeros(B, dtype=torch.int32, device=dev)
    hp1 = torch.full((B, N), -2, dtype=torch.int32, device=dev)
    hp2 = torch.full((B, N), -2, dtype=torch.int32, device=dev)
    hch = torch.full((B, N), -2, dtype=torch.int32, device=dev)
    hd = torch.zeros((B, N), dtype=torch.float32, device=dev)

    _cluster_kernel[(B,)](
        px, py, pz, pe, act, ninit,
        jet_idx, n_jets, hp1, hp2, hch, hd,
        N, 1.0 / (R * R), float(p),
        BLOCK=BLOCK,
    )
    return {
        "jet_idx": jet_idx.long(),
        "n_jets": n_jets.long(),
        "hist_p1": hp1.long(),
        "hist_p2": hp2.long(),
        "hist_child": hch.long(),
        "hist_d": hd,
    }
