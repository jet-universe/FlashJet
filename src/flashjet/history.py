"""Decode per-particle jet assignments from the merge history.

The clustering kernels record, per step, the two parent pseudojet ids and the
child id (or a beam merge).  That tree determines the particle->jet mapping,
so the kernel does not need to maintain a destination array in its hot loop
(which costs global-memory traffic every step at large N).  Instead we
rebuild the mapping afterwards with pointer jumping: O(log N) batched gathers.

On CUDA the whole decode runs as a single Triton launch (one program per
event, ping-pong scratch so each jumping round is synchronous like the eager
gathers); it is bitwise-identical to the eager torch-op loop, which remains
the spec and the CPU path.

Ids: initial particles are 0..n-1 in mask order, children continue from n.
"""

import math
import os

import torch

from .kinematics import rap_phi_kt2
from .triton_backend import HAS_TRITON

_compiled = None


def jet_idx_from_history(hist_p1, hist_p2, hist_child, mask):
    """Returns (jet_idx (B, N) int64 per slot, n_jets (B,) int64).

    jet_idx is -1 for padding slots; jets are numbered in beam-merge order,
    matching the torch backend.

    FLASHJET_COMPILE_DECODE=1 routes the (launch-bound, ~70-kernel) decode
    through torch.compile(mode="reduce-overhead"): ~3x faster, identical
    outputs, at the cost of one compile per process and tensor shape.
    dynamic=False keeps CUDA graphs active (automatic dynamic shapes would
    silently disable them); outputs are cloned out of the graph-owned
    buffers so holding them across later calls stays safe.  torch may emit
    a benign "CUDA Graph is empty" UserWarning during the first capture.
    """
    B, N = hist_p1.shape
    device = hist_p1.device
    if N == 0:
        return (torch.full((B, 0), -1, dtype=torch.long, device=device),
                torch.zeros(B, dtype=torch.long, device=device))
    if hist_p1.is_cuda and os.environ.get("FLASHJET_COMPILE_DECODE", "") not in ("", "0"):
        global _compiled
        if _compiled is None:
            _compiled = torch.compile(_decode, mode="reduce-overhead", dynamic=False)
        jet_idx, n_jets = _compiled(hist_p1, hist_p2, hist_child, mask)
        return jet_idx.clone(), n_jets.clone()
    if hist_p1.is_cuda and HAS_TRITON:
        return _decode_triton(hist_p1, hist_p2, hist_child, mask)
    return _decode(hist_p1, hist_p2, hist_child, mask)


def _resolve_parents(hist_p1, hist_p2, hist_child):
    """Pointer-jumped parent map par (B, 2N): par[b, id] = -(jet + 2) when the
    pseudojet `id` roots in a jet (beam-merge order), else a self/dummy fixed
    point.  Shared by the per-particle decode (_decode) and substructure
    features (splitting_scales_from_history) -- the single source of the
    merge-tree -> jet resolution."""
    B, N = hist_p1.shape
    device = hist_p1.device
    M = 2 * N  # id space; max real child id is 2n-2, so M-1 is a free dummy
    dummy = M - 1

    is_pair = hist_p2 >= 0
    is_beam = hist_p2 == -1  # padded steps are -2
    jetnum = is_beam.long().cumsum(1) - 1

    # parent pointers: par[id] = child id (pair), -(jet+2) (beam), self (none)
    par = torch.arange(M, device=device).expand(B, M).clone()
    child = hist_child.long()
    idx1 = torch.where(is_pair, hist_p1.long(), torch.full_like(child, dummy))
    idx2 = torch.where(is_pair, hist_p2.long(), torch.full_like(child, dummy))
    val = torch.where(is_pair, child, torch.full_like(child, dummy))
    par.scatter_(1, idx1, val)
    par.scatter_(1, idx2, val)
    idxb = torch.where(is_beam, hist_p1.long(), torch.full_like(child, dummy))
    valb = torch.where(is_beam, -(jetnum + 2), torch.full_like(child, dummy))
    par.scatter_(1, idxb, valb)
    # all dummy writes stored `dummy` itself, so par[dummy] == dummy (self)

    # pointer jumping: chains halve every round
    for _ in range(max(1, math.ceil(math.log2(M)))):
        hop = torch.gather(par, 1, par.clamp(0, M - 1))
        par = torch.where(par >= 0, hop, par)
    return par


def _decode(hist_p1, hist_p2, hist_child, mask):
    B, N = hist_p1.shape
    M = 2 * N
    par = _resolve_parents(hist_p1, hist_p2, hist_child)
    # map slots -> initial ids (compact in mask order) -> jet number
    ids = mask.long().cumsum(1) - 1
    rooted = torch.gather(par, 1, ids.clamp(0, M - 1))
    jet_idx = torch.where(mask & (rooted < 0), -rooted - 2, torch.full_like(rooted, -1))
    n_jets = (hist_p2 == -1).sum(1).long()
    return jet_idx, n_jets


def splitting_scales_from_history(hist_p1, hist_p2, hist_child, hist_d, n_jets_max=None):
    """Per-jet sequential-recombination splitting scales, in de-clustering order.

    Returns (B, J, S) float (J = n_jets.max() or n_jets_max, S = the largest
    per-jet merge count): out[b, j, 0] is the LAST merge that formed jet j
    (the widest / d_12 splitting), out[b, j, 1] the next (d_23), ..., zero-
    padded past each jet's merge count.  Jets are numbered in beam-merge order,
    ALIGNED with jet_idx and ClusterOutput.jets_p4 (so per-jet features
    concatenate); a caller that sort_jets_by_pt's the jets must apply the same
    permutation here.

    The value is flashjet's merge distance d = min(w_i, w_j) * dR^2 / R^2 with
    w = kt^(2p).  Entry 0 is the LAST merge that built the jet -- equivalently
    the jet's FIRST de-clustering split, the d_12 scale -- entry 1 the next,
    etc.  For the kt algorithm (p=1) d_min is monotonic, so the sequence is
    additionally value-sorted (d_12 >= d_23 >= ...): the exclusive kt scales.
    For Cambridge/Aachen and anti-kt d_min is NOT monotonic (recombination can
    lower it), so the entries are the de-clustering sequence in merge order but
    not value-sorted.  Recover conventional sqrt(d_ij) as (out * R**2).sqrt().
    """
    B, N = hist_p1.shape
    device = hist_p1.device
    n_jets = (hist_p2 == -1).sum(1).long()
    J = int(n_jets.max()) if (n_jets_max is None and B) else (n_jets_max or 0)
    J = max(J, 1)
    if N == 0:
        return torch.zeros(B, J, 0, device=device, dtype=hist_d.dtype)

    M = 2 * N
    par = _resolve_parents(hist_p1, hist_p2, hist_child)
    is_pair = hist_p2 >= 0
    child = hist_child.long()
    # each pair-merge step -> the jet its child roots in (beam-merge order)
    rooted = torch.gather(par, 1, child.clamp(0, M - 1))
    jet = torch.where(is_pair & (rooted < 0), -rooted - 2, torch.full_like(rooted, J))
    valid = is_pair & (jet < J)  # drop beam/pad steps and jets >= J (truncated)
    jet_c = jet.clamp(0, J)  # safe gather/scatter index (sentinel column J)

    # intra-jet de-clustering rank: one-hot cumsum over jets gives the 1-based
    # forward (clustering-order) rank; reverse it so slot 0 is the last merge.
    oh = torch.zeros(B, N, J + 1, device=device, dtype=torch.long)
    oh.scatter_(2, jet_c.unsqueeze(-1), valid.long().unsqueeze(-1))
    cum = oh.cumsum(1)
    fwd = cum.gather(2, jet_c.unsqueeze(-1)).squeeze(-1)
    counts = cum[:, -1, :]                       # (B, J+1) merges per jet
    rev = (counts.gather(1, jet_c) - fwd).clamp(min=0)  # 0 == last merge

    S = max(int(counts[:, :J].max()), 1)
    # scatter d into out[b, jet, rev] with (jet, rev) flattened.  invalid steps
    # (beam/pad/truncated) go to a per-step UNIQUE tail slot J*S + step, so the
    # scatter has NO duplicate indices in any row -- deterministic and safe
    # under torch.use_deterministic_algorithms(True) -- and the tail is dropped.
    steps = torch.arange(N, device=device).expand(B, N)
    flat = torch.where(valid, jet_c * S + rev.clamp(0, S - 1), J * S + steps)
    out_flat = torch.zeros(B, J * S + N, device=device, dtype=hist_d.dtype)
    out_flat.scatter_(1, flat, hist_d)
    return out_flat[:, : J * S].view(B, J, S)


def _resolve_roots(hist_p1, hist_p2, hist_child, keep_pair):
    """Pointer-jumped root map for an arbitrary *sub-forest* of the merge tree.

    Like `_resolve_parents`, but only the pair-merges flagged True in
    `keep_pair (B, N)` are followed; a *cut* pair-merge (keep_pair == False)
    does not link its parents to their child, so each parent that has no kept
    consumer becomes its own root.  Beam merges are ignored here -- roots are
    plain pseudojet ids (0..2N-1), NOT jet numbers -- so this drives exclusive
    clustering (stop-early forests) rather than the inclusive jet decode.

    Returns par (B, 2N) int64 where par[b, id] is the id of the highest pseudojet
    reachable from `id` through kept merges only (a fixed point == its own root).
    """
    B, N = hist_p1.shape
    device = hist_p1.device
    M = 2 * N
    dummy = M - 1

    is_pair = hist_p2 >= 0
    follow = is_pair & keep_pair  # only kept pair-merges relink parents -> child

    par = torch.arange(M, device=device).expand(B, M).clone()
    child = hist_child.long()
    idx1 = torch.where(follow, hist_p1.long(), torch.full_like(child, dummy))
    idx2 = torch.where(follow, hist_p2.long(), torch.full_like(child, dummy))
    val = torch.where(follow, child, torch.full_like(child, dummy))
    par.scatter_(1, idx1, val)
    par.scatter_(1, idx2, val)
    # all dummy writes store `dummy` itself, so par[dummy] == dummy (self)

    for _ in range(max(1, math.ceil(math.log2(M)))):
        hop = torch.gather(par, 1, par.clamp(0, M - 1))
        # every id is >= 0 here (no beam sentinels), so keep hopping to the root
        par = hop
    return par


def exclusive_jets_from_history(hist_p1, hist_p2, hist_child, hist_d, mask,
                                n_jets=None, d_cut=None):
    """Exclusive-jet particle assignment: undo the last merges of the sequence.

    Exactly one of `n_jets` (stop when this many exclusive jets remain) or
    `d_cut` (undo every pair-merge with d >= d_cut) must be given.  This is the
    kt-family *exclusive* jet definition (FastJet ClusterSequence::
    exclusive_jets / exclusive_jets_ycut) read off the recorded sequence: a
    pair-merge is *kept* only if it happened before the stopping point, and each
    particle is assigned to the highest pseudojet it reaches through kept
    merges.  d_cut compares against the SAME d stored in hist_d
    (min(w_i,w_j) * dR^2 / R^2); for a p_t-style y_cut pass R**2 * d.

    Returns (excl_idx (B, N) int64, n_excl (B,) int64): jet index per particle
    in ascending-root order, -1 for padding.  For the trivial cut
    (n_jets == n_inclusive, or d_cut past the largest d) this reproduces the
    inclusive jet_idx partition (same particles grouped), though jet *numbering*
    follows root order, not beam-merge order.
    """
    if (n_jets is None) == (d_cut is None):
        raise ValueError("pass exactly one of n_jets= or d_cut=")
    B, N = hist_p1.shape
    device = hist_p1.device
    is_pair = hist_p2 >= 0

    if d_cut is not None:
        keep_pair = is_pair & (hist_d < d_cut)
    else:
        # exclusive-n_jets: undo the LAST k pair-merges of the recorded sequence
        # (keep the prefix of k = n_init - n_jets merges), leaving n_jets jets.
        # A prefix is always downward-closed in the tree, so this is a valid
        # sub-forest.  Exclusive jets are a *kt* concept: for kt (p=1) d is
        # monotonic, so the recorded order IS ascending d and undoing the last k
        # == undoing the largest-d k (FastJet exclusive_jets(njets)).  For
        # anti-kt / C-A d is non-monotonic and FastJet does not define exclusive
        # jets meaningfully; we still return the well-defined prefix partition.
        # k == n_pair -> inclusive partition; k == 0 -> singletons.
        pair_rank = is_pair.long().cumsum(1)               # 1..n_pair over pairs
        n_init = mask.sum(1, keepdim=True)
        target = torch.as_tensor(n_jets, device=device).clamp(min=1)
        k_keep = (n_init - target).clamp(min=0)            # (B,1) merges to keep
        keep_pair = is_pair & (pair_rank <= k_keep)

    par = _resolve_roots(hist_p1, hist_p2, hist_child, keep_pair)

    # slot -> initial id (mask order) -> its kept-forest root pseudojet id
    ids = mask.long().cumsum(1) - 1
    M = 2 * N
    root = torch.gather(par, 1, ids.clamp(0, M - 1))
    root = torch.where(mask, root, torch.full_like(root, -1))

    # dense-number the distinct roots per event (ascending id) -> jet index
    excl_idx, n_excl = _dense_number_roots(root, mask)
    return excl_idx, n_excl


def _dense_number_roots(root, mask):
    """Map per-particle root ids (B, N; -1 pad) to a compact 0..J-1 jet index
    per event (ascending root id), returning (idx (B, N) int64, n (B,) int64)."""
    B, N = root.shape
    device = root.device
    big = torch.iinfo(torch.long).max
    keyed = torch.where(mask, root, torch.full_like(root, big))
    order = keyed.argsort(dim=1, stable=True)
    sorted_root = torch.gather(keyed, 1, order)
    # new group whenever the sorted root id changes (and is a real particle)
    is_real = sorted_root < big
    newgrp = torch.ones(B, N, dtype=torch.long, device=device)
    newgrp[:, 1:] = (sorted_root[:, 1:] != sorted_root[:, :-1]).long()
    grp = (newgrp * is_real.long()).cumsum(1) - 1        # 0-based per event
    grp = torch.where(is_real, grp, torch.full_like(grp, -1))
    excl_idx = torch.empty_like(grp).scatter_(1, order, grp)
    n_excl = (grp.max(dim=1).values + 1).clamp(min=0)
    return excl_idx, n_excl


def _pseudojet_p4(hist_p1, hist_p2, hist_child, mask, p4):
    """Reconstruct every pseudojet's E-scheme 4-momentum, keyed by id.

    Returns pj (B, 2N, 4): pj[b, id] is the summed p4 of pseudojet `id`
    (initial particles 0..n-1 in mask order, merged children n.. in step order),
    zero for unused ids.  E-scheme recombination is a plain 4-vector sum, so a
    child's p4 is its two parents' -- filled by a forward scan over merge steps
    (parents always have smaller ids than their child, so one pass suffices).
    Batched over events; the loop is over the <= N merge steps, mirroring the
    clustering loop's structure.
    """
    B, N, _ = p4.shape
    device = p4.device
    dt = p4.dtype if p4.dtype in (torch.float32, torch.float64) else torch.float32
    M = 2 * N
    pj = torch.zeros(B, M, 4, dtype=dt, device=device)

    # seed leaves: initial id = mask-order index; place each real particle's p4
    ids = (mask.long().cumsum(1) - 1).clamp(0, M - 1)
    pj.scatter_(1, ids.unsqueeze(-1).expand(B, N, 4),
                torch.where(mask.unsqueeze(-1), p4.to(dt), torch.zeros_like(p4, dtype=dt)))

    barange = torch.arange(B, device=device)
    is_pair = hist_p2 >= 0
    for s in range(N):
        pair = is_pair[:, s]
        if not bool(pair.any()):
            continue
        c = hist_child[:, s].clamp(0, M - 1)
        a = hist_p1[:, s].clamp(0, M - 1)
        b = hist_p2[:, s].clamp(0, M - 1)
        summed = pj[barange, a] + pj[barange, b]
        pj[barange, c] = torch.where(pair.unsqueeze(-1), summed, pj[barange, c])
    return pj


def lund_coordinates_from_history(hist_p1, hist_p2, hist_child, hist_d, mask, p4,
                                  R, n_jets_max=None):
    """Per-jet, per-split Lund-plane coordinates from the merge history.

    Returns (B, J, S, C) float in the SAME per-jet de-clustering order as
    `splitting_scales_from_history` (slot 0 = the jet's first / widest split),
    zero-padded past each jet's split count. Channels C = 6::

        0 z      = min(pt_i, pt_j) / (pt_i + pt_j)          in (0, 0.5]
        1 dR     = sqrt(dy^2 + dphi^2) of the two parents
        2 kt     = min(pt_i, pt_j) * dR                     (the Lund kt)
        3 ln(1/dR)
        4 ln(kt)
        5 d      = hist_d for that split (== splitting_scales entry; sanity tie)

    Jets are aligned with jet_idx / jets_p4 (beam-merge order); a caller that
    sort_jets_by_pt's the jets applies the same permutation here.  pt is the
    transverse momentum of each parent pseudojet (E-scheme sum), recovered from
    the tree, so this needs the input p4 and mask (unlike splitting_scales).
    """
    B, N = hist_p1.shape
    device = hist_p1.device
    n_jets = (hist_p2 == -1).sum(1).long()
    J = int(n_jets.max()) if (n_jets_max is None and B) else (n_jets_max or 0)
    J = max(J, 1)
    C = 6
    if N == 0:
        return torch.zeros(B, J, 0, C, device=device, dtype=hist_d.dtype)

    M = 2 * N
    par = _resolve_parents(hist_p1, hist_p2, hist_child)
    is_pair = hist_p2 >= 0
    child = hist_child.long()
    rooted = torch.gather(par, 1, child.clamp(0, M - 1))
    jet = torch.where(is_pair & (rooted < 0), -rooted - 2, torch.full_like(rooted, J))
    valid = is_pair & (jet < J)
    jet_c = jet.clamp(0, J)

    # de-clustering rank per jet (0 == last merge == first split), reusing the
    # one-hot cumsum layout of splitting_scales_from_history
    oh = torch.zeros(B, N, J + 1, device=device, dtype=torch.long)
    oh.scatter_(2, jet_c.unsqueeze(-1), valid.long().unsqueeze(-1))
    cum = oh.cumsum(1)
    fwd = cum.gather(2, jet_c.unsqueeze(-1)).squeeze(-1)
    counts = cum[:, -1, :]
    rev = (counts.gather(1, jet_c) - fwd).clamp(min=0)
    S = max(int(counts[:, :J].max()), 1)

    # parent kinematics at each split
    pj = _pseudojet_p4(hist_p1, hist_p2, hist_child, mask, p4)  # (B, M, 4)
    a = hist_p1.long().clamp(0, M - 1)
    b = hist_p2.long().clamp(0, M - 1)
    pa = torch.gather(pj, 1, a.unsqueeze(-1).expand(B, N, 4))
    pb = torch.gather(pj, 1, b.unsqueeze(-1).expand(B, N, 4))
    rap_a, phi_a, kt2_a = rap_phi_kt2(pa[..., 0], pa[..., 1], pa[..., 2], pa[..., 3], xp=torch)
    rap_b, phi_b, kt2_b = rap_phi_kt2(pb[..., 0], pb[..., 1], pb[..., 2], pb[..., 3], xp=torch)
    pt_a, pt_b = kt2_a.clamp_min(0).sqrt(), kt2_b.clamp_min(0).sqrt()
    dphi = (phi_a - phi_b).abs()
    dphi = torch.minimum(dphi, 2 * math.pi - dphi)
    dR = ((rap_a - rap_b) ** 2 + dphi ** 2).clamp_min(0).sqrt()
    pt_min = torch.minimum(pt_a, pt_b)
    pt_sum = (pt_a + pt_b).clamp_min(1e-30)
    z = pt_min / pt_sum
    kt = pt_min * dR
    eps = 1e-30
    chans = torch.stack([
        z, dR, kt,
        torch.log(1.0 / dR.clamp_min(eps)),
        torch.log(kt.clamp_min(eps)),
        hist_d,
    ], dim=-1)  # (B, N, C)

    # scatter each valid split into out[b, jet, rev, :]; invalid -> unique tail
    steps = torch.arange(N, device=device).expand(B, N)
    flat = torch.where(valid, jet_c * S + rev.clamp(0, S - 1), J * S + steps)  # (B,N)
    out_flat = torch.zeros(B, J * S + N, C, device=device, dtype=chans.dtype)
    out_flat.scatter_(1, flat.unsqueeze(-1).expand(B, N, C), chans)
    return out_flat[:, : J * S].view(B, J, S, C)


def _jet_roots(hist_p1, hist_p2, mask, n_jets_max=None):
    """Per-jet root pseudojet id (B, J) int64 in beam-merge order, -1 for
    padded jet columns.  The root of jet j is the pseudojet that beam-merges at
    the (j+1)-th beam step."""
    B, N = hist_p1.shape
    device = hist_p1.device
    is_beam = hist_p2 == -1
    n_jets = is_beam.sum(1).long()
    J = int(n_jets.max()) if (n_jets_max is None and B) else (n_jets_max or 0)
    J = max(J, 1)
    jnum = is_beam.long().cumsum(1) - 1                    # beam step -> jet #
    col = torch.where(is_beam & (jnum < J), jnum, torch.full_like(jnum, J))
    roots = torch.full((B, J + 1), -1, dtype=torch.long, device=device)
    roots.scatter_(1, col, hist_p1.long())
    return roots[:, :J], n_jets


def groom_from_history(hist_p1, hist_p2, hist_child, hist_d, mask, p4, R,
                       z_cut=0.1, beta=0.0, mu=None, n_jets_max=None):
    """Soft-drop / mass-drop grooming by declustering each jet's tree.

    Walks every jet from its root down the HARDER (higher-pt) branch, undoing
    the widest split first (this is the C/A declustering picture, and is exact
    for any recorded tree since we follow the stored merge structure).  At each
    node with parents i, j it tests the soft-drop condition::

        z > z_cut * (dR / R)**beta ,   z = min(pt_i,pt_j)/(pt_i+pt_j)

    (beta=0 is the modified Mass-Drop Tagger / mMDT).  If it passes, that node is
    the groomed jet and the walk stops; otherwise the softer parent is dropped
    and the walk continues into the harder parent.  A jet that declusters to a
    single particle without ever passing is *untagged*.

    If `mu` is given, the additional mass-drop requirement max(m_i,m_j) < mu*m
    (m = mass of the current node) must also hold for a node to pass -- the
    original Mass-Drop Tagger.

    Returns dict of tensors (all B x J, beam-merge order, aligned with
    jets_p4 / splitting_scales)::

        groomed_p4 (B, J, 4): 4-momentum of the groomed subjet (0 if untagged)
        tagged     (B, J) bool: whether a split passed the condition
        z, dR, mu_split (B, J): the passing split's z, dR, and mass ratio
                                max(m_i,m_j)/m (0 where untagged)
        n_drop     (B, J) int64: number of soft branches dropped before passing
    """
    B, N = hist_p1.shape
    device = hist_p1.device
    dt = p4.dtype if p4.dtype in (torch.float32, torch.float64) else torch.float32
    M = 2 * N
    roots, n_jets = _jet_roots(hist_p1, hist_p2, mask, n_jets_max)
    J = roots.shape[1]

    pj = _pseudojet_p4(hist_p1, hist_p2, hist_child, mask, p4)  # (B, M, 4)

    # per-id parent lookup: par1_of[id], par2_of[id] (-1 if id is a leaf).
    # M is a spare column: route beam/pad steps there so col 0 (a real leaf)
    # is never clobbered; then drop the spare.
    par1_of = torch.full((B, M + 1), -1, dtype=torch.long, device=device)
    par2_of = torch.full((B, M + 1), -1, dtype=torch.long, device=device)
    is_pair = hist_p2 >= 0
    ch = torch.where(is_pair, hist_child.long(), torch.full_like(hist_child, M))
    par1_of.scatter_(1, ch, torch.where(is_pair, hist_p1.long(), torch.full_like(ch, -1)))
    par2_of.scatter_(1, ch, torch.where(is_pair, hist_p2.long(), torch.full_like(ch, -1)))
    par1_of, par2_of = par1_of[:, :M], par2_of[:, :M]

    cur = roots.clamp(min=0)                       # current node per jet (B, J)
    alive = roots >= 0                             # jet column in use, not yet stopped
    tagged = torch.zeros(B, J, dtype=torch.bool, device=device)
    out_p4 = torch.zeros(B, J, 4, dtype=dt, device=device)
    out_z = torch.zeros(B, J, dtype=dt, device=device)
    out_dR = torch.zeros(B, J, dtype=dt, device=device)
    out_mu = torch.zeros(B, J, dtype=dt, device=device)
    n_drop = torch.zeros(B, J, dtype=torch.long, device=device)

    def gather_id(arr, idx):  # arr (B, M[,4]), idx (B, J) -> (B, J[,4])
        if arr.dim() == 3:
            return torch.gather(arr, 1, idx.clamp(0, M - 1).unsqueeze(-1).expand(B, J, 4))
        return torch.gather(arr, 1, idx.clamp(0, M - 1))

    def mass(pv):
        m2 = pv[..., 3] ** 2 - pv[..., 0] ** 2 - pv[..., 1] ** 2 - pv[..., 2] ** 2
        return m2.clamp_min(0).sqrt()

    for _ in range(N):                              # <= N declustering levels
        if not bool(alive.any()):
            break
        i = gather_id(par1_of, cur)
        j = gather_id(par2_of, cur)
        is_leaf = (i < 0) | (j < 0)
        # a leaf that is still alive is untagged -> stop it, keep out_p4 = 0
        alive = alive & ~is_leaf

        pi = gather_id(pj, i.clamp(min=0))
        pj_ = gather_id(pj, j.clamp(min=0))
        ri, phii, k2i = rap_phi_kt2(pi[..., 0], pi[..., 1], pi[..., 2], pi[..., 3], xp=torch)
        rj, phij, k2j = rap_phi_kt2(pj_[..., 0], pj_[..., 1], pj_[..., 2], pj_[..., 3], xp=torch)
        pti, ptj = k2i.clamp_min(0).sqrt(), k2j.clamp_min(0).sqrt()
        dphi = (phii - phij).abs()
        dphi = torch.minimum(dphi, 2 * math.pi - dphi)
        dR = ((ri - rj) ** 2 + dphi ** 2).clamp_min(0).sqrt()
        ptmin = torch.minimum(pti, ptj)
        z = ptmin / (pti + ptj).clamp_min(1e-30)
        passes = z > z_cut * (dR / R).clamp_min(1e-30) ** beta

        m_cur = mass(gather_id(pj, cur)).clamp_min(1e-30)
        mu_split = torch.maximum(mass(pi), mass(pj_)) / m_cur
        if mu is not None:
            passes = passes & (mu_split < mu)

        newly = alive & passes
        tagged = tagged | newly
        out_p4 = torch.where(newly.unsqueeze(-1), gather_id(pj, cur), out_p4)
        out_z = torch.where(newly, z, out_z)
        out_dR = torch.where(newly, dR, out_dR)
        out_mu = torch.where(newly, mu_split, out_mu)
        alive = alive & ~passes

        # otherwise descend into the harder (higher-pt) parent
        harder = torch.where(pti >= ptj, i, j)
        n_drop = torch.where(alive, n_drop + 1, n_drop)
        cur = torch.where(alive, harder, cur)

    return {
        "groomed_p4": out_p4,
        "tagged": tagged,
        "z": out_z,
        "dR": out_dR,
        "mu_split": out_mu,
        "n_drop": n_drop,
    }


if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _decode_kernel(
        HP1, HP2, HCH, MSK,             # (B, N) i64 history, (B, N) i8 mask
        PARA, PARB,                     # (B, 2N) i32 scratch (ping-pong)
        JIDX, NJ,                       # outputs (B, N) i64, (B,) i64
        N, M, rounds2,
        BLOCK: tl.constexpr,
    ):
        b = tl.program_id(0)
        base = b * N
        base2 = b * M

        # ---- phase 0: par[id] = id (self) over the full id space ----
        for c in range(0, M, BLOCK):
            offs = c + tl.arange(0, BLOCK)
            m = offs < M
            tl.store(PARA + base2 + offs, offs.to(tl.int32), mask=m)

        tl.debug_barrier()

        # ---- phase 1: apply merge steps; parent ids are unique, so the
        # scatters never collide.  beam value is the eager -(jetnum + 2). ----
        nbeam = N * 0
        for c in range(0, N, BLOCK):
            offs = c + tl.arange(0, BLOCK)
            m = offs < N
            p1 = tl.load(HP1 + base + offs, mask=m, other=0)
            p2 = tl.load(HP2 + base + offs, mask=m, other=-2)
            ch = tl.load(HCH + base + offs, mask=m, other=0)
            is_pair = p2 >= 0
            is_beam = p2 == -1  # padded steps are -2
            jetnum = nbeam + tl.cumsum(is_beam.to(tl.int32), axis=0) - 1
            tl.store(PARA + base2 + p1, ch.to(tl.int32), mask=m & is_pair)
            tl.store(PARA + base2 + p2, ch.to(tl.int32), mask=m & is_pair)
            tl.store(PARA + base2 + p1, -(jetnum + 2), mask=m & is_beam)
            nbeam += tl.sum(is_beam.to(tl.int32), axis=0)
        tl.store(NJ + b, nbeam.to(tl.int64))

        tl.debug_barrier()

        # ---- phase 2: pointer jumping.  Each half writes only the other
        # buffer, so a round is synchronous like the eager full-tensor
        # gather; rounding the count up to A->B->A pairs is harmless (the
        # eager round count already reaches the fixed point). ----
        for _ in range(rounds2):
            for c in range(0, M, BLOCK):
                offs = c + tl.arange(0, BLOCK)
                m = offs < M
                p = tl.load(PARA + base2 + offs, mask=m, other=0)
                addr = tl.minimum(tl.maximum(p, 0), M - 1)
                hop = tl.load(PARA + base2 + addr, mask=m, other=0)
                tl.store(PARB + base2 + offs, tl.where(p >= 0, hop, p), mask=m)
            tl.debug_barrier()
            for c in range(0, M, BLOCK):
                offs = c + tl.arange(0, BLOCK)
                m = offs < M
                p = tl.load(PARB + base2 + offs, mask=m, other=0)
                addr = tl.minimum(tl.maximum(p, 0), M - 1)
                hop = tl.load(PARB + base2 + addr, mask=m, other=0)
                tl.store(PARA + base2 + offs, tl.where(p >= 0, hop, p), mask=m)
            tl.debug_barrier()

        # ---- phase 3: slots -> initial ids (mask order) -> jet number ----
        cnt = N * 0
        for c in range(0, N, BLOCK):
            offs = c + tl.arange(0, BLOCK)
            m = offs < N
            msk = tl.load(MSK + base + offs, mask=m, other=0) != 0
            ids = cnt + tl.cumsum(msk.to(tl.int32), axis=0) - 1
            cnt += tl.sum(msk.to(tl.int32), axis=0)
            addr = tl.minimum(tl.maximum(ids, 0), M - 1)
            rooted = tl.load(PARA + base2 + addr, mask=m, other=0)
            jidx = tl.where(msk & (rooted < 0), -rooted - 2, -1)
            tl.store(JIDX + base + offs, jidx.to(tl.int64), mask=m)


def _decode_triton(hist_p1, hist_p2, hist_child, mask):
    """Single-launch CUDA decode; bitwise-identical to _decode (the spec)."""
    B, N = hist_p1.shape
    device = hist_p1.device
    M = 2 * N
    par_a = torch.empty(B, M, dtype=torch.int32, device=device)
    par_b = torch.empty(B, M, dtype=torch.int32, device=device)
    jet_idx = torch.empty(B, N, dtype=torch.long, device=device)
    n_jets = torch.empty(B, dtype=torch.long, device=device)
    rounds2 = (max(1, math.ceil(math.log2(M))) + 1) // 2
    BLOCK = max(64, min(1024, triton.next_power_of_2(M)))
    _decode_kernel[(B,)](
        hist_p1.contiguous(), hist_p2.contiguous(), hist_child.contiguous(),
        mask.to(torch.int8).contiguous(),
        par_a, par_b, jet_idx, n_jets,
        N, M, rounds2,
        BLOCK=BLOCK, num_warps=4,
    )
    return jet_idx, n_jets
