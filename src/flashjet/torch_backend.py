"""Batched generalized-kt clustering in pure PyTorch (CPU or CUDA).

The whole batch is clustered in lock-step: every iteration of the loop
performs exactly one merge (pair or beam) in every still-active event, fully
vectorized over the batch.  An event with n particles finishes in exactly n
iterations, so the loop runs at most N_max times.

The clustering itself runs under no_grad (the combinatorial assignment is not
differentiable anyway); differentiable jet momenta are recovered afterwards
with a scatter_add over the returned particle->jet assignment, which is exact
for E-scheme recombination.
"""

import math

import torch

from .kinematics import rap_phi_kt2

BEAM = -1
PAD = -2


@torch.no_grad()
def cluster_batch_torch(p4: torch.Tensor, mask: torch.Tensor, R: float, p: float):
    """Cluster a padded batch of events.

    Args:
        p4:   (B, N, 4) float tensor, columns px, py, pz, E.
        mask: (B, N) bool tensor, True for real particles.
        R:    jet radius.
        p:    generalized-kt exponent (-1 anti-kt, 0 C/A, 1 kt).

    Returns dict of tensors (all on p4.device):
        jet_idx   (B, N)  int64: index of the jet each particle ends up in,
                          ordered by beam-merge order; -1 for padding.
        n_jets    (B,)    int64: number of jets per event.
        hist_p1   (B, N)  int64: first parent pseudojet id of merge step s.
        hist_p2   (B, N)  int64: second parent id, BEAM (-1) for beam merges,
                          PAD (-2) for steps past the end of the event.
        hist_child(B, N)  int64: id of the produced pseudojet (BEAM for beam
                          merges).  Initial particles are numbered 0..n-1 in
                          mask order; merged pseudojets continue from n.
        hist_d    (B, N)  float: distance d_min of each merge step.
    """
    B, N, _ = p4.shape
    device = p4.device
    dt = p4.dtype if p4.dtype in (torch.float32, torch.float64) else torch.float32
    INF = torch.finfo(dt).max

    work = p4.to(dt).clone()
    active = mask.clone()
    barange = torch.arange(B, device=device)
    narange = torch.arange(N, device=device)

    # pseudojet ids: initial particles numbered compactly in mask order
    ids = (mask.cumsum(1) - 1).long()
    ids[~mask] = PAD
    next_id = mask.sum(1).long()  # (B,)

    # 'dest' tracks which active slot currently owns each initial particle
    dest = narange.expand(B, N).clone()
    jet_idx = torch.full((B, N), BEAM, dtype=torch.long, device=device)
    n_jets = torch.zeros(B, dtype=torch.long, device=device)

    hist_p1 = torch.full((B, N), PAD, dtype=torch.long, device=device)
    hist_p2 = torch.full((B, N), PAD, dtype=torch.long, device=device)
    hist_child = torch.full((B, N), PAD, dtype=torch.long, device=device)
    hist_d = torch.zeros((B, N), dtype=dt, device=device)

    R2 = R * R
    two_pi = 2 * math.pi

    for step in range(N):
        alive = active.any(1)
        if not bool(alive.any()):
            break

        px, py, pz, E = work.unbind(-1)
        rap, phi, kt2 = rap_phi_kt2(px, py, pz, E, xp=torch)
        w = kt2.clamp_min(1e-30) ** p  # floor matches the other clustering backends
        diB = torch.where(active, w, torch.full_like(w, INF))

        drap = rap.unsqueeze(2) - rap.unsqueeze(1)
        dphi = (phi.unsqueeze(2) - phi.unsqueeze(1)).abs()
        dphi = torch.minimum(dphi, two_pi - dphi)
        dij = torch.minimum(w.unsqueeze(2), w.unsqueeze(1)) * (drap.square() + dphi.square()) / R2
        pair_ok = active.unsqueeze(2) & active.unsqueeze(1)
        pair_ok &= ~torch.eye(N, dtype=torch.bool, device=device)
        dij = torch.where(pair_ok, dij, torch.full_like(dij, INF))

        # per-slot candidate with the beam merge as the default: a pair
        # displaces it only on strictly smaller d (FastJet keeps NN = NULL
        # at dist == R^2), slot and partner ties break toward low indices
        d_pair, jj_pair = dij.min(dim=2)
        cand = torch.minimum(d_pair, diB)
        ii = cand.argmin(1)
        dmin = cand.gather(1, ii.unsqueeze(1)).squeeze(1)

        sel = ii.unsqueeze(1)
        is_pair = d_pair.gather(1, sel).squeeze(1) < diB.gather(1, sel).squeeze(1)
        jj = torch.where(is_pair, jj_pair.gather(1, sel).squeeze(1), ii)

        do_pair = is_pair & alive
        do_beam = (~is_pair) & alive

        pi = work[barange, ii]
        pj = work[barange, jj]
        work[barange, ii] = torch.where(do_pair.unsqueeze(1), pi + pj, pi)

        # record history before ids are overwritten
        sel_p1 = ids[barange, ii]
        sel_p2 = torch.where(do_pair, ids[barange, jj], torch.full_like(ii, BEAM))
        child = torch.where(do_pair, next_id, torch.full_like(ii, BEAM))
        hist_p1[:, step] = torch.where(alive, sel_p1, hist_p1[:, step])
        hist_p2[:, step] = torch.where(alive, sel_p2, hist_p2[:, step])
        hist_child[:, step] = torch.where(alive, child, hist_child[:, step])
        hist_d[:, step] = torch.where(alive, dmin, hist_d[:, step])

        # pair merge: combined pseudojet lives in slot ii, slot jj dies
        ids[barange, ii] = torch.where(do_pair, next_id, ids[barange, ii])
        next_id += do_pair.long()
        dest = torch.where(do_pair.unsqueeze(1) & (dest == jj.unsqueeze(1)), ii.unsqueeze(1), dest)

        # beam merge: slot ii becomes a jet
        owned = dest == ii.unsqueeze(1)
        jet_idx = torch.where(do_beam.unsqueeze(1) & owned & mask, n_jets.unsqueeze(1), jet_idx)
        n_jets += do_beam.long()

        dead = torch.where(do_pair, jj, ii)
        keep = active[barange, dead] & ~alive
        active[barange, dead] = keep

    return {
        "jet_idx": jet_idx,
        "n_jets": n_jets,
        "hist_p1": hist_p1,
        "hist_p2": hist_p2,
        "hist_child": hist_child,
        "hist_d": hist_d,
    }
