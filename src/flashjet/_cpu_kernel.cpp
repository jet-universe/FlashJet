// flashjet CPU kernel: generalized-kt sequential recombination, FastJet's
// N2Plain nearest-neighbour strategy, batched over events with OpenMP.
//
// This is a direct transcription of nn_reference.py / cpu_backend.py, kept
// step-for-step identical to them so the Python mirrors remain the spec:
//
//   * each live slot caches its GEOMETRIC nearest neighbour (min dR^2).  By
//     the Cacciari-Salam lemma the global d_ij minimum is realized at some
//     slot's geometric NN, so the per-slot candidate is
//         min(w_i, w_gnn) * dR2_gnn / R^2   vs the beam distance w_i,
//     and one linear scan of those candidates gives the global minimum.
//   * geometric NNs only go stale when a slot's POSITION moves or dies, so a
//     merge invalidates O(1) rows on average -> O(n^2) per event.
//   * a merge overwrites slot i; slot j dies.  Dead slots keep their index
//     (no compaction) because compacting would renumber slots and change how
//     argmin ties break, and bit-identity with the Python reference is worth
//     more than the ~2x a dense scan would buy.
//
// Two strategies sit on top of that, chosen by event size.  Both are exact:
// every argmin breaks ties on (distance, slot index), so all paths produce
// byte-identical history.
//
//   PLAIN (small n): NN searches scan all n slots and the winning slot comes
//   from a linear scan of the candidate array.  O(n^2), tiny constant.
//
//   TILED + HEAP (large n): mirrors FastJet's N2MinHeapTiled.
//     - Tiling.  Slots live in a (rapidity, phi) grid whose cells are at
//       least R across, so every point within R of a slot lies in its 3x3
//       cell block.  That suffices because a REALIZED merge always has
//       dR < R: the winning pair (a,b) has w_a = min(w_a,w_b), so
//       d_ab = w_a*dR^2/R^2 must beat d_aB = w_a, forcing dR < R.  A slot
//       whose nearest neighbour is farther than R records "no neighbour";
//       its candidate then reads w_a, and it can never spuriously win,
//       because a far pair always costs more than the beam distance of the
//       neighbour it points at -- and that neighbour is itself in the
//       running.  NN maintenance drops from O(n) to O(density * R^2).
//     - Min-heap.  The winning slot comes off an indexed binary heap keyed
//       on (candidate, slot), so a step costs O(log n) instead of O(n), and
//       only the O(1) slots whose candidate changed are re-keyed.
//   Together: O(n log n) per event instead of O(n^2).
//
// No Python C-API here: this compiles to a plain shared library that is
// loaded with ctypes, so the package has no build-time dependency on torch,
// pybind11 or numpy headers.
//
// Ties: every argmin uses a strict `<`, i.e. the FIRST minimum wins, which is
// what np.argmin does.  Pair-vs-beam ties go to the beam (`dpair < w_i`).

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr double TWO_PI = 6.283185307179586;
constexpr double PI = 3.141592653589793;
constexpr double MAX_RAP = 1e5;

template <typename T> struct Traits;
template <> struct Traits<double> { static constexpr double tiny = 1e-300; };
template <> struct Traits<float> { static constexpr float tiny = 1e-30f; };

// FastJet's numerically stable rapidity, matching kinematics.rap_phi_kt2.
template <typename T>
inline void rap_phi_kt2(T px, T py, T pz, T E, T &rap, T &phi, T &kt2) {
  kt2 = px * px + py * py;
  phi = std::atan2(py, px);
  T m2 = E * E - pz * pz - kt2;
  if (!(m2 > T(0))) m2 = T(0);
  const T apz = std::fabs(pz);
  if (kt2 + m2 <= T(0)) {
    rap = (pz >= T(0)) ? T(MAX_RAP) + apz : -(T(MAX_RAP) + apz);
  } else {
    const T denom = (E + apz) * (E + apz);
    const T half_log = T(0.5) * std::log((kt2 + m2) / denom);
    rap = (pz >= T(0)) ? -half_log : half_log;
  }
}

// d_iB = kt^(2p).  The integral exponents are special-cased both because they
// are the only ones anyone uses and because NumPy special-cases them too, so
// the two implementations agree to the last bit for anti-kt / C-A / kt.
template <typename T>
inline T weight(T kt2, double p) {
  const T k = (kt2 > Traits<T>::tiny) ? kt2 : Traits<T>::tiny;
  if (p == -1.0) return T(1) / k;
  if (p == 0.0) return T(1);
  if (p == 1.0) return k;
  return std::pow(k, T(p));
}

// Slots that are at least this many make the tiled/heap machinery pay for its
// setup; below it the flat scans win.  Results are identical either way.
constexpr int64_t TILED_MIN = 12;

// above this many slots, order the cell block by distance and prune it with
// the per-cell max-NN bound (FastJet's Tile::max_NN_dist)
constexpr int64_t CELL_MAX_MIN = 1024;

// mean slots per cell above which the initial build prunes whole neighbour
// cells by distance before enumerating their pairs
constexpr int64_t INIT_PRUNE_OCCUPANCY = 16;

// below this many slots, cap recorded NN distances at R (see cluster_one)
constexpr int64_t NN_CAP_MAX = 512;


// The hot loop is a walk over the slots in a 3x3 cell block, and for each one
// it wants the position (to measure dR^2), the cached NN distance (to test
// for an improvement) and the NN pointer (to test for staleness).  Packed
// together that is one cache line per slot instead of four gathers; split
// across four arrays it was the single biggest cost in the kernel.
/// atan2 gives phi in (-pi, pi]; tile geometry wants [0, 2pi).
template <typename T>
inline T fold01(T phi) {
  return (phi >= T(0)) ? phi : phi + T(TWO_PI);
}

template <typename T>
struct Geo {
  T rap, phi, nnd;   // phi is folded into [0, 2pi), NOT the raw atan2 range
  int32_t nni;
};

// (rapidity, phi) grid with cells >= R across, so a slot's neighbours within
// R all lie in its own cell or the 8 around it.
template <typename T>
struct Tiles {
  int n_rap = 0, n_phi = 0;
  T rap_min = 0, inv_drap = 0, inv_dphi = 0;
  size_t n_cells = 0;
  // Neighbourhood radius in cells.  Cells are R/rad across, so a (2*rad+1)^2
  // block still covers everything within R -- but it hugs the disc of radius R
  // more tightly, which is the whole point of FastJet's Lazy25 over Lazy9:
  // rad=1 sweeps 9R^2 of area, rad=2 sweeps 6.25R^2 for the same guarantee,
  // and the per-cell distance bound then throws away most of what is left.
  int rad = 1;
  T drap = 0, dphi = 0, half_drap = 0, half_dphi = 0;
  // Cell contents are index VECTORS, not intrusive lists: the scan is a
  // sequential walk the prefetcher can run ahead on, whereas chasing a
  // `next` pointer serialises the loads.  Measured 25% slower at N=6000 with
  // linked cells, despite the linked version allocating nothing.
  std::vector<std::vector<int32_t>> cells;
  std::vector<int32_t> cell_of, pos_of;
  // Upper bound on the NN distance of any live slot in the cell (FastJet's
  // Tile::max_NN_dist).  Lets the "who did the new pseudojet get closer to"
  // half of the merge sweep skip a cell outright: if the cell is farther from
  // the new pseudojet than the worst NN distance inside it, nobody there can
  // improve.  Allowed to drift loose -- never tight-but-wrong.
  std::vector<T> cell_max;
  // Per-cell geometry, precomputed once per event.  Profiling put ~19% of the
  // kernel in recomputing cell centres and walking the neighbour stencil with
  // modular arithmetic, on every scan of every step -- all of it invariant.
  std::vector<T> cell_rc, cell_pc;   // cell centre (rapidity, phi)
  // True where a cell's 3x3 block can straddle the phi = 0 seam, so distances
  // out of it need the wrap correction.  Everywhere else the block spans at
  // most three columns of width >= R away from the seam, phi differences
  // inside it cannot exceed pi, and the wrap test is dead weight in the
  // innermost loop of the kernel.  (FastJet marks its tiles the same way.)
  std::vector<uint8_t> periodic;
  std::vector<int32_t> nbr;          // MAX_CELLS neighbour ids per cell
  std::vector<int8_t> nbr_n;         // how many of them are valid

  void build(std::vector<Geo<T>> &geo, int64_t n, double R, int radius) {
    rad = radius;
    T lo = geo[0].rap, hi = geo[0].rap;
    for (int64_t k = 1; k < n; ++k) {
      if (geo[k].rap < lo) lo = geo[k].rap;
      if (geo[k].rap > hi) hi = geo[k].rap;
    }
    // cells must be >= R; WIDENING them is always safe, and widening is what
    // keeps the grid bounded when a beam-like rapidity blows up the range
    const T cell = T(R) / T(rad);
    T drap = cell;
    const T span = hi - lo;
    const T max_rows = T(n) + T(1);
    if (span > drap * max_rows) drap = span / max_rows;
    n_rap = static_cast<int>(span / drap) + 1;
    if (n_rap < 1) n_rap = 1;
    n_phi = static_cast<int>(T(TWO_PI) / cell);
    if (n_phi < 2 * rad + 1) n_phi = 1;  // too few columns to tile in phi  // one column: no phi tiling, still exact
    // A sparse event would otherwise spend more time clearing the grid than
    // scanning it, so keep the cell count O(n).  Cells only ever get WIDER,
    // which never loses a neighbour within R.
    const long long cap = 2LL * n + 8;
    while (1LL * n_rap * n_phi > cap) {
      if (n_rap >= n_phi && n_rap > 1) {
        drap += drap;
        n_rap = static_cast<int>(span / drap) + 1;
      } else if (n_phi > 1) {
        n_phi = (n_phi >= 2 * (2 * rad + 1)) ? n_phi / 2 : 1;
      } else {
        break;
      }
    }
    rap_min = lo;
    inv_drap = T(1) / drap;
    inv_dphi = (n_phi > 1) ? T(n_phi) / T(TWO_PI) : T(0);

    // clear, never reallocate: the Event scratch is reused across every
    // event a thread handles, so the cell buffers keep their capacity
    const size_t want = static_cast<size_t>(n_rap) * n_phi;
    if (cells.size() < want) cells.resize(want);
    for (size_t c = 0; c < want; ++c) cells[c].clear();
    if (cell_max.size() < want) cell_max.resize(want);
    for (size_t c = 0; c < want; ++c) cell_max[c] = std::numeric_limits<T>::infinity();
    n_cells = want;
    this->drap = drap;
    this->dphi = (n_phi > 1) ? T(TWO_PI) / T(n_phi) : T(TWO_PI);
    half_drap = drap * T(0.5);
    half_dphi = this->dphi * T(0.5);
    // precompute cell centres and the neighbour stencil
    if (cell_rc.size() < want) { cell_rc.resize(want); cell_pc.resize(want); }
    if (nbr.size() < want * MAX_CELLS) nbr.resize(want * MAX_CELLS);
    if (nbr_n.size() < want) nbr_n.resize(want);
    if (periodic.size() < want) periodic.resize(want);
    // A block spans 2*rad+1 columns, so phi differences inside it stay below
    // (2*rad+1)*dphi.  The shortcut needs that below pi, i.e. dphi <= pi/(2*rad+1),
    // i.e. at least 2*(2*rad+1) columns in total.
    const bool any_flat = n_phi >= 2 * (2 * rad + 1);
    for (int ir = 0; ir < n_rap; ++ir) {
      for (int ip = 0; ip < n_phi; ++ip) {
        const int c = ir * n_phi + ip;
        cell_rc[c] = rap_min + (T(ir) + T(0.5)) * this->drap;
        cell_pc[c] = (T(ip) + T(0.5)) * this->dphi;
        periodic[c] = (!any_flat || ip < rad || ip >= n_phi - rad) ? 1 : 0;
        // own cell first: it is at distance 0, so it always sorts first and
        // always gets scanned -- putting it there leaves the insertion sort
        // below almost nothing to do
        int cnt = 0;
        nbr[c * MAX_CELLS + cnt++] = c;
        const int lo2 = (ir > rad) ? ir - rad : 0;
        const int hi2 = (ir + rad < n_rap) ? ir + rad : n_rap - 1;
        for (int jr = lo2; jr <= hi2; ++jr) {
          if (n_phi == 1) {
            if (jr != ir) nbr[c * MAX_CELLS + cnt++] = jr;
            continue;
          }
          for (int d = -rad; d <= rad; ++d) {
            int jp = ip + d;
            if (jp < 0) jp += n_phi;
            if (jp >= n_phi) jp -= n_phi;
            const int cc = jr * n_phi + jp;
            if (cc != c) nbr[c * MAX_CELLS + cnt++] = cc;
          }
        }
        nbr_n[c] = static_cast<int8_t>(cnt);
      }
    }
    cell_of.assign(n, -1);
    pos_of.assign(n, -1);
    for (int64_t k = 0; k < n; ++k) insert(static_cast<int32_t>(k), geo[k].rap, geo[k].phi);
  }

  inline int index_of(T rap, T phi01) const {
    int ir = static_cast<int>((rap - rap_min) * inv_drap);
    if (ir < 0) ir = 0;
    if (ir >= n_rap) ir = n_rap - 1;
    int ip = 0;
    if (n_phi > 1) {
      ip = static_cast<int>(phi01 * inv_dphi);
      if (ip < 0) ip = 0;
      if (ip >= n_phi) ip = n_phi - 1;
    }
    return ir * n_phi + ip;
  }

  /// Squared distance from a slot to the nearest point of cell (ir, ip); 0 if
  /// the slot is inside it.  A cell whose bound already exceeds the best
  /// distance so far cannot contain a better neighbour, so it is skipped
  /// whole -- the single biggest saving in the tiled path.
  inline T cell_dist(const Geo<T> &g, int c) const {
    T dr = std::fabs(g.rap - cell_rc[c]) - half_drap;
    if (dr < T(0)) dr = T(0);
    T dp = T(0);
    if (n_phi > 1) {
      dp = std::fabs(g.phi - cell_pc[c]);
      if (dp > T(TWO_PI) - dp) dp = T(TWO_PI) - dp;
      dp -= half_dphi;
      if (dp < T(0)) dp = T(0);
    }
    return dr * dr + dp * dp;
  }

  /// The (2*rad+1)^2 block around slot k, as cell ids plus their (ir, ip).
  /// The precomputed cell block around slot k: pointer into the table, and
  /// how many entries are valid.
  inline const int32_t *neighbour_cells(int32_t k, int &cnt) const {
    const int c = cell_of[k];
    cnt = nbr_n[c];
    return &nbr[static_cast<size_t>(c) * MAX_CELLS];
  }

  static constexpr int MAX_CELLS = 25;  // (2*2+1)^2



  inline void link(int32_t k, int c) {
    cell_of[k] = c;
    pos_of[k] = static_cast<int32_t>(cells[c].size());
    cells[c].push_back(k);
  }

  inline void insert(int32_t k, T rap, T phi01) { link(k, index_of(rap, phi01)); }

  inline void erase(int32_t k) {
    const int c = cell_of[k];
    if (c < 0) return;
    auto &v = cells[c];
    const int32_t moved = v.back();
    v[pos_of[k]] = moved;
    pos_of[moved] = pos_of[k];
    v.pop_back();
    cell_of[k] = -1;
    pos_of[k] = -1;
  }

  inline void move(int32_t k, T rap, T phi01) {
    const int c = index_of(rap, phi01);
    if (c == cell_of[k]) return;
    erase(k);
    link(k, c);
  }

};

// Indexed binary min-heap over the candidate distances.  The key is the pair
// (cand[slot], slot), so the minimum is the LOWEST-INDEXED minimum -- exactly
// what np.argmin returns, which is what keeps the heap path bit-identical to
// the linear scan.
template <typename T>
struct Heap {
  const T *cand = nullptr;
  std::vector<int32_t> node, pos;

  inline bool less(int32_t a, int32_t b) const {
    return cand[a] < cand[b] || (cand[a] == cand[b] && a < b);
  }

  void build(const T *c, int64_t n) {
    cand = c;
    node.resize(n);
    pos.assign(n, -1);
    for (int64_t k = 0; k < n; ++k) node[k] = static_cast<int32_t>(k);
    for (int64_t k = 0; k < n; ++k) pos[k] = static_cast<int32_t>(k);
    for (int64_t k = n / 2; k-- > 0;) down(static_cast<int32_t>(k));
  }

  inline void place(int32_t at, int32_t slot) {
    node[at] = slot;
    pos[slot] = at;
  }

  void up(int32_t at) {
    const int32_t slot = node[at];
    while (at > 0) {
      const int32_t parent = (at - 1) / 2;
      if (!less(slot, node[parent])) break;
      place(at, node[parent]);
      at = parent;
    }
    place(at, slot);
  }

  void down(int32_t at) {
    const int32_t slot = node[at];
    const int32_t n = static_cast<int32_t>(node.size());
    for (;;) {
      int32_t child = 2 * at + 1;
      if (child >= n) break;
      if (child + 1 < n && less(node[child + 1], node[child])) ++child;
      if (!less(node[child], slot)) break;
      place(at, node[child]);
      at = child;
    }
    place(at, slot);
  }

  inline int32_t top() const { return node[0]; }

  // cand[slot] changed, in either direction
  inline void update(int32_t slot) {
    const int32_t at = pos[slot];
    up(at);
    if (pos[slot] == at) down(at);
  }

  inline void erase(int32_t slot) {
    const int32_t at = pos[slot];
    const int32_t last = static_cast<int32_t>(node.size()) - 1;
    const int32_t moved = node[last];
    node.pop_back();
    pos[slot] = -1;
    if (at != last) {
      place(at, moved);
      update(moved);
    }
  }
};

template <typename T>
struct Event {
  std::vector<Geo<T>> geo;   // rap, phi, nnd, nni -- the hot block-scan fields
  std::vector<T> px, py, pz, pe, w, cand;
  std::vector<int32_t> ids, stale, stamp;
  // rev_head[a] starts the list of slots whose geometric NN is a.  Without it,
  // finding the rows invalidated by a merge means sweeping the neighbourhoods
  // of both parents just to read their NN pointers -- which measured as ~35%
  // of all block visits.  Each slot has exactly one NN, so the lists are
  // disjoint and the stale set needs no deduplication.
  std::vector<int32_t> rev_head, rev_next, rev_prev;
  std::vector<uint8_t> act;
  Tiles<T> tiles;
  Heap<T> heap;
  bool tiled = false;
  int32_t stamp_id = 0;
  T R2 = 0, inv_R2 = 0;

  // Nothing farther than R can ever be merged: a realized pair (a,b) has
  // w_a = min(w_a,w_b), so d_ab = w_a*dR^2/R^2 must beat d_aB = w_a.  A slot
  // whose recorded NN sits beyond R therefore never wins the argmin, which is
  // what licenses skipping cells farther than R away -- even though doing so
  // can leave that slot's NN pointer less than truly nearest.

  /// Per-event setup.  Small events skip the tiled bookkeeping entirely --
  /// at n ~ 20 the whole clustering is only a few hundred distance
  /// computations, so initialising state the plain path never reads is a
  /// measurable fraction of the run.
  void prepare(int64_t n, bool use_tiles) {
    tiled = use_tiles;
    geo.resize(n);
    px.resize(n); py.resize(n); pz.resize(n); pe.resize(n);
    w.resize(n); cand.resize(n);
    ids.resize(n); act.assign(n, 1);
    stale.clear();
    if (tiled) {
      for (int64_t k = 0; k < n; ++k) geo[k].nni = -1;  // rev_detach reads this
      stamp.assign(n, -1);
      rev_head.assign(n, -1);
      rev_next.assign(n, -1);
      rev_prev.assign(n, -1);
      stamp_id = 0;
      stale.reserve(n);
    }
  }

  /// dR^2, the innermost expression in the kernel -- ~40% of it at N=6000.
  ///
  /// Signed wrap rather than fold-the-absolute-value.  Only dphi^2 is ever
  /// wanted, so the sign is irrelevant and the fabs is pure waste; and this
  /// is bit-identical to `t = fabs(da); if (2pi - t < t) t = 2pi - t`, not
  /// merely equivalent:
  ///   |da| <= pi   -> neither form folds, and da^2 == |da|^2;
  ///   da  >  pi    -> that form yields 2pi - da, this one da - 2pi, and
  ///                   IEEE makes x-y and y-x exact negatives, so the
  ///                   squares agree to the last bit;
  ///   da  < -pi    -> same argument mirrored.
  static inline T dr2_of(const Geo<T> &a, const Geo<T> &b) {
    T dphi = a.phi - b.phi;
    if (dphi > T(PI)) dphi -= T(TWO_PI);
    else if (dphi < -T(PI)) dphi += T(TWO_PI);
    const T drap = a.rap - b.rap;
    return drap * drap + dphi * dphi;
  }

  /// Same, for pairs known to lie away from the phi = 0 seam, where the wrap
  /// correction provably never fires.  Bit-identical to dr2_of there, since
  /// that branch is simply not taken -- so mixing the two within one event
  /// cannot perturb a comparison.
  static inline T dr2_flat(const Geo<T> &a, const Geo<T> &b) {
    const T dphi = a.phi - b.phi;
    const T drap = a.rap - b.rap;
    return drap * drap + dphi * dphi;
  }

  template <bool CAP, bool WRAP>
  inline void scan_cell(const Geo<T> &gk, int32_t k, const std::vector<int32_t> &cell,
                        T &best, int32_t &bj) const {
    for (int32_t m : cell) {
      if (m == k) continue;
      const T d = WRAP ? dr2_of(gk, geo[m]) : dr2_flat(gk, geo[m]);
      if (CAP && d > R2) continue;
      if (d <= best && (d < best || m < bj)) { best = d; bj = m; }
    }
  }

  inline T candidate(int64_t k, int32_t nn, T dr2v, T inv_R2) const {
    const T INF = std::numeric_limits<T>::infinity();
    const T wk = w[k];
    const T wn = (nn >= 0) ? w[nn] : INF;
    const T pair = ((wk < wn) ? wk : wn) * dr2v * inv_R2;
    return (pair < wk) ? pair : wk;
  }

  inline void rev_detach(int32_t k) {
    const int32_t a = geo[k].nni;
    if (a < 0) return;
    const int32_t pv = rev_prev[k], nx = rev_next[k];
    if (pv >= 0) rev_next[pv] = nx; else rev_head[a] = nx;
    if (nx >= 0) rev_prev[nx] = pv;
    rev_prev[k] = -1;
    rev_next[k] = -1;
  }

  // Point slot k at its new geometric NN, keeping the reverse index in step.
  inline void set_nn(int32_t k, int32_t nn, T d) {
    if (tiled) rev_detach(k);
    geo[k].nnd = d;
    geo[k].nni = nn;
    if (tiled) {
      if (nn >= 0) {
        rev_prev[k] = -1;
        rev_next[k] = rev_head[nn];
        if (rev_head[nn] >= 0) rev_prev[rev_head[nn]] = k;
        rev_head[nn] = k;
      }
      const int c = tiles.cell_of[k];
      if (c >= 0 && d > tiles.cell_max[c]) tiles.cell_max[c] = d;
    }
  }

  // Move every slot currently pointing at `target` into the stale list.
  inline void take_stale(int32_t target, int32_t skip) {
    for (int32_t k = rev_head[target]; k >= 0; k = rev_next[k]) {
      if (k == skip) continue;
      stamp[k] = stamp_id;
      stale.push_back(k);
    }
  }

  // Geometric NN of slot k plus its candidate.  Tiled, this only searches
  // k's 3x3 cell block, so a neighbour farther than R may be missed or
  // recorded wrong -- harmless, because such a slot can only ever win the
  // argmin through its (exact) beam term.  Ties break on the lower index in
  // both modes, which is what np.argmin does and what keeps the two paths
  // producing identical merges.
  template <bool CAP>
  inline void rescan(int64_t k, int64_t n) {
    const T INF = std::numeric_limits<T>::infinity();
    const Geo<T> gk = geo[k];
    T best = INF;
    int32_t bj = -1;
    if (tiled) {
      int nc = 0;
      const int32_t *cs = tiles.neighbour_cells(static_cast<int32_t>(k), nc);
      const bool wrap = tiles.periodic[tiles.cell_of[k]] != 0;
      for (int t = 0; t < nc; ++t) {
        // no neighbour beyond R matters, and no cell farther away than the
        // best so far can hold one -- skip it without touching its contents
        const auto &cell = tiles.cells[cs[t]];
        if (cell.empty()) continue;
        const T bound = (best < R2) ? best : R2;
        if (bound < tiles.cell_dist(gk, cs[t])) continue;
        // cells hold only live slots, so no act[] test is needed here
        if (wrap) scan_cell<CAP, true>(gk, static_cast<int32_t>(k), cell, best, bj);
        else scan_cell<CAP, false>(gk, static_cast<int32_t>(k), cell, best, bj);
      }
    } else {
      for (int64_t m = 0; m < n; ++m) {
        if (!act[m] || m == k) continue;
        const T d = dr2_of(gk, geo[m]);
        if (d <= best && (d < best || m < bj)) { best = d; bj = static_cast<int32_t>(m); }
      }
    }
    set_nn(static_cast<int32_t>(k), bj, best);
    cand[k] = candidate(k, bj, best, inv_R2);
  }

  /// The heap only has to be correct at the top of the next step, and a
  /// slot's candidate often changes more than once within a step, so
  /// re-keying is deferred and de-duplicated (FastJet does the same with its
  /// minheap_update_needed flag).  An O(log n) sift per write was costing
  /// more than the distance computations it was bookkeeping for.
  /// Initial nearest neighbours for every slot, walking each adjacent CELL
  /// PAIR once and updating both endpoints, instead of running an
  /// independent block scan per slot.  Same answer, half the distance
  /// computations, and the cells are traversed in order rather than jumped
  /// around -- this build was 41% of all block visits at N=6000.
  template <bool CAP>
  void init_nn_by_cell_pairs(int64_t n) {
    const T INF = std::numeric_limits<T>::infinity();
    for (int64_t k = 0; k < n; ++k) { geo[k].nnd = INF; geo[k].nni = -1; }

    // Sweep one slot `a` against a run of candidates, updating BOTH sides.
    //
    // `a`'s running best lives in registers across the run instead of being
    // re-read and re-written through geo[] on every pair: each pair then
    // costs one scattered read-modify-write (b's) rather than two.  Safe
    // because b never aliases a here -- within a cell the inner loop starts
    // past a, and across cells the two lists are disjoint.
    //
    // `d <= current` is one comparison and false almost every time; the index
    // tie-break only has to be evaluated inside it.  Written as two ORed
    // conditions, these were ~10% of the whole kernel.
    auto sweep = [&](int32_t a, const int32_t *bs, size_t nb, bool wrap) {
      const Geo<T> ga = geo[a];
      T besta = ga.nnd;
      int32_t bja = ga.nni;
      for (size_t q = 0; q < nb; ++q) {
        const int32_t b = bs[q];
        const T d = wrap ? dr2_of(ga, geo[b]) : dr2_flat(ga, geo[b]);
        if (CAP && d > R2) continue;
        if (d <= besta && (d < besta || b < bja)) { besta = d; bja = b; }
        if (d <= geo[b].nnd && (d < geo[b].nnd || a < geo[b].nni)) {
          geo[b].nnd = d; geo[b].nni = a;
        }
      }
      geo[a].nnd = besta;
      geo[a].nni = bja;
    };

    // The cell-distance test costs one distance and saves a whole cell's
    // worth when it fires, so it pays in proportion to how many slots a cell
    // holds -- occupancy, not event size, is the thing to gate on.
    const int64_t occupancy = n / static_cast<int64_t>(std::max<size_t>(tiles.n_cells, 1));
    const bool prune_cells = occupancy >= INIT_PRUNE_OCCUPANCY;
    const int nr = tiles.n_rap, np = tiles.n_phi;
    for (int ir = 0; ir < nr; ++ir) {
      for (int ip = 0; ip < np; ++ip) {
        const auto &vc = tiles.cells[ir * np + ip];
        if (vc.empty()) continue;
        const bool wrap = tiles.periodic[ir * np + ip] != 0;
        for (size_t x = 0; x + 1 < vc.size(); ++x) {
          sweep(vc[x], vc.data() + x + 1, vc.size() - x - 1, wrap);
        }
        // only "forward" neighbours, so each cell pair in the block is seen
        // once: the rest of this row, then every column of the rows below
        const int rad = tiles.rad;
        int fr[Tiles<T>::MAX_CELLS], fp[Tiles<T>::MAX_CELLS];
        int nf = 0;
        if (np > 1) {
          for (int d = 1; d <= rad; ++d) { fr[nf] = ir; fp[nf] = (ip + d) % np; ++nf; }
        }
        for (int dr = 1; dr <= rad && ir + dr < nr; ++dr) {
          if (np == 1) { fr[nf] = ir + dr; fp[nf] = 0; ++nf; }
          else {
            for (int d = -rad; d <= rad; ++d) {
              int q = ip + d;
              if (q < 0) q += np;
              if (q >= np) q -= np;
              fr[nf] = ir + dr; fp[nf] = q; ++nf;
            }
          }
        }
        for (int t = 0; t < nf; ++t) {
          const int nc2 = fr[t] * np + fp[t];
          const auto &vn = tiles.cells[nc2];
          if (vn.empty()) continue;
          // One cell distance against |vn| pair distances: a slot more than
          // R from the whole neighbour cell can neither find nor be found
          // there.  Only pays once cells are well populated -- below that the
          // test costs about what the pairs it saves cost.
          if (prune_cells) {
            for (int32_t a : vc) {
              if (tiles.cell_dist(geo[a], nc2) > R2) continue;
              sweep(a, vn.data(), vn.size(), wrap);
            }
          } else {
            for (int32_t a : vc) sweep(a, vn.data(), vn.size(), wrap);
          }
        }
      }
    }

    for (int64_t k = 0; k < n; ++k) {
      const int32_t nn = geo[k].nni;
      const T d = geo[k].nnd;
      geo[k].nni = -1;  // so the rev unlink below is a no-op
      set_nn(static_cast<int32_t>(k), nn, d);
      cand[k] = candidate(k, nn, d, inv_R2);
    }
  }

  inline void set_cand(int32_t k, T v) {
    cand[k] = v;
    if (tiled) heap.update(k);
  }


  inline void kill(int32_t k) {
    const T INF = std::numeric_limits<T>::infinity();
    act[k] = 0;
    cand[k] = INF;
    if (tiled) {
      rev_detach(k);
      geo[k].nni = -1;
      heap.erase(k);
      tiles.erase(k);
    }
  }

};

// Cluster one event.  Live particles are gathered densely in mask order, so
// dense index == pseudojet id for the initial particles, exactly as the
// Python backends number them.
template <typename T, bool CAP>
void cluster_one_impl(const T *p4, const uint8_t *mask, int64_t N, double R, T inv_R2,
                      double p, int64_t *hp1, int64_t *hp2, int64_t *hch, T *hd,
                      Event<T> &ev) {
  const T INF = std::numeric_limits<T>::infinity();

  int64_t n = 0;
  for (int64_t k = 0; k < N; ++k) n += (mask[k] != 0);
  if (n == 0) return;
  ev.prepare(n, n >= TILED_MIN);

  {
    int64_t t = 0;
    for (int64_t k = 0; k < N; ++k) {
      if (!mask[k]) continue;
      ev.px[t] = p4[4 * k + 0]; ev.py[t] = p4[4 * k + 1];
      ev.pz[t] = p4[4 * k + 2]; ev.pe[t] = p4[4 * k + 3];
      ev.ids[t] = static_cast<int32_t>(t);
      ++t;
    }
  }
  for (int64_t k = 0; k < n; ++k) {
    T kt2;
    rap_phi_kt2(ev.px[k], ev.py[k], ev.pz[k], ev.pe[k], ev.geo[k].rap, ev.geo[k].phi, kt2);
    ev.geo[k].phi = fold01(ev.geo[k].phi);
    ev.w[k] = weight(kt2, p);
  }

  ev.R2 = T(1) / inv_R2;
  ev.inv_R2 = inv_R2;
  if (ev.tiled) {
    // rad=1 (cells of width R, 3x3 block).  rad=2 -- FastJet's Lazy25 shape,
    // half-size cells sweeping 6.25R^2 instead of 9R^2 -- was measured and is
    // SLOWER here (N=1000: 1.25ms -> 1.48ms): with the per-cell distance
    // bound already discarding most of the 3x3 block, going finer just pays
    // 25 cell-distance evaluations instead of 9 to save candidates the bound
    // was throwing away anyway.
    ev.tiles.build(ev.geo, n, R, 1);
    ev.template init_nn_by_cell_pairs<CAP>(n);
  } else {
    for (int64_t k = 0; k < n; ++k) ev.template rescan<CAP>(k, n);
  }
  if (ev.tiled) ev.heap.build(ev.cand.data(), n);

  int64_t next_id = n;
  for (int64_t step = 0; step < n; ++step) {
    int64_t i;
    if (ev.tiled) {
      i = ev.heap.top();
    } else {
      T gbest = INF;
      i = -1;
      for (int64_t k = 0; k < n; ++k) {
        if (ev.cand[k] < gbest) { gbest = ev.cand[k]; i = k; }
      }
      if (i < 0) break;  // unreachable while any slot is live
    }
    const T gmin = ev.cand[i];

    const T wi = ev.w[i];
    const int32_t j = ev.geo[i].nni;
    const T dpair = (j >= 0) ? ((wi < ev.w[j]) ? wi : ev.w[j]) * ev.geo[i].nnd * inv_R2 : INF;

    ev.stale.clear();
    ++ev.stamp_id;
    if (dpair < wi) {  // ---- pair merge: i absorbs j, j dies
      hp1[step] = ev.ids[i];
      hp2[step] = ev.ids[j];
      hch[step] = next_id;
      hd[step] = gmin;

      // Rows pointing at i or j go stale.  Collect them from the OLD
      // neighbourhoods, before i moves and j leaves the grid.
      // Tiled, the reverse index hands us the invalidated rows directly:
      // everything pointing at i (whose position is about to move) or at j
      // (which is about to die).  The two lists are disjoint.  Untiled there
      // is no index to keep, so the single full sweep below collects them
      // inline instead of paying an extra pass over n.
      if (ev.tiled) {
        ev.stamp[i] = ev.stamp_id;  // i is rebuilt below, j is about to die:
        ev.stamp[j] = ev.stamp_id;  // neither is ever its own stale row
        ev.take_stale(static_cast<int32_t>(i), j);
        ev.take_stale(j, static_cast<int32_t>(i));
      }

      ev.px[i] += ev.px[j]; ev.py[i] += ev.py[j];
      ev.pz[i] += ev.pz[j]; ev.pe[i] += ev.pe[j];
      ev.kill(j);
      T kt2;
      rap_phi_kt2(ev.px[i], ev.py[i], ev.pz[i], ev.pe[i], ev.geo[i].rap, ev.geo[i].phi, kt2);
      ev.geo[i].phi = fold01(ev.geo[i].phi);
      ev.w[i] = weight(kt2, p);
      ev.ids[i] = static_cast<int32_t>(next_id++);
      if (ev.tiled) ev.tiles.move(static_cast<int32_t>(i), ev.geo[i].rap, ev.geo[i].phi);

      // One sweep of the new neighbourhood does both remaining jobs: the new
      // pseudojet's own NN, and the "everyone else can only have improved"
      // update (a slot not adjacent to the new i cannot have improved to
      // within R of it, and improvements beyond R never matter).
      const Geo<T> gi = ev.geo[i];
      const int32_t ii = static_cast<int32_t>(i);
      T best = INF;
      int32_t bj = -1;
      if (ev.tiled) {
        int nc = 0;
        const int32_t *cs = ev.tiles.neighbour_cells(ii, nc);
        const bool wrap = ev.tiles.periodic[ev.tiles.cell_of[ii]] != 0;
        // One sweep still does both jobs, but now a cell farther than R from
        // the new pseudojet is skipped outright: nothing in it can be i's
        // neighbour, and nothing in it can have got closer to i either, since
        // NN distances are capped at R.  That single bound is worth more than
        // splitting the sweep so each half can prune with a tighter one.
        // The block used to be sorted by distance here, so that `best`
        // tightened before the far cells were reached and more of them fell
        // to the bounds below.  That is gone: the stencil already places the
        // slot's OWN cell first, which is where `best` overwhelmingly comes
        // from, and sorting the remaining eight cost more than the extra
        // pruning returned -- 1.08x -> 1.11x at N=6000.  Note the sort was
        // also what licensed breaking out of the loop on the first cell
        // beyond R; unordered, that has to be a `continue`.
        //
        // Worth knowing why this was not obvious from the profile: callgrind
        // counts instructions, and an insertion sort on nine unpredictable
        // branches costs far more cycles than its instruction share suggests.
        // It read as 3% there and was worth ~12%.
        const bool prune = n >= CELL_MAX_MIN;
        for (int t = 0; t < nc; ++t) {
          const T cdist_t = ev.tiles.cell_dist(gi, cs[t]);
          const auto &cell = ev.tiles.cells[cs[t]];
          if (cell.empty()) continue;
          if (cdist_t > ev.R2) continue;  // block is unordered: no break
          // this cell can still matter if it might hold i's own neighbour
          // (bound: the best distance so far) or someone i got closer to
          // (bound: the worst NN distance in the cell)
          if (prune) {
            const T bound = (best < ev.R2) ? best : ev.R2;
            if (cdist_t > bound && cdist_t > ev.tiles.cell_max[cs[t]]) continue;
          }
          T newmax = T(0);
          for (int32_t k : cell) {
            if (k == ii) continue;
            const T d = wrap ? Event<T>::dr2_of(gi, ev.geo[k])
                             : Event<T>::dr2_flat(gi, ev.geo[k]);
            if (CAP && d > ev.R2) continue;
            if (d <= best && (d < best || k < bj)) { best = d; bj = k; }
            // Order matters here.  Without the cap, nnd is a real distance
            // and `d < nnd` is highly selective, so putting it first keeps
            // stamp[] -- a separate array, another cache line -- untouched
            // for most candidates.  WITH the cap, slots that have no
            // neighbour within R carry nnd = inf, `d < nnd` is then always
            // true, and leading with it just adds a test.
            const bool improved = CAP ? (ev.stamp[k] != ev.stamp_id && d < ev.geo[k].nnd)
                                      : (d < ev.geo[k].nnd && ev.stamp[k] != ev.stamp_id);
            if (improved) {
              ev.set_nn(k, ii, d);
              ev.set_cand(k, ev.candidate(k, ii, d, inv_R2));
            }
            if (prune) {
              const T nd = ev.geo[k].nnd;
              newmax = std::max(newmax, (nd < ev.R2) ? nd : ev.R2);
            }
          }
          // we walked the whole cell, so its bound can be retightened
          if (prune) ev.tiles.cell_max[cs[t]] = newmax;
        }
      } else {
        for (int64_t k = 0; k < n; ++k) {
          if (!ev.act[k] || k == i) continue;
          const int32_t kk = static_cast<int32_t>(k);
          const T d = Event<T>::dr2_of(gi, ev.geo[k]);
          if (d < best || (d == best && kk < bj)) { best = d; bj = kk; }
          if (ev.geo[k].nni == ii || ev.geo[k].nni == j) {
            ev.stale.push_back(kk);  // rescanned below
            continue;
          }
          if (d < ev.geo[k].nnd) {
            ev.geo[k].nnd = d;
            ev.geo[k].nni = ii;
            ev.cand[k] = ev.candidate(k, ii, d, inv_R2);
          }
        }
      }
      ev.set_nn(ii, bj, best);
      ev.set_cand(ii, ev.candidate(i, bj, best, inv_R2));
    } else {  // ---- beam merge: i becomes a jet
      hp1[step] = ev.ids[i];
      hp2[step] = -1;
      hch[step] = -1;
      hd[step] = gmin;
      if (ev.tiled) {
        ev.take_stale(static_cast<int32_t>(i), static_cast<int32_t>(i));
      } else {
        for (int64_t k = 0; k < n; ++k) {
          if (ev.act[k] && ev.geo[k].nni == static_cast<int32_t>(i) && k != i) {
            ev.stale.push_back(static_cast<int32_t>(k));
          }
        }
      }
      ev.kill(static_cast<int32_t>(i));
    }
    for (int32_t k : ev.stale) {
      ev.template rescan<CAP>(k, n);
      if (ev.tiled) ev.heap.update(k);
    }
  }
}

/// Capping recorded NN distances at R is a big win on small and medium
/// events: a slot with no neighbour within R then carries no NN pointer at
/// all, so it never enters a reverse-index list and is never rescanned.  On
/// large events it LOSES, because a capped slot reports R^2 and drags
/// Tiles::cell_max up to R^2, which stops that bound pruning anything.  The
/// two exploit the same fact and get in each other's way, so exactly one is
/// active per event.  Compile-time rather than a runtime flag because the
/// test sits in the innermost loop and cost 2-6% there even when it could
/// never fire.  Both settings give identical merge histories.
template <typename T>
inline void cluster_one(const T *p4, const uint8_t *mask, int64_t N, double R, T inv_R2,
                        double p, int64_t *hp1, int64_t *hp2, int64_t *hch, T *hd,
                        Event<T> &ev) {
  int64_t n = 0;
  for (int64_t k = 0; k < N; ++k) n += (mask[k] != 0);
  if (n < NN_CAP_MAX) {
    cluster_one_impl<T, true>(p4, mask, N, R, inv_R2, p, hp1, hp2, hch, hd, ev);
  } else {
    cluster_one_impl<T, false>(p4, mask, N, R, inv_R2, p, hp1, hp2, hch, hd, ev);
  }
}

template <typename T>
void cluster_batch(const T *p4, const uint8_t *mask, int64_t B, int64_t N, double R,
                   double p, int64_t *hp1, int64_t *hp2, int64_t *hch, T *hd,
                   int n_threads) {
  const T inv_R2 = T(1.0 / (R * R));
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads > 0 ? n_threads : 1)
  {
    Event<T> ev;
#pragma omp for schedule(dynamic, 1)
    for (int64_t b = 0; b < B; ++b) {
      cluster_one(p4 + b * N * 4, mask + b * N, N, R, inv_R2, p, hp1 + b * N,
                  hp2 + b * N, hch + b * N, hd + b * N, ev);
    }
  }
#else
  (void)n_threads;
  Event<T> ev;
  for (int64_t b = 0; b < B; ++b) {
    cluster_one(p4 + b * N * 4, mask + b * N, N, R, inv_R2, p, hp1 + b * N, hp2 + b * N,
                hch + b * N, hd + b * N, ev);
  }
#endif
}

}  // namespace

extern "C" {

void flashjet_cluster_f64(const double *p4, const uint8_t *mask, int64_t B, int64_t N,
                          double R, double p, int64_t *hp1, int64_t *hp2, int64_t *hch,
                          double *hd, int n_threads) {
  cluster_batch<double>(p4, mask, B, N, R, p, hp1, hp2, hch, hd, n_threads);
}

void flashjet_cluster_f32(const float *p4, const uint8_t *mask, int64_t B, int64_t N,
                          double R, double p, int64_t *hp1, int64_t *hp2, int64_t *hch,
                          float *hd, int n_threads) {
  cluster_batch<float>(p4, mask, B, N, R, p, hp1, hp2, hch, hd, n_threads);
}

int flashjet_has_openmp(void) {
#ifdef _OPENMP
  return 1;
#else
  return 0;
#endif
}
}
