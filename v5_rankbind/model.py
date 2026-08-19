"""
v5_rankbind/model.py — RankBind model.

  score(L, P) = f(L)^T M g(P) + b     (bilinear head, main model)
  score(L, P) = MLP([f(L); g(P)])     (ablation head)

Encoders are linear projections over pre-computed ChemBERTa / ESM2
embeddings. The encoders do *not* touch the PLM weights: those are frozen
on disk and consumed as fixed-vector inputs, so only the projections and
bilinear core receive gradients.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Heads whose score is read from the DeltaField difference field (free vs coupled
# pass) rather than a pooled dot product. The anti-shortcut zero-field theorem
# applies to every member: mask the cross edges => D ≡ 0 => score ≡ b0. v7
# "gearbind" wraps the same DeltaFieldHead behind two structure GNNs.
FIELD_HEADS = {"deltafield", "gearbind"}


class LigandProjector(nn.Module):
    """ChemBERTa mean-pool (384-d) → d_lig."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProteinProjector(nn.Module):
    """ESM2 mean-pool (1280-d) → d_prot."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidueAttentionPool(nn.Module):
    """Single-head learned-query pool over ESM2 per-residue embeddings.

    Input:
        residues  [B, L, D]   per-residue ESM2 embeddings, padded to L = max_len in batch
        mask      [B, L]      bool, True for real residues, False for padding
    Output:
        pooled    [B, D]      attention-weighted mean over residues
        weights   [B, L]      attention weights (zeroed on padding) — for logging
    """

    def __init__(self, in_dim: int, n_heads: int = 1):
        super().__init__()
        self.in_dim = in_dim
        self.n_heads = n_heads
        # Learned query (one per head). Tiny — ~5KB at D=1280.
        self.q = nn.Parameter(torch.zeros(n_heads, in_dim))
        nn.init.normal_(self.q, std=in_dim ** -0.5)
        # LayerNorm over residue features before scoring keeps the dot-product
        # well-scaled across proteins of very different mean activations.
        self.norm = nn.LayerNorm(in_dim)

    def forward(
        self,
        residues: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # residues: [B, L, D]; mask: [B, L]
        x = self.norm(residues)
        # Single-head version: scores [B, L] = (x · q) / sqrt(D)
        # n_heads kept as scaffolding but we only use head 0 here.
        q = self.q[0]                                          # [D]
        scores = (x * q).sum(dim=-1) / (self.in_dim ** 0.5)     # [B, L]
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)                 # [B, L]
        # Defensive: rows where mask was all False produce NaN; replace with 0.
        weights = torch.nan_to_num(weights, nan=0.0)
        pooled = torch.einsum("bl,bld->bd", weights, residues)  # [B, D]
        return pooled, weights


class BilinearHead(nn.Module):
    """score = f(L)^T M g(P) + b.

    M is parameterised as a low-rank + diagonal factorisation to keep the
    parameter budget tiny:
        M = U V^T + diag(d)
    with U, V ∈ R^{d×r} and d ∈ R^d. For the default config (d_lig=d_prot=256,
    r=32) that is 2 * 256 * 32 + 256 ≈ 16,640 parameters.
    """

    def __init__(self, d_lig: int, d_prot: int, rank: int = 32):
        super().__init__()
        if d_lig != d_prot:
            raise ValueError("BilinearHead assumes d_lig == d_prot for diag term.")
        self.U = nn.Parameter(torch.empty(d_lig, rank))
        self.V = nn.Parameter(torch.empty(d_prot, rank))
        self.d = nn.Parameter(torch.zeros(d_lig))
        self.b = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)

    def forward(self, fL: torch.Tensor, gP: torch.Tensor) -> torch.Tensor:
        # fL: [..., d_lig], gP: [..., d_prot]
        low = (fL @ self.U) * (gP @ self.V)
        lr = low.sum(dim=-1)
        diag = (fL * self.d * gP).sum(dim=-1)
        return lr + diag + self.b


class MLPConcatHead(nn.Module):
    """Ablation head: concat(f(L), g(P)) → 2-layer MLP → scalar.

    Importantly *not* gated by an interaction term — this is the head type
    that lets a model pick up protein-only shortcuts.
    """

    def __init__(self, d_lig: int, d_prot: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_lig + d_prot, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, fL: torch.Tensor, gP: torch.Tensor) -> torch.Tensor:
        x = torch.cat([fL, gP], dim=-1)
        return self.net(x).squeeze(-1)


class _BipartiteBlock(nn.Module):
    """Pre-norm transformer encoder block over the [residues ; atoms] node set.

    Accepts an additive float self-attention mask (``attn_mask`` [N, N]; 0 to
    allow, -inf to block) that encodes the structural regime (free = no
    cross-modal edges, coupled = cross edges open) and a boolean
    ``key_padding_mask`` [B, N] for padding. Dropout defaults to 0 so the
    free/coupled passes are deterministic and the zero-field invariant
    (Section 4 of docs/DELTAFIELD_CONCEPT.md) holds exactly at inference.
    """

    def __init__(self, d: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_mult * d), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ffn_mult * d, d),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask, key_padding_mask):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop(a)
        x = x + self.ffn(self.norm2(x))
        return x


class DeltaFieldHead(nn.Module):
    """Difference-field interaction head (DeltaField-RankBind).

    A single weight-shared bipartite (residue x atom) transformer is run TWICE
    over the projected nodes [u_res ; v_atom]:
      * free pass    — block-diagonal mask, NO cross-modal edges; encodes at
                       most ligand-agnostic protein identity (the shortcut).
      * coupled pass — same weights, cross edges open.
    The binding-induced perturbation field is  D = H_coupled - H_free  per
    node. The score is read ONLY from D, as an additive, mass-conserving sum of
    per-node difference-gated contributions:

        score = b0 + sum_i w_res[i]·(a·Dtilde_res[i]) + sum_a w_atom[a]·(b·Dtilde_atom[a])

    Contribution maps (a·D, b·D) are bias-free linear in D, so D == 0 forces
    score == b0 (a single global constant identical for every protein, hence
    unrankable). This is the anti-shortcut guarantee: the protein-prior lives
    in the free pass and cancels in the subtraction; there is no additive
    protein term in the score. See docs/DELTAFIELD_CONCEPT.md.

    Inputs to forward are ALREADY-PROJECTED per-element tensors:
        u_res  [B, L, d], u_mask [B, L] (bool, True = real residue)
        v_atom [B, A, d], v_mask [B, A] (bool, True = real atom)
    Returns a dict: {score [B], e_res [B, L], e_atom [B, A]} (+ contribution
    maps c_res / c_atom for interpretability).
    """

    NEG_INF = -1e9

    def __init__(self, d: int, n_blocks: int = 2, n_heads: int = 4,
                 ffn_mult: int = 4, prop_layers: int = 0, dropout: float = 0.0):
        super().__init__()
        self.d = d
        self.prop_layers = prop_layers
        # gradient-checkpointing flag (set by RankBind.set_checkpointing); OFF by
        # default so the zero-field unit tests (eval mode) are unaffected.
        self._use_checkpoint = False
        # modality-type embeddings
        self.e_prot = nn.Parameter(torch.zeros(d))
        self.e_lig = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.e_prot, std=d ** -0.5)
        nn.init.normal_(self.e_lig, std=d ** -0.5)
        # weight-shared bipartite encoder (run twice with different masks)
        self.blocks = nn.ModuleList(
            [_BipartiteBlock(d, n_heads, ffn_mult, dropout) for _ in range(n_blocks)]
        )
        # optional difference-graph propagation over D ONLY (keeps invariant exact)
        self.prop = nn.ModuleList(
            [_BipartiteBlock(d, n_heads, ffn_mult, dropout) for _ in range(prop_layers)]
        )
        # difference-gated pocket attention (bias-free so phi(0)=0)
        self.phi_res = nn.Linear(d, 1, bias=False)
        self.phi_atom = nn.Linear(d, 1, bias=False)
        # signed per-node contribution directions (bias-free => c(0)=0, the
        # load-bearing property for the zero-field invariant)
        self.a_res = nn.Linear(d, 1, bias=False)
        self.b_atom = nn.Linear(d, 1, bias=False)
        # the ONLY additive scalar in the score — a single global bias
        self.b0 = nn.Parameter(torch.zeros(1))

    # -- mask construction ---------------------------------------------------
    def _structural_mask(self, L: int, A: int, couple: bool,
                         device) -> torch.Tensor:
        """Boolean self-attention mask [N, N], True = blocked. Residue block
        [0,L), atom block [L,L+A). Bool (not additive float) so it matches the
        bool key_padding_mask type and avoids the MHA mixed-mask warning."""
        N = L + A
        m = torch.zeros(N, N, device=device, dtype=torch.bool)
        if not couple:
            m[:L, L:] = True   # residue queries cannot see atoms
            m[L:, :L] = True   # atom queries cannot see residues
        return m

    def _encode(self, x, structural_mask, key_padding_mask):
        h = x
        for blk in self.blocks:
            # Checkpoint the dominant cost center: each bipartite block, run
            # twice (free + coupled). Only in training — never in eval (the
            # zero-field invariant test runs in eval => checkpointing inactive).
            if self._use_checkpoint and self.training:
                h = checkpoint(blk, h, structural_mask, key_padding_mask,
                               use_reentrant=False)
            else:
                h = blk(h, structural_mask, key_padding_mask)
        return h

    def forward(self, u_res, u_mask, v_atom, v_mask,
                _force_no_coupling: bool = False):
        B, L, d = u_res.shape
        A = v_atom.shape[1]
        # stack nodes with modality-type embeddings
        x = torch.cat([u_res + self.e_prot, v_atom + self.e_lig], dim=1)  # [B, N, d]
        key_padding = ~torch.cat([u_mask, v_mask], dim=1)                 # [B, N] True=pad
        free_mask = self._structural_mask(L, A, couple=False, device=x.device)
        coup_mask = free_mask if _force_no_coupling else \
            self._structural_mask(L, A, couple=True, device=x.device)

        H_free = self._encode(x, free_mask, key_padding)
        H_coup = self._encode(x, coup_mask, key_padding)
        D = H_coup - H_free                                              # [B, N, d]

        # optional difference-graph propagation over D ALONE (coupled mask).
        # NOTE on the invariant: prop blocks contain LayerNorm + FFN biases, so
        # Dt(D=0) != 0 in general. The EXACT zero-field invariant therefore
        # holds only for the default prop_layers=0 (Dt == D). With prop_layers>0
        # it becomes approximate, recovered in practice by the L_neg regulariser
        # that drives D->0 for non-binders. The unit test pins prop_layers=0.
        Dt = D
        for blk in self.prop:
            Dt = blk(Dt, coup_mask, key_padding)

        d_res, d_atom = Dt[:, :L], Dt[:, L:]                             # [B,L,d],[B,A,d]
        e_res = d_res.norm(dim=-1)                                       # [B, L]
        e_atom = d_atom.norm(dim=-1)                                     # [B, A]

        # difference-gated pocket attention (masked softmax over real nodes)
        wr = self.phi_res(d_res).squeeze(-1).masked_fill(~u_mask, self.NEG_INF)
        wr = torch.softmax(wr, dim=-1)                                  # [B, L]
        wa = self.phi_atom(d_atom).squeeze(-1).masked_fill(~v_mask, self.NEG_INF)
        wa = torch.softmax(wa, dim=-1)                                  # [B, A]

        c_res = self.a_res(d_res).squeeze(-1)                           # [B, L]
        c_atom = self.b_atom(d_atom).squeeze(-1)                        # [B, A]

        score = self.b0 + (wr * c_res).sum(-1) + (wa * c_atom).sum(-1)  # [B]
        # zero out reported maps on padding
        e_res = e_res.masked_fill(~u_mask, 0.0)
        e_atom = e_atom.masked_fill(~v_mask, 0.0)
        return {"score": score, "e_res": e_res, "e_atom": e_atom,
                "c_res": c_res, "c_atom": c_atom}


# ──────────────────────────────────────────────────────────────────────────────
# GearBind (v7) — relational structure GNNs feeding the DeltaField head
# ──────────────────────────────────────────────────────────────────────────────

def _rbf(dist: torch.Tensor, num: int, dmin: float, dmax: float) -> torch.Tensor:
    """Gaussian radial basis expansion of `dist` over `num` centers in
    [dmin, dmax]. Returns `[..., num]`. SE(3)-invariant edge geometry feature."""
    centers = torch.linspace(dmin, dmax, num, device=dist.device, dtype=dist.dtype)
    width = (dmax - dmin) / num
    z = (dist.unsqueeze(-1) - centers) / width
    return torch.exp(-(z * z))


class RelationalEdgeConv(nn.Module):
    """GearNet-style relational message passing over a fixed-Kmax neighbor axis.

    message(x_j, edge_attr, rel) = W[rel]([x_j ; edge_mlp(edge_attr)]) * gate(j),
    aggregated by a **masked MEAN** over the Kmax neighbor axis using pure
    gather + mean (no torch_scatter / scatter_softmax). Pre-LayerNorm, residual,
    GELU. `gate` is the pLDDT message-confidence gate for the protein branch and
    all-ones for the ligand branch.
    """

    def __init__(self, d: int, edim: int, n_relations: int = 3):
        super().__init__()
        self.d = d
        self.n_relations = n_relations
        self.norm = nn.LayerNorm(d)
        self.edge_mlp = nn.Sequential(nn.Linear(edim, d), nn.GELU())
        self.rel_w = nn.ModuleList([nn.Linear(2 * d, d) for _ in range(n_relations)])
        self.act = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,          # [B, N, d]
        nbr_idx: torch.Tensor,    # [B, N, K] long
        nbr_type: torch.Tensor,   # [B, N, K] long
        nbr_mask: torch.Tensor,   # [B, N, K] bool
        edge_attr: torch.Tensor,  # [B, N, K, edim]
        gate: torch.Tensor,       # [B, N, K] float (confidence gate; 1 for ligand)
    ) -> torch.Tensor:
        B, N, d = x.shape
        K = nbr_idx.shape[2]
        h = self.norm(x)
        # gather neighbor states: out[b, n*K+k] = h[b, nbr_idx[b, n, k]]
        flat = nbr_idx.reshape(B, N * K)
        x_j = torch.gather(
            h, 1, flat.unsqueeze(-1).expand(B, N * K, d)
        ).reshape(B, N, K, d)
        e = self.edge_mlp(edge_attr)                       # [B, N, K, d]
        mi = torch.cat([x_j, e], dim=-1)                   # [B, N, K, 2d]
        msg = x_j.new_zeros(B, N, K, d)
        for r in range(self.n_relations):
            sel = (nbr_type == r).unsqueeze(-1).to(mi.dtype)
            msg = msg + sel * self.rel_w[r](mi)
        w = nbr_mask.to(msg.dtype) * gate                  # [B, N, K]
        num = (msg * w.unsqueeze(-1)).sum(dim=2)           # [B, N, d]
        den = w.sum(dim=2).clamp(min=1e-6).unsqueeze(-1)
        return x + self.act(num / den)


class ProteinStructureGNN(nn.Module):
    """Relational contact-graph GNN over ESM2 residue node states. 3 relations
    (self+backbone / <8 Å contact / kNN16), JumpingKnowledge concat → Linear."""

    def __init__(self, d: int, n_layers: int = 4, n_relations: int = 3,
                 rbf_num: int = 16, rbf_max: float = 20.0, seqsep_dim: int = 8):
        super().__init__()
        self.rbf_num = rbf_num
        self.rbf_max = rbf_max
        self.seqsep_embed = nn.Embedding(65, seqsep_dim)   # clip(i-j,-32,32)+32
        edim = rbf_num + seqsep_dim
        self.layers = nn.ModuleList(
            [RelationalEdgeConv(d, edim, n_relations) for _ in range(n_layers)]
        )
        self.jk = nn.Linear(n_layers * d, d)
        self._use_checkpoint = False

    def forward(self, h0: torch.Tensor, graph: dict) -> torch.Tensor:
        nbr_idx = graph["nbr_idx"]; nbr_type = graph["nbr_type"]
        nbr_mask = graph["nbr_mask"]; nbr_dist = graph["nbr_dist"]
        nbr_seqsep = graph["nbr_seqsep"]; plddt = graph["plddt"]   # [B, N]
        rbf = _rbf(nbr_dist, self.rbf_num, 0.0, self.rbf_max)      # [B, N, K, 16]
        sep = (nbr_seqsep.clamp(-32, 32) + 32)
        edge_attr = torch.cat([rbf, self.seqsep_embed(sep)], dim=-1)
        # pLDDT confidence gate from the *neighbor's* pLDDT (plddt stored /100).
        B, N, K = nbr_idx.shape
        plddt_j = torch.gather(plddt, 1, nbr_idx.reshape(B, N * K)).reshape(B, N, K)
        gate = torch.sigmoid((plddt_j * 100.0 - 50.0) / 10.0)
        h = h0
        outs = []
        for layer in self.layers:
            if self._use_checkpoint and self.training:
                h = checkpoint(layer, h, nbr_idx, nbr_type, nbr_mask,
                               edge_attr, gate, use_reentrant=False)
            else:
                h = layer(h, nbr_idx, nbr_type, nbr_mask, edge_attr, gate)
            outs.append(h)
        return self.jk(torch.cat(outs, dim=-1))


class LigandStructureGNN(nn.Module):
    """Relational 3D-conformer atom-graph GNN. 3 relations (bond / spatial<4.5 Å
    / spatial-kNN8), no pLDDT gate, JumpingKnowledge concat → Linear."""

    def __init__(self, d: int, n_layers: int = 3, n_relations: int = 3,
                 rbf_num: int = 16, rbf_max: float = 10.0):
        super().__init__()
        self.rbf_num = rbf_num
        self.rbf_max = rbf_max
        self.n_relations = n_relations
        edim = rbf_num + n_relations                       # rbf + relation one-hot
        self.layers = nn.ModuleList(
            [RelationalEdgeConv(d, edim, n_relations) for _ in range(n_layers)]
        )
        self.jk = nn.Linear(n_layers * d, d)
        self._use_checkpoint = False

    def forward(self, h0: torch.Tensor, graph: dict) -> torch.Tensor:
        nbr_idx = graph["nbr_idx"]; nbr_type = graph["nbr_type"]
        nbr_mask = graph["nbr_mask"]; nbr_dist = graph["nbr_dist"]
        rbf = _rbf(nbr_dist, self.rbf_num, 0.0, self.rbf_max)
        onehot = F.one_hot(
            nbr_type.clamp(0, self.n_relations - 1), self.n_relations
        ).to(rbf.dtype)
        edge_attr = torch.cat([rbf, onehot], dim=-1)
        gate = torch.ones_like(nbr_dist)                   # plddt_gate = 1
        h = h0
        outs = []
        for layer in self.layers:
            if self._use_checkpoint and self.training:
                h = checkpoint(layer, h, nbr_idx, nbr_type, nbr_mask,
                               edge_attr, gate, use_reentrant=False)
            else:
                h = layer(h, nbr_idx, nbr_type, nbr_mask, edge_attr, gate)
            outs.append(h)
        return self.jk(torch.cat(outs, dim=-1))


class GearBindHead(nn.Module):
    """v7 head: two structure GNNs that reshape the residue / atom node states
    geometry-aware, feeding the v6 DeltaFieldHead **verbatim**.

    The GNNs run ONCE; their outputs (U_res, V_atom) feed both the free and the
    coupled pass inside DeltaFieldHead. Because the same node states pass through
    the same cross weights in both passes, all intra-molecular geometry cancels
    in D = H_coupled − H_free; only the cross-interaction term survives. The
    zero-field theorem (mask cross edges ⇒ D ≡ 0 ⇒ score ≡ b0) is therefore
    inherited unchanged — do NOT run the GNN separately per pass.
    """

    def __init__(self, d: int, n_blocks: int = 2, n_heads: int = 4,
                 ffn_mult: int = 4, dropout: float = 0.0,
                 prot_gnn_layers: int = 4, lig_gnn_layers: int = 3,
                 prot_in_dim: int = 1280, lig_in_dim: int = 423):
        super().__init__()
        self.d = d
        # own input projectors (do NOT reuse RankBind.prot / RankBind.lig)
        self.prot_in = nn.Linear(prot_in_dim, d)
        self.lig_in = nn.Linear(lig_in_dim, d)
        self.prot_gnn = ProteinStructureGNN(d, n_layers=prot_gnn_layers)
        self.lig_gnn = LigandStructureGNN(d, n_layers=lig_gnn_layers)
        # prop_layers=0 keeps the EXACT zero-field invariant (see DeltaFieldHead).
        self.cross = DeltaFieldHead(
            d, n_blocks=n_blocks, n_heads=n_heads, ffn_mult=ffn_mult,
            prop_layers=0, dropout=dropout,
        )

    def encode(
        self, prot_res: torch.Tensor, prot_mask: torch.Tensor,
        prot_graph: dict, lig_graph: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run both GNNs once. Returns (U_res[B,L,d], V_atom[B,A,d], lig_mask[B,A])."""
        h0 = self.prot_in(prot_res)                        # [B, L, d]
        U_res = self.prot_gnn(h0, prot_graph)              # [B, L, d]
        g0 = self.lig_in(lig_graph["node_feat"])           # [B, A, d]
        V_atom = self.lig_gnn(g0, lig_graph)               # [B, A, d]
        lig_mask = lig_graph["node_mask"]                  # [B, A] bool
        return U_res, V_atom, lig_mask

    def forward(
        self, prot_res: torch.Tensor, prot_mask: torch.Tensor,
        prot_graph: dict, lig_graph: dict,
        _force_no_coupling: bool = False,
    ) -> dict:
        U_res, V_atom, lig_mask = self.encode(prot_res, prot_mask, prot_graph, lig_graph)
        return self.cross(U_res, prot_mask, V_atom, lig_mask,
                          _force_no_coupling=_force_no_coupling)


def _expand_graph_over_k(graph: dict, k: int) -> dict:
    """Expand a batched graph dict `[B, ...]` to `[B*k, ...]` (each item repeated
    k times) — used to broadcast the anchor ligand graph across k negatives."""
    out: dict = {}
    for key, val in graph.items():
        if torch.is_tensor(val):
            B = val.shape[0]
            out[key] = (val.unsqueeze(1)
                        .expand(B, k, *val.shape[1:])
                        .reshape(B * k, *val.shape[1:]))
        else:
            out[key] = val
    return out


def _flatten_bk_graph(graph: dict) -> dict:
    """Reshape a batched graph dict `[B, k, ...]` → `[B*k, ...]` (negatives)."""
    out: dict = {}
    for key, val in graph.items():
        if torch.is_tensor(val):
            B, k = val.shape[0], val.shape[1]
            out[key] = val.reshape(B * k, *val.shape[2:])
        else:
            out[key] = val
    return out


class RankBind(nn.Module):
    """End-to-end model: projections + head.

    Call either:
        score_pairs(lig_emb, prot_emb)   # [B], pointwise
        score_triplet(lig_emb, pos_prot, neg_prot)  # (pos_score [B], neg_score [B, k])

    For head_type='deltafield' the inputs are per-element sequences
    (ligand per-token, protein per-residue) and the field-path methods
    score_pairs_field / score_triplet_field are used instead.
    """

    def __init__(self, config: dict):
        super().__init__()
        m = config["model"]
        self._use_checkpoint = False
        self.d_lig = m["d_lig"]
        self.d_prot = m["d_prot"]
        self.head_type = m["head"]
        self.protein_encoder = m.get("protein_encoder", "mean_pool")
        self.lig = LigandProjector(m["lig_input_dim"], m["d_lig"], m["dropout"])
        self.prot = ProteinProjector(m["prot_input_dim"], m["d_prot"], m["dropout"])
        if self.protein_encoder == "attn_pool":
            self.attn_pool = ResidueAttentionPool(m["prot_input_dim"], n_heads=1)
        elif self.protein_encoder == "mean_pool":
            self.attn_pool = None
        else:
            raise ValueError(
                f"Unknown protein_encoder={self.protein_encoder!r}; "
                "expected 'mean_pool' or 'attn_pool'."
            )
        if self.head_type == "bilinear":
            self.head = BilinearHead(
                m["d_lig"], m["d_prot"],
                rank=m.get("bilinear_rank", 32),
            )
        elif self.head_type == "mlp_concat":
            self.head = MLPConcatHead(
                m["d_lig"], m["d_prot"],
                hidden=m.get("mlp_hidden", 128),
                dropout=m["dropout"],
            )
        elif self.head_type == "deltafield":
            if m["d_lig"] != m["d_prot"]:
                raise ValueError("deltafield head requires d_lig == d_prot (shared node space).")
            if self.protein_encoder != "attn_pool":
                raise ValueError(
                    "deltafield head requires protein_encoder='attn_pool' "
                    "(needs per-residue protein input)."
                )
            self.head = DeltaFieldHead(
                d=m["d_prot"],
                n_blocks=m.get("df_n_blocks", 2),
                n_heads=m.get("df_n_heads", 4),
                ffn_mult=m.get("df_ffn_mult", 4),
                prop_layers=m.get("df_prop_layers", 0),
                dropout=m.get("df_dropout", 0.0),
            )
        elif self.head_type == "gearbind":
            if m["d_lig"] != m["d_prot"]:
                raise ValueError("gearbind head requires d_lig == d_prot (shared node space).")
            if self.protein_encoder != "attn_pool":
                raise ValueError(
                    "gearbind head requires protein_encoder='attn_pool' "
                    "(the structure GNN consumes per-residue ESM2 node inputs)."
                )
            self.head = GearBindHead(
                d=m["d_prot"],
                n_blocks=m.get("df_n_blocks", 2),
                n_heads=m.get("df_n_heads", 4),
                ffn_mult=m.get("df_ffn_mult", 4),
                dropout=m.get("df_dropout", 0.0),
                prot_gnn_layers=m.get("gnn_layers_prot", 4),
                lig_gnn_layers=m.get("gnn_layers_lig", 3),
                prot_in_dim=m["prot_input_dim"],
                # ligand node input = atom_feat[39] ⊕ mean-pool ChemBERTa[lig_input_dim]
                lig_in_dim=m["lig_input_dim"] + 39,
            )
        else:
            raise ValueError(f"Unknown head: {self.head_type}")

    # ── gradient checkpointing ───────────────────────────────────────────────

    def set_checkpointing(self, flag: bool) -> None:
        """Enable/disable activation checkpointing on the field-head cost centers
        (DeltaField bipartite blocks + structure GNN layers). No-op for the
        bilinear / mlp_concat heads. Checkpointing is additionally gated on
        ``module.training`` at call sites, so eval forwards are never affected.
        OFF by default — existing tests that never call this are unchanged."""
        self._use_checkpoint = bool(flag)
        head = self.head
        if isinstance(head, GearBindHead):
            head.cross._use_checkpoint = self._use_checkpoint
            head.prot_gnn._use_checkpoint = self._use_checkpoint
            head.lig_gnn._use_checkpoint = self._use_checkpoint
        elif isinstance(head, DeltaFieldHead):
            head._use_checkpoint = self._use_checkpoint

    # ── forward helpers ────────────────────────────────────────────────────

    def encode_protein(
        self,
        prot_input: torch.Tensor,
        prot_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a protein input to its `d_prot`-dim representation.

        Two input shapes are accepted:
          - mean_pool mode: prot_input is `[B, prot_input_dim]` (already pooled
            on disk by `data.py:load_protein`); `prot_mask` ignored.
          - attn_pool mode: prot_input is `[B, L, prot_input_dim]` per-residue,
            with a `[B, L]` boolean mask marking real residues. Attention pool
            collapses to `[B, prot_input_dim]` first, then the projector runs.
        """
        if self.attn_pool is not None:
            if prot_input.ndim != 3 or prot_mask is None:
                raise ValueError(
                    "attn_pool encoder requires per-residue [B, L, D] input "
                    "with a [B, L] mask; received "
                    f"shape={tuple(prot_input.shape)} mask={None if prot_mask is None else tuple(prot_mask.shape)}."
                )
            pooled, _weights = self.attn_pool(prot_input, prot_mask)
            return self.prot(pooled)
        if prot_input.ndim != 2:
            raise ValueError(
                "mean_pool encoder expects [B, D] input; received "
                f"shape={tuple(prot_input.shape)}."
            )
        return self.prot(prot_input)

    def score_pairs(
        self,
        lig_emb: torch.Tensor,
        prot_emb: torch.Tensor,
        prot_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # FIELD_HEADS: any difference-field head (deltafield / gearbind) has no
        # pooled f(L)·g(P) path — it scores via the full coupled forward_field.
        if self.head_type in FIELD_HEADS:
            raise RuntimeError(
                f"{self.head_type} head operates on sequences/graphs; use "
                "score_pairs_field(...) / forward_field(...) instead."
            )
        fL = self.lig(lig_emb)
        gP = self.encode_protein(prot_emb, prot_mask)
        return self.head(fL, gP)

    # ── difference-field path (head_type='deltafield') ──────────────────────

    def forward_field(
        self,
        lig_tokens: torch.Tensor,   # [B, A, lig_input_dim] ChemBERTa per-token
        lig_mask: torch.Tensor,     # [B, A] bool
        prot_res: torch.Tensor,     # [B, L, prot_input_dim] ESM2 per-residue
        prot_mask: torch.Tensor,    # [B, L] bool
        *,
        prot_graph: dict | None = None,   # gearbind only
        lig_graph: dict | None = None,    # gearbind only
        _force_no_coupling: bool = False,
    ) -> dict:
        """Full field forward: returns {score [B], e_res, e_atom, c_res, c_atom}.

        gearbind routes through the structure GNNs (prot_res = ESM2 node inputs,
        lig_tokens/lig_mask unused — the ligand is read from lig_graph). deltafield
        ignores prot_graph/lig_graph and keeps the projector path verbatim.
        """
        if self.head_type == "gearbind":
            return self.head(prot_res, prot_mask, prot_graph, lig_graph,
                             _force_no_coupling=_force_no_coupling)
        # deltafield projector path (head-specific — not generalised to FIELD_HEADS)
        v = self.lig(lig_tokens)    # projector broadcasts over the token axis -> [B,A,d]
        u = self.prot(prot_res)     # broadcasts over the residue axis        -> [B,L,d]
        return self.head(u, prot_mask, v, lig_mask,
                         _force_no_coupling=_force_no_coupling)

    def score_pairs_field(
        self,
        lig_tokens: torch.Tensor, lig_mask: torch.Tensor,
        prot_res: torch.Tensor, prot_mask: torch.Tensor,
        *,
        prot_graph: dict | None = None,
        lig_graph: dict | None = None,
    ) -> torch.Tensor:
        return self.forward_field(
            lig_tokens, lig_mask, prot_res, prot_mask,
            prot_graph=prot_graph, lig_graph=lig_graph,
        )["score"]

    def score_triplet_field(
        self,
        lig_tokens: torch.Tensor, lig_mask: torch.Tensor,      # [B,A,Dl],[B,A]
        pos_res: torch.Tensor, pos_mask: torch.Tensor,         # [B,Lp,Dp],[B,Lp]
        neg_res: torch.Tensor, neg_mask: torch.Tensor,         # [B,k,Ln,Dp],[B,k,Ln]
        *,
        prot_graph: dict | None = None,        # gearbind: pos protein graphs [B,...]
        neg_prot_graph: dict | None = None,    # gearbind: neg protein graphs [B,k,...]
        lig_graph: dict | None = None,         # gearbind: anchor ligand graphs [B,...]
    ) -> tuple[torch.Tensor, torch.Tensor, dict, dict]:
        """Returns (pos_score [B], neg_score [B,k], pos_field_dict, neg_field_dict).

        The field dicts are surfaced so a later L_neg can read e_res/e_atom over
        the negatives. Both deltafield and gearbind return the dicts (cheap — they
        are already computed)."""
        B, k, Ln, Dp = neg_res.shape
        pos_out = self.forward_field(
            lig_tokens, lig_mask, pos_res, pos_mask,
            prot_graph=prot_graph, lig_graph=lig_graph,
        )
        pos = pos_out["score"]                                         # [B]
        if self.head_type == "gearbind":
            # ligand graph broadcast across k; neg protein graphs flattened to B*k.
            lig_graph_k = _expand_graph_over_k(lig_graph, k)
            neg_graph_flat = _flatten_bk_graph(neg_prot_graph)
            neg_out = self.forward_field(
                lig_tokens, lig_mask,
                neg_res.reshape(B * k, Ln, Dp), neg_mask.reshape(B * k, Ln),
                prot_graph=neg_graph_flat, lig_graph=lig_graph_k,
            )
        else:
            # deltafield: expand the per-token ligand across the k negatives.
            A = lig_tokens.shape[1]
            lt = lig_tokens.unsqueeze(1).expand(-1, k, -1, -1).reshape(B * k, A, -1)
            lm = lig_mask.unsqueeze(1).expand(-1, k, -1).reshape(B * k, A)
            neg_out = self.forward_field(
                lt, lm, neg_res.reshape(B * k, Ln, Dp), neg_mask.reshape(B * k, Ln),
            )
        neg = neg_out["score"].reshape(B, k)
        return pos, neg, pos_out, neg_out

    def score_triplet(
        self,
        lig_emb: torch.Tensor,        # [B, D_lig]
        pos_prot: torch.Tensor,       # [B, D] (mean) or [B, L_pos, D] (attn)
        neg_prot: torch.Tensor,       # [B, k, D] (mean) or [B, k, L_neg, D] (attn)
        pos_mask: torch.Tensor | None = None,  # [B, L_pos] (attn only)
        neg_mask: torch.Tensor | None = None,  # [B, k, L_neg] (attn only)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fL = self.lig(lig_emb)                        # [B, d_lig]
        gP_pos = self.encode_protein(pos_prot, pos_mask)            # [B, d_prot]

        if self.attn_pool is not None:
            B, k, L_neg, D = neg_prot.shape
            gP_neg_flat = self.encode_protein(
                neg_prot.reshape(B * k, L_neg, D),
                neg_mask.reshape(B * k, L_neg) if neg_mask is not None else None,
            )                                                       # [B*k, d_prot]
            gP_neg = gP_neg_flat.reshape(B, k, -1)
        else:
            B, k, _ = neg_prot.shape
            gP_neg = self.prot(neg_prot.reshape(B * k, -1)).reshape(B, k, -1)

        pos_score = self.head(fL, gP_pos)             # [B]
        neg_score = self.head(
            fL.unsqueeze(1).expand(-1, k, -1),
            gP_neg,
        )                                             # [B, k]
        return pos_score, neg_score

    # ── diagnostics ────────────────────────────────────────────────────────

    def count_parameters(self) -> dict:
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {
            "n_parameters_trainable": int(total_trainable),
            "n_parameters_frozen": int(total_frozen),
            "head_type": self.head_type,
            "d_lig": self.d_lig,
            "d_prot": self.d_prot,
        }
