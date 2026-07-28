# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The FRI opening stage: `TraceOpeningClaim` → `TrivialClaim`.

The reference's `pcs.open`/`pcs.verify` (Plonky3 fri/src/two_adic_pcs.rs +
fri prover/verifier at brevis-network/Plonky3@7fbe1908): observe the
out-of-domain opened values, sample the batching α, reduce every column to
one codeword (zorch's DEEP-ALI composition), fold it down with per-layer
commits and βs, grind, and open the queried leaves. The claim carries points
and commitments, never values — the prover computes values while opening,
the verifier checks them against the fold chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array, lax
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from zorch.coding.reed_solomon import BitReversedReedSolomon, eval_domain
from zorch.commit.merkle import MerkleTree, Opening
from zorch.poly.univariate import powers
from zorch.stage import (
    ProveResult,
    ProverStage,
    TrivialClaim,
    VerifierStage,
    VerifyResult,
)
from zorch.transcript import DuplexTranscript

from pico_zorch.challenger.challenger import sample_ext
from pico_zorch.commit.pcs_commit import GENERATOR, CommitData, commit_matrices
from pico_zorch.uni_stark.types import (
    FriOpeningProof,
    FriOpeningWitness,
    FriParams,
    FriProof,
    TraceOpeningClaim,
)


def _eval_columns(coeffs: Array, y: Array) -> Array:
    """`[w, n]` coefficient rows evaluated at the extension point `y`:
    Σ_j coeffs[:, j]·y^j via log-doubling powers. The coefficients are
    promoted to the extension first — a 2-D mixed base×extension multiply
    trips an XLA shape-rewrite RET_CHECK in the frx lowering (as does
    `eval_coeffs`' associative-scan schedule)."""
    ypow = powers(y, coeffs.shape[-1])
    return (coeffs.astype(y.dtype) * ypow[None, :]).sum(axis=-1)


@partial(frx.jit, static_argnames=("opening_pos",))
def _open_head(
    trace,
    chunks,
    chunk_shifts,
    trace_leaves,
    quotient_leaves,
    zeta,
    zeta_next,
    transcript,
    opening_pos,
    domain,
):
    """OOD values -> observe -> α -> reduced openings, one device program."""
    trace_local, trace_next, chunk_values = _ood_values(
        trace, chunks, chunk_shifts, zeta, zeta_next
    )
    opened = fnp.concatenate([trace_local, trace_next, chunk_values.reshape(-1)])
    t = transcript.observe(opened)
    t, alpha_fri = sample_ext(t)
    ro = reduced_openings(
        fnp.concatenate([trace_leaves, trace_leaves, quotient_leaves], axis=1),
        opened,
        fnp.stack([zeta, zeta_next]),
        opening_pos,
        alpha_fri,
        domain,
    )
    return trace_local, trace_next, chunk_values, ro, t


def _ood_values(trace, chunks, chunk_shifts, zeta, zeta_next):
    """All out-of-domain values in one program; the trace interpolates once
    for both ζ and ζ·g. Interpolation is on the plain subgroup (coefficients
    of p̃(y) = p(shift·y)), evaluation at y = z/shift — field-identical to the
    reference's barycentric `interpolate_coset`."""
    n = trace.shape[0]
    coeffs = lax.ntt(trace.T, ntt_type="INTT", ntt_length=n)
    trace_local = _eval_columns(coeffs, zeta)
    trace_next = _eval_columns(coeffs, zeta_next)
    chunk_values = fnp.stack(
        [
            _eval_columns(
                lax.ntt(c.T, ntt_type="INTT", ntt_length=c.shape[0]),
                zeta / s.astype(zeta.dtype),
            )
            for c, s in zip(chunks, chunk_shifts)
        ]
    )
    return trace_local, trace_next, chunk_values


@partial(frx.jit, static_argnames=("opening_pos",))
def reduced_openings(
    columns: Array,
    values: Array,
    points: Array,
    opening_pos: tuple[int, ...],
    alpha: Array,
    domain: Array,
) -> Array:
    """The reference's α-batched opening reduction — zorch's DEEP-ALI
    composition (the per-(matrix, point) α offsets flatten to consecutive
    powers because each offset is the running column count), with the
    per-point quotients merged over ONE inversion:
    Σ_o n_o/(z_o − X) = (Σ_o n_o·Π_{j≠o} d_j) / Π_o d_o. Field-identical to
    `zorch.pcs.deep.deep_composition`, which inverts once per point —
    upstream candidate."""
    m = columns.shape[1]
    alpha_pows = powers(alpha, m)
    numer: dict[int, Array] = {}
    for col in range(m):
        term = alpha_pows[col] * (values[col] - columns[:, col].astype(EF))
        o = opening_pos[col]
        numer[o] = term if o not in numer else numer[o] + term
    keys = sorted(numer)
    denoms = [points[o] - domain.astype(EF) for o in keys]
    prod = denoms[0]
    for d in denoms[1:]:
        prod = prod * d
    total = None
    for i, o in enumerate(keys):
        others = None
        for j, d in enumerate(denoms):
            if j != i:
                others = d if others is None else others * d
        term = numer[o] if others is None else numer[o] * others
        total = term if total is None else total + term
    return total / prod


def sample_query_indices(
    transcript: DuplexTranscript, log_max_height: int, count: int
) -> tuple[DuplexTranscript, np.ndarray]:
    """`count` × the reference's `sample_bits(log_max_height)`: one squeeze
    each, low bits of the canonical value (zorch's `sample_positions` reduces
    the Montgomery bitpattern instead, so it cannot be used here)."""
    t, raw = transcript.sample(count)
    canonical = np.asarray(lax.convert_element_type(raw, fnp.uint32))
    return t, (canonical & ((1 << log_max_height) - 1)).astype(np.int64)


def _opening_pos(width: int, quotient_degree: int) -> tuple[int, ...]:
    """Point index per flattened column: trace at ζ, trace at ζ·g, chunks
    at ζ."""
    return (0,) * width + (1,) * width + (0,) * (4 * quotient_degree)


@partial(frx.jit, static_argnames=("tree",))
def _open_batch(tree: MerkleTree, leaves: Array, digest_layers, indices: Array):
    return frx.vmap(lambda i: tree.open(leaves, digest_layers, i))(indices)


@partial(frx.jit, static_argnames=("tree", "code", "log_blowup"))
def fold_chain(
    tree: MerkleTree,
    code: BitReversedReedSolomon,
    log_blowup: int,
    ro: Array,
    transcript: DuplexTranscript,
):
    """The whole FRI commit phase as one device program: per layer, commit
    the pair matrix, observe the root, sample β, fold; then observe the
    final polynomial. The layer count is static (it follows the codeword
    shape), so the loop unrolls — one dispatch instead of ~46 kernel
    launches per layer."""
    t = transcript
    folded = ro
    roots, layers = [], []
    while folded.shape[0] > (1 << log_blowup):
        leaves = lax.bitcast_convert_type(code.pair_leaves(folded), F).reshape(-1, 8)
        root, digest_layers = tree.commit(leaves)
        t = t.observe(root)
        t, beta = sample_ext(t)
        folded = code.fold(folded, beta)
        roots.append(root)
        layers.append((leaves, tuple(digest_layers)))
    final_poly = folded[0]
    t = t.observe(final_poly)
    return final_poly, tuple(roots), tuple(layers), t


def open_batch(tree: MerkleTree, data: CommitData, indices: Array) -> Opening:
    """All queries' openings of one tree as one jitted, vmapped device call.
    Eager vmap dispatches every per-level gather as its own kernel (the
    profiler read 4,284 launches for 2.7 ms of device work); the jit
    collapses each tree's openings to a single program, cached per height."""
    return _open_batch(tree, data.leaves, data.digest_layers, indices)


def query_opening(batched: Opening, q: int) -> Opening:
    """Query `q`'s view of a batched Opening."""
    return Opening(batched.row[q], [p[q] for p in batched.path])


def _bitrev_lde_domain(lde_height: int) -> Array:
    return lax.bit_reverse(
        eval_domain(F, lde_height, shift=fnp.array(GENERATOR, dtype=F)),
        dimensions=(0,),
    )


@dataclass(frozen=True)
class FriOpener(
    ProverStage[
        TraceOpeningClaim, FriOpeningWitness, TrivialClaim, FriOpeningProof, DuplexTranscript
    ]
):
    tree: MerkleTree
    params: FriParams = FriParams()

    def prove(
        self,
        claim: TraceOpeningClaim,
        witness: FriOpeningWitness,
        transcript: DuplexTranscript,
    ) -> ProveResult[TrivialClaim, FriOpeningProof, DuplexTranscript]:
        one = fnp.ones((), F)
        log_blowup = self.params.log_blowup
        n = witness.trace.shape[0]
        width = witness.trace.shape[1]
        quotient = witness.quotient
        quotient_degree = len(quotient.chunks)

        lde_height = n << log_blowup
        trace_local, trace_next, chunk_values, ro, t = _open_head(
            witness.trace,
            tuple(quotient.chunks),
            tuple(d.shift for d in quotient.qc_domains),
            witness.trace_data.leaves,
            quotient.quotient_data.leaves,
            claim.zeta,
            claim.zeta_next,
            transcript,
            _opening_pos(width, quotient_degree),
            _bitrev_lde_domain(lde_height),
        )

        code = BitReversedReedSolomon(n, 1 << log_blowup, F)
        final_poly, commit_roots, layers, t = fold_chain(
            self.tree, code, log_blowup, ro, t
        )
        phase_data = [
            CommitData((leaves,), leaves, list(digest_layers))
            for leaves, digest_layers in layers
        ]

        t, pow_witness = t.grind(self.params.proof_of_work_bits)
        t, indices = sample_query_indices(
            t, claim.degree_bits + log_blowup, self.params.num_queries
        )

        idx = fnp.asarray(indices.astype(np.int32))
        proof = FriOpeningProof(
            trace_local=trace_local,
            trace_next=trace_next,
            quotient_chunks=chunk_values,
            fri=FriProof(
                commit_phase_roots=commit_roots,
                final_poly=final_poly,
                pow_witness=pow_witness,
                trace_openings=open_batch(self.tree, witness.trace_data, idx),
                quotient_openings=open_batch(
                    self.tree, quotient.quotient_data, idx
                ),
                commit_phase_openings=[
                    open_batch(self.tree, data, (idx >> (layer + 1)))
                    for layer, data in enumerate(phase_data)
                ],
            ),
        )
        return ProveResult(TrivialClaim(), proof, t)


@dataclass(frozen=True)
class FriOpeningVerifier(
    VerifierStage[TraceOpeningClaim, TrivialClaim, FriOpeningProof, DuplexTranscript]
):
    tree: MerkleTree
    # AIR-fixed shape configuration (statement-vs-configuration split).
    width: int
    quotient_degree: int
    params: FriParams = FriParams()

    def verify(
        self,
        claim: TraceOpeningClaim,
        reduction_proof: FriOpeningProof,
        transcript: DuplexTranscript,
    ) -> VerifyResult[TrivialClaim, DuplexTranscript]:
        proof = reduction_proof
        params = self.params
        width, quotient_degree = self.width, self.quotient_degree
        log_blowup = params.log_blowup
        if proof.trace_local.shape != (width,) or proof.trace_next.shape != (width,):
            raise ValueError("opened trace width does not match the AIR")
        if proof.quotient_chunks.shape != (quotient_degree, 4):
            raise ValueError("quotient chunk shape does not match the AIR degree")

        opened = fnp.concatenate(
            [proof.trace_local, proof.trace_next, proof.quotient_chunks.reshape(-1)]
        )
        t = transcript.observe(opened)
        t, alpha_fri = sample_ext(t)

        betas = []
        for root in proof.fri.commit_phase_roots:
            t = t.observe(root)
            t, beta = sample_ext(t)
            betas.append(beta)
        t = t.observe(proof.fri.final_poly)
        t, ok = t.check_witness(
            proof.fri.pow_witness, pow_bits=params.proof_of_work_bits
        )

        log_max_height = claim.degree_bits + log_blowup
        lde_height = 1 << log_max_height
        if len(proof.fri.commit_phase_roots) != log_max_height - log_blowup:
            raise ValueError("commit phase layer count does not match the height")
        t, indices = sample_query_indices(t, log_max_height, params.num_queries)

        xs_br = _bitrev_lde_domain(lde_height)
        points = fnp.stack([claim.zeta, claim.zeta_next])
        opening_pos = _opening_pos(width, quotient_degree)
        code = BitReversedReedSolomon(1 << claim.degree_bits, 1 << log_blowup, F)

        if proof.fri.trace_openings.row.shape != (
            params.num_queries,
            width,
        ) or proof.fri.quotient_openings.row.shape != (
            params.num_queries,
            4 * quotient_degree,
        ):
            raise ValueError("input opening shape mismatch")
        for q, idx in enumerate(map(int, indices)):
            trace_open = query_opening(proof.fri.trace_openings, q)
            quotient_open = query_opening(proof.fri.quotient_openings, q)
            ok = ok & fnp.array_equal(
                self.tree.reconstruct_root(idx, trace_open), claim.trace_root
            )
            ok = ok & fnp.array_equal(
                self.tree.reconstruct_root(idx, quotient_open), claim.quotient_root
            )

            columns = fnp.concatenate(
                [trace_open.row, trace_open.row, quotient_open.row]
            )[None, :]
            value = reduced_openings(
                columns, opened, points, opening_pos, alpha_fri, xs_br[idx : idx + 1]
            )[0]

            for layer, batched in enumerate(proof.fri.commit_phase_openings):
                opening = query_opening(batched, q)
                index_i = idx >> layer
                pair_index = index_i >> 1
                pair = lax.bitcast_convert_type(
                    opening.row.reshape(2, 4), EF
                ).reshape(2)
                # The opened row must bind the running value at this index's
                # slot before it feeds the fold.
                ok = ok & fnp.array_equal(pair[index_i & 1], value)
                ok = ok & fnp.array_equal(
                    self.tree.reconstruct_root(pair_index, opening),
                    proof.fri.commit_phase_roots[layer],
                )
                value = code.fold_values(
                    pair[0:1], pair[1:2], betas[layer], fnp.asarray([pair_index]), layer
                )[0]
            ok = ok & fnp.array_equal(value, proof.fri.final_poly)

        return VerifyResult(TrivialClaim(), t, ok)
