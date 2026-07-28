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

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array, lax
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from zorch.coding.reed_solomon import BitReversedReedSolomon, eval_domain
from zorch.commit.merkle import MerkleTree, Opening
from zorch.pcs.deep import deep_composition
from zorch.poly.univariate import eval_coeffs
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


def eval_matrix_at(evals: Array, shift: Array, z: Array) -> Array:
    """Out-of-domain evaluation of each column of `[n, w]` coset evaluations:
    interpolate on the plain subgroup (coefficients of p̃(y) = p(shift·y)) and
    evaluate at y = z/shift — field-identical to the reference's barycentric
    `interpolate_coset`."""
    n, _ = evals.shape
    coeffs = lax.ntt(evals.T, ntt_type="INTT", ntt_length=n)
    return eval_coeffs(coeffs, z / shift.astype(z.dtype))


def reduced_openings(
    columns: Array,
    values: Array,
    points: Array,
    opening_pos: list[int],
    alpha: Array,
    domain: Array,
) -> Array:
    """The reference's α-batched opening reduction, which is zorch's DEEP-ALI
    composition: Σ_m α^m·(col_m − values[m])/(domain − points[pos_m]). The
    reference's per-(matrix, point) α offsets flatten to consecutive powers
    because each offset is the running column count."""
    ext_none = fnp.zeros((columns.shape[0], 0), dtype=EF)
    return deep_composition(
        columns, ext_none, values, points, opening_pos, alpha, domain
    )


def sample_query_indices(
    transcript: DuplexTranscript, log_max_height: int, count: int
) -> tuple[DuplexTranscript, np.ndarray]:
    """`count` × the reference's `sample_bits(log_max_height)`: one squeeze
    each, low bits of the canonical value (zorch's `sample_positions` reduces
    the Montgomery bitpattern instead, so it cannot be used here)."""
    t, raw = transcript.sample(count)
    canonical = np.asarray(lax.convert_element_type(raw, fnp.uint32))
    return t, (canonical & ((1 << log_max_height) - 1)).astype(np.int64)


def _observe_each(t: DuplexTranscript, values: Array) -> DuplexTranscript:
    for value in values:
        t = t.observe(value)
    return t


def _opening_pos(width: int, quotient_degree: int) -> list[int]:
    """Point index per flattened column: trace at ζ, trace at ζ·g, chunks
    at ζ."""
    return [0] * width + [1] * width + [0] * (4 * quotient_degree)


def open_batch(tree: MerkleTree, data: CommitData, indices: Array) -> Opening:
    """All queries' openings of one tree as a single vmapped device call — the
    per-query eager loop was 95% of the prove's wall clock."""
    return frx.vmap(lambda i: tree.open(data.leaves, data.digest_layers, i))(indices)


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

        trace_local = eval_matrix_at(witness.trace, one, claim.zeta)
        trace_next = eval_matrix_at(witness.trace, one, claim.zeta_next)
        chunk_values = fnp.stack(
            [
                eval_matrix_at(c, d.shift, claim.zeta)
                for c, d in zip(quotient.chunks, quotient.qc_domains)
            ]
        )

        t = _observe_each(
            transcript,
            fnp.concatenate([trace_local, trace_next, chunk_values.reshape(-1)]),
        )
        t, alpha_fri = sample_ext(t)

        lde_height = n << log_blowup
        ro = reduced_openings(
            fnp.concatenate(
                [
                    witness.trace_data.leaves,
                    witness.trace_data.leaves,
                    quotient.quotient_data.leaves,
                ],
                axis=1,
            ),
            fnp.concatenate([trace_local, trace_next, chunk_values.reshape(-1)]),
            fnp.stack([claim.zeta, claim.zeta_next]),
            _opening_pos(width, quotient_degree),
            alpha_fri,
            _bitrev_lde_domain(lde_height),
        )

        code = BitReversedReedSolomon(n, 1 << log_blowup, F)
        folded = ro
        commit_roots: list[Array] = []
        phase_data: list[CommitData] = []
        while folded.shape[0] > (1 << log_blowup):
            pairs_base = lax.bitcast_convert_type(code.pair_leaves(folded), F)
            root, data = commit_matrices(self.tree, [pairs_base.reshape(-1, 8)])
            commit_roots.append(root)
            phase_data.append(data)
            t = t.observe(root)
            t, beta = sample_ext(t)
            folded = code.fold(folded, beta)
        final_poly = folded[0]
        t = t.observe(final_poly)

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
        t = _observe_each(transcript, opened)
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
