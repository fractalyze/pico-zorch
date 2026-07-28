# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The uni-stark prover stage over Pico's KoalaBearPoseidon2 config.

Byte-mirrors the reference flow (Plonky3 uni-stark/src/prover.rs +
fri/src/two_adic_pcs.rs + fri/src/prover.rs at brevis-network/Plonky3@
7fbe1908) on zorch blocks:

  commit trace -> observe(log_degree, trace_root, public_values)
  -> α -> quotient on the disjoint coset, chunked commit -> observe -> ζ
  -> open: observe opened values -> α_fri -> reduced openings
  -> fold layers (observe root, β each) -> observe final -> grind
  -> query indices -> Merkle openings.
"""

from __future__ import annotations

from dataclasses import dataclass

import frx.numpy as fnp
import numpy as np
from frx import Array, lax
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from zorch.coding.reed_solomon import BitReversedReedSolomon, eval_domain
from zorch.commit.merkle import MerkleTree
from zorch.pcs.deep import deep_composition
from zorch.poly.univariate import eval_coeffs
from zorch.stage import ProveResult, ProverStage, TrivialClaim
from zorch.transcript import DuplexTranscript

from pico_zorch.challenger.challenger import sample_ext
from pico_zorch.commit.pcs_commit import GENERATOR, CommitData, bit_reverse_rows, commit_matrices, commit_pcs
from pico_zorch.uni_stark.domain import Coset
from pico_zorch.uni_stark.quotient import flatten_to_base, quotient_values
from pico_zorch.uni_stark.types import (
    FriParams,
    FriProof,
    StarkClaim,
    StarkProof,
    StarkWitness,
)


def _log2_ceil(x: int) -> int:
    return (x - 1).bit_length()


def eval_matrix_at(evals: Array, shift: Array, z: Array) -> Array:
    """Out-of-domain evaluation of each column of `[n, w]` coset evaluations:
    interpolate on the plain subgroup (coefficients of p̃(y) = p(shift·y)) and
    evaluate at y = z/shift — field-identical to the reference's barycentric
    `interpolate_coset`."""
    n, _ = evals.shape
    coeffs = lax.ntt(evals.T, ntt_type="INTT", ntt_length=n)
    return eval_coeffs(coeffs, z / shift.astype(z.dtype))


def reduced_openings(
    columns: Array, values: Array, points: Array, opening_pos: list[int], alpha: Array, domain: Array
) -> Array:
    """The reference's α-batched opening reduction, which is zorch's DEEP-ALI
    composition: Σ_m α^m·(col_m − values[m])/(domain − points[pos_m]). The
    reference's per-(matrix, point) α offsets flatten to consecutive powers
    because each offset is the running column count."""
    ext_none = fnp.zeros((columns.shape[0], 0), dtype=EF)
    return deep_composition(columns, ext_none, values, points, opening_pos, alpha, domain)


def sample_query_indices(
    transcript: DuplexTranscript, log_max_height: int, count: int
) -> tuple[DuplexTranscript, np.ndarray]:
    """`count` × the reference's `sample_bits(log_max_height)`: one squeeze
    each, low bits of the canonical value (zorch's `sample_positions` reduces
    the Montgomery bitpattern instead, so it cannot be used here)."""
    t, raw = transcript.sample(count)
    canonical = np.asarray(lax.convert_element_type(raw, fnp.uint32))
    return t, (canonical & ((1 << log_max_height) - 1)).astype(np.int64)


@dataclass(frozen=True)
class StarkProver(
    ProverStage[StarkClaim, StarkWitness, TrivialClaim, StarkProof, DuplexTranscript]
):
    tree: MerkleTree
    params: FriParams = FriParams()

    def prove(
        self,
        claim: StarkClaim,
        witness: StarkWitness,
        transcript: DuplexTranscript,
    ) -> ProveResult[TrivialClaim, StarkProof, DuplexTranscript]:
        air, pv = claim.air, claim.public_values
        trace = witness.trace
        n = trace.shape[0]
        log_n = claim.degree_bits
        log_blowup = self.params.log_blowup
        one = fnp.ones((), F)
        generator = fnp.array(GENERATOR, dtype=F)

        trace_domain = Coset(log_n, one)
        log_qd = _log2_ceil(max(air.constraint_degree - 1, 1))
        quotient_degree = 1 << log_qd

        # Committing is not a transcript operation; the instance binding is.
        trace_root, trace_data = commit_pcs(self.tree, [trace], log_blowup)
        t = transcript.observe(fnp.array([log_n], dtype=F))
        t = t.observe(trace_root)
        t = t.observe(pv)
        t, alpha = sample_ext(t)

        quotient_domain = Coset(log_n + log_qd, generator)
        trace_on_qd = self._natural_rows(trace_data, 0, quotient_domain.size)
        q_vals = quotient_values(
            air, pv, trace_domain, quotient_domain, trace_on_qd, alpha
        )
        chunks = quotient_domain.split_evals(quotient_degree, flatten_to_base(q_vals))
        qc_domains = quotient_domain.split_domains(quotient_degree)
        quotient_root, quotient_data = commit_pcs(
            self.tree,
            chunks,
            log_blowup,
            shifts=[generator / d.shift for d in qc_domains],
        )
        t = t.observe(quotient_root)
        t, zeta = sample_ext(t)
        zeta_next = trace_domain.next_point(zeta)

        trace_local = eval_matrix_at(trace, one, zeta)
        trace_next = eval_matrix_at(trace, one, zeta_next)
        chunk_values = fnp.stack(
            [eval_matrix_at(c, d.shift, zeta) for c, d in zip(chunks, qc_domains)]
        )

        for value in (*trace_local, *trace_next, *chunk_values.reshape(-1)):
            t = t.observe(value)
        t, alpha_fri = sample_ext(t)

        lde_height = n << log_blowup
        width = air.width
        xs_br = lax.bit_reverse(
            eval_domain(F, lde_height, shift=generator), dimensions=(0,)
        )
        ro = reduced_openings(
            fnp.concatenate(
                [trace_data.leaves, trace_data.leaves, quotient_data.leaves], axis=1
            ),
            fnp.concatenate([trace_local, trace_next, chunk_values.reshape(-1)]),
            fnp.stack([zeta, zeta_next]),
            [0] * width + [1] * width + [0] * (4 * quotient_degree),
            alpha_fri,
            xs_br,
        )

        code = BitReversedReedSolomon(n, 1 << log_blowup, F)
        folded = ro
        commit_roots: list[Array] = []
        phase_data: list[CommitData] = []
        while folded.shape[0] > (1 << log_blowup):
            pairs = code.pair_leaves(folded)
            pairs_base = lax.bitcast_convert_type(pairs, F).reshape(-1, 8)
            root, data = commit_matrices(self.tree, [pairs_base])
            commit_roots.append(root)
            phase_data.append(data)
            t = t.observe(root)
            t, beta = sample_ext(t)
            folded = code.fold(folded, beta)
        final_poly = folded[0]
        t = t.observe(final_poly)

        t, pow_witness = t.grind(self.params.proof_of_work_bits)
        t, indices = sample_query_indices(
            t, log_n + log_blowup, self.params.num_queries
        )

        input_openings = []
        phase_openings = []
        for idx in map(int, indices):
            input_openings.append(
                [
                    self.tree.open(trace_data.leaves, trace_data.digest_layers, idx),
                    self.tree.open(
                        quotient_data.leaves, quotient_data.digest_layers, idx
                    ),
                ]
            )
            phase_openings.append(
                [
                    self.tree.open(data.leaves, data.digest_layers, (idx >> layer) >> 1)
                    for layer, data in enumerate(phase_data)
                ]
            )

        proof = StarkProof(
            trace_root=trace_root,
            quotient_root=quotient_root,
            trace_local=trace_local,
            trace_next=trace_next,
            quotient_chunks=chunk_values,
            fri=FriProof(
                commit_phase_roots=commit_roots,
                final_poly=final_poly,
                pow_witness=pow_witness,
                input_openings=input_openings,
                commit_phase_openings=phase_openings,
            ),
            degree_bits=log_n,
        )
        return ProveResult(TrivialClaim(), proof, t)

    def _natural_rows(self, data: CommitData, mat: int, size: int) -> Array:
        """The reference `get_evaluations_on_domain`: the first `size` rows of
        the bit-reversed LDE, re-bit-reversed — the natural-order sub-coset."""
        return bit_reverse_rows(data.matrices[mat][:size])
