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

Transcript activity all happens through the one `DuplexTranscript` flavour,
so prover and verifier stay in the same claim-reduction seams as every other
zorch scheme.
"""

from __future__ import annotations

from dataclasses import dataclass

import frx.numpy as fnp
import numpy as np
from frx import Array, lax
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from zorch.commit.merkle import MerkleTree
from zorch.poly.univariate import powers
from zorch.stage import ProveResult, ProverStage, TrivialClaim
from zorch.transcript import DuplexTranscript

from pico_zorch.challenger.challenger import sample_ext
from pico_zorch.commit.pcs_commit import (
    GENERATOR,
    CommitData,
    _bit_reverse_indices,
    bit_reverse_rows,
    commit_matrices,
    commit_pcs,
)
from pico_zorch.uni_stark.domain import Coset
from pico_zorch.uni_stark.quotient import flatten_to_base, quotient_values
from pico_zorch.uni_stark.types import (
    CommitPhaseOpening,
    FriParams,
    FriProof,
    StarkClaim,
    StarkProof,
    StarkWitness,
)


def _log2_ceil(x: int) -> int:
    return (x - 1).bit_length()


def eval_matrix_at(evals: Array, shift: Array, z: Array) -> Array:
    """Evaluate each column polynomial of `[n, w]` evaluations on the coset
    shift·<g_n> at the extension point `z` (out-of-domain).

    Interpolate on the plain subgroup (the coefficients of p̃(y) = p(shift·y))
    and Horner at y = z/shift — field-identical to the reference's barycentric
    `interpolate_coset`."""
    n, _ = evals.shape
    coeffs = lax.ntt(evals.T, ntt_type="INTT", ntt_length=n)  # [w, n]
    y = z * (fnp.ones((), shift.dtype) / shift).astype(z.dtype)
    ypow = powers(y, n)  # [n]
    return (coeffs.astype(z.dtype) * ypow[None, :]).sum(axis=-1)


def _fold_points(pair_count: int) -> Array:
    """x-coordinate of pair i in a bit-reversed layer of 2·pair_count values:
    g_{2h}^{rev_{log h}(i)} — the reference fold_matrix's power schedule (no
    coset shift; FRI's fold domain is the plain subgroup)."""
    from zorch.coding.reed_solomon import eval_domain

    dom = eval_domain(F, 2 * pair_count)
    return fnp.take(dom, fnp.asarray(_bit_reverse_indices(pair_count)), axis=0)


def fold_pair_layer(folded: Array, beta: Array) -> Array:
    """One arity-2 fold of a bit-reversed codeword: pair i = (e0, e1) at
    ±g_{2h}^{rev(i)} folds to (e0+e1)/2 + β·(e0−e1)/(2x)."""
    h = folded.shape[0] // 2
    pairs = folded.reshape(h, 2)
    e0, e1 = pairs[:, 0], pairs[:, 1]
    x = _fold_points(h).astype(folded.dtype)
    half = (fnp.ones((), F) / fnp.array(2, dtype=F)).astype(folded.dtype)
    return (e0 + e1) * half + beta * (e0 - e1) * half / x


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

        # -- commit trace (not a transcript operation), then bind the instance.
        trace_root, trace_data = commit_pcs(self.tree, [trace], log_blowup)
        t = transcript.observe(fnp.array([log_n], dtype=F))
        t = t.observe(trace_root)
        t = t.observe(pv)
        t, alpha = sample_ext(t)

        # -- quotient on the disjoint coset.
        quotient_domain = Coset(log_n + log_qd, generator)
        trace_on_qd = self._natural_rows(trace_data, 0, quotient_domain.size)
        q_vals = quotient_values(air, pv, trace_domain, quotient_domain, trace_on_qd, alpha)
        q_flat = flatten_to_base(q_vals)
        chunks = quotient_domain.split_evals(quotient_degree, q_flat)
        qc_domains = quotient_domain.split_domains(quotient_degree)
        chunk_shifts = [generator / d.shift for d in qc_domains]
        quotient_root, quotient_data = commit_pcs(
            self.tree, chunks, log_blowup, shifts=chunk_shifts
        )
        t = t.observe(quotient_root)
        t, zeta = sample_ext(t)
        zeta_next = trace_domain.next_point(zeta)

        # -- out-of-domain opened values.
        trace_local = eval_matrix_at(trace, one, zeta)
        trace_next = eval_matrix_at(trace, one, zeta_next)
        chunk_values = fnp.stack(
            [
                eval_matrix_at(c, d.shift, zeta)
                for c, d in zip(chunks, qc_domains)
            ]
        )  # [quotient_degree, 4] EF

        # -- pcs.open: observe every opened value, then the FRI batching α.
        for ys in (trace_local, trace_next):
            for i in range(ys.shape[0]):
                t = t.observe(ys[i])
        for ci in range(chunk_values.shape[0]):
            for i in range(chunk_values.shape[1]):
                t = t.observe(chunk_values[ci, i])
        t, alpha_fri = sample_ext(t)

        # -- reduced openings over the full bit-reversed LDE height.
        lde_height = n << log_blowup
        xs = self._bitrev_coset_points(lde_height, generator)
        ro = fnp.zeros((lde_height,), dtype=EF)
        num_reduced = 0
        alpha_pows = powers(alpha_fri, 8 * (2 + quotient_degree))
        rounds = [
            (trace_data.matrices[0], zeta, trace_local),
            (trace_data.matrices[0], zeta_next, trace_next),
        ] + [
            (quotient_data.matrices[i], zeta, chunk_values[i])
            for i in range(quotient_degree)
        ]
        for mat, z, ys in rounds:
            width = mat.shape[1]
            y_red = fnp.zeros((), dtype=EF)
            p_red = fnp.zeros((lde_height,), dtype=EF)
            for i in range(width):
                y_red = y_red + alpha_pows[i] * ys[i]
                p_red = p_red + alpha_pows[i] * mat[:, i].astype(EF)
            inv_denom = fnp.ones((), EF) / (z - xs.astype(EF))
            ro = ro + alpha_pows[num_reduced] * (y_red - p_red) * inv_denom
            num_reduced += width

        # -- FRI commit phase.
        folded = ro
        commit_roots: list[Array] = []
        phase_data: list = []
        while folded.shape[0] > (1 << log_blowup):
            pairs_base = lax.bitcast_convert_type(
                folded.reshape(folded.shape[0] // 2, 2), F
            ).reshape(folded.shape[0] // 2, 8)
            root, data = commit_matrices(self.tree, [pairs_base])
            commit_roots.append(root)
            phase_data.append(data)
            t = t.observe(root)
            t, beta = sample_ext(t)
            folded = fold_pair_layer(folded, beta)
        final_poly = folded[0]
        t = t.observe(final_poly)

        # -- proof of work, then query indices (low bits of one squeeze each).
        t, pow_witness = t.grind(self.params.proof_of_work_bits)
        log_max_height = log_n + log_blowup
        t, raw = t.sample(self.params.num_queries)
        canonical = np.asarray(lax.convert_element_type(raw, fnp.uint32))
        indices = (canonical & ((1 << log_max_height) - 1)).astype(np.int64)

        # -- openings.
        input_openings = []
        phase_openings = []
        for idx in indices:
            input_openings.append(
                [
                    self.tree.open(
                        trace_data.leaves, trace_data.digest_layers, int(idx)
                    ),
                    self.tree.open(
                        quotient_data.leaves, quotient_data.digest_layers, int(idx)
                    ),
                ]
            )
            steps = []
            for layer, data in enumerate(phase_data):
                pair_index = (int(idx) >> layer) >> 1
                steps.append(
                    CommitPhaseOpening(
                        self.tree.open(data.leaves, data.digest_layers, pair_index)
                    )
                )
            phase_openings.append(steps)

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

    @staticmethod
    def _bitrev_coset_points(n: int, shift: Array) -> Array:
        from zorch.coding.reed_solomon import eval_domain

        nat = eval_domain(F, n, shift=shift)
        return nat[_bit_reverse_indices(n)]
