# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The uni-stark verifier stage — the prover's explicit dual.

Replays the same transcript schedule (Plonky3 uni-stark/src/verifier.rs +
fri/src/verifier.rs at brevis-network/Plonky3@7fbe1908), re-derives every
challenge, checks each query's Merkle bindings and fold chain against the
final polynomial, and closes with the out-of-domain identity
folded_constraints(ζ)/Z_H(ζ) == quotient(ζ). Algebraic failure lands in
`VerifyResult.ok`; structurally impossible proofs raise."""

from __future__ import annotations

from dataclasses import dataclass

import frx.numpy as fnp
import numpy as np
from frx import Array, lax
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from zorch.coding.reed_solomon import BitReversedReedSolomon, eval_domain
from zorch.commit.merkle import MerkleTree
from zorch.stage import TrivialClaim, VerifierStage, VerifyResult
from zorch.transcript import DuplexTranscript

from pico_zorch.challenger.challenger import sample_ext
from pico_zorch.commit.pcs_commit import GENERATOR
from pico_zorch.uni_stark.domain import Coset
from pico_zorch.uni_stark.prover import (
    _log2_ceil,
    reduced_openings,
    sample_query_indices,
)
from pico_zorch.uni_stark.types import FriParams, StarkClaim, StarkProof


def _ext_monomial(e: int) -> Array:
    limbs = np.zeros(4, dtype=np.uint32)
    limbs[e] = 1
    return lax.bitcast_convert_type(fnp.array(limbs, dtype=F), EF).reshape(())


@dataclass(frozen=True)
class StarkVerifier(
    VerifierStage[StarkClaim, TrivialClaim, StarkProof, DuplexTranscript]
):
    tree: MerkleTree
    params: FriParams = FriParams()

    def verify(
        self,
        claim: StarkClaim,
        reduction_proof: StarkProof,
        transcript: DuplexTranscript,
    ) -> VerifyResult[TrivialClaim, DuplexTranscript]:
        air, pv = claim.air, claim.public_values
        proof = reduction_proof
        log_n = proof.degree_bits
        log_blowup = self.params.log_blowup
        one = fnp.ones((), F)
        generator = fnp.array(GENERATOR, dtype=F)

        log_qd = _log2_ceil(max(air.constraint_degree - 1, 1))
        quotient_degree = 1 << log_qd
        width = air.width
        if proof.trace_local.shape != (width,) or proof.trace_next.shape != (width,):
            raise ValueError("opened trace width does not match the AIR")
        if proof.quotient_chunks.shape != (quotient_degree, 4):
            raise ValueError("quotient chunk shape does not match the AIR degree")

        trace_domain = Coset(log_n, one)
        quotient_domain = Coset(log_n + log_qd, generator)
        qc_domains = quotient_domain.split_domains(quotient_degree)

        t = transcript.observe(fnp.array([log_n], dtype=F))
        t = t.observe(proof.trace_root)
        t = t.observe(pv)
        t, alpha = sample_ext(t)
        t = t.observe(proof.quotient_root)
        t, zeta = sample_ext(t)
        zeta_next = trace_domain.next_point(zeta)

        opened = fnp.concatenate(
            [proof.trace_local, proof.trace_next, proof.quotient_chunks.reshape(-1)]
        )
        for value in opened:
            t = t.observe(value)
        t, alpha_fri = sample_ext(t)

        betas = []
        for root in proof.fri.commit_phase_roots:
            t = t.observe(root)
            t, beta = sample_ext(t)
            betas.append(beta)
        t = t.observe(proof.fri.final_poly)
        t, ok = t.check_witness(
            proof.fri.pow_witness, pow_bits=self.params.proof_of_work_bits
        )

        log_max_height = log_n + log_blowup
        lde_height = 1 << log_max_height
        if len(proof.fri.commit_phase_roots) != log_max_height - log_blowup:
            raise ValueError("commit phase layer count does not match the height")
        t, indices = sample_query_indices(
            t, log_max_height, self.params.num_queries
        )

        xs_br = lax.bit_reverse(
            eval_domain(F, lde_height, shift=generator), dimensions=(0,)
        )
        points = fnp.stack([zeta, zeta_next])
        opening_pos = [0] * width + [1] * width + [0] * (4 * quotient_degree)
        code = BitReversedReedSolomon(1 << log_n, 1 << log_blowup, F)

        for q, idx in enumerate(map(int, indices)):
            trace_open, quotient_open = proof.fri.input_openings[q]
            if trace_open.row.shape != (width,) or quotient_open.row.shape != (
                4 * quotient_degree,
            ):
                raise ValueError("input opening width mismatch")
            ok = ok & fnp.array_equal(
                self.tree.reconstruct_root(idx, trace_open), proof.trace_root
            )
            ok = ok & fnp.array_equal(
                self.tree.reconstruct_root(idx, quotient_open), proof.quotient_root
            )

            columns = fnp.concatenate(
                [trace_open.row, trace_open.row, quotient_open.row]
            )[None, :]
            value = reduced_openings(
                columns, opened, points, opening_pos, alpha_fri, xs_br[idx : idx + 1]
            )[0]

            for layer, opening in enumerate(proof.fri.commit_phase_openings[q]):
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

        # Out-of-domain identity at ζ. zps: each chunk's Lagrange-style factor
        # over the other chunk domains, normalized at the chunk's first point.
        zps = []
        for i, dom in enumerate(qc_domains):
            acc = fnp.ones((), EF)
            for j, other in enumerate(qc_domains):
                if j != i:
                    acc = acc * other.zp_at_point(zeta)
                    acc = acc / other.zp_at_point(dom.shift.astype(EF))
            zps.append(acc)
        quotient = fnp.zeros((), EF)
        for ci in range(quotient_degree):
            for e in range(4):
                quotient = quotient + (
                    zps[ci] * _ext_monomial(e) * proof.quotient_chunks[ci, e]
                )

        sels = trace_domain.selectors_at_point(zeta)
        constraints = air.eval(
            proof.trace_local,
            proof.trace_next,
            sels["is_first_row"],
            sels["is_last_row"],
            sels["is_transition"],
            pv.astype(EF),
        )
        folded = fnp.zeros((), EF)
        for c in constraints:
            folded = folded * alpha + c
        ok = ok & fnp.array_equal(folded * sels["inv_zeroifier"], quotient)

        return VerifyResult(TrivialClaim(), t, ok)
