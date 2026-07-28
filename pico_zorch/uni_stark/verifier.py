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

from zorch.coding.reed_solomon import eval_domain
from zorch.commit.merkle import MerkleTree
from zorch.poly.univariate import powers
from zorch.stage import TrivialClaim, VerifierStage, VerifyResult
from zorch.transcript import DuplexTranscript

from pico_zorch.challenger.challenger import sample_ext
from pico_zorch.commit.pcs_commit import GENERATOR, _bit_reverse_indices
from pico_zorch.uni_stark.domain import Coset
from pico_zorch.uni_stark.prover import _log2_ceil
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
        params = self.params
        log_n = proof.degree_bits
        log_blowup = params.log_blowup
        one = fnp.ones((), F)
        generator = fnp.array(GENERATOR, dtype=F)

        log_qd = _log2_ceil(max(air.constraint_degree - 1, 1))
        quotient_degree = 1 << log_qd
        if proof.trace_local.shape != (air.width,) or proof.trace_next.shape != (
            air.width,
        ):
            raise ValueError("opened trace width does not match the AIR")
        if proof.quotient_chunks.shape != (quotient_degree, 4):
            raise ValueError("quotient chunk shape does not match the AIR degree")

        trace_domain = Coset(log_n, one)
        quotient_domain = Coset(log_n + log_qd, generator)
        qc_domains = quotient_domain.split_domains(quotient_degree)

        # -- instance binding and challenges, the prover's exact schedule.
        t = transcript.observe(fnp.array([log_n], dtype=F))
        t = t.observe(proof.trace_root)
        t = t.observe(pv)
        t, alpha = sample_ext(t)
        t = t.observe(proof.quotient_root)
        t, zeta = sample_ext(t)
        zeta_next = trace_domain.next_point(zeta)

        for ys in (proof.trace_local, proof.trace_next):
            for i in range(ys.shape[0]):
                t = t.observe(ys[i])
        for ci in range(proof.quotient_chunks.shape[0]):
            for i in range(proof.quotient_chunks.shape[1]):
                t = t.observe(proof.quotient_chunks[ci, i])
        t, alpha_fri = sample_ext(t)

        betas = []
        for root in proof.fri.commit_phase_roots:
            t = t.observe(root)
            t, beta = sample_ext(t)
            betas.append(beta)
        t = t.observe(proof.fri.final_poly)
        t, pow_ok = t.check_witness(
            proof.fri.pow_witness, pow_bits=params.proof_of_work_bits
        )

        log_max_height = log_n + log_blowup
        lde_height = 1 << log_max_height
        num_layers = log_max_height - log_blowup
        if len(proof.fri.commit_phase_roots) != num_layers:
            raise ValueError("commit phase layer count does not match the height")

        t, raw = t.sample(params.num_queries)
        canonical = np.asarray(lax.convert_element_type(raw, fnp.uint32))
        indices = (canonical & (lde_height - 1)).astype(np.int64)

        xs_nat = eval_domain(F, lde_height, shift=generator)
        xs_br = fnp.take(xs_nat, fnp.asarray(_bit_reverse_indices(lde_height)), axis=0)
        alpha_pows = powers(alpha_fri, 8 * (2 + quotient_degree))
        fold_doms = [
            eval_domain(F, 1 << (log_max_height - layer))
            for layer in range(num_layers)
        ]

        ok = pow_ok
        for q, idx in enumerate(indices):
            idx = int(idx)
            trace_open, quotient_open = proof.fri.input_openings[q]
            if trace_open.row.shape != (air.width,) or quotient_open.row.shape != (
                4 * quotient_degree,
            ):
                raise ValueError("input opening width mismatch")
            ok = ok & fnp.array_equal(
                self.tree.reconstruct_root(idx, trace_open), proof.trace_root
            )
            ok = ok & fnp.array_equal(
                self.tree.reconstruct_root(idx, quotient_open), proof.quotient_root
            )

            # Reduced opening at the queried point, the prover's formula on the
            # opened rows.
            x = xs_br[idx].astype(EF)
            ro = fnp.zeros((), dtype=EF)
            num_reduced = 0
            legs = [
                (trace_open.row, zeta, proof.trace_local),
                (trace_open.row, zeta_next, proof.trace_next),
            ] + [
                (
                    quotient_open.row[4 * i : 4 * (i + 1)],
                    zeta,
                    proof.quotient_chunks[i],
                )
                for i in range(quotient_degree)
            ]
            for row, z, ys in legs:
                width = row.shape[0]
                y_red = fnp.zeros((), dtype=EF)
                p_red = fnp.zeros((), dtype=EF)
                for i in range(width):
                    y_red = y_red + alpha_pows[i] * ys[i]
                    p_red = p_red + alpha_pows[i] * row[i].astype(EF)
                ro = ro + alpha_pows[num_reduced] * (y_red - p_red) / (z - x)
                num_reduced += width

            # Fold chain against the committed layers.
            value = ro
            for layer, step in enumerate(proof.fri.commit_phase_openings[q]):
                index_i = idx >> layer
                pair_index = index_i >> 1
                pair = lax.bitcast_convert_type(
                    step.opening.row.reshape(2, 4), EF
                ).reshape(2)
                # The row must bind the running value at this index's slot.
                ok = ok & fnp.array_equal(pair[index_i & 1], value)
                ok = ok & fnp.array_equal(
                    self.tree.reconstruct_root(pair_index, step.opening),
                    proof.fri.commit_phase_roots[layer],
                )
                h_pairs = 1 << (log_max_height - layer - 1)
                rev = int(
                    _bit_reverse_indices(h_pairs)[pair_index]
                    if h_pairs > 1
                    else 0
                )
                x0 = fold_doms[layer][rev].astype(EF)
                half = (one / fnp.array(2, dtype=F)).astype(EF)
                e0, e1 = pair[0], pair[1]
                value = (e0 + e1) * half + betas[layer] * (e0 - e1) * half / x0
            ok = ok & fnp.array_equal(value, proof.fri.final_poly)

        # -- out-of-domain identity at zeta.
        zps = []
        for i, dom in enumerate(qc_domains):
            acc = fnp.ones((), EF)
            for j, other in enumerate(qc_domains):
                if j == i:
                    continue
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
        pv_ext = pv.astype(EF)
        constraints = air.eval(
            proof.trace_local,
            proof.trace_next,
            sels["is_first_row"],
            sels["is_last_row"],
            sels["is_transition"],
            pv_ext,
        )
        folded = fnp.zeros((), EF)
        for c in constraints:
            folded = folded * alpha + c
        ok = ok & fnp.array_equal(folded * sels["inv_zeroifier"], quotient)

        return VerifyResult(TrivialClaim(), t, ok)
