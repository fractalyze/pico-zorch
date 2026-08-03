# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Pico's FRI-backed PCS opening prover and verifier.

Everything above this point assumed the commitments hold *polynomials*. A
Merkle root binds an arbitrary function, so what remains is a proximity
test, and it discharges the evaluation claims in the same breath: for a
committed f, the quotient

    (f(X) − f(z)) / (X − z)

is a polynomial exactly when f(z) is the true evaluation, so a low-degree
proof of the batched quotient proves both low-degreeness and every opened
value at once. Drawing z outside the evaluation domain is the DEEP trick —
it denies a prover who committed to something merely *close* to a codeword
the freedom to exploit that gap.

FRI then tests low-degreeness by halving: writing f(X) = f_e(X²) + X·f_o(X²),
a verifier-chosen β collapses the pair to f_e + β·f_o over the squared
domain, one degree halving per layer until a constant remains. The prover
commits each layer first, so the queries that spot-check consecutive layers
land on a codeword it can no longer change.

Mirrors Plonky3 fri/src/two_adic_pcs.rs and the fri prover/verifier at
brevis-network/Plonky3@7fbe1908. The claim carries points and commitments,
never values: the prover computes values while opening, and the verifier
recovers them from the fold chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Any, Sequence

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array, lax
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from zorch.coding.reed_solomon import BitReversedReedSolomon, ReedSolomon
from zorch.commit.merkle import MerkleTree, Opening
from zorch.pcs.fold import from_base_field, to_base_field
from zorch.poly.univariate import powers
from zorch.stage import (
    ProveResult,
    ProverStage,
    TrivialClaim,
    VerifierStage,
    VerifyResult,
)
from pico_zorch.challenger.challenger import PicoTranscript
from pico_zorch.uni_stark.types import (
    CommitData,
    FriParams,
    FriOpeningProof,
    FriOpeningWitness,
    FriProof,
    TraceOpeningClaim,
)

# KoalaBear's multiplicative-group generator: Val::GENERATOR in the reference,
# and the coset shift for a natural (shift-1) evaluation domain.
GENERATOR = 3


def _canonical_shift(shift: Any, dtype: Any) -> int:
    """Materialize a protocol-configured field shift as a static JIT key.

    Forced to compile-time so this holds inside an enclosing trace as well:
    a coset shift is a protocol constant, and staging it would make it a
    tracer that cannot key a jit zone. Callers must pass a shift that is
    itself trace-independent.
    """
    with frx.ensure_compile_time_eval():
        field_shift = fnp.asarray(shift, dtype=dtype)
        canonical = lax.convert_element_type(field_shift, fnp.uint32)
        return int(np.asarray(canonical))


@partial(frx.jit, static_argnames=("tree", "shift_values", "log_blowup"))
def _commit(tree, evals, shift_values, log_blowup):
    ldes = []
    for matrix, shift_value in zip(evals, shift_values):
        shift = fnp.array(shift_value, dtype=matrix.dtype)
        code = ReedSolomon(
            matrix.shape[0],
            1 << log_blowup,
            matrix.dtype,
            coset_shift=shift,
        )
        natural = code.extend(matrix.T)
        ldes.append(lax.bit_reverse(natural, dimensions=(1,)).T)
    leaves = fnp.concatenate(ldes, axis=1) if len(ldes) > 1 else ldes[0]
    raw_root, digest_layers = tree.commit(leaves)
    return tuple(ldes), leaves, raw_root, tuple(digest_layers)


def _eval_columns(coeffs: Array, y: Array) -> Array:
    """`[w, n]` coefficient rows evaluated at the extension point `y`.

    Coefficients are promoted before the multiply: a 2-D mixed
    base×extension operand trips an XLA shape-rewrite RET_CHECK in the frx
    lowering, which is also why this does not call `zorch`'s `eval_coeffs`."""
    ypow = powers(y, coeffs.shape[-1])
    return (coeffs.astype(y.dtype) * ypow[None, :]).sum(axis=-1)


@partial(frx.jit, static_argnames=("width", "quotient_degree"))
def reduced_openings(
    trace_leaves: Array,
    quotient_leaves: Array,
    opened: Array,
    zeta: Array,
    zeta_next: Array,
    alpha: Array,
    domain: Array,
    width: int,
    quotient_degree: int,
) -> Array:
    """Σ_m α^m·(value_m − col_m(X))/(z_m − X) — the reference's α-batched
    opening reduction, with α powers consecutive because each round's offset
    is the running column count.

    Field-identical to the reference while looking unlike it: the trace's
    two rounds (ζ, ζ·g) share one inner column sum because they differ only
    by the offset α^w, the chunks' offsets α^{2w+4i} absorb their inner α^j,
    and the point quotients merge over a single inversion
    (n₀/d₀ + n₁/d₁ = (n₀d₁ + n₁d₀)/(d₀d₁))."""
    ap = powers(alpha, 2 * width + 4 * quotient_degree)
    trace_cols = trace_leaves.astype(EF)
    quotient_cols = quotient_leaves.astype(EF)

    def dot(coeffs: Array, cols: Array) -> Array:
        return (cols * coeffs[None, :]).sum(axis=-1)

    trace_red = dot(ap[:width], trace_cols)
    q_red = dot(ap[2 * width :], quotient_cols)
    y_local = (ap[:width] * opened[:width]).sum()
    y_next = (ap[:width] * opened[width : 2 * width]).sum()
    y_quotient = (ap[2 * width :] * opened[2 * width :]).sum()

    n_zeta = (y_local - trace_red) + (y_quotient - q_red)
    n_next = ap[width] * (y_next - trace_red)
    d_zeta = zeta - domain.astype(EF)
    d_next = zeta_next - domain.astype(EF)
    return (n_zeta * d_next + n_next * d_zeta) / (d_zeta * d_next)


@partial(frx.jit, static_argnames=("width", "quotient_degree"))
def _open_head(
    trace,
    chunks,
    chunk_shifts,
    trace_leaves,
    quotient_leaves,
    zeta,
    zeta_next,
    transcript,
    width,
    quotient_degree,
    domain,
):
    """One device program from the opened values through the reduction: the
    intermediates are full-height codewords, so a stage boundary here would
    pay a round trip per array."""
    trace_local, trace_next, chunk_values = _ood_values(
        trace, chunks, chunk_shifts, zeta, zeta_next
    )
    opened = fnp.concatenate([trace_local, trace_next, chunk_values.reshape(-1)])
    t = transcript.observe(opened)
    t, alpha_fri = t.sample_ext()
    ro = reduced_openings(
        trace_leaves,
        quotient_leaves,
        opened,
        zeta,
        zeta_next,
        alpha_fri,
        domain,
        width,
        quotient_degree,
    )
    return trace_local, trace_next, chunk_values, ro, t


def _ood_values(trace, chunks, chunk_shifts, zeta, zeta_next):
    """Interpolate on the plain subgroup (coefficients of p̃(y) = p(shift·y))
    and evaluate at y = z/shift — field-identical to the reference's
    barycentric `interpolate_coset`."""
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


def sample_query_indices(
    transcript: PicoTranscript, log_max_height: int, count: int
) -> tuple[PicoTranscript, Array]:
    """`count` × the reference's `sample_bits(log_max_height)`: one squeeze
    each, low bits of the *canonical* value. zorch's `sample_positions`
    reduces the Montgomery bitpattern instead, so it draws other indices.

    The indices stay on device. The prover feeds them straight to the batched
    opens, so materializing them here would round-trip to the host mid-proof
    — and would make the prover untraceable as a single program, which the
    Rust binding's exported core requires. The verifier walks its queries in
    Python and converts on its own.
    """
    return transcript.sample_bits_many(log_max_height, count)


@partial(frx.jit, static_argnames=("tree",))
def _open_batch(tree: MerkleTree, leaves: Array, digest_layers, indices: Array):
    return frx.vmap(lambda i: tree.open(leaves, digest_layers, i))(indices)


@partial(frx.jit, static_argnames=("tree", "num_layers"))
def _open_all(tree: MerkleTree, trees, indices: Array, num_layers: int):
    """Fold layer `l` is queried at `index >> (l+1)`: each fold halves the
    codeword, and the pair leaf holding position `i` sits at `i >> 1`."""
    (trace_leaves, trace_digests), (q_leaves, q_digests), *layers = trees
    opens = [
        frx.vmap(lambda i: tree.open(trace_leaves, trace_digests, i))(indices),
        frx.vmap(lambda i: tree.open(q_leaves, q_digests, i))(indices),
    ]
    for layer, (leaves, digests) in enumerate(layers):
        idx = indices >> (layer + 1)
        opens.append(frx.vmap(lambda i: tree.open(leaves, digests, i))(idx))
    return tuple(opens)


@partial(frx.jit, static_argnames=("tree", "code", "log_blowup"))
def fold_chain(
    tree: MerkleTree,
    code: BitReversedReedSolomon,
    log_blowup: int,
    ro: Array,
    transcript: PicoTranscript,
):
    """The FRI commit phase as one device program.

    The layer count follows the codeword shape, not data, so the `while`
    unrolls at trace time — every layer's commit, squeeze and fold land in
    a single dispatch instead of one per layer."""
    t = transcript
    folded = ro
    roots, layers = [], []
    while folded.shape[0] > (1 << log_blowup):
        leaves = to_base_field(code.pair_leaves(folded))
        root, digest_layers = tree.commit(leaves)
        t = t.observe(root)
        t, beta = t.sample_ext()
        folded = code.fold(folded, beta)
        roots.append(root)
        layers.append((leaves, tuple(digest_layers)))
    final_poly = folded[0]
    t = t.observe(final_poly)
    return final_poly, tuple(roots), tuple(layers), t


def open_batch(tree: MerkleTree, data: CommitData, indices: Array) -> Opening:
    """All queries' openings of one tree as one jitted, vmapped call —
    eager `vmap` would dispatch every per-level gather as its own kernel."""
    return _open_batch(tree, data.leaves, data.digest_layers, indices)


def query_opening(batched: Opening, q: int) -> Opening:
    return Opening(batched.row[q], [p[q] for p in batched.path])


@lru_cache(maxsize=None)
def _lde_code(n: int, log_blowup: int) -> BitReversedReedSolomon:
    """The code describing the committed layout: same coset and row order
    `FriOpener.commit` writes, so its `domain()` is the committed x-coordinates.

    Cached because constructing a coset code eagerly builds a block-length
    powers table. Forced to compile-time so the cached code is concrete
    whatever context first asks for it — a code built under a trace would
    carry a tracer coset shift into the cache and leak it into later eager
    callers."""
    with frx.ensure_compile_time_eval():
        return BitReversedReedSolomon(
            n, 1 << log_blowup, F, coset_shift=fnp.array(GENERATOR, dtype=F)
        )


@dataclass(frozen=True)
class FriOpener(
    ProverStage[
        TraceOpeningClaim, FriOpeningWitness, TrivialClaim, FriOpeningProof, PicoTranscript
    ]
):
    tree: MerkleTree
    params: FriParams = FriParams()

    def commit(
        self,
        evals: Sequence[Array],
        *,
        shifts: Sequence[Any],
    ) -> tuple[Array, CommitData]:
        """Commit evaluation matrices in Pico's bit-reversed FRI layout."""
        if not evals:
            raise ValueError("commit requires at least one matrix")
        heights = {matrix.shape[0] for matrix in evals}
        if len(heights) != 1:
            raise ValueError(f"matrices must share a height, got {sorted(heights)}")
        if len(shifts) != len(evals):
            raise ValueError(
                f"expected one shift per matrix, got {len(shifts)} for {len(evals)}"
            )

        dtype = evals[0].dtype
        shift_values = tuple(_canonical_shift(shift, dtype) for shift in shifts)
        ldes, leaves, raw_root, digest_layers = _commit(
            self.tree, tuple(evals), shift_values, self.params.log_blowup
        )
        return raw_root, CommitData(ldes, leaves, list(digest_layers))

    def prove(
        self,
        claim: TraceOpeningClaim,
        witness: FriOpeningWitness,
        transcript: PicoTranscript,
    ) -> ProveResult[TrivialClaim, FriOpeningProof, PicoTranscript]:
        params = self.params
        log_blowup = params.log_blowup
        n = witness.trace.shape[0]
        width = witness.trace.shape[1]
        quotient = witness.quotient
        quotient_degree = len(quotient.chunks)

        trace_local, trace_next, chunk_values, ro, t = _open_head(
            witness.trace,
            tuple(quotient.chunks),
            tuple(d.shift for d in quotient.qc_domains),
            witness.trace_data.leaves,
            quotient.quotient_data.leaves,
            claim.zeta,
            claim.zeta_next,
            transcript,
            width,
            quotient_degree,
            _lde_code(n, log_blowup).domain(),
        )

        code = BitReversedReedSolomon(n, 1 << log_blowup, F)
        final_poly, commit_roots, layers, t = fold_chain(
            self.tree, code, log_blowup, ro, t
        )
        phase_data = [
            CommitData((leaves,), leaves, list(digest_layers))
            for leaves, digest_layers in layers
        ]

        t, pow_witness = t.grind(params.proof_of_work_bits)
        t, idx = sample_query_indices(
            t, claim.degree_bits + log_blowup, params.num_queries
        )

        trees = tuple(
            (d.leaves, tuple(d.digest_layers))
            for d in (witness.trace_data, quotient.quotient_data, *phase_data)
        )
        trace_open, quotient_open, *layer_opens = _open_all(
            self.tree, trees, idx, len(phase_data)
        )
        proof = FriOpeningProof(
            trace_local=trace_local,
            trace_next=trace_next,
            quotient_chunks=chunk_values,
            fri=FriProof(
                commit_phase_roots=commit_roots,
                final_poly=final_poly,
                pow_witness=pow_witness,
                trace_openings=trace_open,
                quotient_openings=quotient_open,
                commit_phase_openings=list(layer_opens),
                query_indices=idx,
            ),
        )
        return ProveResult(TrivialClaim(), proof, t)


@dataclass(frozen=True)
class FriOpeningVerifier(
    VerifierStage[TraceOpeningClaim, TrivialClaim, FriOpeningProof, PicoTranscript]
):
    tree: MerkleTree
    params: FriParams = FriParams()

    def verify(
        self,
        claim: TraceOpeningClaim,
        reduction_proof: FriOpeningProof,
        transcript: PicoTranscript,
    ) -> VerifyResult[TrivialClaim, PicoTranscript]:
        proof = reduction_proof
        params = self.params
        width, quotient_degree = claim.width, claim.quotient_degree
        log_blowup = params.log_blowup
        if proof.trace_local.shape != (width,) or proof.trace_next.shape != (width,):
            raise ValueError("opened trace width does not match the AIR")
        if proof.quotient_chunks.shape != (quotient_degree, 4):
            raise ValueError("quotient chunk shape does not match the AIR degree")

        opened = fnp.concatenate(
            [proof.trace_local, proof.trace_next, proof.quotient_chunks.reshape(-1)]
        )
        t = transcript.observe(opened)
        t, alpha_fri = t.sample_ext()

        betas = []
        for root in proof.fri.commit_phase_roots:
            t = t.observe(root)
            t, beta = t.sample_ext()
            betas.append(beta)
        t = t.observe(proof.fri.final_poly)
        t, ok = t.check_witness(
            proof.fri.pow_witness, pow_bits=params.proof_of_work_bits
        )

        log_max_height = claim.degree_bits + log_blowup
        if len(proof.fri.commit_phase_roots) != log_max_height - log_blowup:
            raise ValueError("commit phase layer count does not match the height")
        t, indices = sample_query_indices(t, log_max_height, params.num_queries)
        # The query loop below is host-side Python, so the indices land here.
        indices = np.asarray(indices)

        xs_br = _lde_code(1 << claim.degree_bits, log_blowup).domain()
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

            value = reduced_openings(
                trace_open.row[None, :],
                quotient_open.row[None, :],
                opened,
                claim.zeta,
                claim.zeta_next,
                alpha_fri,
                xs_br[idx : idx + 1],
                width,
                quotient_degree,
            )[0]

            for layer, batched in enumerate(proof.fri.commit_phase_openings):
                opening = query_opening(batched, q)
                index_i = idx >> layer
                pair_index = index_i >> 1
                pair = from_base_field(opening.row[None, :], EF, 2)[0]
                # Binding before the fold is what ties the chain to the
                # commitment: an unbound row would fold to anything.
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
