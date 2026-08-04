//! Core outputs -> `p3_uni_stark::Proof`.
//!
//! Two layout differences have to be undone here, both of them deliberate on
//! the Python side:
//!
//! * **Query-major batching.** The core opens every query of a tree in one
//!   vmapped call, so an opening arrives as `[queries, ...]` (and a Merkle path
//!   as `[depth, queries, 8]`). The reference nests the other way, one
//!   `QueryProof` per query.
//! * **Pair rows vs siblings.** The core keeps a FRI layer's whole pair row;
//!   the reference stores only the sibling. Which half that is depends on the
//!   parity of `index >> layer`, which is why the core also returns the sampled
//!   query indices.
//!
//! `Proof`, `Commitments` and `OpenedValues` have private fields, so the outer
//! three structs are rebuilt through serde. Everything below them —
//! `FriProof`, `QueryProof`, `CommitPhaseProofStep`, `BatchOpening`, `Hash` —
//! is public and built directly. Both sides of the serde hop use the very same
//! leaf types, so the round trip is a re-wrapping, not a re-encoding.

use std::collections::HashMap;

use p3_commit::Mmcs;
use p3_fri::{BatchOpening, CommitPhaseProofStep, FriProof, QueryProof};
use p3_symmetric::Hash;
use pico_zorch_golden::{Challenge, ChallengeMmcs, MyConfig, Val, ValMmcs};
use serde::Serialize;

use crate::manifest::{Manifest, Spec};
use crate::wire;

/// Digest width of the Merkle commitments (Pico's 8-element roots).
const DIGEST: usize = 8;

type Commitment = <ValMmcs as Mmcs<Val>>::Commitment;
type MerklePath = <ValMmcs as Mmcs<Val>>::Proof;
type InputProof = Vec<BatchOpening<Val, ValMmcs>>;
type OpeningProof = FriProof<Challenge, ChallengeMmcs, Val, InputProof>;

/// The outer three structs, mirrored field for field so serde can rebuild the
/// private originals. `opening_proof` is already the real type.
#[derive(Serialize)]
struct ProofWire {
    commitments: CommitmentsWire,
    opened_values: OpenedValuesWire,
    opening_proof: OpeningProof,
    degree_bits: usize,
}

#[derive(Serialize)]
struct CommitmentsWire {
    trace: Commitment,
    quotient_chunks: Commitment,
}

#[derive(Serialize)]
struct OpenedValuesWire {
    trace_local: Vec<Challenge>,
    trace_next: Vec<Challenge>,
    quotient_chunks: Vec<Vec<Challenge>>,
}

/// The core's outputs, addressable by the manifest's names.
struct Outputs<'a> {
    manifest: &'a Manifest,
    raw: &'a [Vec<u8>],
    index: HashMap<&'a str, usize>,
}

impl<'a> Outputs<'a> {
    fn new(manifest: &'a Manifest, raw: &'a [Vec<u8>]) -> Result<Self, String> {
        if raw.len() != manifest.outputs.len() {
            return Err(format!(
                "core returned {} buffers but its manifest describes {}",
                raw.len(),
                manifest.outputs.len()
            ));
        }
        Ok(Self {
            manifest,
            raw,
            index: manifest.output_index(),
        })
    }

    fn spec(&self, name: &str) -> Result<(&Spec, &[u8]), String> {
        let i = *self
            .index
            .get(name)
            .ok_or_else(|| format!("core has no output {name:?}"))?;
        let spec = &self.manifest.outputs[i];
        let bytes = &self.raw[i];
        if bytes.len() != spec.len() * wire::ELEM {
            return Err(format!(
                "output {name:?}: manifest says {} elements but the buffer holds {} bytes",
                spec.len(),
                bytes.len()
            ));
        }
        Ok((spec, bytes))
    }

    fn vals(&self, name: &str) -> Result<Vec<Val>, String> {
        let (_, bytes) = self.spec(name)?;
        wire::vals(bytes).map_err(|e| format!("output {name:?}: {e}"))
    }

    fn indices(&self, name: &str) -> Result<Vec<usize>, String> {
        let (_, bytes) = self.spec(name)?;
        wire::indices(bytes).map_err(|e| format!("output {name:?}: {e}"))
    }
}

/// One 8-element digest.
fn digest(values: &[Val]) -> Result<[Val; DIGEST], String> {
    <[Val; DIGEST]>::try_from(values)
        .map_err(|_| format!("expected a {DIGEST}-element digest, got {}", values.len()))
}

/// Query `q`'s Merkle path out of a `[depth, queries, 8]` buffer.
fn path_for_query(flat: &[Val], depth: usize, queries: usize, q: usize) -> MerklePath {
    (0..depth)
        .map(|level| {
            let start = (level * queries + q) * DIGEST;
            digest(&flat[start..start + DIGEST]).expect("slice is exactly one digest")
        })
        .collect()
}

/// `[depth, queries, 8]` dims, validated against the query count.
fn path_depth(spec_dims: &[usize], queries: usize, name: &str) -> Result<usize, String> {
    match spec_dims {
        [depth, q, DIGEST] if *q == queries => Ok(*depth),
        other => Err(format!(
            "output {name:?} should be [depth, {queries}, {DIGEST}], got {other:?}"
        )),
    }
}

/// Rebuild the reference proof from one core execution.
pub fn assemble(manifest: &Manifest, raw: &[Vec<u8>]) -> Result<p3_uni_stark::Proof<MyConfig>, String> {
    let out = Outputs::new(manifest, raw)?;
    let us = manifest.uni_stark()?;
    let queries = us.num_queries;
    let layers = manifest.num_fri_layers()?;
    let width = us.width;

    let trace_commit: Commitment = Hash::from(digest(&out.vals("trace_root")?)?);
    let quotient_commit: Commitment = Hash::from(digest(&out.vals("quotient_root")?)?);

    let trace_local = wire::challenges(&out.vals("trace_local")?)?;
    let trace_next = wire::challenges(&out.vals("trace_next")?)?;
    let quotient_chunks = wire::rows(
        &wire::challenges(&out.vals("quotient_chunks")?)?,
        us.quotient_degree,
    )?;

    let commit_phase_commits: Vec<Commitment> =
        wire::rows(&out.vals("commit_phase_roots")?, layers)?
            .iter()
            .map(|row| digest(row).map(Hash::from))
            .collect::<Result<_, _>>()?;

    let final_poly = wire::challenge(&out.vals("final_poly")?)?;
    let pow_witness = *out
        .vals("pow_witness")?
        .first()
        .ok_or("pow_witness output is empty")?;

    let indices = out.indices("query_indices")?;
    if indices.len() != queries {
        return Err(format!(
            "core sampled {} query indices but the config asks for {queries}",
            indices.len()
        ));
    }

    // The two input rounds, in the order the reference opens them.
    let trace_rows = wire::rows(&out.vals("trace_opening_rows")?, queries)?;
    let quotient_rows = wire::rows(&out.vals("quotient_opening_rows")?, queries)?;
    let trace_paths = out.vals("trace_opening_paths")?;
    let quotient_paths = out.vals("quotient_opening_paths")?;
    let trace_depth = path_depth(
        &out.spec("trace_opening_paths")?.0.dims,
        queries,
        "trace_opening_paths",
    )?;
    let quotient_depth = path_depth(
        &out.spec("quotient_opening_paths")?.0.dims,
        queries,
        "quotient_opening_paths",
    )?;
    if trace_rows.first().map_or(0, Vec::len) != width {
        return Err(format!(
            "opened trace rows are {} wide, expected {width}",
            trace_rows.first().map_or(0, Vec::len)
        ));
    }

    // Per-layer pair rows and paths, read once and indexed per query below.
    let mut layer_rows = Vec::with_capacity(layers);
    let mut layer_paths = Vec::with_capacity(layers);
    for layer in 0..layers {
        let rows_name = format!("fri_layer{layer}_rows");
        let paths_name = format!("fri_layer{layer}_paths");
        layer_rows.push(wire::rows(&out.vals(&rows_name)?, queries)?);
        let depth = path_depth(&out.spec(&paths_name)?.0.dims, queries, &paths_name)?;
        layer_paths.push((out.vals(&paths_name)?, depth));
    }

    let query_proofs = (0..queries)
        .map(|q| {
            let input_proof: InputProof = vec![
                BatchOpening {
                    opened_values: vec![trace_rows[q].clone()],
                    opening_proof: path_for_query(&trace_paths, trace_depth, queries, q),
                },
                BatchOpening {
                    opened_values: vec![quotient_rows[q].clone()],
                    opening_proof: path_for_query(&quotient_paths, quotient_depth, queries, q),
                },
            ];
            let commit_phase_openings = (0..layers)
                .map(|layer| {
                    // The pair row holds both halves; the reference keeps the
                    // one the folding does *not* land on. Which that is follows
                    // the parity of the index at this layer.
                    let sibling = ((indices[q] >> layer) & 1) ^ 1;
                    let pair = &layer_rows[layer][q];
                    let coeffs = &pair[sibling * 4..(sibling + 1) * 4];
                    let (flat, depth) = &layer_paths[layer];
                    Ok(CommitPhaseProofStep {
                        sibling_value: wire::challenge(coeffs)?,
                        opening_proof: path_for_query(flat, *depth, queries, q),
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(QueryProof {
                input_proof,
                commit_phase_openings,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    let wire_proof = ProofWire {
        commitments: CommitmentsWire {
            trace: trace_commit,
            quotient_chunks: quotient_commit,
        },
        opened_values: OpenedValuesWire {
            trace_local,
            trace_next,
            quotient_chunks,
        },
        opening_proof: OpeningProof {
            commit_phase_commits,
            query_proofs,
            final_poly,
            pow_witness,
        },
        degree_bits: us.degree_bits,
    };

    // Both ends of this hop use identical leaf types, so it re-wraps the
    // private outer structs without reinterpreting a single field.
    //
    // bincode, not JSON. This hop exists only because `Proof`'s fields are
    // private; it carries no information, so every byte it spends is waste.
    // JSON made it half the wall time of a 2^16 proof — it wrote field names
    // and decimal digits for ~84 queries' worth of Merkle paths, then parsed
    // them back. bincode writes field *order*, which is all a round trip
    // between two identical shapes needs.
    //
    // The cost of dropping names: if p3 ever reorders `Proof`'s fields, this
    // mis-decodes silently where JSON would have failed on a name mismatch.
    // `tests/wire_round_trip.rs` compares a rebuilt proof against a real
    // `p3_uni_stark::prove` output, so that reordering fails there instead.
    let encoded = bincode::serialize(&wire_proof)
        .map_err(|e| format!("serialize the assembled proof: {e}"))?;
    bincode::deserialize(&encoded).map_err(|e| format!("rebuild p3_uni_stark::Proof: {e}"))
}
