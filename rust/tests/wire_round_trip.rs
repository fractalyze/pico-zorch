//! Proof reassembly, exercised against a real reference proof without a GPU.
//!
//! `gpu_byte_match` is the contract, but it needs a device and an exported
//! core. This covers the half of the binding that is pure Rust — manifest
//! binding, buffer sizing, the query-major transposition, path indexing,
//! extension grouping, and the serde hop that rebuilds `Proof`'s private
//! fields — by encoding a genuine `p3_uni_stark::prove` output into the core's
//! wire layout and asserting `assemble` recovers it exactly.
//!
//! What it deliberately does *not* prove: that the core emits this layout.
//! Encoder and decoder here are written to be inverses, so a shared
//! misunderstanding would survive. Only `gpu_byte_match` pins the layout
//! against reality. The two sampled-index-dependent details — which half of a
//! FRI pair row holds the sibling, and hence the parity convention — are in
//! that category, which is why the non-sibling half is filled with a sentinel:
//! picking the wrong half yields the sentinel, not a near-miss.

use p3_field::{FieldAlgebra, FieldExtensionAlgebra, PrimeField32};
use p3_fri::{BatchOpening, FriProof};
use pico_zorch::manifest::{Manifest, Spec};
use pico_zorch::proof::assemble;
use pico_zorch_golden::{
    config, fib_trace, pico_perm, Challenge, ChallengeMmcs, Challenger, FibonacciAir, Val, ValMmcs,
};
use serde::Deserialize;

const DEGREE_BITS: usize = 3;
const WIDTH: usize = 2;
const DIGEST: usize = 8;
const NUM_QUERIES: usize = pico_zorch_golden::NUM_QUERIES;

/// A value no real field element in the proof will collide with, used for the
/// half of a FRI pair row the reference does not carry.
const SENTINEL: u32 = 0x0BAD_BEEF % 0x7F00_0001;

// The real commitment type: `Hash` has its own serde shape, so a bare
// `[Val; 8]` here would not deserialize.
type Commitment = p3_symmetric::Hash<Val, Val, DIGEST>;
type Opening = FriProof<Challenge, ChallengeMmcs, Val, Vec<BatchOpening<Val, ValMmcs>>>;

/// `p3_uni_stark::Proof` has private fields; read it back through serde.
#[derive(Deserialize)]
struct Mirror {
    commitments: Commitments,
    opened_values: OpenedValues,
    opening_proof: Opening,
    #[allow(dead_code)]
    degree_bits: usize,
}

#[derive(Deserialize)]
struct Commitments {
    trace: Commitment,
    quotient_chunks: Commitment,
}

#[derive(Deserialize)]
struct OpenedValues {
    trace_local: Vec<Challenge>,
    trace_next: Vec<Challenge>,
    quotient_chunks: Vec<Vec<Challenge>>,
}

/// The wire encoder — the inverse of `proof::assemble`, in the core's layout.
struct Encoder {
    specs: Vec<Spec>,
    buffers: Vec<Vec<u8>>,
}

impl Encoder {
    fn new() -> Self {
        Self {
            specs: Vec::new(),
            buffers: Vec::new(),
        }
    }

    fn push(&mut self, name: &str, dims: Vec<usize>, values: &[Val]) {
        assert_eq!(
            values.len(),
            dims.iter().product::<usize>().max(1),
            "{name}: value count must match dims"
        );
        self.specs.push(Spec {
            name: name.to_string(),
            dtype: "koalabear_mont".into(),
            dims,
        });
        self.buffers
            .push(values.iter().flat_map(limb_bytes).collect());
    }

    fn push_indices(&mut self, name: &str, indices: &[usize]) {
        self.specs.push(Spec {
            name: name.to_string(),
            dtype: "int32".into(),
            dims: vec![indices.len()],
        });
        self.buffers.push(
            indices
                .iter()
                .flat_map(|&i| (i as i32).to_le_bytes())
                .collect(),
        );
    }
}

/// A field element's Montgomery limb, little-endian — mirroring what the core
/// writes. Goes through the public canonical door so this encoder does not
/// depend on the same `repr` assumption the decoder is testing.
fn limb_bytes(v: &Val) -> [u8; 4] {
    let canonical = u64::from(v.as_canonical_u32());
    let r = (1u64 << 32) % u64::from(Val::ORDER_U32);
    (((canonical * r) % u64::from(Val::ORDER_U32)) as u32).to_le_bytes()
}

fn base_of(c: &Challenge) -> Vec<Val> {
    c.as_base_slice().to_vec()
}

/// Deterministic stand-in for the sampled query indices. The reference proof
/// does not carry them and re-deriving them means replaying Fiat-Shamir, so
/// the round trip supplies its own; see the module docs.
fn synthetic_indices(count: usize, log_max_height: usize) -> Vec<usize> {
    (0..count)
        .map(|q| (q * 2_654_435_761) % (1 << log_max_height))
        .collect()
}

fn encode(mirror: &Mirror, indices: &[usize], quotient_degree: usize) -> (Manifest, Vec<Vec<u8>>) {
    let fri = &mirror.opening_proof;
    let queries = fri.query_proofs.len();
    let layers = fri.commit_phase_commits.len();
    let mut e = Encoder::new();

    e.push(
        "trace_root",
        vec![DIGEST],
        &<[Val; DIGEST]>::from(mirror.commitments.trace),
    );
    e.push(
        "quotient_root",
        vec![DIGEST],
        &<[Val; DIGEST]>::from(mirror.commitments.quotient_chunks),
    );

    let flat = |cs: &[Challenge]| cs.iter().flat_map(base_of).collect::<Vec<_>>();
    e.push(
        "trace_local",
        vec![WIDTH, 4],
        &flat(&mirror.opened_values.trace_local),
    );
    e.push(
        "trace_next",
        vec![WIDTH, 4],
        &flat(&mirror.opened_values.trace_next),
    );
    let chunks: Vec<Challenge> = mirror
        .opened_values
        .quotient_chunks
        .iter()
        .flatten()
        .copied()
        .collect();
    e.push(
        "quotient_chunks",
        vec![quotient_degree, 4, 4],
        &flat(&chunks),
    );

    let roots: Vec<Val> = fri
        .commit_phase_commits
        .iter()
        .flat_map(|c| <[Val; DIGEST]>::from(*c))
        .collect();
    e.push("commit_phase_roots", vec![layers, DIGEST], &roots);
    e.push("final_poly", vec![4], &base_of(&fri.final_poly));
    e.push("pow_witness", vec![], &[fri.pow_witness]);

    // Input rounds: rows are [queries, width], paths [depth, queries, 8].
    for (round, name) in [(0usize, "trace"), (1, "quotient")] {
        let rows: Vec<Val> = fri
            .query_proofs
            .iter()
            .flat_map(|qp| qp.input_proof[round].opened_values[0].clone())
            .collect();
        let row_width = rows.len() / queries;
        e.push(&format!("{name}_opening_rows"), vec![queries, row_width], &rows);

        let depth = fri.query_proofs[0].input_proof[round].opening_proof.len();
        let mut paths = Vec::with_capacity(depth * queries * DIGEST);
        for level in 0..depth {
            for qp in &fri.query_proofs {
                paths.extend_from_slice(&qp.input_proof[round].opening_proof[level]);
            }
        }
        e.push(
            &format!("{name}_opening_paths"),
            vec![depth, queries, DIGEST],
            &paths,
        );
    }

    e.push_indices("query_indices", indices);

    for layer in 0..layers {
        // Pair row: the reference's sibling in the slot its index parity
        // chose, a sentinel in the other.
        let mut rows = Vec::with_capacity(queries * DIGEST);
        for (q, qp) in fri.query_proofs.iter().enumerate() {
            let sibling = ((indices[q] >> layer) & 1) ^ 1;
            let coeffs = base_of(&qp.commit_phase_openings[layer].sibling_value);
            let filler = vec![Val::from_canonical_u32(SENTINEL); 4];
            let (first, second) = if sibling == 0 {
                (&coeffs, &filler)
            } else {
                (&filler, &coeffs)
            };
            rows.extend_from_slice(first);
            rows.extend_from_slice(second);
        }
        e.push(&format!("fri_layer{layer}_rows"), vec![queries, DIGEST], &rows);

        let depth = fri.query_proofs[0].commit_phase_openings[layer]
            .opening_proof
            .len();
        let mut paths = Vec::with_capacity(depth * queries * DIGEST);
        for level in 0..depth {
            for qp in &fri.query_proofs {
                paths.extend_from_slice(&qp.commit_phase_openings[layer].opening_proof[level]);
            }
        }
        e.push(
            &format!("fri_layer{layer}_paths"),
            vec![depth, queries, DIGEST],
            &paths,
        );
    }

    let manifest = Manifest {
        air: "fib".into(),
        degree_bits: DEGREE_BITS,
        width: WIDTH,
        log_blowup: pico_zorch_golden::LOG_BLOWUP,
        num_queries: queries,
        proof_of_work_bits: pico_zorch_golden::POW_BITS,
        quotient_degree,
        inputs: vec![
            Spec {
                name: "trace".into(),
                dtype: "koalabear_mont".into(),
                dims: vec![1 << DEGREE_BITS, WIDTH],
            },
            Spec {
                name: "public_values".into(),
                dtype: "koalabear_mont".into(),
                dims: vec![3],
            },
        ],
        outputs: e.specs,
    };
    (manifest, e.buffers)
}

fn reference_proof() -> (p3_uni_stark::Proof<pico_zorch::MyConfig>, Vec<Val>, FibonacciAir) {
    let (trace, last) = fib_trace(WIDTH, 1 << DEGREE_BITS);
    let public_values = vec![Val::ZERO, Val::ONE, last];
    let air = FibonacciAir { width: WIDTH };
    let mut challenger = Challenger::new(pico_perm());
    let proof = p3_uni_stark::prove(
        &config(),
        &air,
        &mut challenger,
        trace,
        &public_values,
    );
    (proof, public_values, air)
}

#[test]
fn assemble_recovers_a_reference_proof_from_the_wire() {
    let (reference, public_values, air) = reference_proof();
    let mirror: Mirror =
        serde_json::from_value(serde_json::to_value(&reference).unwrap()).expect("read reference");
    let quotient_degree = mirror.opened_values.quotient_chunks.len();

    let indices = synthetic_indices(NUM_QUERIES, DEGREE_BITS + pico_zorch_golden::LOG_BLOWUP);
    let (manifest, buffers) = encode(&mirror, &indices, quotient_degree);

    let rebuilt = assemble(&manifest, &buffers).expect("assemble");

    assert_eq!(
        serde_json::to_value(&rebuilt).unwrap(),
        serde_json::to_value(&reference).unwrap(),
        "round trip through the wire layout changed the proof"
    );

    // A proof that survives the round trip must still verify — the assertion
    // above compares serializations, this one exercises the rebuilt value.
    let mut challenger = Challenger::new(pico_perm());
    p3_uni_stark::verify(&config(), &air, &mut challenger, &rebuilt, &public_values)
        .expect("rebuilt proof must verify");
}

#[test]
fn wrong_buffer_count_is_rejected() {
    let (reference, _, _) = reference_proof();
    let mirror: Mirror =
        serde_json::from_value(serde_json::to_value(&reference).unwrap()).unwrap();
    let quotient_degree = mirror.opened_values.quotient_chunks.len();
    let indices = synthetic_indices(NUM_QUERIES, DEGREE_BITS + pico_zorch_golden::LOG_BLOWUP);
    let (manifest, mut buffers) = encode(&mirror, &indices, quotient_degree);

    buffers.pop();
    // `Proof` has no `Debug`, so unwrap the error by hand rather than with
    // `expect_err`.
    let err = match assemble(&manifest, &buffers) {
        Ok(_) => panic!("short output list must be rejected"),
        Err(e) => e,
    };
    assert!(err.contains("manifest describes"), "got: {err}");
}
