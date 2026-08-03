//! Golden-vector generator for pico-zorch's byte-match tests.
//!
//! Links the Plonky3 fork rev Pico v2.0.0 vendors (brevis-network/Plonky3@
//! 7fbe1908) and instantiates Pico's exact KoalaBearPoseidon2 stark config —
//! round constants from pico v2.0.0 vm/src/primitives/mod.rs, FRI
//! log_blowup=1 / num_queries=84 / proof_of_work_bits=16 from
//! vm/src/configs/stark_config/kb_poseidon2.rs. Emits JSON fixtures under
//! pico_zorch/**/testdata/golden/; regeneration is a no-op unless the
//! reference pin changes.
//!
//! Determinism requires the serial p3-maybe-rayon build (no `parallel`
//! feature): the reference grind is `find_any` over a rayon iterator, which
//! serial execution reduces to "lowest valid witness wins" — the same rule
//! zorch's grind_search implements.
//!
//! Field elements are serialized as canonical u32 (KoalaBear fits JSON
//! numbers exactly); extension elements as [c0, c1, c2, c3].
use std::fs;
use std::path::Path;

use p3_challenger::{
    CanObserve, CanSample, CanSampleBits, FieldChallenger, GrindingChallenger,
};
use p3_commit::Pcs;
use p3_field::{Field, FieldAlgebra, FieldExtensionAlgebra, PrimeField32, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_symmetric::Permutation;
use p3_uni_stark::{prove, verify};
use serde_json::{json, Value};

use pico_zorch_golden::rc::RC_16_30_U32;
use pico_zorch_golden::{
    config, fib_trace, pcs, pico_perm, Challenge, Challenger, Dft, FibonacciAir, MyPcs, Val,
    LOG_BLOWUP, NUM_QUERIES, POW_BITS,
};

fn ser_f(x: Val) -> Value {
    json!(x.as_canonical_u32())
}

fn ser_fs(xs: &[Val]) -> Value {
    Value::Array(xs.iter().copied().map(ser_f).collect())
}

fn ser_ext(x: Challenge) -> Value {
    Value::Array(x.as_base_slice().iter().copied().map(ser_f).collect())
}

fn write_json(path: &str, value: &Value) {
    let path = Path::new(path);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    // Trailing newline so the committed fixture survives end-of-file lint.
    fs::write(path, format!("{:#}\n", value)).unwrap();
    println!("wrote {}", path.display());
}

/// The permutation's own contract: canonical round constants (post
/// from_wrapped_u32) and known input/output vectors.
fn emit_poseidon2(out: &str) {
    let perm = pico_perm();

    let external: Vec<Value> = RC_16_30_U32[..4]
        .iter()
        .chain(RC_16_30_U32[24..28].iter())
        .map(|row| ser_fs(&row.map(Val::from_wrapped_u32)))
        .collect();
    let internal: Vec<Value> = RC_16_30_U32[4..24]
        .iter()
        .map(|row| ser_f(Val::from_wrapped_u32(row[0])))
        .collect();

    let mut vectors = Vec::new();
    for (name, input) in [
        ("zeros", [0u32; 16]),
        ("arange", core::array::from_fn(|i| i as u32)),
        (
            "high",
            // Values at and above the modulus boundary exercise canonical
            // reduction; from_wrapped_u32 folds them like the RC table.
            core::array::from_fn(|i| 0x7f000000u32.wrapping_add(i as u32)),
        ),
    ] {
        let state: [Val; 16] = input.map(Val::from_wrapped_u32);
        let out_state = perm.permute(state);
        vectors.push(json!({
            "name": name,
            "input": ser_fs(&state),
            "output": ser_fs(&out_state),
        }));
    }

    write_json(
        out,
        &json!({
            "external_constants": external,   // 8 rows of 16: 4 initial, 4 terminal
            "internal_constants": internal,   // 20 scalars
            "vectors": vectors,
        }),
    );
}

/// DuplexChallenger observe/sample/sample_bits/grind script. Pins the
/// overwrite-mode buffering, the pop-from-the-back sample order, the low-bits
/// index sampling, and the lowest-witness grind.
fn emit_challenger(out: &str) {
    let mut ch = Challenger::new(pico_perm());
    let mut steps = Vec::new();

    let first: Vec<Val> = (1..=3u32).map(Val::from_canonical_u32).collect();
    ch.observe_slice(&first);
    let s1: Val = ch.sample();
    let s2: Val = ch.sample();
    steps.push(json!({"op": "observe[1,2,3];sample;sample", "out": ser_fs(&[s1, s2])}));

    // 11 observes cross the rate boundary (one mid-stream duplexing).
    let second: Vec<Val> = (100..111u32).map(Val::from_canonical_u32).collect();
    ch.observe_slice(&second);
    let e1: Challenge = ch.sample_ext_element();
    steps.push(json!({"op": "observe[100..111];sample_ext", "out": ser_ext(e1)}));

    let bits: Vec<usize> = vec![ch.sample_bits(4), ch.sample_bits(16), ch.sample_bits(24)];
    steps.push(json!({"op": "sample_bits(4,16,24)", "out": bits}));

    // grind() itself asserts check_witness on the live challenger, advancing
    // it by observe(witness) + sample_bits(8).
    let witness = ch.grind(8);
    steps.push(json!({"op": "grind(8)", "out": ser_f(witness)}));

    let tail: Val = ch.sample();
    steps.push(json!({"op": "sample", "out": ser_f(tail)}));

    write_json(out, &json!({ "steps": steps }));
}

/// TwoAdicFriPcs trace commit: shift-3 coset LDE, bit-reversed rows, Merkle
/// root. The LDE matrix itself is dumped so the Python side can pin the LDE
/// and the tree independently.
fn emit_trace_commit(out: &str) {
    let (trace, _) = fib_trace(2, 8);
    let p = pcs();
    let domain =
        <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&p, trace.height());

    use p3_dft::TwoAdicSubgroupDft;
    let dft = Dft::default();
    let lde = dft
        .coset_lde_batch(trace.clone(), LOG_BLOWUP, Val::GENERATOR)
        .to_row_major_matrix();

    let (commit, _data) =
        <MyPcs as Pcs<Challenge, Challenger>>::commit(&p, vec![(domain, trace.clone())]);
    let commit: [Val; 8] = commit.into();

    write_json(
        out,
        &json!({
            "trace": (0..trace.height()).map(|r| ser_fs(&trace.row_slice(r))).collect::<Vec<_>>(),
            "lde_natural_order": (0..lde.height()).map(|r| ser_fs(&lde.row_slice(r))).collect::<Vec<_>>(),
            "root": ser_fs(&commit),
        }),
    );
}


/// A **mixed-height** batched commit — the shape Pico's `commit_main` actually
/// uses, where every chip's main trace goes into one `pcs.commit` at whatever
/// height that chip ran to.
///
/// This is the case a single-matrix commit cannot stand in for. Plonky3 sorts
/// the matrices tallest-first, hashes the tallest into the leaf layer, then at
/// each subsequent layer *injects* the matrices whose padded height equals that
/// layer's length, hashing their rows alongside the compression output. Two
/// matrices share the tallest height here on purpose: that path hashes several
/// matrices' rows into one leaf, which is distinct from the injection path.
///
/// Heights are exact powers of two because Plonky3 rejects a batch where two
/// matrices round up to the same power of two without being equal.
fn emit_batch_commit(out: &str) {
    // (height, width) per matrix, in the order they are handed to `commit`.
    // Deliberately not sorted by height: the reference sorts internally, so
    // feeding it pre-sorted would hide an ordering bug in a consumer.
    const SHAPES: [(usize, usize); 4] = [(4, 3), (16, 2), (8, 1), (16, 4)];

    use p3_dft::TwoAdicSubgroupDft;

    let p = pcs();
    let dft = Dft::default();

    let matrices: Vec<RowMajorMatrix<Val>> = SHAPES
        .iter()
        .enumerate()
        .map(|(m, &(height, width))| {
            // Distinct, position-dependent values so a matrix landing in the
            // wrong slot cannot coincidentally still match.
            let values = (0..height * width)
                .map(|i| Val::from_canonical_usize(1 + i + 100 * m))
                .collect();
            RowMajorMatrix::new(values, width)
        })
        .collect();

    let domains_and_matrices = matrices
        .iter()
        .map(|mat| {
            let domain = <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
                &p,
                mat.height(),
            );
            (domain, mat.clone())
        })
        .collect::<Vec<_>>();

    let (commit, _data) =
        <MyPcs as Pcs<Challenge, Challenger>>::commit(&p, domains_and_matrices);
    let commit: [Val; 8] = commit.into();

    // Each matrix's coset LDE in natural order, so the Python side can pin the
    // extension separately from the mixed-height tree it feeds.
    let per_matrix = matrices
        .iter()
        .map(|mat| {
            let lde = dft
                .coset_lde_batch(mat.clone(), LOG_BLOWUP, Val::GENERATOR)
                .to_row_major_matrix();
            json!({
                "height": mat.height(),
                "width": mat.width(),
                "values": (0..mat.height()).map(|r| ser_fs(&mat.row_slice(r))).collect::<Vec<_>>(),
                "lde_natural_order": (0..lde.height())
                    .map(|r| ser_fs(&lde.row_slice(r)))
                    .collect::<Vec<_>>(),
            })
        })
        .collect::<Vec<_>>();

    write_json(
        out,
        &json!({
            "log_blowup": LOG_BLOWUP,
            "matrices": per_matrix,
            "root": ser_fs(&commit),
        }),
    );
}

/// The full uni-stark proof for the reference Fibonacci AIR, plus the
/// challenges a verifier derives up to the PCS opening — enough to byte-match
/// each pipeline stage before the proof as a whole.
fn emit_fib_prove(out: &str) {
    let (trace, last) = fib_trace(2, 8);
    let pis = vec![Val::ZERO, Val::ONE, last];

    let config = config();
    let mut challenger = Challenger::new(pico_perm());
    let air = FibonacciAir { width: 2 };
    let proof = prove(&config, &air, &mut challenger, trace.clone(), &pis);

    let mut challenger = Challenger::new(pico_perm());
    verify(&config, &air, &mut challenger, &proof, &pis).expect("golden proof must verify");

    let mut proof_json = serde_json::to_value(&proof).unwrap();

    // The proof is generated and verified under Pico's real 84-query config,
    // but only checkpoint query proofs are committed: the pre-query
    // transcript (roots, challenges, final poly, PoW witness) already pins
    // every earlier byte, and 84 full Merkle path sets would be ~17k lines
    // of fixture. The last query is a checkpoint on purpose — it pins the
    // tail of the index-sampling stream, which a first-only prefix would
    // miss. The consumer must still prove with all 84 queries and compares
    // the stored checkpoints by position.
    let all_queries = proof_json["opening_proof"]["query_proofs"]
        .as_array()
        .unwrap()
        .clone();
    let checkpoints = [0usize, 1, 2, all_queries.len() - 1];
    let stored: Vec<Value> = checkpoints
        .iter()
        .map(|&i| all_queries[i].clone())
        .collect();
    proof_json["opening_proof"]["query_proofs"] = Value::Array(stored);
    proof_json["opening_proof"]["stored_query_indices"] =
        serde_json::to_value(checkpoints).unwrap();

    // Re-derive the pre-opening challenges the way the verifier does, so the
    // Python pipeline can byte-match alpha/zeta without parsing FRI first.
    let commitments = &proof_json["commitments"];
    // Hash<F, F, 8> serializes as {"value": [8 canonical u32], "_marker": null}.
    let read_commit = |v: &Value| -> Vec<Val> {
        v["value"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| Val::from_canonical_u32(x.as_u64().unwrap() as u32))
            .collect()
    };
    let mut ch = Challenger::new(pico_perm());
    ch.observe(Val::from_canonical_usize(3)); // log_degree
    ch.observe_slice(&read_commit(&commitments["trace"]));
    ch.observe_slice(&pis);
    let alpha: Challenge = ch.sample_ext_element();
    ch.observe_slice(&read_commit(&commitments["quotient_chunks"]));
    let zeta: Challenge = ch.sample();

    write_json(
        out,
        &json!({
            "public_values": ser_fs(&pis),
            "trace": (0..trace.height()).map(|r| ser_fs(&trace.row_slice(r))).collect::<Vec<_>>(),
            "fri_config": {
                "log_blowup": LOG_BLOWUP,
                "num_queries": NUM_QUERIES,
                "proof_of_work_bits": POW_BITS,
            },
            "alpha": ser_ext(alpha),
            "zeta": ser_ext(zeta),
            "two_adic_generator_log8": ser_f(Val::two_adic_generator(3)),
            "proof": proof_json,
        }),
    );
}

fn main() {
    let root = std::env::args().nth(1).unwrap_or_else(|| "..".to_string());
    assert!(
        std::env::var("FRI_QUERIES").is_err(),
        "FRI_QUERIES would silently change the reference num_queries"
    );
    assert!(
        !cfg!(feature = "parallel"),
        "the `parallel` feature makes the reference grind's find_any pick an \
         arbitrary witness, so fixtures written under it are not reproducible"
    );

    emit_poseidon2(&format!(
        "{root}/pico_zorch/poseidon2/testdata/golden/poseidon2.json"
    ));
    emit_challenger(&format!(
        "{root}/pico_zorch/challenger/testdata/golden/challenger.json"
    ));
    emit_trace_commit(&format!(
        "{root}/pico_zorch/uni_stark/testdata/golden/trace_commit.json"
    ));
    emit_batch_commit(&format!(
        "{root}/pico_zorch/uni_stark/testdata/golden/batch_commit.json"
    ));
    emit_fib_prove(&format!(
        "{root}/pico_zorch/uni_stark/testdata/golden/fib_prove.json"
    ));
}
