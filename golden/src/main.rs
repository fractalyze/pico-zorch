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

mod fib_air;
mod rc;

use std::fs;
use std::path::Path;

use p3_challenger::{
    CanObserve, CanSample, CanSampleBits, DuplexChallenger, FieldChallenger, GrindingChallenger,
};
use p3_commit::{ExtensionMmcs, Pcs};
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, FieldAlgebra, FieldExtensionAlgebra, PrimeField32, TwoAdicField};
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
use p3_matrix::Matrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_poseidon2::ExternalLayerConstants;
use p3_symmetric::{PaddingFreeSponge, Permutation, TruncatedPermutation};
use p3_uni_stark::{prove, verify, StarkConfig};
use serde_json::{json, Value};

use fib_air::{generate_trace_rows, FibonacciAir};
use rc::RC_16_30_U32;

type Val = KoalaBear;
type Perm = Poseidon2KoalaBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
type Challenge = BinomialExtensionField<Val, 4>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type Dft = Radix2DitParallel<Val>;
type MyPcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
type MyConfig = StarkConfig<MyPcs, Challenge, Challenger>;

/// Pico's FRI parameters for the RISCV phase:
/// KoalaBearPoseidon2::new() in vm/src/configs/stark_config/kb_poseidon2.rs.
const LOG_BLOWUP: usize = 1;
const NUM_QUERIES: usize = 84;
const POW_BITS: usize = 16;

/// pico_poseidon2kb_init() from pico v2.0.0 vm/src/primitives/mod.rs:
/// RC_16_30 rows 0..4 initial external, rows 4..24 column 0 internal,
/// rows 24..28 terminal external (rows 28..30 unused), each element reduced
/// with from_wrapped_u32.
fn pico_perm() -> Perm {
    const ROUNDS_F: usize = 8;
    const ROUNDS_P: usize = 20;

    let rc: Vec<[Val; 16]> = RC_16_30_U32
        .iter()
        .map(|row| row.map(Val::from_wrapped_u32))
        .collect();
    let internal: Vec<Val> = rc[(ROUNDS_F / 2)..(ROUNDS_F / 2 + ROUNDS_P)]
        .iter()
        .map(|row| row[0])
        .collect();
    let external = ExternalLayerConstants::new(
        rc[..(ROUNDS_F / 2)].to_vec(),
        rc[(ROUNDS_F / 2 + ROUNDS_P)..(ROUNDS_F + ROUNDS_P)].to_vec(),
    );
    Perm::new(external, internal)
}

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

fn pcs() -> MyPcs {
    let perm = pico_perm();
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let fri_config = FriConfig {
        log_blowup: LOG_BLOWUP,
        num_queries: NUM_QUERIES,
        proof_of_work_bits: POW_BITS,
        mmcs: ChallengeMmcs::new(val_mmcs.clone()),
    };
    MyPcs::new(Dft::default(), val_mmcs, fri_config)
}

/// TwoAdicFriPcs trace commit: shift-3 coset LDE, bit-reversed rows, Merkle
/// root. The LDE matrix itself is dumped so the Python side can pin the LDE
/// and the tree independently.
fn emit_trace_commit(out: &str) {
    let trace = generate_trace_rows::<Val>(0, 1, 8);
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

/// The full uni-stark proof for the reference Fibonacci AIR, plus the
/// challenges a verifier derives up to the PCS opening — enough to byte-match
/// each pipeline stage before the proof as a whole.
fn emit_fib_prove(out: &str) {
    let n = 8usize;
    let trace = generate_trace_rows::<Val>(0, 1, n);
    let pis = vec![
        Val::from_canonical_u64(0),
        Val::from_canonical_u64(1),
        Val::from_canonical_u64(21),
    ];

    let config = MyConfig::new(pcs());
    let mut challenger = Challenger::new(pico_perm());
    let proof = prove(&config, &FibonacciAir {}, &mut challenger, trace.clone(), &pis);

    let mut challenger = Challenger::new(pico_perm());
    verify(&config, &FibonacciAir {}, &mut challenger, &proof, &pis)
        .expect("golden proof must verify");

    let mut proof_json = serde_json::to_value(&proof).unwrap();

    // The proof is generated and verified under Pico's real 84-query config,
    // but only the first STORED_QUERY_PROOFS query proofs are committed: the
    // pre-query transcript (roots, challenges, final poly, PoW witness)
    // already pins every earlier byte, and 84 full Merkle path sets would be
    // ~17k lines of fixture. The consumer must still prove with all 84
    // queries; it just compares the stored prefix.
    const STORED_QUERY_PROOFS: usize = 4;
    let stored: Vec<Value> = proof_json["opening_proof"]["query_proofs"]
        .as_array()
        .unwrap()
        .iter()
        .take(STORED_QUERY_PROOFS)
        .cloned()
        .collect();
    proof_json["opening_proof"]["query_proofs"] = Value::Array(stored);

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

    emit_poseidon2(&format!(
        "{root}/pico_zorch/poseidon2/testdata/golden/poseidon2.json"
    ));
    emit_challenger(&format!(
        "{root}/pico_zorch/challenger/testdata/golden/challenger.json"
    ));
    emit_trace_commit(&format!(
        "{root}/pico_zorch/commit/testdata/golden/trace_commit.json"
    ));
    emit_fib_prove(&format!(
        "{root}/pico_zorch/uni_stark/testdata/golden/fib_prove.json"
    ));
}
