//! The GPU `Pcs::open` is the reference's, proof and challenger alike.
//!
//! The proof is the contract, but the challenger matters just as much: Pico
//! keeps transcripting after the PCS returns, so a prover whose challenger
//! diverges here produces a proof that verifies in isolation and fails in the
//! machine. Both are checked.
//!
//! Needs a GPU and cores exported for this argument's shape:
//!
//! ```sh
//! bazel run //export:export_pcs_commit_core -- --shapes=16x2,8x1
//! bazel run //export:export_pcs_open_core -- --rounds='16x2:2,8x1:1;16x4:2,4x3:1'
//! cargo test --test pcs_open -- --ignored --test-threads=1
//! ```

use std::path::PathBuf;

use p3_commit::Pcs;
use p3_field::FieldAlgebra;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use pico_zorch::{pcs::TwoAdicFriPcs, Challenge, Challenger, Val};
use pico_zorch_golden::{pcs, pico_perm, MyPcs};

/// The shape `golden`'s `emit_pcs_open` pins: two rounds, mixed heights inside
/// a round, uneven point counts per matrix.
const ROUNDS: [&[(usize, usize)]; 2] = [&[(16, 2), (8, 1)], &[(16, 4), (4, 3)]];

fn artifacts() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate has a parent directory")
        .join("artifacts")
}

fn matrices(round: usize) -> Vec<RowMajorMatrix<Val>> {
    ROUNDS[round]
        .iter()
        .enumerate()
        .map(|(m, &(height, width))| {
            let values = (0..height * width)
                .map(|i| Val::from_canonical_usize(1 + i + 100 * (m + 10 * round)))
                .collect();
            RowMajorMatrix::new(values, width)
        })
        .collect()
}

/// The first matrix of each round is opened at two points, the rest at one —
/// the uneven case a uniform fixture would not reach.
fn points_for(round: usize, zeta: Challenge, zeta_next: Challenge) -> Vec<Vec<Challenge>> {
    ROUNDS[round]
        .iter()
        .enumerate()
        .map(|(i, _)| {
            if i == 0 {
                vec![zeta, zeta_next]
            } else {
                vec![zeta]
            }
        })
        .collect()
}

fn commit_core(round: usize) -> PathBuf {
    let stem: Vec<String> = ROUNDS[round]
        .iter()
        .map(|(h, w)| format!("{h}x{w}"))
        .collect();
    artifacts().join(format!("pcs_commit_core_{}.mlirbc", stem.join("_")))
}

fn open_core() -> PathBuf {
    artifacts().join("pcs_open_core_16x2p2_8x1p1__16x4p2_4x3p1.mlirbc")
}

fn gpu_pcs(round: usize) -> TwoAdicFriPcs {
    TwoAdicFriPcs::new(pcs(), &commit_core(round), &open_core()).unwrap_or_else(|e| {
        panic!(
            "{e}\n  export the cores — see the module docs for the exact commands"
        )
    })
}

#[test]
#[ignore = "needs a GPU and exported cores (see the module docs)"]
fn opening_matches_the_reference() {
    let reference = pcs();
    let zeta = Challenge::from_canonical_u32(97);
    let zeta_next = Challenge::from_canonical_u32(1234567);

    let domains_and_mats = |round: usize| {
        matrices(round)
            .into_iter()
            .map(|m| {
                let domain = <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
                    &reference,
                    m.height(),
                );
                (domain, m)
            })
            .collect::<Vec<_>>()
    };

    // Reference: commit both rounds, then open them together.
    let mut want_data = Vec::new();
    for round in 0..ROUNDS.len() {
        let (_, data) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&reference, domains_and_mats(round));
        want_data.push(data);
    }
    let mut want_challenger = Challenger::new(pico_perm());
    let (want_opened, want_proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
        &reference,
        want_data
            .iter()
            .enumerate()
            .map(|(r, d)| (d, points_for(r, zeta, zeta_next)))
            .collect(),
        &mut want_challenger,
    );

    // Ours: same instances, commit on the device, open from what it left there.
    let mut got_data = Vec::new();
    for round in 0..ROUNDS.len() {
        let gpu = gpu_pcs(round);
        let (_, data) = Pcs::<Challenge, Challenger>::commit(&gpu, domains_and_mats(round));
        got_data.push(data);
    }
    let gpu = gpu_pcs(0);
    let mut got_challenger = Challenger::new(pico_perm());
    let (got_opened, got_proof) = Pcs::<Challenge, Challenger>::open(
        &gpu,
        got_data
            .iter()
            .enumerate()
            .map(|(r, d)| (d, points_for(r, zeta, zeta_next)))
            .collect(),
        &mut got_challenger,
    );

    assert_eq!(
        serde_json::to_value(&got_opened).unwrap(),
        serde_json::to_value(&want_opened).unwrap(),
        "opened values differ from TwoAdicFriPcs::open"
    );
    assert_eq!(
        serde_json::to_value(&got_proof).unwrap(),
        serde_json::to_value(&want_proof).unwrap(),
        "FRI proof differs from TwoAdicFriPcs::open"
    );

    // The consumer keeps transcripting after this returns.
    assert_eq!(got_challenger.sponge_state, want_challenger.sponge_state);
    assert_eq!(got_challenger.input_buffer, want_challenger.input_buffer);
    assert_eq!(got_challenger.output_buffer, want_challenger.output_buffer);
}
