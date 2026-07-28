//! The Pico stark config, shared by the fixture generator and the bench so
//! the two cannot describe different provers.
//!
//! Everything here mirrors Pico v2.0.0's `KoalaBearPoseidon2`
//! (vm/src/configs/stark_config/kb_poseidon2.rs) over the Plonky3 fork it
//! vendors, brevis-network/Plonky3@7fbe1908.

pub mod rc;

use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, FieldAlgebra};
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_poseidon2::ExternalLayerConstants;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::StarkConfig;

use rc::RC_16_30_U32;

pub type Val = KoalaBear;
pub type Perm = Poseidon2KoalaBear<16>;
pub type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
pub type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
pub type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
pub type Challenge = BinomialExtensionField<Val, 4>;
pub type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
pub type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
pub type Dft = Radix2DitParallel<Val>;
pub type MyPcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
pub type MyConfig = StarkConfig<MyPcs, Challenge, Challenger>;

/// Pico's FRI parameters for the RISCV phase: `KoalaBearPoseidon2::new()`.
pub const LOG_BLOWUP: usize = 1;
pub const NUM_QUERIES: usize = 84;
pub const POW_BITS: usize = 16;

/// `pico_poseidon2kb_init()`: RC_16_30 rows 0..4 initial external, rows 4..24
/// column 0 internal, rows 24..28 terminal external, rows 28..30 unused, each
/// element reduced with `from_wrapped_u32`.
pub fn pico_perm() -> Perm {
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

pub fn pcs() -> MyPcs {
    let perm = pico_perm();
    let val_mmcs = ValMmcs::new(MyHash::new(perm.clone()), MyCompress::new(perm.clone()));
    let fri_config = FriConfig {
        log_blowup: LOG_BLOWUP,
        num_queries: NUM_QUERIES,
        proof_of_work_bits: POW_BITS,
        mmcs: ChallengeMmcs::new(val_mmcs.clone()),
    };
    MyPcs::new(Dft::default(), val_mmcs, fri_config)
}

pub fn config() -> MyConfig {
    MyConfig::new(pcs())
}

/// The Fibonacci AIR from the fork's own uni-stark tests, widened: the
/// recurrence occupies columns 0-1 and any further columns are unconstrained
/// padding, so trace width sweeps without the constraint set changing.
pub struct FibonacciAir {
    pub width: usize,
}

impl<F> BaseAir<F> for FibonacciAir {
    fn width(&self) -> usize {
        self.width
    }
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for FibonacciAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let pis = builder.public_values();
        let (a, b, x) = (pis[0], pis[1], pis[2]);
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let (l0, l1) = (local[0], local[1]);
        let (n0, n1) = (next[0], next[1]);

        let mut first = builder.when_first_row();
        first.assert_eq(l0, a);
        first.assert_eq(l1, b);
        let mut trans = builder.when_transition();
        trans.assert_eq(l1, n0);
        trans.assert_eq(l0 + l1, n1);
        builder.when_last_row().assert_eq(l1, x);
    }
}

/// `(trace, last)` for the width-`width` Fibonacci trace of height `n`;
/// `last` is the final right-hand value, the AIR's third public value.
pub fn fib_trace(width: usize, n: usize) -> (RowMajorMatrix<Val>, Val) {
    assert!(n.is_power_of_two() && width >= 2);
    let mut values = Val::zero_vec(n * width);
    let (mut a, mut b) = (Val::ZERO, Val::ONE);
    for row in 0..n {
        values[row * width] = a;
        values[row * width + 1] = b;
        let next = a + b;
        a = b;
        b = next;
    }
    let last = values[(n - 1) * width + 1];
    (RowMajorMatrix::new(values, width), last)
}
