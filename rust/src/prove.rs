//! The drop-in: `pico_zorch::prove` for `p3_uni_stark::prove`.

use std::path::Path;

use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use pico_zorch_golden::{MyConfig, Val};

use crate::gpu::{self, Core, Phases};
use crate::proof::assemble;

/// Environment variable naming the exported core, for callers that prove one
/// shape per process.
pub const CORE_ENV: &str = "PICO_ZORCH_CORE_MLIRBC";

/// Prove `trace` against the core at `PICO_ZORCH_CORE_MLIRBC`.
///
/// Byte-identical to `p3_uni_stark::prove(&config(), &air, &mut
/// Challenger::new(pico_perm()), trace, public_values)` — the AIR and the FRI
/// config are baked into the core, so they are not arguments here; the core's
/// manifest records them and [`prove_with`] checks the shape it was exported
/// for.
pub fn prove(
    trace: &RowMajorMatrix<Val>,
    public_values: &[Val],
) -> Result<p3_uni_stark::Proof<MyConfig>, String> {
    let path = std::env::var(CORE_ENV)
        .map_err(|_| format!("set {CORE_ENV} to an exported uni-stark core .mlirbc"))?;
    prove_at(Path::new(&path), trace, public_values)
}

/// Like [`prove`], but with an explicit core path — for driving several
/// instance shapes from one process. The core is compiled on first use and
/// cached for the life of the process.
pub fn prove_at(
    core_path: &Path,
    trace: &RowMajorMatrix<Val>,
    public_values: &[Val],
) -> Result<p3_uni_stark::Proof<MyConfig>, String> {
    prove_with(gpu::load(core_path)?, trace, public_values).map(|(proof, _)| proof)
}

/// Prove with an already-loaded core, also returning the host-side phase
/// breakdown.
///
/// # Transcript
///
/// The core bakes in the fresh (all-zero) challenger, matching
/// `Challenger::new(pico_perm())`. There is no way to seed it, which is why
/// this takes no challenger argument: accepting one and ignoring its state
/// would silently prove against a different transcript.
pub fn prove_with(
    core: &Core,
    trace: &RowMajorMatrix<Val>,
    public_values: &[Val],
) -> Result<(p3_uni_stark::Proof<MyConfig>, Phases), String> {
    let degree_bits = log2_strict(trace.height())?;
    core.manifest.expect_instance(degree_bits, trace.width())?;

    let (raw, mut phases) = gpu::run(core, &trace.values, public_values)?;
    let t = std::time::Instant::now();
    let proof = assemble(&core.manifest, &raw)?;
    phases.assemble = t.elapsed();
    Ok((proof, phases))
}

fn log2_strict(n: usize) -> Result<usize, String> {
    if !n.is_power_of_two() {
        return Err(format!("trace height {n} is not a power of two"));
    }
    Ok(n.trailing_zeros() as usize)
}
