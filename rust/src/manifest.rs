//! The exported core's self-description.
//!
//! `export/export_uni_stark_core.py` writes one of these beside every `.mlirbc`.
//! Binding buffers by name rather than by position is what keeps the two sides
//! honest: adding a proof field shifts every later index, and an off-by-one
//! there would not fail loudly — it would decode one field's bytes as another
//! and produce a wrong proof.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

/// One input or output buffer: its element type and logical shape.
#[derive(Debug, Clone, Deserialize)]
pub struct Spec {
    pub name: String,
    /// `koalabear` for field values (canonical u32, standard form), `int32`
    /// for the query indices.
    pub dtype: String,
    pub dims: Vec<usize>,
}

impl Spec {
    /// Elements in this buffer — the product of its dims, so a scalar is 1.
    pub fn len(&self) -> usize {
        self.dims.iter().product()
    }
}

/// A core's buffers, plus whatever its kind adds.
///
/// Every core describes its inputs and outputs the same way; only some of them
/// prove. The uni-stark core carries the proof's shape too — AIR, degree bits,
/// query count — while a PCS commit core has no notion of any of it, so those
/// fields are optional here and reached through [`Manifest::uni_stark`], which
/// fails loudly rather than defaulting.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub inputs: Vec<Spec>,
    pub outputs: Vec<Spec>,
    #[serde(default)]
    pub air: Option<String>,
    #[serde(default)]
    pub degree_bits: Option<usize>,
    #[serde(default)]
    pub width: Option<usize>,
    #[serde(default)]
    pub log_blowup: Option<usize>,
    #[serde(default)]
    pub num_queries: Option<usize>,
    #[serde(default)]
    pub proof_of_work_bits: Option<usize>,
    #[serde(default)]
    pub quotient_degree: Option<usize>,
}

/// The proof-shaped parameters only a uni-stark core carries.
#[derive(Debug, Clone)]
pub struct UniStark {
    pub degree_bits: usize,
    pub width: usize,
    pub num_queries: usize,
    pub quotient_degree: usize,
}

impl Manifest {
    /// Read the manifest sitting beside `core_path` (same stem, `.json`).
    pub fn beside(core_path: &Path) -> Result<Self, String> {
        let json = Self::manifest_path(core_path);
        let text = std::fs::read_to_string(&json)
            .map_err(|e| format!("read core manifest {}: {e}", json.display()))?;
        let manifest: Manifest = serde_json::from_str(&text)
            .map_err(|e| format!("parse core manifest {}: {e}", json.display()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    fn manifest_path(core_path: &Path) -> PathBuf {
        core_path.with_extension("json")
    }

    /// Output index by name, for binding decoded buffers to proof fields.
    pub fn output_index(&self) -> HashMap<&str, usize> {
        self.outputs
            .iter()
            .enumerate()
            .map(|(i, s)| (s.name.as_str(), i))
            .collect()
    }

    /// The FRI layer count, read off the commitments the core returns rather
    /// than recomputed — the exporter and the executable agree by construction,
    /// and a recomputation here would be a second opinion that could differ.
    pub fn num_fri_layers(&self) -> Result<usize, String> {
        let roots = self
            .outputs
            .iter()
            .find(|s| s.name == "commit_phase_roots")
            .ok_or("core manifest has no commit_phase_roots output")?;
        roots
            .dims
            .first()
            .copied()
            .ok_or_else(|| "commit_phase_roots must be [layers, 8]".to_string())
    }

    /// The uni-stark parameters, or an error naming what is missing.
    pub fn uni_stark(&self) -> Result<UniStark, String> {
        let missing = |f: &str| format!("core manifest has no {f:?} — not a uni-stark core");
        Ok(UniStark {
            degree_bits: self.degree_bits.ok_or_else(|| missing("degree_bits"))?,
            width: self.width.ok_or_else(|| missing("width"))?,
            num_queries: self.num_queries.ok_or_else(|| missing("num_queries"))?,
            quotient_degree: self.quotient_degree.ok_or_else(|| missing("quotient_degree"))?,
        })
    }

    /// Fail before any PJRT call if the core was exported for a different
    /// instance — otherwise a mismatch surfaces as an opaque plugin abort or,
    /// worse, a silently wrong proof.
    pub fn expect_instance(&self, degree_bits: usize, width: usize) -> Result<(), String> {
        let us = self.uni_stark()?;
        if us.degree_bits != degree_bits || us.width != width {
            return Err(format!(
                "core was exported for degree_bits={} width={} but this instance is \
                 degree_bits={degree_bits} width={width} — re-export: \
                 bazel run //export:export_uni_stark_core -- --degree_bits={degree_bits} \
                 --width={width}",
                us.degree_bits, us.width
            ));
        }
        Ok(())
    }

    fn validate(&self) -> Result<(), String> {
        if self.air.is_none() {
            // Not a uni-stark core; its arity and outputs are its own business
            // (a PCS commit core takes one matrix per chip).
            return Ok(());
        }
        if self.inputs.len() != 2 {
            return Err(format!(
                "a uni-stark core takes (trace, public_values), got {} inputs",
                self.inputs.len()
            ));
        }
        for required in [
            "trace_root",
            "quotient_root",
            "trace_local",
            "trace_next",
            "quotient_chunks",
            "commit_phase_roots",
            "final_poly",
            "pow_witness",
            "query_indices",
        ] {
            if !self.outputs.iter().any(|s| s.name == required) {
                return Err(format!("core manifest is missing output {required:?}"));
            }
        }
        Ok(())
    }
}
