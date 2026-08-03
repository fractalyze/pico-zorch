//! The KoalaBear wire boundary — a cast, not a conversion.
//!
//! The core emits raw Montgomery limbs, which is bit-for-bit Plonky3's
//! in-memory `KoalaBear`: `MontyField31` is `#[repr(transparent)]` over `u32`
//! (upstream annotates the attribute "Packed field implementations rely on
//! this!"), same modulus `0x7f000001`, same `R = 2^32`. So a buffer of field
//! elements crosses this boundary by reinterpretation.
//!
//! The tidier-looking alternative — canonical form on the wire, rebuilt with
//! `from_canonical_u32` — costs a Montgomery reduction per element on both
//! sides. For a 2^20 x 32 trace that is 33M reductions per proof on the host
//! critical path; Plonky3's own note on that constructor says it "should be
//! avoided in performance critical implementations".
//!
//! The representation agreement is load-bearing and therefore pinned by a
//! test on the Python side (`export_pico_core_test.WireRepresentationTest`),
//! so a change in either library fails as an assertion rather than as an
//! inexplicable proof mismatch.
//!
//! Extension elements arrive already widened to their four base coefficients,
//! so everything here is base field.

use p3_field::{FieldExtensionAlgebra, PrimeField32};
use pico_zorch_golden::{Challenge, Val};

/// Bytes per element on the wire.
pub const ELEM: usize = std::mem::size_of::<u32>();

// The cast below is only sound while `Val` really is a `u32` in disguise.
const _: () = assert!(std::mem::size_of::<Val>() == std::mem::size_of::<u32>());
const _: () = assert!(std::mem::align_of::<Val>() == std::mem::align_of::<u32>());

/// A validated Montgomery limb as a field element.
#[inline]
fn from_monty_limb(limb: u32) -> Val {
    // SAFETY: `MontyField31` is `#[repr(transparent)]` over its `u32` limb
    // (plus a zero-sized `PhantomData`), asserted above. The caller has
    // checked `limb < ORDER`, which is the reducedness invariant the
    // representation carries — an unreduced limb would not be unsound, but it
    // would make the element's arithmetic silently wrong.
    unsafe { std::mem::transmute::<u32, Val>(limb) }
}

/// Field elements out of a little-endian wire buffer.
pub fn vals(bytes: &[u8]) -> Result<Vec<Val>, String> {
    if bytes.len() % ELEM != 0 {
        return Err(format!(
            "wire buffer of {} bytes is not a whole number of 4-byte elements",
            bytes.len()
        ));
    }
    bytes
        .chunks_exact(ELEM)
        .map(|c| {
            let limb = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            if limb >= Val::ORDER_U32 {
                return Err(format!(
                    "wire limb {limb} is not reduced mod {} — the core and this \
                     binding disagree about the field representation",
                    Val::ORDER_U32
                ));
            }
            Ok(from_monty_limb(limb))
        })
        .collect()
}

/// Field elements as wire bytes, without copying or touching the values.
///
/// The trace goes out this way, so this is the one direction where a per-element
/// conversion would actually be felt.
pub fn as_bytes(values: &[Val]) -> &[u8] {
    // SAFETY: `Val` is `#[repr(transparent)]` over `u32` (asserted above), so
    // the slice is exactly `values.len() * 4` initialized bytes. The lifetime
    // is tied to `values`, and `u8` has no alignment requirement.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * ELEM) }
}

/// `int32` query indices out of a wire buffer.
pub fn indices(bytes: &[u8]) -> Result<Vec<usize>, String> {
    if bytes.len() % ELEM != 0 {
        return Err(format!(
            "index buffer of {} bytes is not a whole number of int32s",
            bytes.len()
        ));
    }
    bytes
        .chunks_exact(ELEM)
        .map(|c| {
            let signed = i32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            usize::try_from(signed).map_err(|_| format!("negative query index {signed}"))
        })
        .collect()
}

/// Consecutive groups of four base coefficients as extension elements.
pub fn challenges(base: &[Val]) -> Result<Vec<Challenge>, String> {
    if base.len() % 4 != 0 {
        return Err(format!(
            "{} base coefficients do not group into quartic extension elements",
            base.len()
        ));
    }
    Ok(base.chunks_exact(4).map(Challenge::from_base_slice).collect())
}

/// One extension element from exactly four base coefficients.
pub fn challenge(base: &[Val]) -> Result<Challenge, String> {
    match challenges(base)?[..] {
        [only] => Ok(only),
        ref many => Err(format!("expected one extension element, got {}", many.len())),
    }
}

/// Split a flat buffer into `count` equal-length rows.
pub fn rows<T: Clone>(flat: &[T], count: usize) -> Result<Vec<Vec<T>>, String> {
    if count == 0 || flat.len() % count != 0 {
        return Err(format!(
            "cannot split {} elements into {count} equal rows",
            flat.len()
        ));
    }
    Ok(flat
        .chunks_exact(flat.len() / count)
        .map(<[T]>::to_vec)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::FieldAlgebra;

    /// The cast is the whole design; a round trip through it must be identity
    /// for values built the ordinary way.
    #[test]
    fn round_trips_through_the_wire() {
        let values: Vec<Val> = (0..64u32).map(Val::from_canonical_u32).collect();
        let decoded = vals(as_bytes(&values)).expect("decode");
        assert_eq!(values, decoded);
    }

    /// The Montgomery image of 1 is `R = 2^32 mod P`, the same constant the
    /// exporter's test pins on the Python side.
    #[test]
    fn montgomery_constant_matches_the_exporter() {
        let one = as_bytes(&[Val::ONE]);
        let limb = u32::from_le_bytes([one[0], one[1], one[2], one[3]]);
        assert_eq!(limb, ((1u64 << 32) % u64::from(Val::ORDER_U32)) as u32);
    }

    #[test]
    fn rejects_an_unreduced_limb() {
        let bad = Val::ORDER_U32.to_le_bytes();
        assert!(vals(&bad).is_err());
    }
}
