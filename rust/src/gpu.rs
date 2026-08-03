//! The PJRT side: compile a core once, then run it per proof.
//!
//! A whole Pico proof is one executable, so there is exactly one dispatch and
//! one readback per call. The trace goes up as a reinterpreted byte view (see
//! [`crate::wire`]) — no per-element work on the host in either direction.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use pico_zorch_golden::Val;
use xla_pjrt::sys::PJRT_Buffer_Type_KOALABEAR_MONT as KOALABEAR_MONT;

use crate::manifest::Manifest;
use crate::wire;

/// The persistent client plus a cache of compiled cores keyed by path.
struct Gpu {
    session: xla_pjrt::Session,
    cores: RefCell<HashMap<String, &'static Core>>,
}

thread_local! {
    /// One [`Gpu`] per thread, leaked to `'static` — a second PJRT client in
    /// one process aborts, and tearing the client down against a live CUDA
    /// context can fault, so its destructor never runs.
    static GPU: RefCell<Option<&'static Gpu>> = const { RefCell::new(None) };
}

fn gpu() -> &'static Gpu {
    GPU.with(|cell| {
        *cell.borrow_mut().get_or_insert_with(|| {
            let session = unsafe { xla_pjrt::Session::new() };
            Box::leak(Box::new(Gpu {
                session,
                cores: RefCell::new(HashMap::new()),
            }))
        })
    })
}

/// A compiled core and the manifest describing its buffers.
pub struct Core {
    exe: xla_pjrt::Executable,
    pub manifest: Manifest,
}

/// Host-side phase timings for one proof (profiling aid).
#[derive(Clone, Copy, Default, Debug)]
pub struct Phases {
    /// Uploading the trace and public values.
    pub h2d: Duration,
    /// Enqueueing the executable.
    pub dispatch: Duration,
    /// Reading the proof back — dominated by waiting on the computation.
    pub readback: Duration,
}

/// Compile (once) the core at `path`, or return the cached one.
///
/// Compilation is per-process and can take a while for a large instance, so a
/// caller proving repeatedly should hold the returned reference rather than
/// calling this per proof (it is cheap after the first call, but not free).
pub fn load(path: &Path) -> Result<&'static Core, String> {
    let key = path.to_string_lossy().into_owned();
    let g = gpu();
    if let Some(core) = g.cores.borrow().get(&key) {
        return Ok(core);
    }
    let manifest = Manifest::beside(path)?;
    let code = std::fs::read(path).map_err(|e| format!("read core {}: {e}", path.display()))?;
    let exe = unsafe { g.session.compile(&code) };
    let core: &'static Core = Box::leak(Box::new(Core { exe, manifest }));
    g.cores.borrow_mut().insert(key, core);
    Ok(core)
}

/// Run `core` on one instance, returning its raw output buffers in manifest
/// order.
pub fn run(
    core: &Core,
    trace: &[Val],
    public_values: &[Val],
) -> Result<(Vec<Vec<u8>>, Phases), String> {
    let m = &core.manifest;
    let expected_trace = m.inputs[0].len();
    if trace.len() != expected_trace {
        return Err(format!(
            "trace has {} elements but the core expects {expected_trace} ({:?})",
            trace.len(),
            m.inputs[0].dims
        ));
    }
    let expected_pv = m.inputs[1].len();
    if public_values.len() != expected_pv {
        return Err(format!(
            "{} public values but the core expects {expected_pv}",
            public_values.len()
        ));
    }

    let s = gpu();
    let dims = |d: &[usize]| d.iter().map(|&n| n as i64).collect::<Vec<_>>();

    let t = Instant::now();
    let trace_buf = unsafe {
        s.session
            .input_buffer(wire::as_bytes(trace), &dims(&m.inputs[0].dims), KOALABEAR_MONT)
    };
    let pv_buf = unsafe {
        s.session.input_buffer(
            wire::as_bytes(public_values),
            &dims(&m.inputs[1].dims),
            KOALABEAR_MONT,
        )
    };
    let h2d = t.elapsed();

    let (outs, dispatch, readback) = unsafe {
        s.session
            .run_buffers_timed(&core.exe, &[&trace_buf, &pv_buf], m.outputs.len())
    };
    Ok((
        outs,
        Phases {
            h2d,
            dispatch,
            readback,
        },
    ))
}
