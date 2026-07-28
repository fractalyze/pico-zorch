"""Shared Starlark helpers for pico-zorch BUILD files."""

load("@pico_zorch_pip//:requirements.bzl", "requirement")

# GPU runtime plugins (frx-cuda12 PJRT + plugin). Carried by every frx-using
# py_test — CI's GPU leg (FRX_PLATFORMS=cuda) initializes the device; the CPU
# leg never initializes the plugin, so they are inert there — and by the bench.
GPU_PLUGIN_DEPS = [
    requirement("frx_cuda12_plugin"),
    requirement("frx_cuda12_pjrt"),
]
