"""Import mlx_whisper without numba and scipy installed.

mlx_whisper/timing.py imports numba and scipy.signal at module level, and
mlx_whisper/__init__ imports it, but they only run for word timestamps
(`word_timestamps=True`), which we never request. Leaving them out of the
lock saves about 250MB on disk (llvmlite alone is 125MB); see
docs/development.md.

Importing this module imports mlx_whisper with stand-ins for whichever of the
two is missing, then removes the stand-ins from sys.modules: mlx_whisper keeps
its own references, and other libraries (transformers probes for scipy with
importlib.util.find_spec) see the packages as simply not installed. Import it
before the first `import mlx_whisper` or `mlx_whisper.*` import.
"""

import importlib.util
import sys
import types


def _stub_numba() -> dict[str, types.ModuleType]:
    numba = types.ModuleType("numba")

    def jit(*args, **kwargs):
        # Bare @numba.jit decorates directly; @numba.jit(...) returns a decorator.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda fn: fn

    numba.jit = jit
    return {"numba": numba}


def _stub_scipy() -> dict[str, types.ModuleType]:
    scipy = types.ModuleType("scipy")
    signal = types.ModuleType("scipy.signal")

    def _missing(name: str):
        raise RuntimeError(
            f"scipy.signal.{name} is unavailable: scipy is left out of the lock, so "
            "mlx_whisper word timestamps aren't supported (see snoozy_scribe/whisper_stubs.py)."
        )

    signal.__getattr__ = _missing
    scipy.signal = signal
    return {"scipy": scipy, "scipy.signal": signal}


def _import_mlx_whisper() -> None:
    if "mlx_whisper" in sys.modules:
        return
    stubs: dict[str, types.ModuleType] = {}
    if importlib.util.find_spec("numba") is None:
        stubs |= _stub_numba()
    if importlib.util.find_spec("scipy") is None:
        stubs |= _stub_scipy()
    sys.modules.update(stubs)
    try:
        import mlx_whisper  # noqa: F401
    finally:
        for name, module in stubs.items():
            if sys.modules.get(name) is module:
                del sys.modules[name]


_import_mlx_whisper()
