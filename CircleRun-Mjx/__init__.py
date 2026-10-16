
import os

os.environ.setdefault("MUJOCO_GL", "disabled")

def _patch_mjx_finfo() -> None:
    try:
        import jax.numpy as _jnp
        import numpy as _np
        from mujoco.mjx._src import collision_convex as _cc
    except Exception:
        return

    _real_jp = _cc.jp

    def _patched_finfo(dtype):
        if dtype is float or _np.dtype(dtype) == _np.float64:
            return _real_jp.finfo(_jnp.float32)
        return _real_jp.finfo(dtype)

    class _JpWrap:
        def __getattr__(self, name):  # pragma: no cover - trivial pass-through
            if name == "finfo":
                return _patched_finfo
            return getattr(_real_jp, name)

    _cc.jp = _JpWrap()

_patch_mjx_finfo()

__all__: list[str] = []
