"""Compatibility shim for the former transition-era WNM runtime location.

The maintained implementation now lives in :mod:`shared.wnm`. This module
re-exports every non-dunder symbol so historical scripts and old pickle module
references remain loadable during the repository transition.
"""
from shared import wnm as _impl

globals().update({
    name: getattr(_impl, name)
    for name in dir(_impl)
    if not name.startswith("__")
})

__all__ = [name for name in dir(_impl) if not name.startswith("_")]
