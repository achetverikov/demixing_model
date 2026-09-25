"""Compatibility shim retained during the WNM architecture transition."""
from surface_computation import legacy_sample_import as _impl

globals().update({
    name: getattr(_impl, name)
    for name in dir(_impl)
    if not name.startswith("__")
})

__all__ = [name for name in dir(_impl) if not name.startswith("_")]
