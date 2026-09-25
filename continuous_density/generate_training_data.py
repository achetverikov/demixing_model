"""Compatibility shim retained during the WNM architecture transition."""
from surface_computation import generate_wnm_training_data as _impl

globals().update({
    name: getattr(_impl, name)
    for name in dir(_impl)
    if not name.startswith("__")
})

__all__ = [name for name in dir(_impl) if not name.startswith("_")]

if __name__ == "__main__":
    _impl.main()
