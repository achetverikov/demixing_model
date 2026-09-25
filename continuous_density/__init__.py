"""Transition compatibility and historical WNM development workspace.

Maintained runtime math now lives in shared.wnm; simulation and training-data
generation live in surface_computation; training and packaging live in
surrogate_training.wnm. Files left here are either compatibility shims or
development/transition workflows awaiting relocation under development/ and
docs/history/. No maintained production path should import this package.
"""
