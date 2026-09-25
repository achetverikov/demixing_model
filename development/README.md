# Development-only code

This tree contains model-development, comparison, validation, and diagnostic
workflows that are useful while developing the demixing model but are not part
of the maintained production architecture.

- `wnm_transition/`: validation and comparison work used during the WNM cutover.
- `wnm_representation/`: alternative/local representation experiments and diagnostics.

Code under `development/` may depend on maintained production modules, but
maintained production modules must not depend on `development/`.

This tree is intentionally separable from the production PR. The default pytest
suite does not collect its colocated tests.
