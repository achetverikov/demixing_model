"""One checkpoint resolver and one family-aware loader for both surrogates.

Two model families answer the same scientific question -- given
``(sd_feat1, sd_feat2, sd_spat, feat_diff)``, what is the distribution of the
component-1 bias?  The historical family is the **surface NN**, which emits a
180x90 log-density surface; the replacement is the **conditional wrapped-normal
mixture** (WNM), which emits mixture parameters and is continuous in both the
bias and the feature difference.

Every public entry point should reach a checkpoint through :func:`load_surrogate`
rather than opening a path itself, so that exactly one place knows which
families exist, which file belongs to which observer sample count, and what
identity travels with a fitted run.

Three rules this module exists to enforce:

* **Metadata decides the family, never the filename.**  A packaged WNM artifact
  declares its own family and sample count; a request that contradicts what the
  file says raises rather than silently winning.
* **No identity is guessed from a substring.**  Historical surface checkpoints
  carry no sample count, so theirs comes from :data:`SURFACE_CHECKPOINT_REGISTRY`
  or from an explicit argument.  An unregistered surface file has no sample
  identity and must be given one.
* **Production never loads a research fit.**  The training scripts write a fit
  dictionary that lacks the architecture; it must go through the packager first.
  Loading one here raises and says so.
"""
from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_DIR = REPO_ROOT / "pretrained"

FAMILY_SURFACE_NN = "surface_nn"
FAMILY_WNM = "wnm"
FAMILIES = (FAMILY_SURFACE_NN, FAMILY_WNM)

#: The family used when nothing asks for one.  Stays on the surface NN until the
#: WNM artifacts have passed acceptance for both sample counts; the transition
#: plan flips it in its promotion step, not before.
DEFAULT_FAMILY = FAMILY_SURFACE_NN

#: Sample identity for the historical surface checkpoints, which predate any
#: metadata.  This is a recorded fact about specific files, not a parsing rule:
#: a surface checkpoint that is not listed here has no known sample count and
#: must be given one explicitly.
SURFACE_CHECKPOINT_REGISTRY = {
    "model_epoch1425_10ktrain_20samples.pkl": 20,
    "model_epoch1500_10ktrain_20samples.pkl": 20,
    "model_epoch1500_10ktrain_100samples.pkl": 100,
    "model_epoch1500_8ktrain_20samples_old.pkl": 20,
    "model_epoch1500_8ktrain_100samples_old.pkl": 100,
}

#: Which surface checkpoint a bare ``(family, n_samples)`` request resolves to.
SURFACE_DEFAULTS = {
    20: PRETRAINED_DIR / "model_epoch1425_10ktrain_20samples.pkl",
    100: PRETRAINED_DIR / "model_epoch1500_10ktrain_100samples.pkl",
}

#: Same, for the WNM artifacts.  Populated by the packaging step; a missing file
#: is reported as "not installed yet" rather than as a bad request.
WNM_DEFAULTS = {
    20: PRETRAINED_DIR / "wnm_k12_20samples.pkl",
    100: PRETRAINED_DIR / "wnm_k12_100samples.pkl",
}

SUPPORTED_SAMPLE_COUNTS = (20, 100)

#: Which family the production artifacts come from. Exactly two checkpoints are
#: production at any moment -- one per observer sample count -- and this is the
#: single switch that says which pair. Promotion flips it once, after acceptance
#: (transition plan step 6); until then production is the surface network.
#:
#: Scripts and pipelines should ask for an observer model by ``n_samples`` and
#: let :func:`production_checkpoint` answer, rather than naming a file. A caller
#: that hardcodes a filename keeps pointing at that file after promotion, which
#: is how a plot or a prediction ends up computed from a different model than the
#: fit it accompanies.
PRODUCTION_FAMILY = FAMILY_SURFACE_NN


def production_checkpoint(n_samples: int) -> Path:
    """The production artifact for one observer model.

    ``n_samples`` is the parameter: 20 and 100 are two different observer models,
    and everything else about which file to load follows from
    :data:`PRODUCTION_FAMILY`.
    """
    if n_samples not in SUPPORTED_SAMPLE_COUNTS:
        raise ValueError(
            f"n_samples={n_samples!r} is not one of {SUPPORTED_SAMPLE_COUNTS}. These are two "
            "different observer models, not a resolution setting, so there is nothing "
            "sensible between or beyond them.")
    table = SURFACE_DEFAULTS if PRODUCTION_FAMILY == FAMILY_SURFACE_NN else WNM_DEFAULTS
    path = table[n_samples]
    if not path.exists():
        raise FileNotFoundError(
            f"the production {PRODUCTION_FAMILY} artifact for n_samples={n_samples} is not "
            f"installed at {path}")
    return path


def load_production(n_samples: int) -> "LoadedSurrogate":
    """Load the production artifact for one observer model."""
    return load_surrogate(checkpoint_path=production_checkpoint(n_samples))

#: The surface network's training domain, in model degrees.  Its parameter grid
#: was swept over [5, 200] on all three SDs (see ``pretrained/README.md``), and
#: unlike the WNM artifacts it carries no metadata, so the fact is recorded here.
#: It is narrower than the WNM domain on the feature axis: the surface corpus
#: never went below 5, which is precisely the coverage the replacement adds.
SURFACE_DOMAIN = {
    "sd_feat1": (5.0, 200.0),
    "sd_feat2": (5.0, 200.0),
    "sd_spat": (5.0, 200.0),
    "feat_diff": (2.0, 180.0),
}


def search_bounds(domain) -> dict:
    """Fitting bounds for the two searched axes, from a surrogate's domain.

    A search must not propose parameters its own surrogate was never trained on,
    and the two families differ: the surface network stops at 5 degrees on every
    axis, while the mixture reaches 2.5 on the feature SDs but still stops at 5
    on the spatial one, because ``sd_spat`` is ``42/d'`` and d-prime was capped.
    Bounds therefore travel with the loaded surrogate rather than living in a
    module-level constant that is right for whichever family was current when it
    was written.

    The two feature SDs share one interval: they are exchangeable, and a search
    that could reach a value for one but not the other would break that symmetry.

    Returns:
        ``{"sd_feat": (low, high), "sd_spat": (low, high)}``.
    """
    feat_low = max(domain["sd_feat1"][0], domain["sd_feat2"][0])
    feat_high = min(domain["sd_feat1"][1], domain["sd_feat2"][1])
    return {"sd_feat": (float(feat_low), float(feat_high)),
            "sd_spat": (float(domain["sd_spat"][0]), float(domain["sd_spat"][1]))}


@dataclass(frozen=True)
class LoadedSurrogate:
    """A checkpoint plus the identity that has to travel with anything it produces.

    Attributes:
        family: one of :data:`FAMILIES`.
        n_samples: observer evidence samples per trial (20 or 100).  This is a
            property of the simulated observer the checkpoint was trained on,
            not a precision knob, and results from two sample counts are not
            interchangeable.
        path: the file actually loaded, resolved and absolute.
        payload: family-specific loaded content.  For ``surface_nn`` a
            ``{'state', 'checkpoint_info'}`` dict from
            :func:`shared.utils.load_checkpoint`; for ``wnm`` a
            ``{'model', 'variables'}`` dict from
            :func:`continuous_density.wrapped_mixture_model.load_model`.
        meta: the artifact's recorded provenance.  Empty for historical surface
            checkpoints, which have none.
    """

    family: str
    n_samples: int
    path: Path
    payload: dict = field(repr=False)
    meta: dict = field(default_factory=dict, repr=False)

    def identity(self) -> dict:
        """The fields a run fingerprint or an exported fit should record."""
        identity = {
            "surrogate_family": self.family,
            "surrogate_n_samples": self.n_samples,
            "surrogate_artifact": self.path.name,
        }
        for key in ("artifact_schema", "source_checkpoint_digest", "corpus_stage"):
            if key in self.meta:
                identity[f"surrogate_{key}"] = self.meta[key]
        return identity


def _ensure_surface_imports() -> None:
    """Put ``neural_network_optimization`` on the path before unpickling.

    Surface checkpoints pickle a Flax module by reference, so unpickling one
    imports ``mirror_aware_model``.  The optimizer used to arrange this itself,
    immediately before its own load; now that a checkpoint's family is read from
    its content, the import has to be available to whoever opens the file first.
    """
    directory = str(REPO_ROOT / "neural_network_optimization")
    if directory not in sys.path:
        sys.path.insert(0, directory)


def _read_blob(path: Path) -> Any:
    _ensure_surface_imports()
    with open(path, "rb") as handle:
        return pickle.load(handle)


def detect_family(path, blob=None) -> str:
    """Return the family of the checkpoint at ``path`` from its content.

    Never inspects the filename.  Raises for a research WNM fit dictionary,
    which carries weights but no architecture and so cannot be loaded without
    the training script that wrote it.
    """
    path = Path(path)
    blob = _read_blob(path) if blob is None else blob
    if not isinstance(blob, dict):
        raise ValueError(f"{path}: not a checkpoint (expected a dict, got {type(blob).__name__})")
    if "model_config" in blob and "variables" in blob:
        return FAMILY_WNM
    if "apply_fn" in blob and "params" in blob:
        return FAMILY_SURFACE_NN
    if "variables" in blob and "selected_step" in blob:
        raise ValueError(
            f"{path} is a research WNM fit (variables + selected_step), not a production "
            "artifact: it records no architecture, so nothing can reconstruct the network "
            "from it alone. Package it first with continuous_density/package_wnm_artifact.py.")
    raise ValueError(f"{path}: unrecognised checkpoint layout, keys {sorted(blob)}")


def resolve_checkpoint(family: Optional[str] = None,
                       n_samples: Optional[int] = None,
                       checkpoint_path=None) -> Path:
    """Resolve a checkpoint request to a path without loading it.

    An explicit ``checkpoint_path`` always wins and is returned as given; the
    family/sample-count arguments are then checked against it at load time.
    Otherwise the default artifact for ``(family, n_samples)`` is returned.
    """
    if checkpoint_path is not None:
        return Path(checkpoint_path)

    family = DEFAULT_FAMILY if family is None else family
    if family not in FAMILIES:
        raise ValueError(f"unknown surrogate family {family!r}; expected one of {FAMILIES}")
    if n_samples is None:
        raise ValueError(
            "n_samples is required to resolve a default checkpoint: 20 and 100 are two "
            "different observer models, so there is no sensible default between them.")
    if n_samples not in SUPPORTED_SAMPLE_COUNTS:
        raise ValueError(f"n_samples={n_samples} is not one of {SUPPORTED_SAMPLE_COUNTS}")

    table = SURFACE_DEFAULTS if family == FAMILY_SURFACE_NN else WNM_DEFAULTS
    path = table[n_samples]
    if not path.exists():
        raise FileNotFoundError(
            f"no {family} artifact installed for n_samples={n_samples}: expected {path}")
    return path


def _surface_sample_count(path: Path, requested: Optional[int]) -> int:
    known = SURFACE_CHECKPOINT_REGISTRY.get(path.name)
    if known is None:
        if requested is None:
            raise ValueError(
                f"{path.name} is not in SURFACE_CHECKPOINT_REGISTRY and surface checkpoints "
                "record no sample count, so its observer model is unknown. Pass n_samples "
                "explicitly, or register the file if it is a known artifact. Its identity is "
                "not inferable from its name.")
        return requested
    if requested is not None and requested != known:
        raise ValueError(
            f"{path.name} is the n_samples={known} checkpoint but n_samples={requested} was "
            "requested; these are different observer models and their results are not "
            "interchangeable.")
    return known


def load_surrogate(family: Optional[str] = None,
                   n_samples: Optional[int] = None,
                   checkpoint_path=None) -> LoadedSurrogate:
    """Resolve and load a surrogate, with the file's own metadata deciding family.

    Args:
        family: requested family, or ``None`` for :data:`DEFAULT_FAMILY`.  When
            an explicit ``checkpoint_path`` disagrees with this, the call raises
            rather than quietly honouring one of the two.
        n_samples: requested observer sample count, checked the same way.
        checkpoint_path: explicit artifact to load, overriding the defaults.

    Returns:
        :class:`LoadedSurrogate`.
    """
    path = resolve_checkpoint(family, n_samples, checkpoint_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"surrogate checkpoint not found: {path}")

    blob = _read_blob(path)
    actual = detect_family(path, blob)
    if family is not None and family != actual:
        raise ValueError(
            f"{path.name} is a {actual} checkpoint but family={family!r} was requested. "
            "The artifact's own metadata decides its family.")

    if actual == FAMILY_WNM:
        from continuous_density.wrapped_mixture_model import ConditionalWrappedMixture

        meta = dict(blob.get("meta") or {})
        declared = meta.get("n_samples")
        if declared is None:
            raise ValueError(
                f"{path.name} declares no n_samples in its metadata; a WNM artifact must "
                "record the observer model it was trained on.")
        if n_samples is not None and int(n_samples) != int(declared):
            raise ValueError(
                f"{path.name} was trained at n_samples={declared} but n_samples={n_samples} "
                "was requested; these are different observer models.")
        model = ConditionalWrappedMixture(**blob["model_config"])
        return LoadedSurrogate(family=FAMILY_WNM, n_samples=int(declared), path=path,
                               payload={"model": model, "variables": blob["variables"]},
                               meta=meta)

    from shared.utils import load_checkpoint

    resolved_samples = _surface_sample_count(path, n_samples)
    state, checkpoint_info = load_checkpoint(str(path))
    return LoadedSurrogate(family=FAMILY_SURFACE_NN, n_samples=resolved_samples, path=path,
                           payload={"state": state, "checkpoint_info": checkpoint_info},
                           meta={})


def add_surrogate_arguments(parser) -> None:
    """Add the standard surrogate-selection flags to an argparse parser.

    Entry points keep their own ``--checkpoint-path`` default for now; these
    flags select among installed artifacts when no explicit path is given.
    """
    parser.add_argument("--surrogate", choices=list(FAMILIES), default=None,
                        help="Surrogate family. Default: the artifact's own metadata when "
                             "--checkpoint-path is given, otherwise "
                             f"{DEFAULT_FAMILY}.")
    parser.add_argument("--surrogate-n-samples", type=int,
                        choices=list(SUPPORTED_SAMPLE_COUNTS), default=None,
                        help="Observer evidence samples per trial (20 or 100). Two different "
                             "observer models, not a precision setting.")
