"""
src/models/classifier.py
========================
ECG arrhythmia stacking ensemble architecture.

This module is responsible solely for *defining and configuring* the model
architecture.  Training, evaluation, and threshold calibration live in
``train_evaluate.py``.  Keeping architecture separate from the training loop
makes hyperparameter sweeps, serialisation, and unit-testing straightforward.

Stacking architecture
---------------------

  Level-0  (Base Learners — trained on K-fold out-of-fold splits)
  ┌─────────────────────────────────────────────────────────────────┐
  │  SVC          │  RandomForest       │  HistGradientBoosting     │
  │  RBF kernel   │  balanced_subsample │  CPU-optimised histogram  │
  │  balanced     │  n_jobs=-1          │  native NaN support       │
  └───────┬───────┴──────────┬──────────┴──────────────┬────────────┘
          │  predict_proba   │  predict_proba           │  predict_proba
          └──────────────────┴──────────────────────────┘
                             ↓
               Meta-feature matrix  shape (n_train, n_classes × 3)
                             ↓
  Level-1  (Meta-Learner — trained on OOF meta-features)
  ┌──────────────────────────────────────────────┐
  │  LogisticRegression                          │
  │  multinomial · balanced · C=1.0 · lbfgs      │
  └──────────────────────────────────────────────┘

Why these three base learners?
-------------------------------
SVC (RBF)
    A kernel machine that finds the maximum-margin decision boundary in an
    implicit high-dimensional feature space.  ``class_weight='balanced'``
    rescales the C penalty by inverse class frequency, giving minority classes
    (S, V, F, Q) proportionally larger margin violations — the kernel-space
    equivalent of oversampling.  ``probability=True`` activates internal
    Platt scaling so ``predict_proba`` is available for stacking.

RandomForestClassifier
    An ensemble of independently trained decision trees on bootstrap samples.
    ``class_weight='balanced_subsample'`` recomputes per-class weights on each
    bootstrap draw (rather than the global class distribution), making the
    correction adaptive per tree.  Provides natural feature importance rankings
    and out-of-bag error estimates.  ``n_jobs=-1`` saturates all CPU cores via
    joblib during both fit and predict.

HistGradientBoostingClassifier
    A histogram-binned gradient boosted tree implementation modelled after
    LightGBM.  On Intel CPUs it uses BLAS-accelerated bin computation and is
    substantially faster than ``GradientBoostingClassifier`` for large datasets.
    Key properties:
    - Native NaN support: ``td_pre_rr`` / ``td_post_rr`` boundary NaNs do not
      require imputation before HGB sees the data.
    - ``early_stopping='auto'`` halts training when validation loss plateaus,
      preventing overfitting on sparse F and Q classes.
    - ``l2_regularization`` shrinks leaf values, reducing variance on minority
      classes that have fewer training examples even after SMOTE.

Why LogisticRegression as the meta-learner?
--------------------------------------------
The meta-learner's input is a (n_train, 15) matrix of out-of-fold class
probabilities (3 learners × 5 AAMI classes).  Logistic Regression with
softmax loss learns a linear combination of these probability columns that
minimises cross-entropy — effectively discovering which base learner is most
trustworthy for each class.  L2 regularisation (C=1.0) prevents the meta-
learner from collapsing to a single base learner when one is dominant.

The ``balanced`` class weight on the meta-learner provides an additional guard
against the residual imbalance that persists in OOF predictions even after
SMOTE resampling of the training set.

5-fold cross-validation in StackingClassifier
----------------------------------------------
``cv=5`` means the training set is split into 5 folds.  For each fold:
  - 4 folds are used to fit each base learner.
  - The held-out fold receives ``predict_proba`` from that fit.
This produces out-of-fold (OOF) predictions for every training sample —
predictions that were never seen during that base learner's training.
The meta-learner is then trained on the full OOF matrix, meaning it learns
on samples the base learners had never seen.  This prevents the meta-learner
from learning to exploit the base learners' overfit patterns.

After generating OOF predictions, sklearn re-fits each base learner on the
*full* training set so the final ``predict`` / ``predict_proba`` calls use
models trained on all available data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical AAMI class ordering — imported by train_evaluate.py
# ---------------------------------------------------------------------------
AAMI_CLASSES: List[str] = ["N", "S", "V", "F", "Q"]
V_CLASS_IDX: int = AAMI_CLASSES.index("V")


# ---------------------------------------------------------------------------
# Hyperparameter containers
# ---------------------------------------------------------------------------

@dataclass
class SVCConfig:
    """Hyperparameters for the SVC base learner."""
    C: float = 1.0
    kernel: str = "rbf"
    gamma: str = "scale"
    class_weight: str = "balanced"
    probability: bool = True           # required for predict_proba in stacking
    random_state: int = 42
    # Note: SVC does not accept n_jobs directly; parallelism is available only
    # at the cross-validation level inside StackingClassifier.


@dataclass
class RandomForestConfig:
    """Hyperparameters for the RandomForestClassifier base learner."""
    n_estimators: int = 500
    max_features: str = "sqrt"         # ~sqrt(46) ≈ 7 features per split
    max_depth: Optional[int] = None    # grow fully to minimise bias
    min_samples_leaf: int = 2          # slight smoothing for minority leaves
    class_weight: str = "balanced_subsample"
    n_jobs: int = -1                   # use all available CPU cores
    random_state: int = 42


@dataclass
class HGBConfig:
    """Hyperparameters for the HistGradientBoostingClassifier base learner."""
    learning_rate: float = 0.05
    max_iter: int = 300
    max_leaf_nodes: int = 31           # LightGBM default; controls complexity
    min_samples_leaf: int = 20         # prevents tiny leaves on minority classes
    l2_regularization: float = 0.1
    early_stopping: str = "auto"       # halts when val loss plateaus
    validation_fraction: float = 0.1   # fraction of training data held for ES
    n_iter_no_change: int = 20         # patience for early stopping
    scoring: str = "loss"
    random_state: int = 42


@dataclass
class MetaLearnerConfig:
    """Hyperparameters for the LogisticRegression meta-learner."""
    C: float = 1.0
    penalty: str = "l2"
    multi_class: str = "multinomial"
    solver: str = "lbfgs"
    max_iter: int = 1000
    class_weight: str = "balanced"
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class StackingConfig:
    """Top-level stacking ensemble configuration."""
    cv: int = 5                         # cross-validation folds for OOF generation
    stack_method: str = "predict_proba" # richer signal than hard predictions
    passthrough: bool = False           # do NOT concatenate raw features to meta input
    # passthrough=False rationale: with 46 raw features added to 15 OOF features,
    # the meta-learner can overfit the raw space and bypass the stacking benefit.

    svc: SVCConfig = field(default_factory=SVCConfig)
    rf: RandomForestConfig = field(default_factory=RandomForestConfig)
    hgb: HGBConfig = field(default_factory=HGBConfig)
    meta: MetaLearnerConfig = field(default_factory=MetaLearnerConfig)


# ---------------------------------------------------------------------------
# Ensemble class
# ---------------------------------------------------------------------------

class ECGStackingEnsemble:
    """
    Factory and container for the ECG arrhythmia stacking classifier.

    This class is not itself a sklearn estimator — it holds configuration
    and constructs a ``StackingClassifier`` via ``build()``.  Separating
    construction from the sklearn interface makes it straightforward to:
    - Swap hyperparameters without subclassing sklearn estimators.
    - Serialise config independently of fitted weights.
    - Run multiple builds with different configs in a hyperparameter search.

    Parameters
    ----------
    config : StackingConfig, optional
        Full hyperparameter configuration.  Defaults to production-ready
        settings tuned for the MIT-BIH 46-feature dataset.

    Examples
    --------
    Default configuration:

    >>> ensemble_factory = ECGStackingEnsemble()
    >>> model = ensemble_factory.build()
    >>> model.fit(X_train_resampled, y_train_resampled)
    >>> y_pred = model.predict(X_test)

    Custom configuration:

    >>> config = StackingConfig(cv=3)
    >>> config.rf.n_estimators = 200
    >>> config.hgb.max_iter = 150
    >>> model = ECGStackingEnsemble(config).build()
    """

    def __init__(self, config: Optional[StackingConfig] = None) -> None:
        self.config: StackingConfig = config or StackingConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> StackingClassifier:
        """
        Construct and return an *unfitted* StackingClassifier.

        The estimator is returned unfitted so the caller has full control
        over when and on what data ``.fit()`` is called.  This prevents
        accidental training on test data and makes the object serialisable
        before any data is seen.

        Returns
        -------
        StackingClassifier
            Ready to call ``.fit(X_train, y_train)``.
        """
        cfg = self.config

        svc      = self._build_svc(cfg.svc)
        rf       = self._build_rf(cfg.rf)
        hgb      = self._build_hgb(cfg.hgb)
        meta_lr  = self._build_meta(cfg.meta)

        estimators = [
            ("svc", svc),
            ("rf",  rf),
            ("hgb", hgb),
        ]

        stack = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_lr,
            cv=cfg.cv,
            stack_method=cfg.stack_method,
            passthrough=cfg.passthrough,
            # n_jobs at the stacking level controls cross-fold parallelism.
            # Set to 1 here to avoid nested joblib parallelism deadlocks when
            # RF and SVC already use n_jobs=-1 internally.
            n_jobs=1,
            verbose=0,
        )

        logger.info(
            "ECGStackingEnsemble built | base=%s | cv=%d | "
            "stack_method=%s | passthrough=%s | meta=%s",
            [name for name, _ in estimators],
            cfg.cv,
            cfg.stack_method,
            cfg.passthrough,
            type(meta_lr).__name__,
        )
        return stack

    def config_summary(self) -> Dict:
        """
        Return a flat dictionary of all hyperparameters for logging / MLflow.

        Returns
        -------
        dict
            Keys follow the pattern ``{component}__{param}``.
        """
        cfg = self.config
        return {
            # Stacking
            "stacking__cv":            cfg.cv,
            "stacking__stack_method":  cfg.stack_method,
            "stacking__passthrough":   cfg.passthrough,
            # SVC
            "svc__C":                  cfg.svc.C,
            "svc__kernel":             cfg.svc.kernel,
            "svc__gamma":              cfg.svc.gamma,
            "svc__class_weight":       cfg.svc.class_weight,
            # Random Forest
            "rf__n_estimators":        cfg.rf.n_estimators,
            "rf__max_features":        cfg.rf.max_features,
            "rf__max_depth":           cfg.rf.max_depth,
            "rf__min_samples_leaf":    cfg.rf.min_samples_leaf,
            "rf__class_weight":        cfg.rf.class_weight,
            # HGB
            "hgb__learning_rate":      cfg.hgb.learning_rate,
            "hgb__max_iter":           cfg.hgb.max_iter,
            "hgb__max_leaf_nodes":     cfg.hgb.max_leaf_nodes,
            "hgb__l2_regularization":  cfg.hgb.l2_regularization,
            "hgb__early_stopping":     cfg.hgb.early_stopping,
            # Meta-learner
            "meta__C":                 cfg.meta.C,
            "meta__class_weight":      cfg.meta.class_weight,
            "meta__solver":            cfg.meta.solver,
        }

    # ------------------------------------------------------------------
    # Private builder helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_svc(c: SVCConfig) -> SVC:
        """Construct the SVC base learner from config."""
        return SVC(
            C=c.C,
            kernel=c.kernel,
            gamma=c.gamma,
            class_weight=c.class_weight,
            probability=c.probability,
            random_state=c.random_state,
        )

    @staticmethod
    def _build_rf(c: RandomForestConfig) -> RandomForestClassifier:
        """Construct the RandomForestClassifier base learner from config."""
        return RandomForestClassifier(
            n_estimators=c.n_estimators,
            max_features=c.max_features,
            max_depth=c.max_depth,
            min_samples_leaf=c.min_samples_leaf,
            class_weight=c.class_weight,
            n_jobs=c.n_jobs,
            random_state=c.random_state,
        )

    @staticmethod
    def _build_hgb(c: HGBConfig) -> HistGradientBoostingClassifier:
        """Construct the HistGradientBoostingClassifier base learner from config."""
        return HistGradientBoostingClassifier(
            learning_rate=c.learning_rate,
            max_iter=c.max_iter,
            max_leaf_nodes=c.max_leaf_nodes,
            min_samples_leaf=c.min_samples_leaf,
            l2_regularization=c.l2_regularization,
            early_stopping=c.early_stopping,
            validation_fraction=c.validation_fraction,
            n_iter_no_change=c.n_iter_no_change,
            scoring=c.scoring,
            random_state=c.random_state,
        )

    @staticmethod
    def _build_meta(c: MetaLearnerConfig) -> LogisticRegression:
        """Construct the LogisticRegression meta-learner from config."""
        return LogisticRegression(
            C=c.C,
            penalty=c.penalty,
            multi_class=c.multi_class,
            solver=c.solver,
            max_iter=c.max_iter,
            class_weight=c.class_weight,
            random_state=c.random_state,
            n_jobs=c.n_jobs,
        )


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )

    print("=" * 60)
    print("  classifier.py — architecture smoke-test")
    print("=" * 60)

    factory = ECGStackingEnsemble()
    model   = factory.build()

    print("\n[1] StackingClassifier estimators:")
    for name, est in model.estimators:
        print(f"    {name:>4} → {type(est).__name__}")
    print(f"    meta → {type(model.final_estimator).__name__}")

    print("\n[2] Config summary (sample):")
    for k, v in list(factory.config_summary().items())[:8]:
        print(f"    {k:<35}: {v}")
    print("    ...")

    print("\n[3] Quick fit on toy data (n=300, features=46) ...")
    np.random.seed(0)
    X_toy = np.random.randn(300, 46).astype("float32")
    y_toy = np.array(["N"] * 200 + ["V"] * 50 + ["S"] * 30 + ["F"] * 10 + ["Q"] * 10)
    model.fit(X_toy, y_toy)

    proba = model.predict_proba(X_toy[:5])
    print(f"    predict_proba shape: {proba.shape}  (expect (5, 5))")
    print(f"    classes_: {list(model.classes_)}")
    print("\n[✓] classifier.py verified.\n")