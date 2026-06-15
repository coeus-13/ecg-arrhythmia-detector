"""
src/models/preprocessing.py
============================
Data splitting and class-imbalance mitigation for ECG arrhythmia classification.

Two cardinal rules enforced by this module
-------------------------------------------

RULE 1 — PATIENT-LEVEL SPLIT (inter-patient protocol)
    Beats from the same patient record MUST NOT appear in both the training
    and test sets.  This is non-negotiable for MIT-BIH evaluations.

    If you split beat-by-beat (intra-patient), the model learns each patient's
    personal ECG morphology and achieves deceptively high accuracy (~99%) that
    collapses on new patients.  The inter-patient split forces the model to
    learn class-generic features.  This is the DS2 evaluation scheme defined by
    Chazal et al. (2004) and is the accepted standard in the arrhythmia
    classification literature.

    Reference: de Chazal P, et al.  "Automatic classification of heartbeats
    using ECG morphology and heartbeat interval features."
    IEEE Trans Biomed Eng. 2004;51(7):1196–1206.

RULE 2 — SMOTE IS FIT ONLY ON TRAINING DATA
    Synthetic Minority Over-sampling Technique (SMOTE) generates new samples
    by interpolating between existing minority-class samples in feature space.
    If SMOTE is applied before the train/test split — or if the resampler is
    fit on any test-set beats — synthetic samples will be interpolated using
    information from the test set, leaking future-knowledge into the model.

    This is a form of data leakage that inflates all reported metrics
    (sensitivity, specificity, F1) and produces a model that cannot generalise.

    This module enforces the correct order at the API level:
        split_by_patient()  →  apply_resampling_pipeline(X_train, y_train only)
    and raises AssertionError if the caller attempts to resample test data.

MIT-BIH inter-patient partition (DS1 / DS2)
---------------------------------------------
DS1 (training):  101,106,108,109,112,114,115,116,118,119,
                 122,124,201,203,205,207,208,209,215,220,
                 223,230
DS2 (test):      100,103,105,111,113,117,121,123,200,202,
                 210,212,213,214,219,221,222,228,231,232,
                 233,234

These partitions are drawn from de Chazal et al. (2004) Table I and are the
standard split used by the majority of published MIT-BIH classifiers.  Using
any other split makes results incomparable to the literature.

Class imbalance in MIT-BIH (approximate)
-----------------------------------------
N  ~90 500 beats   ~75%    ← massive majority
V  ~7 100 beats    ~6%
S  ~2 800 beats    ~2%
F  ~800  beats     ~0.6%
Q  ~800  beats     ~0.6%

Strategy: two-stage resampling inside an imblearn Pipeline
    Stage 1 — RandomUnderSampler  : reduce N to a configurable ceiling
    Stage 2 — SMOTE               : over-sample S, V, F, Q toward targets
    Result  : a balanced training set without discarding all majority data
              and without requiring real minority beats to be collected.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


# ── Canonical DS1 / DS2 partition from de Chazal et al. (2004) ───────────────

DS1_TRAIN_RECORDS: List[str] = [
    "101", "106", "108", "109", "112", "114", "115", "116",
    "118", "119", "122", "124", "201", "203", "205", "207",
    "208", "209", "215", "220", "223", "230",
]

DS2_TEST_RECORDS: List[str] = [
    "100", "103", "105", "111", "113", "117", "121", "123",
    "200", "202", "210", "212", "213", "214", "219", "221",
    "222", "228", "231", "232", "233", "234",
]

# Sanity-check: the two sets must be disjoint and together cover all 44 records
# that carry beat annotations (4 of the 48 MIT-BIH records carry only rhythm).
assert set(DS1_TRAIN_RECORDS).isdisjoint(set(DS2_TEST_RECORDS)), (
    "DS1 and DS2 overlap — partition definition is corrupt."
)

AAMI_CLASSES: List[str] = ["N", "S", "V", "F", "Q"]


# ═════════════════════════════════════════════════════════════════════════════
# 1.  Patient-level split
# ═════════════════════════════════════════════════════════════════════════════

def split_by_patient(
    df: pd.DataFrame,
    train_records: Optional[List[str]] = None,
    test_records:  Optional[List[str]] = None,
    feature_cols:  Optional[List[str]] = None,
    label_col:     str = "label",
    record_col:    str = "record",
    drop_unknown_labels: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    Split a beat-level DataFrame into train and test sets by patient record ID.

    WHY PATIENT-LEVEL SPLITTING MATTERS
    ------------------------------------
    Each MIT-BIH record is a 30-minute continuous ECG from one patient.  A
    single patient contributes ~2 000 beats.  If we split randomly at the beat
    level, ~80% of a patient's beats go to training and ~20% to test.  The
    model then learns that patient's personal morphology (T-wave shape, QRS
    amplitude, baseline offset) and scores well on *their* held-out beats but
    fails on unseen patients.

    The inter-patient (DS1/DS2) split ensures every test-set patient is
    completely invisible during training, producing an honest estimate of
    real-world generalisation.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``FeatureExtractor.extract_batch``, containing a ``record``
        column with MIT-BIH record IDs and a ``label`` column with AAMI codes.
    train_records : list of str, optional
        Record IDs to assign to the training set.
        Defaults to ``DS1_TRAIN_RECORDS`` (de Chazal et al. 2004).
    test_records : list of str, optional
        Record IDs to assign to the test set.
        Defaults to ``DS2_TEST_RECORDS`` (de Chazal et al. 2004).
    feature_cols : list of str, optional
        Columns to treat as model input features.  Defaults to all columns
        whose names begin with ``td_``, ``fd_``, or ``wd_``.
    label_col : str
        Name of the AAMI label column.  Default ``"label"``.
    record_col : str
        Name of the patient record ID column.  Default ``"record"``.
    drop_unknown_labels : bool
        If True, rows whose label is NaN or not in AAMI_CLASSES are removed
        before splitting.  Default True.

    Returns
    -------
    X_train : np.ndarray, shape (n_train, n_features), float64
    X_test  : np.ndarray, shape (n_test,  n_features), float64
    y_train : np.ndarray, shape (n_train,), str
    y_test  : np.ndarray, shape (n_test,),  str
    train_meta : pd.Series — record IDs for training rows (same index as X_train)
    test_meta  : pd.Series — record IDs for test rows    (same index as X_test)

    Raises
    ------
    ValueError
        If train_records and test_records are not disjoint, or if either
        partition is empty after filtering.
    KeyError
        If ``record_col`` or ``label_col`` are not found in *df*.
    """

    # ── Parameter defaults ────────────────────────────────────────────────────
    if train_records is None:
        train_records = DS1_TRAIN_RECORDS
    if test_records is None:
        test_records = DS2_TEST_RECORDS

    train_set = set(str(r) for r in train_records)
    test_set  = set(str(r) for r in test_records)

    # ── Validate inputs ───────────────────────────────────────────────────────
    if record_col not in df.columns:
        raise KeyError(f"Record column '{record_col}' not found in DataFrame.")
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in DataFrame.")

    overlap = train_set & test_set
    if overlap:
        raise ValueError(
            f"train_records and test_records share {len(overlap)} record(s): "
            f"{sorted(overlap)}. "
            "Patient leakage would invalidate all reported metrics."
        )

    # ── Optional: remove unlabelled / out-of-vocabulary rows ─────────────────
    working = df.copy()
    if drop_unknown_labels:
        valid_mask = working[label_col].isin(AAMI_CLASSES)
        n_dropped = (~valid_mask).sum()
        if n_dropped:
            logger.warning(
                "Dropping %d rows with unknown/NaN labels before split.",
                n_dropped,
            )
        working = working[valid_mask].reset_index(drop=True)

    # ── Infer feature columns ─────────────────────────────────────────────────
    if feature_cols is None:
        feature_cols = [
            c for c in working.columns
            if c.startswith(("td_", "fd_", "wd_"))
        ]
        if not feature_cols:
            raise ValueError(
                "No feature columns found with prefixes td_/fd_/wd_. "
                "Pass explicit feature_cols."
            )

    # ── Partition by record ID ────────────────────────────────────────────────
    rec_col_str = working[record_col].astype(str)

    train_mask = rec_col_str.isin(train_set)
    test_mask  = rec_col_str.isin(test_set)

    unassigned = working[~(train_mask | test_mask)][record_col].unique()
    if len(unassigned):
        logger.warning(
            "%d record(s) not assigned to either partition and will be "
            "excluded: %s",
            len(unassigned), sorted(unassigned),
        )

    train_df = working[train_mask]
    test_df  = working[test_mask]

    if len(train_df) == 0:
        raise ValueError(
            "Training partition is empty.  Check that train_records IDs match "
            f"the 'record' column values.  Available records: "
            f"{sorted(working[record_col].unique()[:20])}"
        )
    if len(test_df) == 0:
        raise ValueError(
            "Test partition is empty.  Check that test_records IDs match "
            f"the 'record' column values."
        )

    X_train = train_df[feature_cols].to_numpy(dtype=np.float64, na_value=np.nan)
    X_test  = test_df[feature_cols].to_numpy(dtype=np.float64,  na_value=np.nan)
    y_train = train_df[label_col].to_numpy(dtype=str)
    y_test  = test_df[label_col].to_numpy(dtype=str)

    train_meta = train_df[record_col].reset_index(drop=True)
    test_meta  = test_df[record_col].reset_index(drop=True)

    # ── Integrity assertions (defence-in-depth) ───────────────────────────────
    # These fire even if the caller modifies DS1/DS2 to custom partitions.
    assert len(set(train_meta.unique()) & set(test_meta.unique())) == 0, (
        "CRITICAL: Patient leakage detected — the same record ID appears in "
        "both X_train and X_test.  This invalidates all evaluation metrics."
    )
    assert X_train.shape[1] == X_test.shape[1], (
        f"Feature count mismatch: X_train has {X_train.shape[1]} features "
        f"but X_test has {X_test.shape[1]}."
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    logger.info(
        "Patient-level split complete:\n"
        "  Train — %d beats from %d records  | label dist: %s\n"
        "  Test  — %d beats from %d records  | label dist: %s",
        len(X_train), train_df[record_col].nunique(),
        _label_dist_str(y_train),
        len(X_test),  test_df[record_col].nunique(),
        _label_dist_str(y_test),
    )

    return X_train, X_test, y_train, y_test, train_meta, test_meta


# ═════════════════════════════════════════════════════════════════════════════
# 2.  SMOTE resampling pipeline
# ═════════════════════════════════════════════════════════════════════════════

def apply_resampling_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    # -- Undersampling configuration ------------------------------------------
    n_majority_target:  Optional[int]   = None,
    majority_ratio:     float           = 0.4,
    # -- SMOTE configuration --------------------------------------------------
    smote_k_neighbors:  int             = 5,
    minority_target:    Optional[int]   = None,
    # -- Reproducibility ------------------------------------------------------
    random_state:       int             = 42,
    # -- Safety ---------------------------------------------------------------
    X_test:             Optional[np.ndarray] = None,
    y_test:             Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[str, int]]:
    """
    Apply a two-stage imblearn resampling pipeline to the TRAINING data only.

    ╔══════════════════════════════════════════════════════════════════════════╗
    ║  DATA LEAKAGE WARNING — READ BEFORE MODIFYING THIS FUNCTION             ║
    ║                                                                          ║
    ║  SMOTE works by selecting a minority-class sample, finding its k        ║
    ║  nearest neighbours in feature space, and synthesising a new sample     ║
    ║  somewhere on the line segment between them.                            ║
    ║                                                                          ║
    ║  If test-set samples are included when fitting SMOTE, those samples     ║
    ║  can become neighbours of training-set minority samples.  The synthetic ║
    ║  points will then encode information about the test distribution,       ║
    ║  creating a statistical dependency between training and test data.      ║
    ║                                                                          ║
    ║  The same leakage mechanism applies to RandomUnderSampler: deciding     ║
    ║  which N-class samples to remove based on proximity to test samples     ║
    ║  implicitly incorporates test information into the training set.        ║
    ║                                                                          ║
    ║  This function accepts X_test / y_test ONLY to assert they were not    ║
    ║  accidentally passed as X_train / y_train.  They are never used for     ║
    ║  any computation.                                                        ║
    ╚══════════════════════════════════════════════════════════════════════════╝

    Resampling strategy (two-stage Pipeline)
    ----------------------------------------
    Stage 1 — RandomUnderSampler
        Reduces the dominant N class from ~75% of the dataset to a configurable
        ceiling (default: 40% of its original count, ``majority_ratio=0.4``).
        Removing *some* majority samples before SMOTE is important because:
        a) it reduces the computational cost of SMOTE's kNN search, and
        b) it prevents the model from being overwhelmed by the N class even
           after over-sampling — the boundary between N and minority classes
           becomes better defined with a balanced neighbourhood.

    Stage 2 — SMOTE
        Synthesises new minority-class samples (S, V, F, Q) until each class
        reaches ``minority_target`` samples.  Default: the post-undersampling
        N count, producing a 1:1:1:1:1 class ratio.

        SMOTE interpolates in *feature* space, not raw signal space.  For
        wavelet / time / frequency features this is well-conditioned; for raw
        waveform data it would produce physiologically implausible beats.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, n_features)
        Feature matrix for training beats ONLY.
    y_train : np.ndarray, shape (n_train,)
        AAMI class labels for training beats ONLY.
    n_majority_target : int, optional
        Explicit target count for the N class after undersampling.
        Overrides ``majority_ratio`` if provided.
    majority_ratio : float
        Fraction of original N-class count to retain.  Default 0.4.
    smote_k_neighbors : int
        Number of nearest neighbours for SMOTE interpolation.  Smaller k
        values generate more diverse synthetic samples but risk producing
        samples outside the true class distribution.  Default 5.
    minority_target : int, optional
        Target count for each minority class after SMOTE.  If None,
        defaults to the post-undersampling N count (balanced 1:1 ratio).
    random_state : int
        Seed for reproducibility.  Both stages use this seed.
    X_test : np.ndarray, optional
        Accepted ONLY to detect accidental misuse.  Must NOT be passed
        as X_train.  Triggers an AssertionError if identical to X_train.
    y_test : np.ndarray, optional
        Accepted ONLY to detect accidental misuse (see X_test).

    Returns
    -------
    X_resampled : np.ndarray, shape (n_resampled, n_features)
        Resampled feature matrix.
    y_resampled : np.ndarray, shape (n_resampled,)
        Resampled labels (string AAMI codes).
    counts_before : dict
        Class counts before resampling  {class: count}.
    counts_after  : dict
        Class counts after  resampling  {class: count}.

    Raises
    ------
    AssertionError
        If X_test is identical to X_train (misuse detection).
        If any test-set sample is found inside X_train.
    ValueError
        If a minority class has fewer samples than ``smote_k_neighbors``
        (SMOTE cannot run) — the function raises with a clear message.
    """

    # ── ANTI-LEAKAGE ASSERTIONS ───────────────────────────────────────────────
    #
    # These checks are the last line of defence against accidentally passing
    # test data into the resampling pipeline.
    #
    # Check 1: if the caller passed X_test explicitly, verify it is different.
    if X_test is not None:
        assert not np.array_equal(X_train, X_test), (
            "DATA LEAKAGE: X_train and X_test are identical arrays. "
            "You must call apply_resampling_pipeline with X_train ONLY. "
            "The test set must never be passed to the resampler."
        )
        # Check 2: no test row should be byte-identical to any training row.
        # np.array_equal on the full matrix is expensive for large datasets;
        # we use a row-hash approach that is O(n_test) rather than O(n_train * n_test).
        train_row_hashes = set(
            hash(row.tobytes()) for row in X_train
        )
        leaked = sum(
            1 for row in X_test if hash(row.tobytes()) in train_row_hashes
        )
        assert leaked == 0, (
            f"DATA LEAKAGE: {leaked} test-set row(s) found inside X_train. "
            "The resampling pipeline must be fit only on training data."
        )
        logger.info("Anti-leakage check passed: X_test is disjoint from X_train.")

    # ── Pre-resampling class inventory ────────────────────────────────────────
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    counts_before: Dict[str, int] = dict(zip(unique_classes, class_counts.tolist()))

    logger.info(
        "Class distribution BEFORE resampling:\n%s",
        _format_class_table(counts_before),
    )

    # Validate that all present classes have enough samples for SMOTE
    for cls, cnt in counts_before.items():
        if cls != "N" and cnt <= smote_k_neighbors:
            raise ValueError(
                f"Class '{cls}' has only {cnt} sample(s), but SMOTE requires "
                f"at least k_neighbors+1 = {smote_k_neighbors + 1} samples. "
                f"Either collect more '{cls}' beats, reduce smote_k_neighbors, "
                f"or merge rare classes."
            )

    # ── Stage 1: compute undersampling target for N class ─────────────────────
    n_class_original = counts_before.get("N", 0)

    if n_class_original == 0:
        logger.warning("No 'N' class samples found — skipping undersampling stage.")
        under_sampling_strategy: dict | str = "auto"
    else:
        if n_majority_target is not None:
            n_target = min(n_majority_target, n_class_original)
        else:
            n_target = max(
                int(n_class_original * majority_ratio),
                # Floor: N must never be reduced below the largest minority class
                # to avoid inverting the imbalance ratio
                max((v for k, v in counts_before.items() if k != "N"), default=1),
            )
        # RandomUnderSampler expects {class: target_count} for each class to
        # be undersampled; classes not listed are left unchanged.
        under_sampling_strategy = {cls: cnt for cls, cnt in counts_before.items()}
        under_sampling_strategy["N"] = n_target

    # ── Stage 2: compute SMOTE target for minority classes ────────────────────
    # After undersampling, N will be at n_target.  Default SMOTE target = n_target
    # so all classes end at the same count → balanced 1:1:1:1:1.
    n_post_under = n_target if n_class_original > 0 else max(counts_before.values())

    if minority_target is not None:
        smote_target_count = minority_target
    else:
        smote_target_count = n_post_under

    # Build SMOTE sampling_strategy: over-sample each minority class to target.
    # Classes already at or above the target are left unchanged.
    smote_strategy: Dict[str, int] = {}
    for cls in unique_classes:
        if cls == "N":
            continue
        current = counts_before[cls]
        if current < smote_target_count:
            smote_strategy[cls] = smote_target_count

    # ── Construct the imblearn Pipeline ───────────────────────────────────────
    #
    # imblearn.pipeline.Pipeline is used (not sklearn.pipeline.Pipeline) because
    # it correctly handles resamplers that change the number of samples — a
    # standard sklearn Pipeline would raise at the fit step.
    #
    # The Pipeline chains:
    #   Step 1 "under" → RandomUnderSampler  (reduces N)
    #   Step 2 "smote" → SMOTE               (grows minority classes)
    #
    # Critically: Pipeline.fit_resample(X_train, y_train) fits BOTH steps on
    # X_train/y_train only.  There is no separate transform step that could
    # accidentally be applied to test data.
    #
    steps = []

    if n_class_original > 0:
        steps.append((
            "under",
            RandomUnderSampler(
                sampling_strategy=under_sampling_strategy,
                random_state=random_state,
                # replacement=False: discard original N samples; do NOT
                # bootstrap-resample (which would create duplicate rows and
                # inflate apparent diversity without adding information).
                replacement=False,
            ),
        ))

    if smote_strategy:
        steps.append((
            "smote",
            SMOTE(
                sampling_strategy=smote_strategy,
                k_neighbors=smote_k_neighbors,
                random_state=random_state,
                # n_jobs=-1 would use all cores via joblib; kept at 1 here
                # because the outer training loop already parallelises at the
                # model-fitting level, and nested parallelism can deadlock on
                # some Linux configurations with MKL-threaded NumPy.
                n_jobs=1,
            ),
        ))

    if not steps:
        logger.warning(
            "No resampling steps configured (N class absent, no minority "
            "classes below target).  Returning original data unchanged."
        )
        counts_after = counts_before.copy()
        return X_train.copy(), y_train.copy(), counts_before, counts_after

    pipeline = ImbPipeline(steps=steps)

    # ── CRITICAL: fit_resample on TRAINING DATA ONLY ──────────────────────────
    #
    # This is the single point where the resampler learns from data.
    # X_test and y_test are NEVER passed here.  The pipeline object produced
    # by fit_resample must also NEVER be called with .transform(X_test) or
    # .resample(X_test) downstream — doing so would constitute data leakage
    # by applying training-set-derived synthetic boundaries to test data.
    #
    logger.info(
        "Fitting resampling pipeline on training data ONLY "
        "(%d samples, %d features) …",
        X_train.shape[0], X_train.shape[1],
    )
    X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)

    # ── Post-resampling audit ─────────────────────────────────────────────────
    unique_after, counts_after_arr = np.unique(y_resampled, return_counts=True)
    counts_after: Dict[str, int] = dict(zip(unique_after, counts_after_arr.tolist()))

    logger.info(
        "Class distribution AFTER resampling:\n%s",
        _format_class_table(counts_after),
    )
    logger.info(
        "Resampling summary: %d → %d samples  (×%.2f expansion)",
        len(X_train), len(X_resampled),
        len(X_resampled) / max(len(X_train), 1),
    )

    # ── Post-resampling integrity assertions ──────────────────────────────────
    #
    # Verify the resampled set does not re-introduce test data through
    # SMOTE synthetic points that happen to match test rows exactly
    # (extremely unlikely but architecturally important to assert).
    if X_test is not None:
        resampled_hashes = set(hash(row.tobytes()) for row in X_resampled)
        test_hashes = [hash(row.tobytes()) for row in X_test]
        leaked_after = sum(1 for h in test_hashes if h in resampled_hashes)
        # Allow matches only from the original training rows (not synthetic)
        original_hashes = set(hash(row.tobytes()) for row in X_train)
        new_leaked = sum(
            1 for h in test_hashes
            if h in resampled_hashes and h not in original_hashes
        )
        assert new_leaked == 0, (
            f"DATA LEAKAGE: {new_leaked} SMOTE-generated sample(s) are "
            "byte-identical to test-set rows.  This is a statistical anomaly "
            "that should be investigated immediately."
        )

    return X_resampled, y_resampled, counts_before, counts_after


# ═════════════════════════════════════════════════════════════════════════════
# 3.  NaN imputation  (required before SMOTE — sklearn transformers reject NaN)
# ═════════════════════════════════════════════════════════════════════════════

def impute_features(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    strategy: str = "median",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Impute missing values in feature matrices using training-set statistics.

    NaN values arise legitimately from:
    • ``td_pre_rr``  — first beat of a record has no preceding beat
    • ``td_post_rr`` — last  beat of a record has no following beat
    • ``td_rr_ratio``/ ``td_rr_diff`` — propagated from the above
    • Rare edge cases where QRS detection fails (flat-line segment)

    LEAKAGE RULE
    ~~~~~~~~~~~~
    The imputer is FIT on X_train only.  X_test is transformed using the
    training-set median/mean — never fit on X_test.  Fitting on X_test would
    reveal the test-set distribution to the imputer and bias the transformation.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, n_features)
    X_test  : np.ndarray, shape (n_test,  n_features)
    strategy : str
        Imputation statistic: ``'median'`` (default, robust to outliers)
        or ``'mean'``.

    Returns
    -------
    X_train_imp : np.ndarray — imputed training features
    X_test_imp  : np.ndarray — imputed test features (using training stats)
    """
    from sklearn.impute import SimpleImputer

    # FIT on training data only
    imputer = SimpleImputer(strategy=strategy, keep_empty_features=True)
    X_train_imp = imputer.fit_transform(X_train)

    # TRANSFORM test data using training-set statistics — no fit on test
    X_test_imp  = imputer.transform(X_test)

    n_nan_train = np.isnan(X_train).sum()
    n_nan_test  = np.isnan(X_test).sum()
    logger.info(
        "Imputation (%s) — filled %d NaNs in X_train, %d NaNs in X_test "
        "(using training statistics).",
        strategy, n_nan_train, n_nan_test,
    )

    return X_train_imp, X_test_imp


# ═════════════════════════════════════════════════════════════════════════════
# 4.  Feature scaling  (fit on train, transform both)
# ═════════════════════════════════════════════════════════════════════════════

def scale_features(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    scaler_type: str = "robust",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a feature scaler on X_train and apply it to X_train and X_test.

    LEAKAGE RULE
    ~~~~~~~~~~~~
    The scaler is FIT on X_train only.  The same fitted scaler is then
    applied to X_test.  Fitting on X_test (or the full dataset) would leak
    the test-set distribution (min, max, median) into the scaling parameters,
    giving the model implicit knowledge of the test set.

    Scaler choices
    --------------
    ``'robust'``  (default)
        Scales using training-set median and IQR.  Preferred for ECG features
        because wavelet energy and RR intervals have heavy-tailed distributions
        with occasional extreme outlier beats (artefacts).  RobustScaler does
        not compress the bulk of the distribution to accommodate outliers.

    ``'standard'``
        Zero-mean, unit-variance.  Appropriate if outliers have been removed
        and the feature distributions are approximately Gaussian.

    ``'minmax'``
        Scales to [0, 1] using training-set min/max.  Sensitive to outliers;
        not recommended for raw ECG features.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, n_features)
    X_test  : np.ndarray, shape (n_test,  n_features)
    scaler_type : str
        One of ``'robust'``, ``'standard'``, ``'minmax'``.

    Returns
    -------
    X_train_scaled : np.ndarray
    X_test_scaled  : np.ndarray
    """
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    scalers = {
        "robust":   RobustScaler(),
        "standard": StandardScaler(),
        "minmax":   MinMaxScaler(),
    }
    if scaler_type not in scalers:
        raise ValueError(
            f"Unknown scaler_type '{scaler_type}'. "
            f"Choose from: {list(scalers.keys())}"
        )

    scaler = scalers[scaler_type]

    # FIT on training data only — captures training-set distribution
    X_train_scaled = scaler.fit_transform(X_train)

    # TRANSFORM test using training statistics — no information from test
    X_test_scaled  = scaler.transform(X_test)

    logger.info(
        "Feature scaling (%s) applied — fit on X_train (%d, %d), "
        "transformed X_test (%d, %d).",
        scaler_type,
        X_train.shape[0], X_train.shape[1],
        X_test.shape[0],  X_test.shape[1],
    )

    return X_train_scaled, X_test_scaled


# ═════════════════════════════════════════════════════════════════════════════
# Private helpers
# ═════════════════════════════════════════════════════════════════════════════

def _label_dist_str(y: np.ndarray) -> str:
    """Compact label distribution string for logging."""
    unique, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    parts = [f"{cls}={cnt}({cnt/total*100:.1f}%)" for cls, cnt in zip(unique, counts)]
    return "  ".join(parts)


def _format_class_table(counts: Dict[str, int]) -> str:
    """ASCII table of class counts and percentages for logging."""
    total = sum(counts.values())
    lines = ["  Class  Count    Pct", "  ─────  ──────  ──────"]
    for cls in AAMI_CLASSES:
        cnt = counts.get(cls, 0)
        pct = cnt / total * 100 if total > 0 else 0.0
        lines.append(f"    {cls}    {cnt:6d}   {pct:5.1f}%")
    lines.append(f"  Total  {total:6d}  100.0%")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Standalone demo
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import textwrap

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 72)
    print("  preprocessing.py — Phase 4 Demo")
    print("=" * 72)

    # ── Build a realistic synthetic DataFrame mimicking FeatureExtractor output
    np.random.seed(42)

    N_FEATURES = 46        # matches Phase 3 output (14 td + 8 fd + 24 wd)
    RECORDS    = DS1_TRAIN_RECORDS + DS2_TEST_RECORDS

    # Simulate ~2 000 beats per record with realistic class imbalance
    CLASS_PROBS = {"N": 0.75, "V": 0.08, "S": 0.05, "F": 0.03, "Q": 0.09}
    rows = []
    for rec in RECORDS:
        n_beats = np.random.randint(1800, 2200)
        labels  = np.random.choice(
            list(CLASS_PROBS.keys()),
            size=n_beats,
            p=list(CLASS_PROBS.values()),
        )
        feats = np.random.randn(n_beats, N_FEATURES).astype(np.float32)
        # Inject some NaNs to mimic boundary RR features
        nan_idx = np.random.choice(n_beats, size=max(1, n_beats // 20), replace=False)
        feats[nan_idx, 0] = np.nan   # td_pre_rr
        feats[nan_idx, 1] = np.nan   # td_post_rr
        for i, lbl in enumerate(labels):
            row = {f"td_feat_{j}" if j < 14 else
                   (f"fd_feat_{j-14}" if j < 22 else f"wd_feat_{j-22}"): feats[i, j]
                   for j in range(N_FEATURES)}
            row["record"] = rec
            row["label"]  = lbl
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\n[0] Synthetic dataset: {len(df):,} beats | "
          f"{df['record'].nunique()} records | {N_FEATURES} features")
    print(f"    Overall class dist: {_label_dist_str(df['label'].to_numpy())}")

    # ── Step 1: Patient-level split ───────────────────────────────────────────
    print("\n" + "─" * 72)
    print("[1] Patient-level split (DS1/DS2 inter-patient protocol)")
    print("─" * 72)

    X_train, X_test, y_train, y_test, train_meta, test_meta = split_by_patient(df)

    print(f"\n    X_train : {X_train.shape}  |  y_train unique records: "
          f"{train_meta.nunique()}")
    print(f"    X_test  : {X_test.shape}   |  y_test  unique records: "
          f"{test_meta.nunique()}")

    overlap = set(train_meta.unique()) & set(test_meta.unique())
    print(f"\n    ✓ Patient overlap between train and test: {len(overlap)} records "
          f"(must be 0)")

    # ── Step 2: NaN imputation (fit on train only) ────────────────────────────
    print("\n" + "─" * 72)
    print("[2] NaN imputation (fit on X_train, transform both)")
    print("─" * 72)

    X_train_imp, X_test_imp = impute_features(X_train, X_test, strategy="median")
    print(f"    NaN in X_train after imputation : {np.isnan(X_train_imp).sum()}")
    print(f"    NaN in X_test  after imputation : {np.isnan(X_test_imp).sum()}")

    # ── Step 3: Feature scaling (fit on train only) ───────────────────────────
    print("\n" + "─" * 72)
    print("[3] Feature scaling — RobustScaler (fit on X_train only)")
    print("─" * 72)

    X_train_sc, X_test_sc = scale_features(X_train_imp, X_test_imp, "robust")
    print(f"    X_train scaled — mean={X_train_sc.mean():.4f}  "
          f"std={X_train_sc.std():.4f}")
    print(f"    X_test  scaled — mean={X_test_sc.mean():.4f}  "
          f"std={X_test_sc.std():.4f}  (different from train — correct)")

    # ── Step 4: SMOTE resampling (training data ONLY) ─────────────────────────
    print("\n" + "─" * 72)
    print("[4] SMOTE resampling pipeline (training data ONLY)")
    print("─" * 72)

    X_res, y_res, counts_before, counts_after = apply_resampling_pipeline(
        X_train=X_train_sc,
        y_train=y_train,
        majority_ratio=0.4,
        smote_k_neighbors=5,
        random_state=42,
        X_test=X_test_sc,    # passed for anti-leakage assertion only
        y_test=y_test,
    )

    print(f"\n    Before resampling : {len(X_train_sc):>7,} samples")
    print(f"    After  resampling : {len(X_res):>7,} samples")
    print(f"\n    Class counts BEFORE resampling:")
    print(textwrap.indent(_format_class_table(counts_before), "      "))
    print(f"\n    Class counts AFTER  resampling:")
    print(textwrap.indent(_format_class_table(counts_after), "      "))

    # ── Step 5: Leakage demonstration (expected to fail) ─────────────────────
    print("\n" + "─" * 72)
    print("[5] Anti-leakage guard demonstration")
    print("─" * 72)
    print("    Attempting to resample with X_test passed as X_train …")
    try:
        apply_resampling_pipeline(
            X_train=X_test_sc,    # ← deliberately wrong: passing test as train
            y_train=y_test,
            X_test=X_test_sc,     # ← same array → should trigger assertion
            y_test=y_test,
        )
        print("    ✗ LEAKAGE NOT DETECTED — this should not happen.")
    except AssertionError as exc:
        print(f"    ✓ AssertionError raised as expected:\n"
              f"      {str(exc)[:120]} …")

    print("\n[✓] Phase 4 preprocessing complete — "
          "X_res / y_res ready for Phase 5 model training.\n")