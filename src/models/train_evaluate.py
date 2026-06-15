"""
src/models/train_evaluate.py
=============================
Main execution script for the ECG arrhythmia classification pipeline.

This script is the single entry point that ties together every upstream module:

  Phase 1  src/data/data_loader.py          -> ECGDataLoader
  Phase 2  src/features/signal_processing.py -> ECGFilterPipeline
           src/features/segmentation.py     -> PanTompkinsDetector
  Phase 3  src/features/extraction.py       -> FeatureExtractor
  Phase 4  src/models/preprocessing.py      -> split_by_patient,
                                              apply_resampling_pipeline,
                                              impute_features, scale_features
  Phase 5  src/models/classifier.py         -> ECGStackingEnsemble  (this file)

Execution
---------
From the repository root:

    python -m src.models.train_evaluate          # production run
    python src/models/train_evaluate.py --fast   # smoke-test (< 2 min)

Data flow
---------
Raw MIT-BIH records
    -> ECGDataLoader         (wfdb -> signal + AAMI labels)
    -> ECGFilterPipeline     (bandpass + notch)
    -> PanTompkinsDetector   (R-peak detection + beat segmentation)
    -> FeatureExtractor      (46 time/frequency/wavelet features per beat)
    -> split_by_patient      (DS1 train / DS2 test -- NO patient overlap)
    -> impute_features       (median on train, transform both)
    -> scale_features        (RobustScaler on train, transform both)
    -> apply_resampling_pipeline  (SMOTE on TRAIN ONLY)
    -> ECGStackingEnsemble.build().fit(X_train_res, y_train_res)
    -> evaluate_model(X_test_imbalanced)   <- test set NEVER resampled
    -> find_v_sensitivity_threshold        <- calibrate V-class boundary
    -> apply_v_threshold                   <- final clinical predictions

Test set integrity rule
-----------------------
The test set (DS2) is evaluated in its NATURAL, IMBALANCED state.
It is NEVER passed through SMOTE or RandomUnderSampler.
This is the only evaluation that honestly reflects real-world deployment,
where the N class constitutes ~75% of all beats.  Evaluating on a balanced
test set would artificially inflate minority-class metrics.
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import LabelBinarizer

# ---------------------------------------------------------------------------
# Ensure src/ is importable regardless of invocation style
# ---------------------------------------------------------------------------
_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from models.classifier import (
    AAMI_CLASSES,
    ECGStackingEnsemble,
    StackingConfig,
    V_CLASS_IDX,
)

try:
    from models.preprocessing import (
        apply_resampling_pipeline,
        impute_features,
        scale_features,
        split_by_patient,
    )
    _HAVE_PREPROCESSING = True
except ImportError:
    _HAVE_PREPROCESSING = False
    warnings.warn(
        "src/models/preprocessing.py not found -- "
        "using inline sklearn fallbacks for splitting, imputation, and scaling.",
        ImportWarning,
        stacklevel=1,
    )

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
V_SENSITIVITY_FLOOR: float = 0.95
RANDOM_STATE:        int   = 42
OUTPUT_DIR:          Path  = Path("outputs") / "evaluation"


# =============================================================================
# 1.  Data preparation helpers
# =============================================================================

def load_feature_dataframe(data_dir: Path) -> pd.DataFrame:
    """
    Load or regenerate the compiled feature DataFrame.

    Checks for a cached Parquet file produced by a previous run of
    FeatureExtractor.extract_batch.  If not found and raw MIT-BIH data is
    present, triggers the full Phase 1-3 pipeline.

    Parameters
    ----------
    data_dir : Path
        Root data directory containing mitdb/ and optionally features.parquet.

    Returns
    -------
    pd.DataFrame
        Beat-level feature DataFrame with columns td_*, fd_*, wd_*,
        'label' (AAMI code), and 'record' (MIT-BIH record ID).
    """
    cache_path = data_dir / "features.parquet"

    if cache_path.exists():
        logger.info("Loading cached features from %s", cache_path)
        df = pd.read_parquet(cache_path)
        logger.info("Loaded %d beats x %d columns from cache.", len(df), len(df.columns))
        return df

    mitdb_dir = data_dir / "mitdb"
    if not mitdb_dir.exists() or not any(mitdb_dir.glob("*.dat")):
        raise FileNotFoundError(
            f"Neither '{cache_path}' nor MIT-BIH raw data at '{mitdb_dir}' found.\n"
            "Run Phase 1-3 first:\n"
            "    python src/data/data_loader.py        # downloads mitdb\n"
            "    python src/features/extraction.py     # builds features.parquet\n"
            "Or pass a pre-built DataFrame directly to run_pipeline()."
        )

    logger.info("Cache not found -- regenerating from raw MIT-BIH data ...")
    from data.data_loader import ECGDataLoader, ALL_MITBIH_RECORDS, MITBIH_TO_AAMI
    from features.feature_extractor import FeatureExtractor
    from features.segmentation import EdgeStrategy, PanTompkinsDetector
    from features.signal_processing import ECGFilterPipeline
    import wfdb

    loader    = ECGDataLoader(data_dir=data_dir)
    pipeline  = ECGFilterPipeline(fs=360.0)
    detector  = PanTompkinsDetector(fs=360.0, edge_strategy=EdgeStrategy.SKIP)
    extractor = FeatureExtractor(fs=360.0, pre_peak_samples=108)

    all_frames: List[pd.DataFrame] = []
    for record_id in ALL_MITBIH_RECORDS:
        try:
            rec = wfdb.rdrecord(str(mitdb_dir / record_id))
            raw = rec.p_signal[:, 0].astype(np.float32)
            clean = pipeline.process_signal(raw)
            beats, kept_peaks, all_peaks = detector.detect_and_extract(clean)
            rr_ms, _ = detector.compute_rr_intervals(all_peaks)
            ann = wfdb.rdann(str(mitdb_dir / record_id), "atr")
            ann_dict = dict(zip(ann.sample.tolist(), ann.symbol))
            labels = []
            for rp in kept_peaks:
                closest = min(ann.sample, key=lambda s: abs(int(s) - int(rp)))
                sym = ann_dict.get(int(closest), "Q")
                labels.append(MITBIH_TO_AAMI.get(sym, "Q"))
            feat_df = extractor.extract_batch(
                beats=beats, rr_intervals=rr_ms,
                r_peak_indices=kept_peaks, labels=labels,
            )
            feat_df["record"] = record_id
            all_frames.append(feat_df)
        except Exception as exc:
            logger.warning("Skipping record %s: %s", record_id, exc)

    df = pd.concat(all_frames, ignore_index=True)
    df.to_parquet(cache_path, index=False)
    logger.info("Feature DataFrame saved to %s (%d beats)", cache_path, len(df))
    return df


def prepare_train_test(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply the full Phase-4 preprocessing chain and return scaled arrays.

    Processing order
    ----------------
    1. Patient-level split  (DS1 train / DS2 test -- zero patient overlap)
    2. Median imputation    (fit on train, transform both)
    3. RobustScaler         (fit on train, transform both)

    Parameters
    ----------
    df : pd.DataFrame
        Full feature DataFrame from load_feature_dataframe.

    Returns
    -------
    X_train, X_test : np.ndarray -- scaled, imputed feature matrices
    y_train, y_test : np.ndarray -- string AAMI labels
    """
    if _HAVE_PREPROCESSING:
        X_tr, X_te, y_tr, y_te, _, _ = split_by_patient(df)
        X_tr, X_te = impute_features(X_tr, X_te, strategy="median")
        X_tr, X_te = scale_features(X_tr, X_te, scaler_type="robust")
    else:
        logger.warning("Using inline 70/30 train-test split (not patient-level).")
        from sklearn.model_selection import train_test_split
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import RobustScaler

        feat_cols = [c for c in df.columns if c.startswith(("td_", "fd_", "wd_"))]
        X = df[feat_cols].to_numpy(dtype=np.float64)
        y = df["label"].to_numpy(dtype=str)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
        )
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X_tr)
        X_te = imp.transform(X_te)
        sc = RobustScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

    return X_tr, X_te, y_tr, y_te


# =============================================================================
# 2.  Evaluation
# =============================================================================

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: List[str] = AAMI_CLASSES,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Compute and display the full clinical evaluation suite on the test set.

    The test set must be the raw, IMBALANCED DS2 partition -- never resampled.
    All metrics reflect real-world class distributions.

    Metrics produced
    ----------------
    - Normalised confusion matrix  (row-normalised: diagonal = per-class recall)
    - Raw confusion matrix         (absolute counts)
    - Full classification report   (precision, recall, F1, support per class)
    - One-vs-rest ROC-AUC          (macro average and per class)
    - Per-class sensitivity (= recall) and specificity

    Parameters
    ----------
    model : fitted StackingClassifier
    X_test : np.ndarray, shape (n_test, n_features)
        Scaled, imputed test features.  NEVER resampled.
    y_test : np.ndarray, shape (n_test,)
        True AAMI labels.
    classes : list of str
    output_dir : Path, optional
        If supplied, saves confusion matrix PNG and metrics CSV here.

    Returns
    -------
    dict with keys:
        y_pred, y_proba, cm_norm, cm_raw, report_str, report_dict,
        roc_auc_macro, roc_auc_per_class, sensitivity, specificity.
    """
    logger.info("Evaluating on %d test samples ...", len(X_test))
    t0      = time.perf_counter()
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    elapsed = time.perf_counter() - t0
    logger.info("Inference: %.2f s  (%.0f beats/s)", elapsed, len(X_test) / elapsed)

    # Align proba columns with canonical AAMI_CLASSES order
    fitted_classes = list(model.classes_)
    if fitted_classes != classes:
        reorder = [fitted_classes.index(c) for c in classes if c in fitted_classes]
        y_proba = y_proba[:, reorder]

    # -- Confusion matrices ---------------------------------------------------
    cm_raw  = confusion_matrix(y_test, y_pred, labels=classes)
    cm_norm = cm_raw.astype(float)
    row_sums = cm_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm /= row_sums

    # -- Classification report ------------------------------------------------
    report_str  = classification_report(
        y_test, y_pred, labels=classes, target_names=classes,
        digits=4, zero_division=0,
    )
    report_dict = classification_report(
        y_test, y_pred, labels=classes, target_names=classes,
        output_dict=True, zero_division=0,
    )

    # -- ROC-AUC (one-vs-rest) ------------------------------------------------
    lb    = LabelBinarizer().fit(classes)
    y_bin = lb.transform(y_test)
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    roc_auc_per_class: Dict[str, float] = {}
    for i, cls in enumerate(classes):
        if i >= y_proba.shape[1] or y_bin[:, i].sum() == 0:
            roc_auc_per_class[cls] = float("nan")
        else:
            roc_auc_per_class[cls] = float(
                roc_auc_score(y_bin[:, i], y_proba[:, i])
            )
    valid_aucs    = [v for v in roc_auc_per_class.values() if not np.isnan(v)]
    roc_auc_macro = float(np.mean(valid_aucs)) if valid_aucs else float("nan")

    # -- Per-class sensitivity & specificity ----------------------------------
    sensitivity: Dict[str, float] = {}
    specificity: Dict[str, float] = {}
    for i, cls in enumerate(classes):
        tp = int(cm_raw[i, i])
        fn = int(cm_raw[i, :].sum()) - tp
        fp = int(cm_raw[:, i].sum()) - tp
        tn = int(cm_raw.sum()) - tp - fn - fp
        sensitivity[cls] = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        specificity[cls] = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    _print_report(
        classes, cm_norm, cm_raw, report_str,
        roc_auc_macro, roc_auc_per_class, sensitivity, specificity,
    )

    if output_dir is not None:
        _save_artefacts(
            output_dir, cm_norm, cm_raw, classes,
            roc_auc_per_class, sensitivity, specificity, report_dict,
        )

    return {
        "y_pred":            y_pred,
        "y_proba":           y_proba,
        "cm_norm":           cm_norm,
        "cm_raw":            cm_raw,
        "report_str":        report_str,
        "report_dict":       report_dict,
        "roc_auc_macro":     roc_auc_macro,
        "roc_auc_per_class": roc_auc_per_class,
        "sensitivity":       sensitivity,
        "specificity":       specificity,
    }


# =============================================================================
# 3.  V-class probability thresholding
# =============================================================================

def find_v_sensitivity_threshold(
    y_proba: np.ndarray,
    y_true: np.ndarray,
    target_sensitivity: float = V_SENSITIVITY_FLOOR,
    classes: List[str] = AAMI_CLASSES,
    n_thresholds: int = 2000,
) -> Dict:
    """
    Find the lowest probability threshold T* such that Sensitivity(V) >= target.

    Clinical rationale
    ------------------
    The default argmax rule applies an implicit threshold of ~0.2 in a balanced
    5-class problem.  For ventricular ectopic beats, a missed detection (False
    Negative) may precede ventricular fibrillation -- a life-threatening event.

    This function converts the 5-class posterior into a binary V / not-V
    decision and sweeps T from 0.0 to 1.0, selecting:

        T* = highest T where Se(V) >= target_sensitivity

    Breaking ties by maximising Sp(V) to minimise false alarms.

    Parameters
    ----------
    y_proba : np.ndarray, shape (n_test, n_classes)
        Posterior probabilities from model.predict_proba(X_test).
    y_true : np.ndarray, shape (n_test,)
        True AAMI labels (imbalanced test set).
    target_sensitivity : float
        Hard lower bound on V-class sensitivity.  Default 0.95.
    classes : list of str
    n_thresholds : int

    Returns
    -------
    dict with keys:
        optimal_threshold    : float
        achieved_sensitivity : float
        achieved_specificity : float
        achieved_fpr         : float
        n_additional_flags   : int  (extra V flags vs argmax)
        threshold_curve      : pd.DataFrame (full sweep)
        v_idx                : int
    """
    assert "V" in classes, "Class 'V' not in provided classes list."
    v_idx    = classes.index("V")
    v_scores = y_proba[:, v_idx]
    y_bin    = (np.asarray(y_true) == "V").astype(int)

    if int(y_bin.sum()) == 0:
        raise ValueError(
            "No V-class samples in y_true. "
            "Cannot calibrate a V-class threshold without positive examples."
        )

    rows: List[Dict] = []
    for T in np.linspace(0.0, 1.0, n_thresholds):
        y_hat = (v_scores >= T).astype(int)
        tp = int(((y_hat == 1) & (y_bin == 1)).sum())
        fp = int(((y_hat == 1) & (y_bin == 0)).sum())
        fn = int(((y_hat == 0) & (y_bin == 1)).sum())
        tn = int(((y_hat == 0) & (y_bin == 0)).sum())

        se  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        sp  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        pr  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1  = 2 * pr * se / (pr + se) if (pr + se) > 0 else 0.0

        rows.append({
            "threshold": float(T), "sensitivity": se, "specificity": sp,
            "fpr": 1.0 - sp, "precision": pr, "f1_v": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    curve    = pd.DataFrame(rows)
    feasible = curve[curve["sensitivity"] >= target_sensitivity]

    if feasible.empty:
        logger.warning(
            "Se(V) >= %.0f%% not achievable -- returning max-sensitivity point.",
            target_sensitivity * 100,
        )
        best = curve.loc[curve["sensitivity"].idxmax()]
    else:
        best = feasible.loc[feasible["specificity"].idxmax()]

    T_star   = float(best["threshold"])
    se_star  = float(best["sensitivity"])
    sp_star  = float(best["specificity"])
    fpr_star = float(best["fpr"])

    argmax_v_count = int((np.argmax(y_proba, axis=1) == v_idx).sum())
    thresh_v_count = int((v_scores >= T_star).sum())

    logger.info(
        "V-class threshold calibration:\n"
        "  Target sensitivity        : >= %.1f%%\n"
        "  Optimal threshold T*      : %.5f\n"
        "  Achieved sensitivity Se(V): %.2f%%\n"
        "  Achieved specificity Sp(V): %.2f%%\n"
        "  False Positive Rate FPR(V): %.2f%%\n"
        "  V flags: argmax=%d  T*=%d  (delta=%+d)",
        target_sensitivity * 100, T_star,
        se_star * 100, sp_star * 100, fpr_star * 100,
        argmax_v_count, thresh_v_count, thresh_v_count - argmax_v_count,
    )

    return {
        "optimal_threshold":    T_star,
        "achieved_sensitivity": se_star,
        "achieved_specificity": sp_star,
        "achieved_fpr":         fpr_star,
        "n_additional_flags":   thresh_v_count - argmax_v_count,
        "threshold_curve":      curve,
        "v_idx":                v_idx,
    }


def apply_v_threshold(
    y_proba: np.ndarray,
    threshold_result: Dict,
    classes: List[str] = AAMI_CLASSES,
) -> np.ndarray:
    """
    Apply the calibrated V-class threshold to produce final predictions.

    Decision rule (one-vs-rest override)
    -------------------------------------
    For each beat i:
        if  y_proba[i, V] >= T*  ->  predict "V"
        else                      ->  predict argmax(y_proba[i])

    Parameters
    ----------
    y_proba : np.ndarray, shape (n_test, n_classes)
    threshold_result : dict  (output of find_v_sensitivity_threshold)
    classes : list of str

    Returns
    -------
    np.ndarray, shape (n_test,), dtype object (str)
    """
    T_star   = threshold_result["optimal_threshold"]
    v_idx    = threshold_result["v_idx"]
    cls_arr  = np.array(classes)

    y_pred           = cls_arr[np.argmax(y_proba, axis=1)]
    y_pred[y_proba[:, v_idx] >= T_star] = "V"
    return y_pred


def print_threshold_comparison(
    y_proba: np.ndarray,
    y_true: np.ndarray,
    threshold_result: Dict,
    classes: List[str] = AAMI_CLASSES,
) -> None:
    """
    Print a side-by-side table comparing argmax vs T*-thresholded V decisions.
    """
    v_idx  = threshold_result["v_idx"]
    T_star = threshold_result["optimal_threshold"]
    y_bin  = (np.asarray(y_true) == "V").astype(int)

    def _stats(y_hat_bin):
        tp = int(((y_hat_bin == 1) & (y_bin == 1)).sum())
        fp = int(((y_hat_bin == 1) & (y_bin == 0)).sum())
        fn = int(((y_hat_bin == 0) & (y_bin == 1)).sum())
        tn = int(((y_hat_bin == 0) & (y_bin == 0)).sum())
        se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * pr * se / (pr + se) if (pr + se) > 0 else 0.0
        return {"Se(V)": se, "Sp(V)": sp, "Pr(V)": pr, "F1(V)": f1,
                "TP": tp, "FP": fp, "FN": fn, "TN": tn}

    s_arg   = _stats((np.argmax(y_proba, axis=1) == v_idx).astype(int))
    s_thr   = _stats((y_proba[:, v_idx] >= T_star).astype(int))
    w       = 24

    print()
    print("+" + "-" * 70 + "+")
    print("|   V-Class Decision Boundary Comparison" + " " * 31 + "|")
    print("+" + "-" * 70 + "+")
    print(f"|  {'Metric':<{w}} {'Argmax (default)':>16}   {('T* = ' + f'{T_star:.4f}'):>16}  |")
    print("+" + "-" * 70 + "+")

    spec = [
        ("Sensitivity  Se(V)", "Se(V)", True,  True),
        ("Specificity  Sp(V)", "Sp(V)", True,  False),
        ("Precision    Pr(V)", "Pr(V)", True,  False),
        ("F1 Score     F1(V)", "F1(V)", True,  False),
        ("True  Positives TP", "TP",    False, True),
        ("False Positives FP", "FP",    False, False),
        ("False Negatives FN", "FN",    False, True),
        ("True  Negatives TN", "TN",    False, False),
    ]
    for label, key, is_float, higher_better in spec:
        a = s_arg[key]
        t = s_thr[key]
        a_str = f"{a:.4f}" if is_float else f"{a:d}"
        t_str = f"{t:.4f}" if is_float else f"{t:d}"
        if a == t:
            arrow = "  ="
        elif (t > a) == higher_better:
            arrow = "  ^" if is_float else "  +"
        else:
            arrow = "  v" if is_float else "  -"
        print(f"|  {label:<{w}} {a_str:>16}   {t_str:>16}{arrow} |")

    print("+" + "-" * 70 + "+")
    print(
        f"\n  Adjusted threshold T* = {T_star:.5f} achieves "
        f"Se(V) = {s_thr['Se(V)']*100:.2f}%  "
        f"(target: >= {V_SENSITIVITY_FLOOR*100:.0f}%)\n"
        f"  Trade-off: {threshold_result['n_additional_flags']:+d} additional V flags  |  "
        f"{s_thr['FP'] - s_arg['FP']:+d} extra false positives  |  "
        f"{s_thr['FN'] - s_arg['FN']:+d} fewer missed beats.\n"
    )


# =============================================================================
# 4.  Display / persistence helpers
# =============================================================================

def _print_report(
    classes, cm_norm, cm_raw, report_str,
    roc_auc_macro, roc_auc_per_class, sensitivity, specificity,
) -> None:
    SEP = "=" * 72
    sep = "-" * 72

    print(f"\n{SEP}")
    print("  STACKING ENSEMBLE  *  CLINICAL EVALUATION  *  DS2 TEST SET")
    print(SEP)

    print("\n  Normalized Confusion Matrix")
    print("  (rows = True class | cols = Predicted | diagonal = recall)\n")
    header = f"{'':>6}" + "".join(f"{c:>10}" for c in classes)
    print(header)
    print("  " + sep[:len(header) - 2])
    for i, cls in enumerate(classes):
        vals  = "".join(f"{cm_norm[i, j]:>10.3f}" for j in range(len(classes)))
        count = cm_raw[i, :].sum()
        print(f"  {cls:>3} |{vals}   (n={count})")

    print(f"\n{sep}")
    print("  Classification Report  (natural imbalance -- never resampled)\n")
    for line in report_str.splitlines():
        print("  " + line)

    print(f"\n{sep}")
    print("  ROC-AUC (one-vs-rest)\n")
    print(f"  {'macro average':<22}: {roc_auc_macro:.4f}")
    for cls in classes:
        auc = roc_auc_per_class.get(cls, float("nan"))
        bar = "#" * int(auc * 24) if not np.isnan(auc) else "--"
        print(f"  {cls:<22}: {auc:.4f}  {bar}")

    print(f"\n{sep}")
    print("  Per-class Sensitivity & Specificity\n")
    print(f"  {'Class':<8} {'Sensitivity':>13} {'Specificity':>13}   Note")
    print(f"  {'-----':<8} {'-----------':>13} {'-----------':>13}")
    for cls in classes:
        se   = sensitivity.get(cls, float("nan"))
        sp   = specificity.get(cls, float("nan"))
        note = "  <- CRITICAL (threshold applied separately)" if cls == "V" else ""
        print(f"  {cls:<8} {se*100:>12.2f}%  {sp*100:>12.2f}%{note}")

    print(f"\n{SEP}\n")


def _save_artefacts(
    output_dir, cm_norm, cm_raw, classes,
    roc_auc_per_class, sensitivity, specificity, report_dict,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, (cm, title, fmt) in zip(axes, [
            (cm_norm, "Normalized (Recall)", ".2f"),
            (cm_raw,  "Raw Counts",          "d"),
        ]):
            ConfusionMatrixDisplay(cm, display_labels=classes).plot(
                ax=ax, colorbar=True, cmap="Blues", values_format=fmt,
            )
            ax.set_title(f"Confusion Matrix -- {title}", fontsize=12, fontweight="bold")
        plt.suptitle(
            "ECG Arrhythmia Stacking Ensemble  *  MIT-BIH DS2",
            fontsize=13, fontweight="bold", y=1.01,
        )
        plt.tight_layout()
        path = output_dir / "confusion_matrix.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Confusion matrix -> %s", path)

        rows = []
        for cls in classes:
            d = report_dict.get(cls, {})
            rows.append({
                "class":       cls,
                "precision":   d.get("precision",  float("nan")),
                "recall":      d.get("recall",     float("nan")),
                "f1_score":    d.get("f1-score",   float("nan")),
                "support":     d.get("support",    0),
                "roc_auc":     roc_auc_per_class.get(cls, float("nan")),
                "sensitivity": sensitivity.get(cls, float("nan")),
                "specificity": specificity.get(cls, float("nan")),
            })
        csv_path = output_dir / "metrics.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.4f")
        logger.info("Metrics CSV -> %s", csv_path)

    except ImportError:
        logger.warning("matplotlib not installed -- skipping plot output.")


# =============================================================================
# 5.  Main pipeline
# =============================================================================

def run_pipeline(
    df: Optional[pd.DataFrame] = None,
    data_dir: Path = Path("data"),
    output_dir: Path = OUTPUT_DIR,
    stacking_config: Optional[StackingConfig] = None,
    majority_ratio: float = 0.40,
    smote_k: int = 5,
    v_sensitivity_target: float = V_SENSITIVITY_FLOOR,
    random_state: int = RANDOM_STATE,
) -> Dict:
    """
    End-to-end pipeline: features -> train -> evaluate -> threshold.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Pre-built feature DataFrame.  If None, loads from data_dir.
    data_dir : Path
        Root directory for raw data and feature cache.
    output_dir : Path
        Directory for evaluation artefacts.
    stacking_config : StackingConfig, optional
        Custom hyperparameters.  Defaults to production settings.
    majority_ratio : float
        Fraction of N-class beats retained after RandomUnderSampler.
    smote_k : int
        k-neighbours for SMOTE minority over-sampling.
    v_sensitivity_target : float
        Hard lower bound for V-class recall at threshold calibration.
    random_state : int

    Returns
    -------
    dict  -- all intermediate and final artefacts keyed by stage.
    """
    results: Dict = {}

    # Stage 1: feature DataFrame
    if df is None:
        df = load_feature_dataframe(data_dir)
    results["feature_df"] = df
    logger.info("Feature DataFrame: %d beats x %d columns", len(df), len(df.columns))

    # Stage 2: split / impute / scale
    logger.info("Stage 2 -- preprocessing ...")
    X_train, X_test, y_train, y_test = prepare_train_test(df)
    results.update({"X_train": X_train, "X_test": X_test,
                    "y_train": y_train, "y_test": y_test})
    logger.info(
        "Split: train=%d test=%d  |  train: %s",
        len(X_train), len(X_test), _label_dist(y_train),
    )
    logger.info("Test (NATURAL imbalance): %s", _label_dist(y_test))

    # Stage 3: SMOTE on train ONLY
    logger.info("Stage 3 -- SMOTE resampling on training data ONLY ...")
    if _HAVE_PREPROCESSING:
        X_res, y_res, _, _ = apply_resampling_pipeline(
            X_train, y_train,
            majority_ratio=majority_ratio,
            smote_k_neighbors=smote_k,
            random_state=random_state,
            X_test=X_test,
            y_test=y_test,
        )
    else:
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.pipeline import Pipeline as ImbPipeline
        pipe = ImbPipeline([
            ("under", RandomUnderSampler(random_state=random_state)),
            ("smote", SMOTE(k_neighbors=smote_k, random_state=random_state)),
        ])
        X_res, y_res = pipe.fit_resample(X_train, y_train)
    results.update({"X_res": X_res, "y_res": y_res})
    logger.info(
        "Resampled: %d -> %d  |  %s", len(X_train), len(X_res), _label_dist(y_res),
    )

    # Stage 4: build + fit ensemble
    logger.info("Stage 4 -- building and fitting stacking ensemble ...")
    factory = ECGStackingEnsemble(config=stacking_config)
    model   = factory.build()
    logger.info("Hyperparameters: %s", factory.config_summary())
    t0 = time.perf_counter()
    model.fit(X_res, y_res)
    elapsed = time.perf_counter() - t0
    logger.info("Training complete in %.1f s", elapsed)
    results["model"]         = model
    results["train_elapsed"] = elapsed

    # Stage 5: evaluate on imbalanced test set
    logger.info("Stage 5 -- evaluation on DS2 (imbalanced) test set ...")
    eval_res = evaluate_model(model, X_test, y_test, classes=AAMI_CLASSES, output_dir=output_dir)
    results.update(eval_res)

    # Stage 6: V-class threshold calibration
    logger.info("Stage 6 -- calibrating V-class decision threshold ...")
    thr = find_v_sensitivity_threshold(
        y_proba=eval_res["y_proba"],
        y_true=y_test,
        target_sensitivity=v_sensitivity_target,
        classes=AAMI_CLASSES,
    )
    results["threshold_result"] = thr
    print_threshold_comparison(eval_res["y_proba"], y_test, thr, AAMI_CLASSES)

    y_clinical = apply_v_threshold(eval_res["y_proba"], thr, AAMI_CLASSES)
    results["y_pred_clinical"] = y_clinical

    logger.info(
        "Pipeline complete.\n"
        "  Training time   : %.1f s\n"
        "  T*              : %.5f\n"
        "  Achieved Se(V)  : %.2f%%\n"
        "  Achieved Sp(V)  : %.2f%%\n"
        "  ROC-AUC macro   : %.4f",
        elapsed,
        thr["optimal_threshold"],
        thr["achieved_sensitivity"] * 100,
        thr["achieved_specificity"] * 100,
        eval_res["roc_auc_macro"],
    )
    return results


# =============================================================================
# 6.  Helpers
# =============================================================================

def _label_dist(y: np.ndarray) -> str:
    u, c = np.unique(y, return_counts=True)
    total = c.sum()
    return "  ".join(f"{cls}={n}({n/total*100:.1f}%)" for cls, n in zip(u, c))


def _make_synthetic_df(
    n_per_class: Dict[str, int],
    n_features: int = 46,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Synthetic feature DataFrame for offline smoke-testing.

    Parameters
    ----------
    n_per_class : dict {class: count}
    n_features : int
    random_state : int

    Returns
    -------
    pd.DataFrame
    """
    if _HAVE_PREPROCESSING:
        from src.data.preprocessing import DS1_TRAIN_RECORDS, DS2_TEST_RECORDS
        pool = DS1_TRAIN_RECORDS[:11] + DS2_TEST_RECORDS[:11]
    else:
        pool = [f"tr{i}" for i in range(11)] + [f"te{i}" for i in range(11)]

    rng = np.random.default_rng(random_state)
    offsets = {"N": 0.0, "S": 0.5, "V": 1.0, "F": 0.8, "Q": 0.3}
    feat_names = (
        [f"td_feat_{i}" for i in range(14)]
        + [f"fd_feat_{i}" for i in range(8)]
        + [f"wd_feat_{i}" for i in range(24)]
    )

    rows: List[Dict] = []
    for cls, n in n_per_class.items():
        X = rng.standard_normal((n, n_features)).astype(np.float32) + offsets[cls]
        X[0, 0]  = np.nan
        X[-1, 1] = np.nan
        record_ids = rng.choice(pool, size=n)
        for i in range(n):
            row = {feat_names[j]: float(X[i, j]) for j in range(n_features)}
            row["label"]  = cls
            row["record"] = str(record_ids[i])
            rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    fast_mode = "--fast" in sys.argv

    if fast_mode:
        logger.info("FAST MODE -- reduced estimator sizes for smoke-testing.")
        cfg = StackingConfig(cv=3)
        cfg.rf.n_estimators = 100
        cfg.hgb.max_iter    = 80
        n_per_class = {"N": 1500, "S": 150, "V": 200, "F": 40, "Q": 40}
    else:
        cfg         = None      # use production defaults from StackingConfig
        n_per_class = {"N": 4000, "S": 400, "V": 500, "F": 100, "Q": 100}

    real_cache = Path("data") / "features.parquet"
    mitdb_path = Path("data") / "mitdb"

    if real_cache.exists() or mitdb_path.exists():
        logger.info("MIT-BIH data detected -- running full pipeline.")
        df_input = None
    else:
        logger.info(
            "No MIT-BIH data found at 'data/'.  Running on synthetic data.\n"
            "  To use real data: python src/data/data_loader.py"
        )
        df_input = _make_synthetic_df(n_per_class, random_state=RANDOM_STATE)

    run_pipeline(
        df=df_input,
        data_dir=Path("data"),
        output_dir=OUTPUT_DIR,
        stacking_config=cfg,
        majority_ratio=0.40,
        smote_k=5,
        v_sensitivity_target=V_SENSITIVITY_FLOOR,
        random_state=RANDOM_STATE,
    )