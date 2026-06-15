"""
src/features/extraction.py
===========================
Multi-domain feature engineering for ECG beat classification.

Feature taxonomy
----------------
Three complementary domains capture the distinct morphological and rhythmic
signatures of each AAMI superclass:

┌─────────────────────────────────────────────────────────────────────────────┐
│ Domain       │ Features                       │ Clinical rationale          │
├─────────────────────────────────────────────────────────────────────────────┤
│ Time         │ Pre/Post RR intervals,          │ Ectopic beats (V, S) have  │
│              │ RR ratio, local heart rate,     │ abnormal coupling intervals │
│              │ QRS amplitude, QRS duration,    │ and wider / taller QRS.    │
│              │ beat morphology statistics      │                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ Frequency    │ LF band power (0.04–0.15 Hz),   │ LF/HF reflects autonomic   │
│              │ HF band power (0.15–0.40 Hz),   │ balance; V beats shift      │
│              │ VLF power, LF/HF ratio,         │ spectral energy into HF.   │
│              │ dominant frequency, total power │                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ Wavelet      │ db4 DWT at 5 levels:            │ QRS sharpness lives in D1  │
│  (db4, L=5) │ energy per sub-band,            │ (high freq); T/P wave      │
│              │ relative energy ratio,          │ content in A5 (low freq).  │
│              │ coefficient statistics          │ V beats have high D1 energy.│
└─────────────────────────────────────────────────────────────────────────────┘

Wavelet sub-band → frequency mapping (fs = 360 Hz)
---------------------------------------------------
Level  Sub-band   Approx. freq range     Clinical content
  D1   cD1        90–180 Hz              High-freq noise / fine QRS detail
  D2   cD2        45–90  Hz              QRS complex high-frequency notches
  D3   cD3        22.5–45 Hz             QRS main energy
  D4   cD4        11.25–22.5 Hz          QRS + ST segment
  D5   cD5        5.6–11.25 Hz           ST segment, early T-wave
  A5   cA5        0–5.6 Hz               Baseline, P-wave, T-wave

CPU optimisation notes
----------------------
• All NumPy operations are vectorised; no Python-level loops over samples.
• PyWavelets `wavedec` uses an optimised C backend with SIMD on x86.
• ``extract_batch`` pre-allocates the output DataFrame via a list-of-dicts
  pattern — avoids repeated ``pd.concat`` which copies the entire frame each
  call.
• ``scipy.signal.welch`` uses FFTPACK / MKL-linked FFT when NumPy is built
  against MKL (standard in the Intel distribution and conda-forge).
• Float32 inputs are upcast to float64 only inside signal.welch (required for
  PSD accuracy); all intermediate arrays stay float32.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pywt
from scipy.signal import welch
from scipy.stats import kurtosis, skew

logger = logging.getLogger(__name__)

# ── Frequency band definitions (Hz) ──────────────────────────────────────────
# Standard Task-Force HRV frequency bands (Task Force of ESC/NASPE, 1996).
# Applied here to single-beat PSDs as morphological descriptors rather than
# true HRV bands — they capture how QRS energy is distributed spectrally.
VLF_BAND: Tuple[float, float] = (0.003, 0.04)
LF_BAND:  Tuple[float, float] = (0.04,  0.15)
HF_BAND:  Tuple[float, float] = (0.15,  0.40)

# Wavelet configuration
WAVELET:        str = "db4"
DWT_LEVELS:     int = 5
DWT_LEVEL_NAMES: List[str] = [f"cD{i}" for i in range(1, DWT_LEVELS + 1)] + ["cA5"]


class FeatureExtractor:
    """
    Extract time-domain, frequency-domain, and wavelet-domain features
    from a single segmented ECG beat window.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.  MIT-BIH = 360 Hz.
    pre_peak_samples : int
        Number of samples before the R-peak in the beat window.
        Must match the value used in ``PanTompkinsDetector`` (default 108
        samples = 300 ms at 360 Hz).
    welch_nperseg : int | None
        Segment length for Welch PSD.  ``None`` → use the full beat window
        as a single segment (equivalent to a periodogram).  For the 288-sample
        default window, ``nperseg=64`` gives 5 frequency bins per Hz.
    qrs_search_ms : float
        Half-width of the search window around the R-peak used to compute
        QRS amplitude and duration (ms).  Default 50 ms (±18 samples @ 360 Hz).
    normalize_beats : bool
        If True, z-score each beat window before feature extraction.
        Removes amplitude scale differences between leads / patients.
        Default True.

    Examples
    --------
    Single beat
    >>> extractor = FeatureExtractor(fs=360, pre_peak_samples=108)
    >>> features = extractor.extract(beat_window, pre_rr=0.83, post_rr=0.80,
    ...                              label="N")

    Full dataset
    >>> df = extractor.extract_batch(beats, rr_intervals, r_peak_indices, labels)
    """

    def __init__(
        self,
        fs: float = 360.0,
        pre_peak_samples: int = 108,          # 300 ms @ 360 Hz
        welch_nperseg: Optional[int] = 64,
        qrs_search_ms: float = 50.0,
        normalize_beats: bool = True,
    ) -> None:
        self.fs = float(fs)
        self.pre_peak_samples = pre_peak_samples
        self.welch_nperseg = welch_nperseg
        self.qrs_search_ms = qrs_search_ms
        self.normalize_beats = normalize_beats

        # Pre-compute QRS search half-width in samples
        self._qrs_half: int = max(1, int(round(qrs_search_ms * fs / 1000.0)))

        # Validate wavelet decomposition level against a typical window
        max_level = pywt.dwt_max_level(
            data_len=288,      # default window length (108 + 180)
            filter_len=pywt.Wavelet(WAVELET).dec_len,
        )
        if DWT_LEVELS > max_level:
            raise ValueError(
                f"DWT_LEVELS={DWT_LEVELS} exceeds pywt max level {max_level} "
                f"for wavelet '{WAVELET}' and window length 288."
            )

        logger.info(
            "FeatureExtractor ready | fs=%.1f Hz | pre_peak=%d samp | "
            "qrs_search=±%d samp | wavelet=%s L=%d | normalize=%s",
            self.fs, self.pre_peak_samples, self._qrs_half,
            WAVELET, DWT_LEVELS, self.normalize_beats,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # Public API
    # ═════════════════════════════════════════════════════════════════════════

    def extract(
        self,
        beat: np.ndarray,
        pre_rr: Optional[float] = None,
        post_rr: Optional[float] = None,
        label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract all features from a single beat window.

        Parameters
        ----------
        beat : np.ndarray, shape (window_len,)
            Segmented, filtered beat window centred on the R-peak.
            The R-peak is located at index ``pre_peak_samples``.
        pre_rr : float | None
            RR interval *before* this beat in seconds.  None → NaN feature.
        post_rr : float | None
            RR interval *after* this beat in seconds.  None → NaN feature.
        label : str | None
            AAMI superclass label (N / S / V / F / Q).  Appended as-is.

        Returns
        -------
        dict
            Flat dictionary of feature_name → scalar value.
            Key groups (prefix):
            • ``td_``   — time domain
            • ``fd_``   — frequency domain (Welch PSD)
            • ``wd_``   — wavelet domain (db4 DWT)
            • ``label`` — AAMI class label (str or NaN)
        """
        beat = np.asarray(beat, dtype=np.float64)

        if self.normalize_beats:
            beat = self._zscore(beat)

        features: Dict[str, Any] = {}
        features.update(self._time_domain(beat, pre_rr, post_rr))
        features.update(self._frequency_domain(beat))
        features.update(self._wavelet_domain(beat))
        features["label"] = label if label is not None else float("nan")

        return features

    def extract_batch(
        self,
        beats: np.ndarray,
        rr_intervals: np.ndarray,
        r_peak_indices: np.ndarray,
        labels: Optional[List[Optional[str]]] = None,
    ) -> pd.DataFrame:
        """
        Extract features for an array of beats and return a tidy DataFrame.

        RR interval derivation
        ----------------------
        ``rr_intervals`` is expected to be the **sample-to-sample** RR array
        produced by ``PanTompkinsDetector.compute_rr_intervals`` (shape
        ``(M-1,)`` where M = number of detected peaks).

        ``r_peak_indices`` is the array of accepted R-peak positions
        (shape ``(M,)``).  Beat *i* corresponds to peak *i*:

            pre_rr[i]  = rr_intervals[i-1]   (NaN for i=0)
            post_rr[i] = rr_intervals[i]     (NaN for i=M-1)

        Parameters
        ----------
        beats : np.ndarray, shape (M, window_len)
            Matrix of segmented beat windows.
        rr_intervals : np.ndarray, shape (M-1,)
            RR intervals in **milliseconds** (from
            ``PanTompkinsDetector.compute_rr_intervals``).
        r_peak_indices : np.ndarray, shape (M,)
            Sample indices of R-peaks (used only to derive pre/post RR;
            ``rr_intervals`` already encodes the intervals).
        labels : list of str | None, length M
            AAMI superclass labels.  Pass None for unlabelled (inference) data.

        Returns
        -------
        pd.DataFrame
            Shape ``(M, n_features + 1)``.  The ``label`` column is last.
            Index is reset (0 … M-1).
        """
        n_beats = len(beats)
        if labels is None:
            labels = [None] * n_beats

        # Convert RR from ms → seconds for feature storage
        rr_sec = rr_intervals / 1000.0 if len(rr_intervals) > 0 else np.array([])

        rows: List[Dict[str, Any]] = []

        for i in range(n_beats):
            # Pre-RR: interval ending at this beat (index i-1 in rr array)
            pre_rr = float(rr_sec[i - 1]) if i > 0 and len(rr_sec) >= i else None

            # Post-RR: interval starting at this beat (index i in rr array)
            post_rr = float(rr_sec[i]) if i < len(rr_sec) else None

            row = self.extract(
                beat=beats[i],
                pre_rr=pre_rr,
                post_rr=post_rr,
                label=labels[i],
            )
            rows.append(row)

        df = pd.DataFrame(rows)

        # ── Column ordering: td_ → fd_ → wd_ → label ─────────────────────────
        td_cols = sorted(c for c in df.columns if c.startswith("td_"))
        fd_cols = sorted(c for c in df.columns if c.startswith("fd_"))
        wd_cols = sorted(c for c in df.columns if c.startswith("wd_"))
        feat_cols = td_cols + fd_cols + wd_cols
        df = df[feat_cols + ["label"]]

        # ── Step 1: fill NaNs in numerical feature columns ONLY ───────────────
        #
        # scipy.signal.welch can return NaN for frequency-domain features when
        # the beat window is shorter than nperseg (e.g. the 288-sample default
        # window with nperseg=64 is fine, but edge beats after resampling can
        # be shorter).  Similarly, td_pre_rr / td_post_rr are legitimately NaN
        # for the first and last beat of every record.
        #
        # We zero-fill ONLY the typed numeric columns identified by their
        # td_ / fd_ / wd_ prefix.  Applying fillna to the whole DataFrame would
        # coerce NaN labels (Python float('nan')) to the integer 0, producing a
        # mixed-type object column that pyarrow rejects with:
        #   ArrowTypeError: Expected bytes, got a 'int' object
        df[feat_cols] = df[feat_cols].fillna(0.0)

        # ── Step 2: drop beats with no valid clinical annotation ───────────────
        #
        # Beats that could not be matched to an MIT-BIH annotation — or whose
        # annotation symbol mapped to None in MITBIH_TO_AAMI (rhythm markers,
        # waveform-onset labels, etc.) — arrive here with label=None / NaN.
        # These are not classifiable beats and must be excluded before training.
        # Keeping them would introduce rows with a null target into the feature
        # matrix, causing downstream errors in sklearn estimators and corrupting
        # class-distribution counts.
        n_before = len(df)
        df = df.dropna(subset=["label"]).reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped:
            logger.warning(
                "extract_batch: dropped %d beat(s) with no valid label "
                "(non-beat MIT-BIH annotations or unmatched R-peaks).",
                n_dropped,
            )

        # ── Step 3: guarantee label column is a clean str dtype ───────────────
        #
        # After dropna the label column still has dtype=object (Python mixed
        # container).  Pyarrow's Parquet writer maps object columns by
        # inspecting the runtime type of each element; if *any* element is not
        # bytes/str it raises ArrowTypeError.  Explicit astype(str) forces every
        # cell to a Python str, giving pyarrow a uniform large_string column it
        # can serialise without inspection.
        #
        # This also handles the edge case where a caller passed integer class
        # codes (0, 1, 2 …) instead of AAMI strings — they become '0', '1', …
        # which will surface as a downstream mismatch rather than a silent crash.
        df["label"] = df["label"].astype(str)

        logger.info(
            "extract_batch complete | beats=%d | dropped=%d | features=%d "
            "| label_dist=%s",
            len(df),
            n_dropped,
            len(feat_cols),
            df["label"].value_counts().to_dict(),
        )

        return df

    # ═════════════════════════════════════════════════════════════════════════
    # Time-domain features
    # ═════════════════════════════════════════════════════════════════════════

    def _time_domain(
        self,
        beat: np.ndarray,
        pre_rr: Optional[float],
        post_rr: Optional[float],
    ) -> Dict[str, float]:
        """
        Compute 14 time-domain features.

        RR interval features
        --------------------
        ``td_pre_rr``    — preceding RR interval (s); NaN at recording start.
        ``td_post_rr``   — following RR interval (s); NaN at recording end.
        ``td_rr_ratio``  — pre_rr / post_rr; detects compensatory pauses (V beats).
        ``td_local_hr``  — instantaneous HR = 60 / mean(pre_rr, post_rr) (bpm).
        ``td_rr_diff``   — post_rr − pre_rr; premature vs. delayed coupling.

        QRS morphology features
        -----------------------
        ``td_qrs_amplitude`` — peak-to-peak amplitude in ±qrs_search window (mV).
        ``td_r_peak_value``  — raw signal value at the R-peak index.
        ``td_qrs_duration``  — estimated QRS duration (s): width above 50% of
                               R-peak amplitude within the search window.
        ``td_qrs_area``      — absolute area under QRS (integral of |signal|).

        Global beat statistics
        ----------------------
        ``td_beat_mean``     — mean of the full beat window.
        ``td_beat_std``      — standard deviation.
        ``td_beat_skewness`` — skewness (asymmetry; + → right-tailed).
        ``td_beat_kurtosis`` — excess kurtosis (peakedness).
        ``td_beat_energy``   — sum of squared samples (signal energy proxy).
        """
        r_idx = self.pre_peak_samples   # R-peak is at this fixed index
        n = len(beat)

        # ── RR interval features ──────────────────────────────────────────────
        pre_rr_val  = pre_rr  if pre_rr  is not None else float("nan")
        post_rr_val = post_rr if post_rr is not None else float("nan")

        valid_rr = [v for v in (pre_rr_val, post_rr_val) if not np.isnan(v)]
        rr_ratio  = pre_rr_val / post_rr_val if (pre_rr is not None and post_rr is not None and post_rr_val != 0) else float("nan")
        local_hr  = 60.0 / np.mean(valid_rr) if valid_rr else float("nan")
        rr_diff   = post_rr_val - pre_rr_val if (pre_rr is not None and post_rr is not None) else float("nan")

        # ── QRS morphology ────────────────────────────────────────────────────
        half = self._qrs_half
        qrs_lo = max(0, r_idx - half)
        qrs_hi = min(n, r_idx + half)
        qrs_segment = beat[qrs_lo:qrs_hi]

        r_peak_value   = float(beat[r_idx])
        qrs_amplitude  = float(qrs_segment.max() - qrs_segment.min()) if len(qrs_segment) > 0 else float("nan")
        qrs_area       = float(np.trapz(np.abs(qrs_segment))) / self.fs

        # QRS duration: samples where |signal| > 50% of peak absolute value
        if len(qrs_segment) > 0:
            threshold_50 = 0.5 * np.abs(r_peak_value)
            above = np.where(np.abs(qrs_segment) >= threshold_50)[0]
            qrs_duration = float(len(above)) / self.fs if len(above) > 0 else float("nan")
        else:
            qrs_duration = float("nan")

        # ── Global beat statistics ────────────────────────────────────────────
        beat_mean     = float(np.mean(beat))
        beat_std      = float(np.std(beat))
        beat_skew     = float(skew(beat))
        beat_kurt     = float(kurtosis(beat))       # excess kurtosis
        beat_energy   = float(np.sum(beat ** 2))

        return {
            # RR
            "td_pre_rr":        pre_rr_val,
            "td_post_rr":       post_rr_val,
            "td_rr_ratio":      rr_ratio,
            "td_local_hr":      local_hr,
            "td_rr_diff":       rr_diff,
            # QRS
            "td_qrs_amplitude": qrs_amplitude,
            "td_r_peak_value":  r_peak_value,
            "td_qrs_duration":  qrs_duration,
            "td_qrs_area":      qrs_area,
            # Global
            "td_beat_mean":     beat_mean,
            "td_beat_std":      beat_std,
            "td_beat_skewness": beat_skew,
            "td_beat_kurtosis": beat_kurt,
            "td_beat_energy":   beat_energy,
        }

    # ═════════════════════════════════════════════════════════════════════════
    # Frequency-domain features  (Welch PSD)
    # ═════════════════════════════════════════════════════════════════════════

    def _frequency_domain(self, beat: np.ndarray) -> Dict[str, float]:
        """
        Compute 8 frequency-domain features via Welch's method.

        Implementation notes
        --------------------
        ``scipy.signal.welch`` divides the beat into overlapping segments
        (50% overlap, Hann window), computes the periodogram for each, and
        averages.  This trades frequency resolution for variance reduction —
        appropriate for the 288-sample window.

        For a 288-sample beat at 360 Hz:
        • ``nperseg=64``  → frequency resolution Δf = 360/64 ≈ 5.6 Hz/bin
        • The VLF/LF/HF bands contain very few bins at this scale — the values
          are morphological descriptors rather than true HRV PSD estimates.

        Features
        --------
        ``fd_vlf_power``      — VLF band power (0.003–0.04 Hz)
        ``fd_lf_power``       — LF  band power (0.04–0.15 Hz)
        ``fd_hf_power``       — HF  band power (0.15–0.40 Hz)
        ``fd_lf_hf_ratio``    — LF / HF ratio
        ``fd_total_power``    — integral of PSD over [0, fs/2]
        ``fd_dominant_freq``  — frequency of PSD maximum (Hz)
        ``fd_spectral_entropy``— normalised Shannon entropy of PSD (complexity)
        ``fd_mean_freq``      — spectral centroid (Hz)
        """
        # nperseg must not exceed signal length
        nperseg = self.welch_nperseg
        if nperseg is None or nperseg > len(beat):
            nperseg = len(beat)

        freqs, psd = welch(
            beat,
            fs=self.fs,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            scaling="density",
        )

        # Band power integrations via the trapezoidal rule
        def band_power(lo: float, hi: float) -> float:
            mask = (freqs >= lo) & (freqs < hi)
            if mask.sum() < 2:
                return float("nan")
            return float(np.trapz(psd[mask], freqs[mask]))

        vlf_power   = band_power(*VLF_BAND)
        lf_power    = band_power(*LF_BAND)
        hf_power    = band_power(*HF_BAND)
        total_power = float(np.trapz(psd, freqs))

        lf_hf_ratio = (
            lf_power / hf_power
            if (not np.isnan(hf_power) and hf_power > 1e-12)
            else float("nan")
        )

        dominant_freq = float(freqs[np.argmax(psd)])

        # Spectral centroid (mean frequency weighted by PSD)
        psd_sum = psd.sum()
        mean_freq = float(np.sum(freqs * psd) / psd_sum) if psd_sum > 1e-12 else float("nan")

        # Normalised spectral entropy  H = -Σ p·log(p)  where p = psd / Σpsd
        p = psd / (psd_sum + 1e-12)
        spectral_entropy = float(-np.sum(p * np.log(p + 1e-12)) / np.log(len(p) + 1e-12))

        return {
            "fd_vlf_power":        vlf_power,
            "fd_lf_power":         lf_power,
            "fd_hf_power":         hf_power,
            "fd_lf_hf_ratio":      lf_hf_ratio,
            "fd_total_power":      total_power,
            "fd_dominant_freq":    dominant_freq,
            "fd_spectral_entropy": spectral_entropy,
            "fd_mean_freq":        mean_freq,
        }

    # ═════════════════════════════════════════════════════════════════════════
    # Wavelet-domain features  (db4, 5 levels)
    # ═════════════════════════════════════════════════════════════════════════

    def _wavelet_domain(self, beat: np.ndarray) -> Dict[str, float]:
        """
        Compute 24 wavelet-domain features using db4 DWT at 5 levels.

        Decomposition
        -------------
        ``pywt.wavedec`` returns [cA5, cD5, cD4, cD3, cD2, cD1] (coarsest
        to finest).  We re-order to [cD1, cD2, cD3, cD4, cD5, cA5] to align
        with increasing decomposition level for naming clarity.

        Sub-band frequency ranges (fs = 360 Hz)
        ----------------------------------------
        cD1  90–180 Hz   high-frequency noise / sharp QRS edges
        cD2  45–90  Hz   QRS fine structure
        cD3  22.5–45 Hz  QRS main energy (strongest for normal beats)
        cD4  11.25–22.5 Hz  QRS tail / ST onset
        cD5  5.6–11.25 Hz   T-wave onset
        cA5  0–5.6 Hz    baseline / P-wave / T-wave peak

        Per sub-band features (4 features × 6 sub-bands = 24 total)
        -------------------------------------------------------------
        ``wd_{band}_energy``     — Σ c² (Parseval energy in sub-band)
        ``wd_{band}_rel_energy`` — sub-band energy / total DWT energy
        ``wd_{band}_mean_abs``   — mean |coefficient|  (robust amplitude)
        ``wd_{band}_std``        — std of coefficients (intra-band variability)
        """
        coeffs = pywt.wavedec(beat, wavelet=WAVELET, level=DWT_LEVELS)
        # wavedec order: [cA5, cD5, cD4, cD3, cD2, cD1]
        # Reverse so index 0 = cD1 (finest), last = cA5 (coarsest)
        coeffs_ordered = list(reversed(coeffs))  # [cD1, cD2, cD3, cD4, cD5, cA5]

        # Total energy across all sub-bands (denominator for relative energy)
        total_energy = sum(float(np.sum(c ** 2)) for c in coeffs_ordered)
        total_energy = total_energy if total_energy > 1e-12 else 1.0

        features: Dict[str, float] = {}

        for name, c in zip(DWT_LEVEL_NAMES, coeffs_ordered):
            energy     = float(np.sum(c ** 2))
            rel_energy = energy / total_energy
            mean_abs   = float(np.mean(np.abs(c)))
            std        = float(np.std(c))

            features[f"wd_{name}_energy"]     = energy
            features[f"wd_{name}_rel_energy"] = rel_energy
            features[f"wd_{name}_mean_abs"]   = mean_abs
            features[f"wd_{name}_std"]        = std

        return features

    # ═════════════════════════════════════════════════════════════════════════
    # Private helpers
    # ═════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _zscore(beat: np.ndarray) -> np.ndarray:
        """
        Z-score normalise a beat window.

        If the standard deviation is essentially zero (flat-line artefact),
        the beat is returned unchanged to avoid division-by-zero.
        """
        std = beat.std()
        if std < 1e-8:
            return beat
        return (beat - beat.mean()) / std

    # ═════════════════════════════════════════════════════════════════════════
    # Introspection utilities
    # ═════════════════════════════════════════════════════════════════════════

    def feature_names(self) -> List[str]:
        """
        Return the ordered list of feature names produced by ``extract``.

        Useful for constructing column lists before running the full pipeline.
        Excludes the ``label`` column.

        Returns
        -------
        list of str
        """
        # Generate from a dummy zero beat
        dummy = np.zeros(self.pre_peak_samples + 180, dtype=np.float64)
        feats = self.extract(dummy, pre_rr=0.8, post_rr=0.8, label=None)
        return [k for k in feats.keys() if k != "label"]

    def feature_groups(self) -> Dict[str, List[str]]:
        """
        Return feature names grouped by domain prefix.

        Returns
        -------
        dict with keys 'time_domain', 'frequency_domain', 'wavelet_domain'.
        """
        names = self.feature_names()
        return {
            "time_domain":      [n for n in names if n.startswith("td_")],
            "frequency_domain": [n for n in names if n.startswith("fd_")],
            "wavelet_domain":   [n for n in names if n.startswith("wd_")],
        }


# ── Standalone demo ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import textwrap
    from pathlib import Path

    print("=" * 72)
    print("  FeatureExtractor — Phase 3 Demo")
    print("=" * 72)

    FS          = 360.0
    PRE_SAMPLES = 108          # 300 ms @ 360 Hz
    POST_SAMPLES = 180         # 500 ms @ 360 Hz
    WIN_LEN     = PRE_SAMPLES + POST_SAMPLES   # 288 samples

    extractor = FeatureExtractor(
        fs=FS,
        pre_peak_samples=PRE_SAMPLES,
        welch_nperseg=64,
        qrs_search_ms=50.0,
        normalize_beats=True,
    )

    # ── Try loading real MIT-BIH data if available ────────────────────────────
    DATA_DIR = Path("./data/mitdb")
    RECORD    = "100"
    use_real  = (DATA_DIR / f"{RECORD}.dat").exists()

    if use_real:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        import wfdb
        from features.signal_processing import ECGFilterPipeline
        from features.segmentation import PanTompkinsDetector, EdgeStrategy

        print(f"\n[0] Loading MIT-BIH record '{RECORD}' …")
        rec        = wfdb.rdrecord(str(DATA_DIR / RECORD))
        raw_signal = rec.p_signal[:, 0].astype(np.float32)
        ann        = wfdb.rdann(str(DATA_DIR / RECORD), "atr")

        pipeline  = ECGFilterPipeline(fs=FS)
        clean_ecg = pipeline.process_signal(raw_signal)

        detector  = PanTompkinsDetector(
            fs=FS,
            pre_peak_ms=300.0,
            post_peak_ms=500.0,
            edge_strategy=EdgeStrategy.SKIP,
        )
        beats, kept_peaks, all_peaks = detector.detect_and_extract(clean_ecg)
        rr_ms, _ = detector.compute_rr_intervals(all_peaks)

        # Map annotation symbols → AAMI using the Phase 1 mapping
        from data.data_loader import MITBIH_TO_AAMI
        sym_at_peak: Dict = {s: sym for s, sym in zip(ann.sample, ann.symbol)}
        labels = []
        for rp in kept_peaks:
            closest = min(ann.sample, key=lambda s: abs(int(s) - int(rp)))
            sym = sym_at_peak.get(closest, "Q")
            labels.append(MITBIH_TO_AAMI.get(sym, "Q"))

        print(f"    Beats extracted : {beats.shape}")

    else:
        # ── Synthetic dataset: 5 classes × 20 beats ──────────────────────────
        print("\n[0] MIT-BIH data not found — generating synthetic dataset …")
        np.random.seed(42)

        AAMI_CLASSES   = ["N", "S", "V", "F", "Q"]
        BEATS_PER_CLASS = 20
        N_BEATS        = BEATS_PER_CLASS * len(AAMI_CLASSES)

        def make_synthetic_beat(cls: str, rng: np.random.Generator) -> np.ndarray:
            """Crude class-differentiated synthetic beat."""
            t = np.linspace(-PRE_SAMPLES, POST_SAMPLES, WIN_LEN) / FS
            # QRS Gaussian pulse — V beats wider, S beats earlier
            qrs_width = {"N": 0.03, "S": 0.025, "V": 0.06, "F": 0.045, "Q": 0.05}[cls]
            qrs_shift = {"N": 0.0,  "S": -0.01, "V": 0.0,  "F": 0.005, "Q": 0.0}[cls]
            qrs_amp   = {"N": 1.0,  "S": 0.8,   "V": 1.4,  "F": 1.1,   "Q": 0.6}[cls]
            beat = qrs_amp * np.exp(-0.5 * ((t - qrs_shift) / qrs_width) ** 2)
            # T-wave
            beat += 0.3 * np.exp(-0.5 * ((t - 0.25) / 0.06) ** 2)
            # Noise
            beat += 0.05 * rng.standard_normal(WIN_LEN)
            return beat.astype(np.float64)

        rng    = np.random.default_rng(0)
        beats  = np.stack(
            [make_synthetic_beat(cls, rng)
             for cls in AAMI_CLASSES
             for _ in range(BEATS_PER_CLASS)],
            axis=0,
        ).astype(np.float32)
        labels = [cls for cls in AAMI_CLASSES for _ in range(BEATS_PER_CLASS)]

        # Synthetic RR: ~833 ms (72 bpm) with mild jitter
        rr_ms      = 833.0 + 20.0 * rng.standard_normal(N_BEATS - 1)
        kept_peaks = np.arange(N_BEATS) * 360   # dummy positions

        print(f"    Synthetic beats : {beats.shape}")

    # ── Feature extraction ────────────────────────────────────────────────────
    print("\n[1] Feature names and groups:")
    groups = extractor.feature_groups()
    for domain, names in groups.items():
        print(f"    {domain:<20s} → {len(names):2d} features")
        print("      " + ", ".join(names[:4]) + (" …" if len(names) > 4 else ""))

    total_features = sum(len(v) for v in groups.values())
    print(f"\n    Total features (excl. label): {total_features}")

    # ── Single-beat extraction demo ───────────────────────────────────────────
    print("\n[2] Single-beat feature extraction (beat index 0):")
    feat0 = extractor.extract(
        beat    = beats[0],
        pre_rr  = float(rr_ms[0]) / 1000.0 if len(rr_ms) > 0 else None,
        post_rr = float(rr_ms[1]) / 1000.0 if len(rr_ms) > 1 else None,
        label   = labels[0] if labels else None,
    )

    td_subset = {k: v for k, v in feat0.items() if k.startswith("td_")}
    fd_subset = {k: v for k, v in feat0.items() if k.startswith("fd_")}
    wd_subset = {k: v for k, v in feat0.items() if k.startswith("wd_") and "energy" in k}

    print("  Time-domain:")
    for k, v in td_subset.items():
        print(f"    {k:<28s}: {v:.6f}" if not np.isnan(float(v)) else f"    {k:<28s}: NaN")
    print("  Frequency-domain:")
    for k, v in fd_subset.items():
        print(f"    {k:<28s}: {v:.6f}" if not np.isnan(float(v)) else f"    {k:<28s}: NaN")
    print("  Wavelet (energy only):")
    for k, v in wd_subset.items():
        print(f"    {k:<28s}: {v:.6f}")

    # ── Full batch extraction ─────────────────────────────────────────────────
    print("\n[3] Batch extraction …")
    df = extractor.extract_batch(
        beats        = beats,
        rr_intervals = rr_ms,
        r_peak_indices = kept_peaks,
        labels       = labels,
    )

    print(f"    DataFrame shape  : {df.shape}")
    print(f"    Memory usage     : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print(f"    NaN count        : {df.isna().sum().sum()}")
    print(f"\n    Class distribution:")
    print(textwrap.indent(df["label"].value_counts().to_string(), "      "))

    # ── Per-class feature statistics ──────────────────────────────────────────
    print("\n[4] Per-class mean for key discriminative features:")
    key_feats = ["td_pre_rr", "td_qrs_amplitude", "td_qrs_duration",
                 "fd_lf_hf_ratio", "wd_cD3_energy", "wd_cD1_rel_energy"]
    available = [f for f in key_feats if f in df.columns]
    summary   = df.groupby("label")[available].mean().round(4)
    print(textwrap.indent(summary.to_string(), "    "))

    # ── Data types & dtypes ───────────────────────────────────────────────────
    print("\n[5] Column dtypes (sample):")
    print(textwrap.indent(df.dtypes.value_counts().to_string(), "    "))

    print("\n[✓] Phase 3 feature extraction complete — DataFrame ready for Phase 4 modelling.\n")