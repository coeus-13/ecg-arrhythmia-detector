"""
src/features/segmentation.py
=============================
Pan-Tompkins QRS detector and R-peak-centred beat segmentation.

Algorithm reference
-------------------
Pan J, Tompkins WJ. "A real-time QRS detection algorithm."
IEEE Trans Biomed Eng. 1985;32(3):230–236.

Pipeline (all stages operate on the *already band-pass filtered* signal)
------------------------------------------------------------------------
1. Derivative filter  — accentuates QRS slope, suppresses P/T waves.
2. Squaring           — makes all values positive; amplifies large slopes
                        nonlinearly (QRS >> T wave).
3. Moving-window integration (MWI) — integrates energy over a 150 ms window
                        to produce a smooth envelope.
4. Adaptive dual-threshold detection — maintains two running estimates:
   • SPKI  (signal peak)  — updated when a peak exceeds threshold.
   • NPKI  (noise peak)   — updated when a peak is below threshold.
   • Threshold = NPKI + 0.25 × (SPKI − NPKI).
   The 200 ms refractory period enforces the physiological minimum RR interval.
5. R-peak back-projection — detected peaks in the MWI signal are mapped back
   to the maximum of the original band-pass signal within ±150 ms, giving
   sub-sample accurate R-peak positions.

Beat segmentation
-----------------
Fixed window: 300 ms pre-peak + 500 ms post-peak (800 ms total at 360 Hz →
108 + 180 = 288 samples).  Beats too close to the recording boundary are
handled by one of three configurable strategies: ``skip``, ``zero-pad``,
or ``edge-pad``.

CPU optimisation notes
----------------------
• All intermediate arrays are ``float32`` / ``int32`` — halves memory bandwidth.
• The derivative, squaring, and MWI stages are vectorised NumPy operations;
  no Python loops until the peak-picking stage.
• ``np.convolve`` with ``mode='same'`` is used for MWI (equivalent to a
  rectangular FIR and faster than ``scipy.ndimage.uniform_filter1d`` for
  short kernels at this signal length on x86 with MKL-linked NumPy).
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class EdgeStrategy(str, Enum):
    """Strategy for beats whose window exceeds the signal boundaries."""

    SKIP = "skip"          # Discard the beat entirely
    ZERO_PAD = "zero_pad"  # Pad missing samples with zeros
    EDGE_PAD = "edge_pad"  # Repeat the nearest boundary sample (replicate)


class PanTompkinsDetector:
    """
    Real-time Pan-Tompkins QRS detector with beat segmentation.

    The detector is designed for single-lead ECG signals that have already
    been conditioned by ``ECGFilterPipeline.process_signal``.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.  MIT-BIH = 360 Hz.
    pre_peak_ms : float
        Milliseconds to include *before* each R-peak in the extracted window.
        Default 300 ms.
    post_peak_ms : float
        Milliseconds to include *after* each R-peak in the extracted window.
        Default 500 ms.
    mwi_window_ms : float
        Moving-window integration window in milliseconds.  Pan & Tompkins
        recommend 150 ms.
    refractory_ms : float
        Minimum interval between consecutive R-peaks (physiological floor for
        HR ≤ 300 bpm).  Default 200 ms.
    edge_strategy : EdgeStrategy | str
        How to handle beats at the recording boundary.  Default ``'skip'``.
    search_back_ms : float
        When no peak is found in an interval > 1.66 × mean RR (bradycardia /
        missed beat), the algorithm searches back *search_back_ms* in the MWI
        signal using a halved threshold.  Default 1000 ms.

    Examples
    --------
    >>> detector = PanTompkinsDetector(fs=360)
    >>> r_peaks = detector.detect_r_peaks(filtered_ecg)
    >>> beats, indices, mask = detector.extract_beats(filtered_ecg, r_peaks)
    """

    def __init__(
        self,
        fs: float = 360.0,
        pre_peak_ms: float = 300.0,
        post_peak_ms: float = 500.0,
        mwi_window_ms: float = 150.0,
        refractory_ms: float = 200.0,
        edge_strategy: EdgeStrategy | str = EdgeStrategy.SKIP,
        search_back_ms: float = 1000.0,
    ) -> None:
        self.fs = float(fs)
        self.pre_peak_ms = pre_peak_ms
        self.post_peak_ms = post_peak_ms
        self.mwi_window_ms = mwi_window_ms
        self.refractory_ms = refractory_ms
        self.edge_strategy = EdgeStrategy(edge_strategy)
        self.search_back_ms = search_back_ms

        # Derived sample counts (computed once)
        self._pre_samples: int = int(round(pre_peak_ms * fs / 1000.0))
        self._post_samples: int = int(round(post_peak_ms * fs / 1000.0))
        self._window_len: int = self._pre_samples + self._post_samples
        self._mwi_samples: int = max(1, int(round(mwi_window_ms * fs / 1000.0)))
        self._refractory_samples: int = int(round(refractory_ms * fs / 1000.0))
        self._search_back_samples: int = int(round(search_back_ms * fs / 1000.0))
        # ±half-window for back-projecting MWI peak → signal R-peak
        self._backproject_samples: int = int(round(150.0 * fs / 1000.0))

        logger.info(
            "PanTompkinsDetector ready | fs=%.1f Hz | window=%d+%d=%d samples "
            "(%.0f+%.0f ms) | MWI=%d samples | refractory=%d samples | edge=%s",
            self.fs,
            self._pre_samples, self._post_samples, self._window_len,
            pre_peak_ms, post_peak_ms,
            self._mwi_samples,
            self._refractory_samples,
            self.edge_strategy.value,
        )

    # ── Pan-Tompkins processing chain ─────────────────────────────────────────

    def _derivative_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Five-point derivative filter (Pan & Tompkins, 1985 eq. 2).

        H(z) = (1/8T)(−z⁻² − 2z⁻¹ + 2z + z²)

        Approximated with a simple central-difference variant that is
        causal-equivalent for offline processing:

            y[n] = (1/8) × (−x[n−2] − 2x[n−1] + 2x[n+1] + x[n+2])

        Parameters
        ----------
        signal : np.ndarray, shape (N,)

        Returns
        -------
        np.ndarray, shape (N,), dtype float32
        """
        # Coefficients from Pan & Tompkins (1985)
        kernel = np.array([-1, -2, 0, 2, 1], dtype=np.float32) / 8.0
        # np.convolve with 'same' preserves length
        deriv = np.convolve(signal.astype(np.float32), kernel, mode="same")
        return deriv

    def _squaring(self, signal: np.ndarray) -> np.ndarray:
        """
        Point-wise squaring — makes all values positive and amplifies peaks.

        y[n] = x[n]²

        Parameters
        ----------
        signal : np.ndarray

        Returns
        -------
        np.ndarray, dtype float32
        """
        return np.square(signal, dtype=np.float32)

    def _moving_window_integration(self, signal: np.ndarray) -> np.ndarray:
        """
        Rectangular moving-window integration over ``_mwi_samples``.

        y[n] = (1/N) × Σ_{k=n−N+1}^{n} x[k]

        Implemented as convolution with a normalised rectangular window,
        which is O(N·W) but cache-efficient for the window sizes used here
        (W ≈ 54 samples at 360 Hz).

        Parameters
        ----------
        signal : np.ndarray, shape (N,)

        Returns
        -------
        np.ndarray, shape (N,), dtype float32
        """
        kernel = np.ones(self._mwi_samples, dtype=np.float32) / self._mwi_samples
        mwi = np.convolve(signal, kernel, mode="same")
        return mwi.astype(np.float32)

    def _compute_mwi_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Execute the derivative → squaring → MWI sub-pipeline.

        Parameters
        ----------
        signal : np.ndarray, shape (N,)
            Band-pass filtered single-lead ECG.

        Returns
        -------
        np.ndarray, shape (N,), dtype float32
            MWI envelope used for adaptive thresholding.
        """
        d = self._derivative_filter(signal)
        sq = self._squaring(d)
        mwi = self._moving_window_integration(sq)
        return mwi

    # ── Adaptive thresholding & peak detection ────────────────────────────────

    def _find_local_maxima(
        self,
        signal: np.ndarray,
        min_distance: int,
    ) -> np.ndarray:
        """
        Return indices of local maxima separated by at least *min_distance*.

        Uses a sliding-comparison approach that is vectorised except for the
        minimum-distance enforcement loop (which operates on the much smaller
        candidate set, not the full signal).

        Parameters
        ----------
        signal : np.ndarray, shape (N,)
        min_distance : int
            Minimum sample gap between accepted maxima.

        Returns
        -------
        np.ndarray, shape (M,), dtype int64
            Sorted array of local maximum positions.
        """
        # A sample is a local maximum if it is greater than its immediate
        # neighbours (strict inequality suppresses plateaux).
        candidates = np.where(
            (signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:])
        )[0] + 1  # +1 to correct for the slice offset

        if len(candidates) == 0:
            return candidates

        # Enforce minimum distance (greedy left-to-right)
        accepted: List[int] = [candidates[0]]
        for idx in candidates[1:]:
            if idx - accepted[-1] >= min_distance:
                accepted.append(idx)
        return np.array(accepted, dtype=np.int64)

    def _adaptive_threshold(
        self,
        mwi: np.ndarray,
        candidates: np.ndarray,
    ) -> np.ndarray:
        """
        Pan-Tompkins adaptive dual-threshold classifier.

        Running estimates
        -----------------
        SPKI : signal peak index estimate
            Updated with 1/8 weight when a candidate exceeds THRESHOLD1.
        NPKI : noise peak index estimate
            Updated with 1/8 weight otherwise.
        THRESHOLD1 = NPKI + 0.25 × (SPKI − NPKI)

        Search-back rule
        ----------------
        If the interval since the last accepted R-peak exceeds 1.66 × mean RR,
        search the preceding ``search_back_samples`` in the MWI signal for a
        peak exceeding THRESHOLD1 / 2.

        Parameters
        ----------
        mwi : np.ndarray, shape (N,)
        candidates : np.ndarray
            Local maxima indices in *mwi*.

        Returns
        -------
        np.ndarray
            Accepted R-peak indices (in the MWI signal coordinate).
        """
        if len(candidates) == 0:
            return np.array([], dtype=np.int64)

        # Initialise thresholds from the first 2 seconds of signal
        init_window = candidates[candidates < int(2.0 * self.fs)]
        if len(init_window) == 0:
            init_window = candidates[:8]

        peak_vals = mwi[init_window]
        spki = float(np.max(peak_vals)) if len(peak_vals) > 0 else float(mwi.max())
        npki = float(np.mean(peak_vals)) * 0.5 if len(peak_vals) > 0 else spki * 0.1
        threshold1 = npki + 0.25 * (spki - npki)

        accepted_peaks: List[int] = []
        rr_intervals: List[int] = []
        last_peak: int = -self._refractory_samples  # allow detection from sample 0

        for idx in candidates:
            # Enforce absolute refractory period
            if idx - last_peak < self._refractory_samples:
                continue

            peak_val = float(mwi[idx])

            # Search-back: if RR interval is too long, lower the threshold
            if accepted_peaks:
                rr_mean = int(np.mean(rr_intervals)) if rr_intervals else 0
                missed_limit = int(1.66 * rr_mean) if rr_mean > 0 else int(1.66 * self.fs)
                gap = idx - last_peak
                if gap > missed_limit:
                    # Search in the interval [last_peak, idx] with half-threshold
                    search_start = max(0, last_peak + self._refractory_samples)
                    sub_mwi = mwi[search_start:idx]
                    if len(sub_mwi) > 0:
                        sub_max_idx = int(np.argmax(sub_mwi))
                        sub_max_val = float(sub_mwi[sub_max_idx])
                        if sub_max_val > threshold1 / 2.0:
                            sb_peak = search_start + sub_max_idx
                            accepted_peaks.append(sb_peak)
                            rr_intervals.append(sb_peak - last_peak)
                            spki = 0.875 * spki + 0.125 * sub_max_val
                            threshold1 = npki + 0.25 * (spki - npki)
                            last_peak = sb_peak

            # Main threshold decision
            if peak_val >= threshold1:
                accepted_peaks.append(idx)
                if last_peak >= 0:
                    rr_intervals.append(idx - last_peak)
                # Limit RR buffer to last 8 intervals (Pan & Tompkins spec)
                rr_intervals = rr_intervals[-8:]
                spki = 0.875 * spki + 0.125 * peak_val
                last_peak = idx
            else:
                npki = 0.875 * npki + 0.125 * peak_val

            threshold1 = npki + 0.25 * (spki - npki)

        return np.array(accepted_peaks, dtype=np.int64)

    def _backproject_to_signal(
        self,
        signal: np.ndarray,
        mwi_peaks: np.ndarray,
    ) -> np.ndarray:
        """
        Map MWI-domain peak positions to the true R-peak in the filtered signal.

        The MWI introduces a group delay of ≈ MWI_window/2 samples.  For each
        MWI peak, we search for the maximum of |signal| in a ±150 ms window
        around the MWI peak position and return that as the R-peak location.

        Parameters
        ----------
        signal : np.ndarray, shape (N,)
            Band-pass filtered ECG.
        mwi_peaks : np.ndarray
            Peak indices in the MWI signal.

        Returns
        -------
        np.ndarray
            R-peak indices in *signal* coordinate space.
        """
        n = len(signal)
        r_peaks: List[int] = []
        half = self._backproject_samples

        for mwi_idx in mwi_peaks:
            lo = max(0, int(mwi_idx) - half)
            hi = min(n, int(mwi_idx) + half)
            local = signal[lo:hi]
            r = lo + int(np.argmax(np.abs(local)))
            r_peaks.append(r)

        # Remove duplicates that can arise from nearby MWI peaks
        r_peaks_arr = np.array(r_peaks, dtype=np.int64)
        if len(r_peaks_arr) > 1:
            keep_mask = np.concatenate(
                [[True], np.diff(r_peaks_arr) >= self._refractory_samples]
            )
            r_peaks_arr = r_peaks_arr[keep_mask]

        return r_peaks_arr

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_r_peaks(self, signal: np.ndarray) -> np.ndarray:
        """
        Run the full Pan-Tompkins pipeline and return R-peak sample indices.

        Parameters
        ----------
        signal : np.ndarray, shape (N,)
            Band-pass filtered single-lead ECG (output of
            ``ECGFilterPipeline.process_signal``).

        Returns
        -------
        np.ndarray, shape (M,), dtype int64
            Sample indices of detected R-peaks, sorted ascending.
        """
        signal = np.asarray(signal, dtype=np.float32)
        n = len(signal)

        if n < self._mwi_samples * 2:
            logger.warning(
                "Signal length %d is very short for Pan-Tompkins (MWI=%d). "
                "Results may be unreliable.",
                n, self._mwi_samples,
            )

        # ── Steps 1–3: derivative → squaring → MWI ───────────────────────────
        mwi = self._compute_mwi_signal(signal)

        # ── Step 4a: find local maxima in MWI with refractory spacing ─────────
        candidates = self._find_local_maxima(mwi, min_distance=self._refractory_samples)

        # ── Step 4b: adaptive dual-threshold classifier ───────────────────────
        mwi_peaks = self._adaptive_threshold(mwi, candidates)

        # ── Step 5: back-project to signal R-peaks ────────────────────────────
        r_peaks = self._backproject_to_signal(signal, mwi_peaks)

        logger.debug(
            "detect_r_peaks | signal_len=%d | candidates=%d | accepted=%d",
            n, len(candidates), len(r_peaks),
        )

        return r_peaks

    def extract_beats(
        self,
        signal: np.ndarray,
        r_peaks: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract fixed-length windows centred on each R-peak.

        Window layout
        -------------
        [ ←── pre_samples ──→ | R-peak | ←── post_samples ──→ ]
          = pre_peak_ms (300 ms)             post_peak_ms (500 ms)
          = 108 samples @ 360 Hz             180 samples @ 360 Hz
          Total: 288 samples @ 360 Hz (~800 ms)

        Edge handling (configurable via ``edge_strategy``)
        ---------------------------------------------------
        SKIP      — Beats whose window would exceed [0, N) are omitted.
                    The returned ``mask`` indicates which input peaks were kept.
        ZERO_PAD  — Out-of-bounds samples are filled with 0.0.
        EDGE_PAD  — Out-of-bounds samples are filled by replicating the
                    nearest boundary sample (``np.pad`` 'edge' mode).

        Parameters
        ----------
        signal : np.ndarray, shape (N,)
            Band-pass filtered ECG (same array used for detection).
        r_peaks : np.ndarray, shape (M,)
            R-peak sample indices (from ``detect_r_peaks``).

        Returns
        -------
        beats : np.ndarray, shape (K, window_len), dtype float32
            Beat windows.  K ≤ M depending on edge strategy.
        kept_peaks : np.ndarray, shape (K,), dtype int64
            R-peak indices corresponding to rows in *beats*.
        mask : np.ndarray, shape (M,), dtype bool
            Boolean mask of which input *r_peaks* were retained.
        """
        signal = np.asarray(signal, dtype=np.float32)
        n = len(signal)
        pre = self._pre_samples
        post = self._post_samples
        win = self._window_len
        strategy = self.edge_strategy

        beats_list: List[np.ndarray] = []
        kept_peaks_list: List[int] = []
        mask = np.zeros(len(r_peaks), dtype=bool)

        for i, rp in enumerate(r_peaks):
            rp = int(rp)
            start = rp - pre
            end = rp + post  # exclusive

            # ── Case 1: window fully within signal ────────────────────────────
            if start >= 0 and end <= n:
                beat = signal[start:end].copy()
                beats_list.append(beat)
                kept_peaks_list.append(rp)
                mask[i] = True
                continue

            # ── Case 2: boundary conditions ───────────────────────────────────
            if strategy == EdgeStrategy.SKIP:
                logger.debug(
                    "Beat at sample %d skipped (window [%d, %d) out of [0, %d)).",
                    rp, start, end, n,
                )
                continue  # mask[i] stays False

            elif strategy == EdgeStrategy.ZERO_PAD:
                beat = np.zeros(win, dtype=np.float32)
                # Compute valid overlap between [start, end) and [0, n)
                src_lo = max(0, start)
                src_hi = min(n, end)
                dst_lo = src_lo - start          # offset into beat buffer
                dst_hi = dst_lo + (src_hi - src_lo)
                beat[dst_lo:dst_hi] = signal[src_lo:src_hi]
                beats_list.append(beat)
                kept_peaks_list.append(rp)
                mask[i] = True

            elif strategy == EdgeStrategy.EDGE_PAD:
                # Pad signal *temporarily* to avoid boundary checks
                pad_left = max(0, -start)
                pad_right = max(0, end - n)
                padded = np.pad(signal, (pad_left, pad_right), mode="edge")
                adj_start = start + pad_left
                adj_end = end + pad_left
                beat = padded[adj_start:adj_end].copy()
                beats_list.append(beat)
                kept_peaks_list.append(rp)
                mask[i] = True

        if not beats_list:
            logger.warning("No beats extracted from signal of length %d.", n)
            return (
                np.empty((0, win), dtype=np.float32),
                np.empty(0, dtype=np.int64),
                mask,
            )

        beats = np.stack(beats_list, axis=0)   # (K, win)
        kept_peaks = np.array(kept_peaks_list, dtype=np.int64)

        logger.debug(
            "extract_beats | r_peaks=%d | extracted=%d | edge_strategy=%s",
            len(r_peaks), len(beats_list), strategy.value,
        )

        return beats, kept_peaks, mask

    def detect_and_extract(
        self,
        signal: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convenience method: detect R-peaks then extract beat windows.

        Parameters
        ----------
        signal : np.ndarray, shape (N,)

        Returns
        -------
        beats : np.ndarray, shape (K, window_len), dtype float32
        r_peaks : np.ndarray, shape (K,), dtype int64
            Back-projected R-peak positions for the *extracted* beats.
        all_r_peaks : np.ndarray, shape (M,), dtype int64
            All detected R-peaks before edge filtering.
        """
        all_r_peaks = self.detect_r_peaks(signal)
        beats, kept_peaks, _ = self.extract_beats(signal, all_r_peaks)
        return beats, kept_peaks, all_r_peaks

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def compute_rr_intervals(
        self, r_peaks: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Compute RR intervals in milliseconds and basic HRV statistics.

        Parameters
        ----------
        r_peaks : np.ndarray, shape (M,)
            Sorted R-peak sample indices.

        Returns
        -------
        rr_ms : np.ndarray, shape (M-1,)
            RR intervals in milliseconds.
        stats : dict
            Keys: mean_rr_ms, std_rr_ms, min_rr_ms, max_rr_ms,
                  mean_hr_bpm, sdnn_ms, rmssd_ms.
        """
        if len(r_peaks) < 2:
            return np.array([]), {}

        rr_samples = np.diff(r_peaks.astype(np.float64))
        rr_ms = rr_samples * 1000.0 / self.fs

        diff_rr = np.diff(rr_ms)
        stats = {
            "mean_rr_ms": float(np.mean(rr_ms)),
            "std_rr_ms": float(np.std(rr_ms)),
            "min_rr_ms": float(np.min(rr_ms)),
            "max_rr_ms": float(np.max(rr_ms)),
            "mean_hr_bpm": float(60_000.0 / np.mean(rr_ms)),
            "sdnn_ms": float(np.std(rr_ms)),
            "rmssd_ms": float(np.sqrt(np.mean(diff_rr ** 2))),
        }
        return rr_ms.astype(np.float32), stats

    @property
    def window_length(self) -> int:
        """Total beat window length in samples."""
        return self._window_len

    @property
    def window_length_ms(self) -> float:
        """Total beat window length in milliseconds."""
        return self._window_len / self.fs * 1000.0


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path
    import sys

    # ── Try to use a real MIT-BIH record if the data folder exists ────────────
    DATA_DIR = Path("./data/mitdb")
    RECORD = "100"
    FS = 360.0

    use_real = (DATA_DIR / f"{RECORD}.dat").exists()

    if use_real:
        import wfdb
        from signal_processing import ECGFilterPipeline  # noqa: E402

        print(f"Loading real MIT-BIH record '{RECORD}' …")
        rec = wfdb.rdrecord(str(DATA_DIR / RECORD))
        raw_signal = rec.p_signal[:, 0].astype(np.float32)

        pipeline = ECGFilterPipeline(fs=FS)
        clean_signal = pipeline.process_signal(raw_signal)

        ann = wfdb.rdann(str(DATA_DIR / RECORD), "atr")
        reference_peaks = ann.sample[
            [s in {"N", "L", "R", "A", "V", "F", "/"} for s in ann.symbol]
        ]
    else:
        # ── Synthetic ECG: sum of Gaussian QRS pulses ─────────────────────────
        print("MIT-BIH data not found — using synthetic ECG …")
        T = 20.0
        t = np.linspace(0, T, int(FS * T), endpoint=False)
        # Simulate ~60 bpm: R-peaks every 360 samples, add jitter
        np.random.seed(0)
        r_truth = np.arange(180, int(FS * T) - 180, 360)
        r_truth = (r_truth + np.random.randint(-10, 10, size=r_truth.shape)).clip(0)

        clean_signal = np.zeros(len(t), dtype=np.float32)
        for rp in r_truth:
            width = 0.015 * FS  # ~15 ms QRS width
            gaussian = np.exp(-0.5 * ((np.arange(len(t)) - rp) / width) ** 2)
            clean_signal += gaussian.astype(np.float32)

        # Add noise
        clean_signal += (0.05 * np.random.randn(len(t))).astype(np.float32)
        reference_peaks = r_truth

    # ── Run detector ──────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  PanTompkinsDetector — Demo")
    print("=" * 68)

    detector = PanTompkinsDetector(
        fs=FS,
        pre_peak_ms=300.0,
        post_peak_ms=500.0,
        edge_strategy=EdgeStrategy.SKIP,
    )

    beats, kept_peaks, all_peaks = detector.detect_and_extract(clean_signal)

    print(f"\n[1] Signal length     : {len(clean_signal)} samples ({len(clean_signal)/FS:.1f} s)")
    print(f"    Detected R-peaks  : {len(all_peaks)}")
    print(f"    Extracted beats   : {beats.shape}  (samples × window)")
    print(f"    Beat window       : {detector.window_length} samples "
          f"({detector.window_length_ms:.0f} ms)")

    # ── RR statistics ─────────────────────────────────────────────────────────
    rr_ms, hrv = detector.compute_rr_intervals(all_peaks)
    if hrv:
        print("\n[2] HRV statistics:")
        for k, v in hrv.items():
            print(f"    {k:<18s}: {v:.2f}")

    # ── Detection accuracy vs reference (synthetic or annotated) ─────────────
    if len(reference_peaks) > 0 and len(all_peaks) > 0:
        tolerance = int(round(0.05 * FS))  # 50 ms tolerance
        tp = sum(
            any(abs(int(d) - int(rp)) <= tolerance for rp in all_peaks)
            for d in reference_peaks
        )
        sensitivity = tp / len(reference_peaks) * 100
        ppv = tp / len(all_peaks) * 100 if len(all_peaks) > 0 else 0.0
        print(f"\n[3] Detection accuracy (±{tolerance} sample tolerance):")
        print(f"    Sensitivity (Se) : {sensitivity:.1f}%   (TP={tp} / Ref={len(reference_peaks)})")
        print(f"    Precision  (PPV) : {ppv:.1f}%   (TP={tp} / Det={len(all_peaks)})")

    # ── Beat sample ───────────────────────────────────────────────────────────
    if len(beats) > 0:
        b0 = beats[0]
        print(f"\n[4] Beat [0] — mean={b0.mean():.4f}  std={b0.std():.4f}  "
              f"min={b0.min():.4f}  max={b0.max():.4f}  dtype={b0.dtype}")

    print("\n[✓] segmentation.py verified.\n")
