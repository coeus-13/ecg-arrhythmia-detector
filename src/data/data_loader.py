"""
src/data/data_loader.py
=======================
ECG data loading and annotation mapping for the MIT-BIH Arrhythmia Database.

Clinical reference
------------------
Beat classification follows the ANSI/AAMI EC57:1998 standard, which collapses
the 15+ MIT-BIH annotation codes into 5 superclasses:

  N  – Normal beat and normal-variant beats
  S  – Supraventricular ectopic beat
  V  – Ventricular ectopic beat
  F  – Fusion beat (ventricular + normal)
  Q  – Unknown / unclassifiable beat (and paced beats per AAMI)

Reference: Moody GB, Mark RG. "The impact of the MIT-BIH Arrhythmia Database."
           IEEE Eng Med Biol Mag. 2001;20(3):45–50.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wfdb

# ── Module-level logger ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── AAMI EC57 annotation mapping ──────────────────────────────────────────────
#
# Every symbol that appears in MIT-BIH annotations is listed here so that
# nothing is silently dropped.  Non-beat annotations (rhythm labels, signal
# quality markers, etc.) are mapped to None and filtered out downstream.
#
MITBIH_TO_AAMI: Dict[str, Optional[str]] = {
    # ── N  (Normal / normal-variant) ─────────────────────────────────────────
    "N": "N",   # Normal beat
    "L": "N",   # Left bundle branch block beat
    "R": "N",   # Right bundle branch block beat
    "B": "N",   # Bundle branch block beat (unspecified)
    "e": "N",   # Atrial escape beat
    "j": "N",   # Nodal (junctional) escape beat

    # ── S  (Supraventricular ectopic) ────────────────────────────────────────
    "A": "S",   # Atrial premature beat
    "a": "S",   # Aberrated atrial premature beat
    "J": "S",   # Nodal (junctional) premature beat
    "S": "S",   # Supraventricular premature or ectopic beat (generic)

    # ── V  (Ventricular ectopic) ─────────────────────────────────────────────
    "V": "V",   # Premature ventricular contraction
    "E": "V",   # Ventricular escape beat

    # ── F  (Fusion) ──────────────────────────────────────────────────────────
    "F": "F",   # Fusion of ventricular and normal beat

    # ── Q  (Unknown / paced) ─────────────────────────────────────────────────
    "/": "Q",   # Paced beat
    "f": "Q",   # Fusion of paced and normal beat
    "Q": "Q",   # Unclassifiable beat

    # ── Non-beat annotations (rhythm / signal-quality / waveform markers) ─────
    # These are present in .atr files but do NOT represent individual beats.
    # Mapped to None so they are excluded from the beat-level dataset.
    "[": None,  # Start of ventricular flutter/fibrillation
    "!": None,  # Ventricular flutter wave
    "]": None,  # End of ventricular flutter/fibrillation
    "x": None,  # Non-conducted P-wave (blocked APC)
    "(": None,  # Rhythm change annotation
    ")": None,  # Rhythm change annotation
    "p": None,  # Peak of P-wave (waveform onset marker)
    "t": None,  # Peak of T-wave (waveform onset marker)
    "u": None,  # Peak of U-wave
    "`": None,  # PQ junction
    "'": None,  # J-point
    "^": None,  # Non-conducted pacer spike
    "|": None,  # Isolated QRS-like artifact
    "~": None,  # Signal quality change
    "+": None,  # Rhythm change (start of new rhythm annotation)
    "s": None,  # ST segment change
    "T": None,  # T-wave change
    "*": None,  # Systole marker
    "D": None,  # Diastole marker
    "=": None,  # Measurement annotation
    '"': None,  # Comment annotation
    "@": None,  # Link to external data
}

# All 48 MIT-BIH records (both standard and "noisy" subsets)
ALL_MITBIH_RECORDS: List[str] = [
    "100", "101", "102", "103", "104", "105", "106", "107",
    "108", "109", "111", "112", "113", "114", "115", "116",
    "117", "118", "119", "121", "122", "123", "124", "200",
    "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]


class ECGDataLoader:
    """
    Load, segment, and label ECG beats from the MIT-BIH Arrhythmia Database.

    The class handles:
    - Downloading the PhysioNet ``mitdb`` database on first use.
    - Reading raw WFDB signal (.dat) and annotation (.atr) files.
    - Mapping annotation symbols to the 5 AAMI EC57 superclasses.
    - Windowing individual beats around their R-peak annotation samples.
    - Returning a tidy ``pandas.DataFrame`` ready for feature extraction.

    Parameters
    ----------
    data_dir : str | Path
        Root directory where ``mitdb/`` will be stored or already exists.
        Default: ``./data``.
    window_before : int
        Number of samples to include *before* each R-peak annotation.
        Default: 90  (~250 ms at 360 Hz).
    window_after : int
        Number of samples to include *after* each R-peak annotation.
        Default: 110  (~306 ms at 360 Hz).
        Total window = window_before + window_after = 200 samples (~556 ms).
    lead : int
        Which lead channel to use (0 = MLII, 1 = V1 for most records).
        Default: 0.
    """

    # PhysioNet database identifier
    _DB_NAME = "mitdb"

    def __init__(
        self,
        data_dir: str | Path = "./data",
        window_before: int = 90,
        window_after: int = 110,
        lead: int = 0,
    ) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.db_dir = self.data_dir / self._DB_NAME
        self.window_before = window_before
        self.window_after = window_after
        self.lead = lead

        self.db_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ECGDataLoader initialised — data directory: %s", self.db_dir)

    # ── Public API ────────────────────────────────────────────────────────────

    def download_database(self, overwrite: bool = False) -> None:
        """
        Download all MIT-BIH records from PhysioNet.

        Uses ``wfdb.dl_database`` which performs incremental downloads;
        existing files are skipped unless *overwrite* is True.

        Parameters
        ----------
        overwrite : bool
            If True, re-download even if files already exist.
        """
        existing = list(self.db_dir.glob("*.dat"))
        if existing and not overwrite:
            logger.info(
                "Found %d existing .dat files in %s — skipping download. "
                "Pass overwrite=True to force.",
                len(existing),
                self.db_dir,
            )
            return

        logger.info("Downloading MIT-BIH database to %s …", self.db_dir)
        wfdb.dl_database(self._DB_NAME, dl_dir=str(self.db_dir))
        logger.info("Download complete.")

    def load_record(self, record_id: str) -> pd.DataFrame:
        """
        Load a single MIT-BIH record and return a beat-level DataFrame.

        Parameters
        ----------
        record_id : str
            Record name without extension, e.g. ``"100"``.

        Returns
        -------
        pd.DataFrame
            One row per beat with columns:

            - ``record``        : record identifier string
            - ``sample``        : R-peak sample index in the raw signal
            - ``symbol``        : original MIT-BIH annotation symbol
            - ``aami_label``    : AAMI superclass  (N / S / V / F / Q)
            - ``signal``        : 1-D numpy array of length
                                  (window_before + window_after)

        Raises
        ------
        FileNotFoundError
            If the record files are not present in ``db_dir``.
        ValueError
            If the lead index is out of range for this record.
        """
        record_path = str(self.db_dir / record_id)
        self._assert_record_exists(record_id)

        # ── 1. Read raw signal ────────────────────────────────────────────────
        record = wfdb.rdrecord(record_path)
        if self.lead >= record.n_sig:
            raise ValueError(
                f"Lead index {self.lead} is out of range for record {record_id} "
                f"which has {record.n_sig} channel(s)."
            )
        signal: np.ndarray = record.p_signal[:, self.lead].astype(np.float32)
        fs: int = record.fs  # sampling frequency (360 Hz for MIT-BIH)
        n_samples: int = len(signal)

        logger.debug(
            "Record %s | fs=%d Hz | samples=%d | lead=%d (%s)",
            record_id, fs, n_samples, self.lead, record.sig_name[self.lead],
        )

        # ── 2. Read annotations ───────────────────────────────────────────────
        annotation = wfdb.rdann(record_path, extension="atr")
        ann_samples: np.ndarray = annotation.sample      # R-peak positions
        ann_symbols: List[str] = annotation.symbol       # raw symbol strings

        # ── 3. Map symbols → AAMI superclasses ───────────────────────────────
        rows: List[dict] = []
        skipped_non_beat = 0
        skipped_boundary = 0
        skipped_unknown_symbol = 0

        for sample, symbol in zip(ann_samples, ann_symbols):
            # Unknown symbols not in our map (future-proof)
            if symbol not in MITBIH_TO_AAMI:
                logger.debug(
                    "Record %s | Unknown annotation symbol '%s' at sample %d — skipping.",
                    record_id, symbol, sample,
                )
                skipped_unknown_symbol += 1
                continue

            aami_label = MITBIH_TO_AAMI[symbol]

            # Non-beat rhythmic / waveform annotation
            if aami_label is None:
                skipped_non_beat += 1
                continue

            # Window boundary check (exclude beats too close to record edges)
            start = sample - self.window_before
            end = sample + self.window_after
            if start < 0 or end > n_samples:
                skipped_boundary += 1
                continue

            beat_signal = signal[start:end].copy()

            rows.append(
                {
                    "record": record_id,
                    "sample": sample,
                    "symbol": symbol,
                    "aami_label": aami_label,
                    "signal": beat_signal,
                }
            )

        logger.info(
            "Record %s | beats extracted=%d | skipped_non_beat=%d | "
            "skipped_boundary=%d | skipped_unknown=%d",
            record_id,
            len(rows),
            skipped_non_beat,
            skipped_boundary,
            skipped_unknown_symbol,
        )

        return pd.DataFrame(rows)

    def load_records(
        self,
        record_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load multiple records and concatenate into a single DataFrame.

        Parameters
        ----------
        record_ids : list of str, optional
            Records to load. Defaults to all 48 MIT-BIH records
            (``ALL_MITBIH_RECORDS``).

        Returns
        -------
        pd.DataFrame
            Concatenated beat-level DataFrame (see :meth:`load_record`).
        """
        if record_ids is None:
            record_ids = ALL_MITBIH_RECORDS

        frames: List[pd.DataFrame] = []
        failed: List[str] = []

        for rid in record_ids:
            try:
                df = self.load_record(rid)
                frames.append(df)
            except FileNotFoundError as exc:
                logger.warning("Skipping record %s — %s", rid, exc)
                failed.append(rid)
            except Exception as exc:  # noqa: BLE001
                logger.error("Error loading record %s: %s", rid, exc, exc_info=True)
                failed.append(rid)

        if failed:
            logger.warning("Failed to load %d record(s): %s", len(failed), failed)

        if not frames:
            raise RuntimeError(
                "No records could be loaded. Run download_database() first."
            )

        combined = pd.concat(frames, ignore_index=True)
        logger.info(
            "Loaded %d records | total beats=%d", len(frames), len(combined)
        )
        return combined

    # ── Static / class helpers ────────────────────────────────────────────────

    @staticmethod
    def class_distribution(df: pd.DataFrame) -> pd.Series:
        """
        Return beat counts and percentages per AAMI superclass.

        Parameters
        ----------
        df : pd.DataFrame
            Output from :meth:`load_record` or :meth:`load_records`.

        Returns
        -------
        pd.DataFrame
            Indexed by AAMI label with columns ``count`` and ``pct``.
        """
        counts = df["aami_label"].value_counts().rename("count")
        pct = (counts / counts.sum() * 100).round(2).rename("pct_%")
        return pd.concat([counts, pct], axis=1)

    @staticmethod
    def symbol_distribution(df: pd.DataFrame) -> pd.Series:
        """
        Return raw MIT-BIH symbol counts (useful for EDA / debugging).

        Parameters
        ----------
        df : pd.DataFrame
            Output from :meth:`load_record` or :meth:`load_records`.

        Returns
        -------
        pd.Series
            Symbol counts sorted descending.
        """
        return df["symbol"].value_counts()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _assert_record_exists(self, record_id: str) -> None:
        """Raise FileNotFoundError if the .dat file is missing."""
        dat_path = self.db_dir / f"{record_id}.dat"
        if not dat_path.exists():
            raise FileNotFoundError(
                f"Record file not found: {dat_path}\n"
                f"Run ECGDataLoader.download_database() first, or check that "
                f"'{self.db_dir}' is the correct data directory."
            )


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import textwrap

    # ── Configuration ─────────────────────────────────────────────────────────
    DATA_DIR = Path("./data")
    RECORD = "100"

    print("=" * 68)
    print("  ECGDataLoader — Phase 1 Demo  (MIT-BIH record '100')")
    print("=" * 68)

    # ── Instantiate loader ────────────────────────────────────────────────────
    loader = ECGDataLoader(
        data_dir=DATA_DIR,
        window_before=90,   # ~250 ms before R-peak  @ 360 Hz
        window_after=110,   # ~306 ms after  R-peak  @ 360 Hz
        lead=0,             # MLII lead (default for MIT-BIH)
    )

    # ── Download (no-op if files already present) ─────────────────────────────
    loader.download_database(overwrite=False)

    # ── Load a single record ──────────────────────────────────────────────────
    print(f"\n[1] Loading record '{RECORD}' …")
    df = loader.load_record(RECORD)

    print(f"\n    DataFrame shape : {df.shape}")
    print(f"    Columns         : {list(df.columns)}")
    print(f"    Beat window     : {df['signal'].iloc[0].shape[0]} samples")
    print(f"    Signal dtype    : {df['signal'].iloc[0].dtype}")

    # ── AAMI class distribution ───────────────────────────────────────────────
    print(f"\n[2] AAMI EC57 class distribution for record '{RECORD}':")
    dist = loader.class_distribution(df)
    print(textwrap.indent(dist.to_string(), prefix="    "))

    # ── Raw symbol breakdown ──────────────────────────────────────────────────
    print(f"\n[3] Raw MIT-BIH symbol breakdown for record '{RECORD}':")
    sym_dist = loader.symbol_distribution(df)
    print(textwrap.indent(sym_dist.to_string(), prefix="    "))

    # ── Peek at one beat per class ────────────────────────────────────────────
    print("\n[4] One example beat per AAMI class:")
    for label, group in df.groupby("aami_label"):
        row = group.iloc[0]
        sig = row["signal"]
        print(
            f"    [{label}]  sample={row['sample']:6d}  "
            f"symbol='{row['symbol']}'  "
            f"signal_mean={sig.mean():.4f}  "
            f"signal_std={sig.std():.4f}"
        )

    print("\n[✓] Phase 1 data loading complete — ready for feature extraction.\n")
