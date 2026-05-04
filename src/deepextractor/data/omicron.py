"""Omicron glitch-trigger utilities.

!! DISCLAIMER !!
fetch_omicron_triggers() relies on gwtrigfind locating Omicron output files on
the local filesystem.  These files are only present on the computing clusters
co-located with each LIGO detector site:

    L1 (Livingston)  →  LLO cluster  (ldas-pcdev*.ligo-la.caltech.edu)
    H1 (Hanford)     →  LHO cluster  (ldas-pcdev*.ligo-wa.caltech.edu)

Do NOT run trigger-fetching code on CIT, LDAS, or any off-site cluster — it
will fail to locate trigger files and raise RuntimeError.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Observing run GPS boundaries
# Source: https://gwosc.org/timeline/
# O1–O3 values are authoritative; O4 values are approximate — verify against
# GWOSC before use in production analyses.
# ---------------------------------------------------------------------------

# O1: 2015-09-12 to 2016-01-19
O1_START = 1126051217
O1_END   = 1137254417

# O2: 2016-11-30 to 2017-08-25
O2_START = 1164556817
O2_END   = 1187733618

# O3a: 2019-04-01 to 2019-10-01
O3A_START = 1238166018
O3A_END   = 1253977218

# O3b: 2019-11-01 to 2020-03-27
O3B_START = 1256655618
O3B_END   = 1269363618

# O4a: 2023-05-24 to 2024-01-16  (approximate)
O4A_START = 1368975618
O4A_END   = 1389398418

# O4b: 2024-04-10 to 2024-10-19  (approximate)
O4B_START = 1396742418
O4B_END   = 1413331218

RUN_PERIODS: dict[str, tuple[int, int]] = {
    "O1":  (O1_START,  O1_END),
    "O2":  (O2_START,  O2_END),
    "O3a": (O3A_START, O3A_END),
    "O3b": (O3B_START, O3B_END),
    # Combined spans — gwtrigfind silently skips gaps (engineering runs, etc.)
    "O3":  (O3A_START, O3B_END),
    "O4a": (O4A_START, O4A_END),
    "O4b": (O4B_START, O4B_END),
    "O4":  (O4A_START, O4B_END),
}

# Default calibrated strain channels per IFO
_DEFAULT_CHANNELS: dict[str, str] = {
    "L1": "L1:GDS-CALIB_STRAIN",
    "H1": "H1:GDS-CALIB_STRAIN",
}


def find_clean_gaps(
    peak_times: np.ndarray,
    durations: np.ndarray,
    run_start: int | float,
    run_end: int | float,
    trigger_buffer: float = 0.5,
) -> list[tuple[float, float]]:
    """Return trigger-free time intervals within a GPS span.

    Builds excluded windows from each trigger's peak time and duration, merges
    overlapping exclusions, then returns their complement within
    ``[run_start, run_end)``.

    Args:
        peak_times: Omicron trigger peak times (GPS seconds).
        durations: Trigger durations (seconds), matched 1-to-1 with peak_times.
        run_start: GPS start of the observing run (inclusive).
        run_end: GPS end of the observing run (exclusive).
        trigger_buffer: Extra padding (seconds) on each side of every trigger
            beyond the raw duration.  Accounts for ring-down and edge effects.

    Returns:
        List of ``(gap_start, gap_end)`` tuples (GPS seconds), sorted by start
        time, all within ``[run_start, run_end)``.
    """
    excluded: list[list[float]] = []
    for t, d in zip(peak_times, durations):
        half = max(float(d) / 2.0, 0.0)
        excluded.append([t - half - trigger_buffer, t + half + trigger_buffer])

    excluded.sort(key=lambda x: x[0])

    merged: list[list[float]] = []
    for seg in excluded:
        if merged and seg[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], seg[1])
        else:
            merged.append(list(seg))

    gaps: list[tuple[float, float]] = []
    cursor = float(run_start)
    for excl_start, excl_end in merged:
        if excl_start > cursor:
            g_start = max(cursor, float(run_start))
            g_end   = min(excl_start, float(run_end))
            if g_end > g_start:
                gaps.append((g_start, g_end))
        cursor = max(cursor, excl_end)

    if cursor < run_end:
        gaps.append((max(cursor, float(run_start)), float(run_end)))

    return gaps


def fetch_omicron_triggers(
    ifo: str,
    gps_start: int,
    gps_end: int,
    channel: str | None = None,
) -> dict[str, np.ndarray]:
    """Return Omicron trigger parameters for a GPS interval.

    Args:
        ifo: Detector prefix — ``"L1"`` or ``"H1"``.
        gps_start: GPS start time (inclusive).
        gps_end: GPS end time (exclusive).
        channel: Full channel name, e.g. ``"L1:GDS-CALIB_STRAIN"``.
            Defaults to the standard strain channel for *ifo*.

    Returns:
        Dict with 1-D float64 arrays per trigger:
          - ``peak_time``: GPS time of peak SNR
          - ``tstart``:    GPS start of the Omicron Q-tile
          - ``tend``:      GPS end of the Omicron Q-tile
          - ``duration``:  tend - tstart (tile width in seconds)

    Raises:
        ValueError: If *ifo* is unrecognised and *channel* is not given.
        RuntimeError: If no trigger files are found.  Usually means the script
            is not running on the correct site cluster.
    """
    if channel is None:
        if ifo not in _DEFAULT_CHANNELS:
            raise ValueError(
                f"Unknown IFO '{ifo}'.  Pass channel= explicitly or use 'L1'/'H1'."
            )
        channel = _DEFAULT_CHANNELS[ifo]

    try:
        from gwtrigfind import find_trigger_files
        from gwpy.table import EventTable
    except ImportError as exc:
        raise ImportError(
            "fetch_omicron_triggers() requires gwtrigfind and gwpy, which are "
            "only available on LIGO site clusters.  "
            "Install them or run this function at LLO/LHO."
        ) from exc

    logger.info(
        "Locating Omicron files: channel=%s  GPS [%d, %d) ...",
        channel, gps_start, gps_end,
    )
    trigger_files = find_trigger_files(channel, "omicron", gps_start, gps_end)
    if not trigger_files:
        raise RuntimeError(
            f"No Omicron trigger files found for {channel} GPS [{gps_start}, {gps_end}).  "
            "Confirm this script is running on the site cluster for this IFO."
        )

    # gwtrigfind may return file:// URIs — strip to plain paths for h5py/readers
    trigger_files = [str(f).replace("file://", "") for f in trigger_files]
    logger.info("Found %d trigger file(s) — reading ...", len(trigger_files))

    # Omicron writes LIGO-LW XML for older runs and HDF5 for newer runs.
    # Detect from the first file's extension.
    first = trigger_files[0]
    if first.endswith(".h5") or first.endswith(".hdf5"):
        fmt = "hdf5"
    else:
        fmt = "ligolw"
    logger.info("Detected trigger file format: %s", fmt)

    if fmt == "hdf5":
        events = EventTable.read(trigger_files, path="triggers", format=fmt)
        peak_times = np.asarray(events["time"],   dtype=np.float64)
        tstarts    = np.asarray(events["tstart"], dtype=np.float64)
        tends      = np.asarray(events["tend"],   dtype=np.float64)
    else:
        events = EventTable.read(trigger_files, tablename="sngl_burst", format=fmt)
        peak_times = np.asarray(events["peak_time"],             dtype=np.float64)
        tstarts    = np.asarray(events["start_time"],            dtype=np.float64)
        tends      = np.asarray(events["start_time"]
                                + events["duration"],            dtype=np.float64)
    durations = tends - tstarts
    logger.info("Loaded %d triggers", len(peak_times))
    return {
        "peak_time": peak_times,
        "tstart":    tstarts,
        "tend":      tends,
        "duration":  durations,
    }


def save_omicron_triggers(
    triggers: dict[str, np.ndarray],
    prefix: str,
    output_dir: str | Path = ".",
) -> None:
    """Save trigger arrays to ``<output_dir>/<prefix>_<key>.npy`` for each key.

    Args:
        triggers: Dict returned by fetch_omicron_triggers (peak_time, tstart,
            tend, duration).
        prefix: Filename stem, e.g. ``"l1_o3a"``.
        output_dir: Destination directory (created if absent).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for key, arr in triggers.items():
        np.save(out / f"{prefix}_{key}", arr)
    logger.info(
        "Saved %s_{%s}.npy → %s",
        prefix, ",".join(triggers.keys()), out.resolve(),
    )
