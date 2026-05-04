"""
Build whitened background datasets for H1 and L1 across O3a and O3b.

Whitening strategy:
  - Each 36s context window is whitened independently using exactly 36s of data,
    consistent with the procedure used at test time.
  - GWpy's whiten() (fftlength=PSD_DURATION) corrupts fftlength/2 seconds at each
    edge; we manually crop MAX_FILTER_DUR = fftlength/2 seconds from each end,
    leaving a 32s usable window per context.
  - The context window advances by CONTEXT_STRIDE = 32s (the usable window size),
    so usable regions tile without gaps or overlaps — no GPS time is whitened twice.

Sample extraction:
  - A 8s window slides through the 32s usable region with step DELTA_T = 4s,
    yielding 7 overlapping samples per context window.
  - Adjacent samples share 4s of content (50% overlap) but are distinct windows.

Train/test split note:
  - Always split by GPS time range, not sample index — overlapping samples within
    a context share the same underlying noise realization.
"""

import numpy as np
import pickle
import time

from gwpy.segments import DataQualityFlag, SegmentList, Segment
from gwpy.timeseries import TimeSeries

# ── Run configuration ─────────────────────────────────────────────────────────

RUNS = {
    'O3a': {'start': 1238166018, 'end': 1253977218},
    'O3b': {'start': 1256655618, 'end': 1269363618},
}
IFOS    = ['H1', 'L1']
IFO_SITE = {'H1': 'hanford', 'L1': 'livingston'}

# ── Processing parameters ─────────────────────────────────────────────────────

OMICRON_DIR      = 'omicron_triggers/'
SAMPLE_RATE      = 4096   # Hz

CONTEXT_DURATION = 36     # s — data fetched and whitened per window
MAX_FILTER_DUR   = 2      # s — whitening filter length; removed from each edge
PSD_DURATION     = 4      # s — Welch sub-segment length for PSD estimation
CONTEXT_STRIDE   = CONTEXT_DURATION - 2 * MAX_FILTER_DUR  # = 32s, usable window size

SAMPLE_DURATION  = 8      # s — output sample length
SAMPLE_LENGTH    = SAMPLE_DURATION * SAMPLE_RATE           # 32768 samples
DELTA_T          = 2      # s — sliding step within usable window
DELTA_T_SAMPLES  = DELTA_T * SAMPLE_RATE                   # 16384 samples

# Samples per context: floor((CONTEXT_STRIDE - SAMPLE_DURATION) / DELTA_T) + 1 = 13
SAMPLES_PER_CONTEXT = (CONTEXT_STRIDE - SAMPLE_DURATION) // DELTA_T + 1

TRIGGER_BUFFER   = 2.0    # s — safety margin added to each side of a trigger
MIN_SEG_DUR      = CONTEXT_DURATION
TARGET_COUNT     = 40_000

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_glitch_segments(peak_times, durations, buffer):
    """Coalesced SegmentList covering each trigger plus a safety buffer."""
    segs = SegmentList([
        Segment(t - d / 2.0 - buffer, t + d / 2.0 + buffer)
        for t, d in zip(peak_times, durations)
    ])
    segs.coalesce()
    return segs


def whiten_and_slice(ts):
    """
    Whiten a CONTEXT_DURATION-second GWpy TimeSeries and return overlapping
    SAMPLE_DURATION windows from the usable (edge-trimmed) region.

    GWpy's whiten() corrupts fftlength/2 seconds at each edge, so we manually
    crop MAX_FILTER_DUR = PSD_DURATION/2 seconds from each end.
    """
    whitened = ts.whiten(fftlength=PSD_DURATION, overlap=PSD_DURATION // 2)
    pad = MAX_FILTER_DUR * SAMPLE_RATE
    data = np.array(whitened, dtype=np.float32)[pad:-pad]

    windows = []
    for start in range(0, len(data) - SAMPLE_LENGTH + 1, DELTA_T_SAMPLES):
        w = data[start : start + SAMPLE_LENGTH]
        if np.isfinite(w).all():
            windows.append(w)
    return windows

# ── Main loop ─────────────────────────────────────────────────────────────────

backgrounds = {run: {ifo: [] for ifo in IFOS} for run in RUNS}

for run, cfg in RUNS.items():
    run_start, run_end = cfg['start'], cfg['end']

    for ifo in IFOS:
        print(f"\n{'─' * 60}")
        print(f"  {ifo}  {run}  (GPS {run_start} – {run_end})")
        print(f"  Context: {CONTEXT_DURATION}s  |  Stride: {CONTEXT_STRIDE}s  |  "
              f"{SAMPLES_PER_CONTEXT} samples/context  |  target: {TARGET_COUNT}")
        print(f"{'─' * 60}")

        site = IFO_SITE[ifo]
        peak_times = np.load(f"{OMICRON_DIR}{site}_{run.lower()}_peak_times.npy")
        durations  = np.load(f"{OMICRON_DIR}{site}_{run.lower()}_durations.npy")

        flag = f'{ifo}:DMT-ANALYSIS_READY:1'
        print(f"  Querying {flag} ...")
        science_segs = DataQualityFlag.query(flag, run_start, run_end).active

        glitch_segs = build_glitch_segments(peak_times, durations, TRIGGER_BUFFER)
        clean_segs  = science_segs - glitch_segs

        n_qualifying = sum(1 for s in clean_segs if float(s[1] - s[0]) >= MIN_SEG_DUR)
        total_clean_h = sum(float(s[1] - s[0]) for s in clean_segs) / 3600
        print(f"  Clean time: {total_clean_h:.1f} h  |  "
              f"segments >= {MIN_SEG_DUR}s: {n_qualifying}")

        samples = []
        t0 = time.time()
        failed = 0

        for seg in clean_segs:
            if len(samples) >= TARGET_COUNT:
                break
            if float(seg[1] - seg[0]) < MIN_SEG_DUR:
                continue

            context_start = float(seg[0])
            while context_start + CONTEXT_DURATION <= float(seg[1]):
                if len(samples) >= TARGET_COUNT:
                    break

                try:
                    ts = TimeSeries.get(
                        f'{ifo}:GDS-CALIB_STRAIN',
                        context_start,
                        context_start + CONTEXT_DURATION,
                        verbose=False,
                    )
                    if ts.sample_rate.value != SAMPLE_RATE:
                        ts = ts.resample(SAMPLE_RATE)

                    new = whiten_and_slice(ts)
                    samples.extend(new)

                except Exception as e:
                    failed += 1
                    print(f"\n  ! GPS {context_start:.0f}: {e}")

                context_start += CONTEXT_STRIDE

                elapsed = time.time() - t0
                rate = len(samples) / max(elapsed, 1e-6)
                eta  = (TARGET_COUNT - len(samples)) / max(rate, 1e-6)
                m, s = divmod(int(eta), 60)
                print(
                    f"\r  {len(samples):>6}/{TARGET_COUNT}  |  "
                    f"{rate:.1f} samples/s  |  ETA {m}m {s:02d}s  |  "
                    f"failed fetches: {failed}",
                    end='',
                )

        backgrounds[run][ifo] = np.array(samples[:TARGET_COUNT], dtype=np.float32)
        print(f"\n  Done: {len(backgrounds[run][ifo])} samples  |  "
              f"failed fetches: {failed}")

with open('real_backgrounds_dict.pkl', 'wb') as f:
    pickle.dump(backgrounds, f)

print("\nSaved real_backgrounds_dict.pkl")
print(f"Dataset shape per IFO/run: ({TARGET_COUNT}, {SAMPLE_LENGTH})  "
      f"[{TARGET_COUNT * SAMPLE_LENGTH * 4 / 1e9:.2f} GB each]")
