"""Fetch and save Omicron trigger peak times and durations for L1.

Must be run on the LLO site cluster (ldas-pcdev*.ligo-la.caltech.edu)
where Omicron trigger files are available via gwtrigfind.

Output files are saved as <out>/<ifo>_<run>_peak_times.npy and
<out>/<ifo>_<run>_durations.npy, ready to be committed to the repo
and used by get_clean_backgrounds.py on CIT.

Example
-------
    python scripts/get_omicron_triggers_L1.py --runs O3a O3b --out triggers/
    python scripts/get_omicron_triggers_L1.py --runs O2 O3 O4 --out triggers/
"""

import argparse
from pathlib import Path

from deepextractor.data.omicron import (
    RUN_PERIODS,
    fetch_omicron_triggers,
    save_omicron_triggers,
)

IFO = 'L1'


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        '--runs', nargs='+', required=True,
        choices=sorted(RUN_PERIODS),
        metavar='RUN',
        help=f'Observing run(s) to process. Choices: {sorted(RUN_PERIODS)}',
    )
    p.add_argument(
        '--out', type=Path, default=Path('triggers'),
        help='Output directory for .npy files (default: triggers/)',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for run in args.runs:
        gps_start, gps_end = RUN_PERIODS[run]
        print(f'\n{IFO}  {run}  GPS [{gps_start}, {gps_end})')
        triggers = fetch_omicron_triggers(IFO, gps_start, gps_end)
        print(f'  Loaded {len(triggers["peak_time"])} triggers')
        prefix = f'{IFO.lower()}_{run.lower()}'
        save_omicron_triggers(triggers, prefix, args.out)


if __name__ == '__main__':
    main()
