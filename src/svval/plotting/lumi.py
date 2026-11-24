"""
Recorded luminosity of the slices, from brilcalc CSV files computed once and committed in
configs/lumi_brilcalc/ (see tools/make_lumi_csv.sh), so brilcalc is never needed at plot time.

Two brilcalc outputs are understood (the format is recognised from the CSV header):
  per run   brilcalc lumi       ... --output-style csv   '#run:fill,time,nls,ncms,delivered(/fb),recorded(/fb)'
  per LS    brilcalc lumi --byls ... --output-style csv  '#run:fill,ls,time,beamstatus,E(GeV),delivered,recorded,...'
Any unit in the header (/fb, /pb, /nb, /ub) is converted to fb^-1. The '#Summary' block is ignored.

Per-run values are matched to what was actually processed: a run's luminosity is scaled by the
fraction of its golden lumi sections present in the merge (CRAB failures, missing jobs), so the
denominator always matches the events counted. With per-LS values the present sections are summed.
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

DEFAULT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'configs', 'lumi_brilcalc')
GOLDEN_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'configs', 'golden_json')
_TO_FB = {'/fb': 1.0, '/pb': 1e-3, '/nb': 1e-6, '/ub': 1e-9}


@dataclass
class LumiTable:
    by_run: Dict[int, float] = field(default_factory=dict)          # fb^-1
    by_ls: Dict[Tuple[int, int], float] = field(default_factory=dict)
    files: list = field(default_factory=list)

    def __bool__(self):
        return bool(self.by_run or self.by_ls)


def read_brilcalc(paths: Iterable[str]) -> LumiTable:
    out = LumiTable()
    for path in paths:
        out.files.append(os.path.basename(path))
        cols, scale, i_rec, byls = None, None, None, False
        with open(path) as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s.startswith('#'):
                    h = s.lstrip('#').strip().split(',')
                    if h[0].startswith('run'):
                        cols = h
                        rec = next((c for c in h if c.startswith('recorded')), None)
                        if rec is None:
                            raise ValueError(f"{path}: no 'recorded' column in {h}")
                        unit = rec[rec.find('(') + 1:rec.find(')')] if '(' in rec else '/ub'
                        if unit not in _TO_FB:
                            raise ValueError(f"{path}: unknown luminosity unit '{unit}'")
                        scale, i_rec, byls = _TO_FB[unit], h.index(rec), (len(h) > 1 and h[1] == 'ls')
                    else:
                        cols = None          # '#Summary:' and its own header: stop reading rows
                    continue
                if cols is None:
                    continue
                p = s.split(',')
                run = int(p[0].split(':')[0])
                val = float(p[i_rec]) * scale
                if byls:
                    out.by_ls[(run, int(p[1].split(':')[0]))] = val
                else:
                    out.by_run[run] = out.by_run.get(run, 0.0) + val
    return out


def load_default(directory: Optional[str] = None) -> LumiTable:
    d = os.path.abspath(directory or DEFAULT_DIR)
    files = sorted(glob.glob(os.path.join(d, '*.csv')))
    return read_brilcalc(files) if files else LumiTable()


def golden_ls_per_run(directory: Optional[str] = None) -> Dict[int, int]:
    d = os.path.abspath(directory or GOLDEN_DIR)
    out = defaultdict(int)
    for path in sorted(glob.glob(os.path.join(d, '*.json')) + glob.glob(os.path.join(d, '*.txt'))):
        with open(path) as f:
            for run, ranges in json.load(f).items():
                out[int(run)] += sum(b - a + 1 for a, b in ranges)
    return dict(out)


def slice_lumis(t: np.ndarray, table: LumiTable, golden_mask, golden_counts: Dict[int, int]):
    """
    t: merged lumi table (slice, run, lumi columns; one row per file and lumi section read).
    Returns ({slice: fb^-1}, {slice: info}) with info = present / golden LS, runs without a value.
    """
    good = golden_mask(t['run'], t['lumi'])
    keys = np.unique(np.stack([t['run'][good].astype(np.int64), t['lumi'][good].astype(np.int64)], 1), axis=0) \
        if good.any() else np.zeros((0, 2), np.int64)
    # one slice per (run, lumi): slices are split by run, so take the slice of the first row
    first = {}
    for s, r, l in zip(t['slice'][good], t['run'][good], t['lumi'][good]):
        first.setdefault((int(r), int(l)), str(s))
    present = defaultdict(lambda: defaultdict(int))                   # slice -> run -> n present LS
    lumi = defaultdict(float)
    for r, l in keys:
        s = first[(int(r), int(l))]
        present[s][int(r)] += 1
        if table.by_ls:
            lumi[s] += table.by_ls.get((int(r), int(l)), 0.0)
    info = {}
    for s, runs in present.items():
        missing_runs = []
        n_present = sum(runs.values())
        n_golden = sum(golden_counts.get(r, 0) for r in runs)
        if not table.by_ls:
            for r, n in runs.items():
                if r in table.by_run and golden_counts.get(r):
                    lumi[s] += table.by_run[r] * n / golden_counts[r]
                else:
                    missing_runs.append(r)
        info[s] = {'ls_present': n_present, 'ls_golden': n_golden, 'runs_without_lumi': sorted(missing_runs)}
    return dict(lumi), info
