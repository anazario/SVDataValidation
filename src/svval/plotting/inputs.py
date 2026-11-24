"""
Reading a merged tag (svval merge): slices, summed histograms, lumi-table rates.

Units of comparison
  slice  the finest unit: '2016F_preVFP', '2018B_HEM', '2024G' (era + detector / HEM period)
  year   the sum of its slices
  run    Run 2 (2016-2018) or Run 3 (2022-), the sum of its years
A Group is a set of units compared with each other: the slices of one year, or the years of a Run.
"""

from __future__ import annotations

import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

_PERIOD_ORDER = {'': 0, 'preVFP': 0, 'postVFP': 1, 'preEE': 0, 'postEE': 1, 'preBPix': 0, 'postBPix': 1}
_HEM_ORDER = {'': 0, 'preHEM': 0, 'HEM': 1}
RUN2_YEARS = (2016, 2017, 2018)


@dataclass(frozen=True)
class Slice:
    name: str
    year: int
    era: str
    period: str = ''
    hem: str = ''

    @property
    def label(self) -> str:
        return ' '.join(x for x in (f"{self.year}{self.era}", self.period, self.hem) if x)

    @property
    def run(self) -> str:
        return 'Run2' if self.year in RUN2_YEARS else 'Run3'

    def sort_key(self):
        return (self.year, self.era, _PERIOD_ORDER.get(self.period, 9), _HEM_ORDER.get(self.hem, 9))


def parse_slice(name: str) -> Slice:
    m = re.fullmatch(r'(20\d\d)([A-Z])((?:_[A-Za-z]+)*)', name)
    if not m:
        raise ValueError(f"not a slice name: '{name}'")
    parts = [p for p in m.group(3).split('_') if p]
    hem = parts.pop() if parts and parts[-1] in ('preHEM', 'HEM') else ''
    if len(parts) > 1:
        raise ValueError(f"not a slice name: '{name}'")
    return Slice(name, int(m.group(1)), m.group(2), parts[0] if parts else '', hem)


@dataclass
class Group:
    """Units (label -> slices) compared with each other."""
    name: str
    title: str
    units: "OrderedDict[str, List[Slice]]" = field(default_factory=OrderedDict)

    @property
    def slices(self) -> List[Slice]:
        return [s for v in self.units.values() for s in v]

    @property
    def years(self) -> List[int]:
        return sorted({s.year for s in self.slices})


def make_groups(slices: Sequence[Slice]) -> List[Group]:
    """One group per year (its slices) and one per Run (its years), skipping groups of one unit."""
    slices = sorted(slices, key=Slice.sort_key)
    groups = []
    for year in sorted({s.year for s in slices}):
        g = Group(f"y{year}", f"{year}: slices")
        for s in slices:
            if s.year == year:
                g.units[s.label] = [s]
        if len(g.units) > 1:
            groups.append(g)
    for run in ('Run2', 'Run3'):
        g = Group(run.lower(), f"{run[:3]} {run[3]}: years")
        for year in sorted({s.year for s in slices if s.run == run}):
            g.units[str(year)] = [s for s in slices if s.year == year]
        if len(g.units) > 1:
            groups.append(g)
    return groups


# ----------------------------------------------------------------------------- merged tag

class Merged:
    def __init__(self, directory: str, datasets_config: Optional[str] = None):
        import ROOT
        self.dir = directory
        self._f = ROOT.TFile.Open(os.path.join(directory, 'all.root'))
        if not self._f or self._f.IsZombie():
            raise OSError(f"cannot open {directory}/all.root")
        self.slices = sorted((parse_slice(k.GetName()) for k in self._f.GetListOfKeys()), key=Slice.sort_key)
        self.by_name = {s.name: s for s in self.slices}
        with open(os.path.join(directory, 'checks.json')) as f:
            self.checks = json.load(f)
        with open(os.path.join(directory, 'cutflow.json')) as f:
            self.cutflow = json.load(f)
        self.complete = not self.checks.get('missing')
        # years whose every catalogued ntuple file is in this merge (merges before file_coverage
        # existed have none: no luminosity is used for them)
        self.full_years = {int(y) for y, c in self.checks.get('file_coverage', {}).items() if c >= 1 - 1e-9}
        self._keys = None
        self._lumis = None
        self._datasets_config = datasets_config

    # ------------------------------------------------------------------ histograms
    def keys(self) -> List[str]:
        """Histogram paths inside a slice (taken from the first slice; all slices share them)."""
        if self._keys is None:
            out = []

            def walk(d, prefix):
                for k in d.GetListOfKeys():
                    obj = k.ReadObj()
                    path = f"{prefix}/{k.GetName()}" if prefix else k.GetName()
                    if obj.InheritsFrom('TDirectory'):
                        walk(obj, path)
                    else:
                        out.append((path, obj.ClassName()))
            walk(self._f.Get(self.slices[0].name), '')
            self._keys = out
        return [k for k, _ in self._keys]

    def is_2d(self, key: str) -> bool:
        self.keys()
        return dict(self._keys)[key].startswith('TH2')

    def hist(self, key: str, slices: Sequence[Slice]):
        """Sum of the histogram over the slices (a detached clone), or None if no slice has entries."""
        out = None
        for s in slices:
            h = self._f.Get(f"{s.name}/{key}")
            if not h:
                continue
            if out is None:
                out = h.Clone(f"{key.replace('/', '_')}__{'_'.join(x.name for x in slices)}"[:200])
                out.SetDirectory(0)
            else:
                out.Add(h)
        if out is None or out.GetEntries() == 0:
            return None
        return out

    # ------------------------------------------------------------------ lumi table
    @property
    def lumis(self) -> np.ndarray:
        if self._lumis is None:
            with np.load(os.path.join(self.dir, 'lumis.npz')) as z:
                t, paths = z['lumis'], [str(p) for p in z['paths']]
            if 'slice' not in t.dtype.names:            # merged before the slice column existed
                t = _add_slice_column(t, paths, self._datasets_config)
            self._lumis = t
        return self._lumis

    def rows(self, slices: Sequence[Slice]) -> np.ndarray:
        names = np.array([s.name for s in slices])
        t = self.lumis
        return t[np.isin(t['slice'], names)]


def _add_slice_column(t, paths, datasets_config):
    """Derive the slice of each row from year / era / run and the file path (HIPM), as fill does."""
    import numpy.lib.recfunctions as rfn
    from .. import catalog as cat
    from ..fill import slice_names
    cfg = cat.load_config(datasets_config or os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                                          'configs', 'datasets_v34.yaml'))
    names = np.empty(len(t), dtype='U24')
    keys = np.stack([t['path'], t['year'].astype(np.int64)], 1)
    for (pi, year), idx in _group_indices(keys).items():
        era = str(t['era'][idx[0]])
        period = cat.period_of(cfg, int(year), era, 'HIPM' in paths[pi])
        names[idx] = slice_names(int(year), era, period, t['run'][idx])
    return rfn.append_fields(t, 'slice', names, usemask=False)


def _group_indices(keys: np.ndarray) -> Dict[tuple, np.ndarray]:
    uk, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    order = np.argsort(inv, kind='stable')
    splits = np.split(order, np.cumsum(np.bincount(inv))[:-1])
    return {tuple(int(x) for x in k): s for k, s in zip(uk, splits)}


# ----------------------------------------------------------------------------- rates

# name -> (numerator, denominator, kind, label); kind 'count' (Poisson numerator) or 'frac' (binomial)
RATES = OrderedDict([
    ('sv_per_event', ('n_sv', 'n_sel', 'count', 'SVs per event')),
    ('lep2_mu_per_event', ('n_lep2_mu_base', 'n_sel', 'count', '#mu#mu SVs (baseline) per event')),
    ('lep2_e_per_event', ('n_lep2_e_base', 'n_sel', 'count', 'ee SVs (baseline) per event')),
    ('lep2_both_per_event', ('n_lep2_both_base', 'n_sel', 'count', '#mu+e ID SVs (baseline) per event')),
    ('lep2_noid_per_event', ('n_lep2_noid_base', 'n_sel', 'count', '2-track no-ID SVs (baseline) per event')),
    ('lep2_all_per_event', ('n_lep2_all', 'n_sel', 'count', '2-track SVs per event')),
    ('trk3_per_event', ('n_trk3_base', 'n_sel', 'count', '3-track SVs (baseline) per event')),
    ('had_id_per_event', ('n_had_id_base', 'n_sel', 'count', 'hadronic ID SVs (baseline) per event')),
    ('had_noid_per_event', ('n_had_noid_base', 'n_sel', 'count', 'hadronic no-ID SVs (baseline) per event')),
    ('had_all_per_event', ('n_had_all', 'n_sel', 'count', '#geq4-track SVs per event')),
    ('badErr_frac', ('n_badErr', 'n_sv_plus_badErr', 'frac', 'fraction of SVs with invalid #sigma_{d_{xy}}')),
    ('negCov_frac', ('n_negCov', 'n_sv', 'frac', 'fraction of SVs with C_{ii} < 0')),
    ('shared_per_sv', ('n_vtrk_shared', 'n_sv', 'count', 'shared SV tracks per SV')),
    ('fakePV_frac', ('n_fakePV', 'n_sel', 'frac', 'fraction of events with N_{PV} = 1')),
    ('mean_npv', ('sum_npv', 'n_sel', 'mean', '#LT N_{PV} #GT')),
])


def _column(t, name):
    if name == 'n_sv_plus_badErr':
        return t['n_sv'].astype(np.float64) + t['n_badErr']
    return t[name].astype(np.float64)


def rate(t: np.ndarray, name: str):
    """(value, error) of a RATES entry over the rows t; (nan, nan) if the denominator is 0."""
    num, den, kind, _ = RATES[name]
    n, d = _column(t, num).sum(), _column(t, den).sum()
    if d <= 0:
        return float('nan'), float('nan')
    v = n / d
    if kind == 'count':
        e = np.sqrt(n) / d
    elif kind == 'frac':
        e = np.sqrt(max(v * (1 - v), 0) / d)
    else:
        e = 0.0
    return float(v), float(e)


def per_run(t: np.ndarray, name: str, min_events: int = 100):
    """Runs with at least min_events selected events -> (runs, values, errors)."""
    runs = np.unique(t['run'])
    out_r, out_v, out_e = [], [], []
    order = np.argsort(t['run'], kind='stable')
    ts = t[order]
    bounds = np.searchsorted(ts['run'], runs, side='left')
    ends = np.searchsorted(ts['run'], runs, side='right')
    for r, a, b in zip(runs, bounds, ends):
        rows = ts[a:b]
        if rows['n_sel'].sum() < min_events:
            continue
        v, e = rate(rows, name)
        if np.isfinite(v):
            out_r.append(int(r)); out_v.append(v); out_e.append(e)
    return np.array(out_r), np.array(out_v), np.array(out_e)
