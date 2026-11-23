"""
Statistical comparison of units (slices of a year, years of a Run) on every 1D histogram.

Each unit is compared with the *complement* in its group (the rest of the year / of the Run), so a
unit is never compared with a sample that contains it. With these statistics every small drift is
significant, so a comparison is flagged only when it is both significant and large:
    KS p-value < P_MAX   and   Kolmogorov distance D >= D_MIN
D (the largest difference of the two normalized cumulative distributions) is the effect size;
chi2/ndf of the normalized shapes and the mean shift (in RMS of the reference) are reported too.
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional

D_MIN = 0.02
P_MAX = 1e-3
MIN_ENTRIES = 50


@dataclass
class Result:
    group: str
    unit: str
    key: str
    n: int
    n_ref: int
    ks_d: float
    ks_p: float
    chi2_ndf: float
    chi2_p: float
    mean_shift: float
    flagged: bool


def compare_hists(h, ref, d_min: float = D_MIN, p_max: float = P_MAX):
    """Shape comparison of two count histograms; None if either has fewer than MIN_ENTRIES."""
    n, nr = h.GetEntries(), ref.GetEntries()
    if n < MIN_ENTRIES or nr < MIN_ENTRIES or h.Integral() <= 0 or ref.Integral() <= 0:
        return None
    ks_p = h.KolmogorovTest(ref)
    ks_d = h.KolmogorovTest(ref, "M")
    chi2_p = h.Chi2Test(ref, "UU NORM")
    chi2_ndf = h.Chi2Test(ref, "UU NORM CHI2/NDF")
    rms = ref.GetStdDev()
    shift = (h.GetMean() - ref.GetMean()) / rms if rms > 0 else 0.0
    return dict(n=int(n), n_ref=int(nr), ks_d=float(ks_d), ks_p=float(ks_p), chi2_ndf=float(chi2_ndf),
                chi2_p=float(chi2_p), mean_shift=float(shift), flagged=bool(ks_p < p_max and ks_d >= d_min))


def compare_group(merged, group, keys: Iterable[str], d_min: float = D_MIN, p_max: float = P_MAX,
                  log=None) -> List[Result]:
    out = []
    keys = [k for k in keys if not merged.is_2d(k)]
    for i, key in enumerate(keys):
        per_unit = {u: merged.hist(key, sl) for u, sl in group.units.items()}
        for unit, h in per_unit.items():
            if h is None:
                continue
            others = [x for u, x in per_unit.items() if u != unit and x is not None]
            if not others:
                continue
            ref = others[0].Clone(); ref.SetDirectory(0)
            for x in others[1:]:
                ref.Add(x)
            r = compare_hists(h, ref, d_min, p_max)
            if r:
                out.append(Result(group=group.name, unit=unit, key=key, **r))
        if log and (i + 1) % 200 == 0:
            log(f"    {group.name}: {i + 1}/{len(keys)} histograms compared")
    return out


def write_csv(path: str, results: List[Result]):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(Result.__dataclass_fields__))
        w.writeheader()
        for r in results:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in asdict(r).items()})


def unit_summary(results: List[Result]):
    """Per (group, unit): comparisons, flagged count, largest D and where it is."""
    acc = {}
    for r in results:
        a = acc.setdefault((r.group, r.unit), {'n': 0, 'flagged': 0, 'max_d': 0.0, 'max_key': ''})
        a['n'] += 1
        a['flagged'] += r.flagged
        if r.ks_d > a['max_d']:
            a['max_d'], a['max_key'] = r.ks_d, r.key
    return acc


def fmt_p(p: float) -> str:
    if p <= 0:
        return '<1e-300'
    return f"{p:.1e}" if p < 0.01 else f"{p:.2f}"


def finite(x: Optional[float]) -> bool:
    return x is not None and math.isfinite(x)
