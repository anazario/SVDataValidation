"""
Stage 5 driver: every plot and comparison of a merged tag, plus an index.html.

  <outdir>/overlays/<group>/   shape overlays (units of the group, ratio to the group total)
  <outdir>/maps/<year>/        2D SV position maps
  <outdir>/rates/<group>/      SV rates per event and per fb^-1 per unit
  <outdir>/trends/<year>/      the same rates run by run
  <outdir>/stats/              comparisons.csv (every 1D histogram, every unit vs the rest of its
                               group) and Kolmogorov-distance maps per group and section
  <outdir>/index.html          summary: checks, cutflow, outlier units, largest flagged differences
"""

from __future__ import annotations

import csv
import html
import json
import os
import time
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import compare as cmp
from . import inputs as inp
from . import plots


# ----------------------------------------------------------------------------- luminosity

def slice_luminosity(merged: inp.Merged, lumi_csv: Optional[Sequence[str]] = None):
    """
    ({slice: fb^-1}, {slice: info}) from brilcalc CSVs (--lumi-csv, else configs/lumi_brilcalc/*.csv),
    or ({}, {}) if there are none. A slice with a run that has no value gets no luminosity.
    """
    from . import lumi as L
    from ..lumimask import LumiMask
    table = L.read_brilcalc(lumi_csv) if lumi_csv else L.load_default()
    if not table:
        return {}, {}
    vals, info = L.slice_lumis(merged.lumis, table, LumiMask.from_dir(), L.golden_ls_per_run())
    ok = {s: v for s, v in vals.items() if v > 0 and not info[s]['runs_without_lumi']}
    return ok, info


def unit_lumis(group: inp.Group, merged: inp.Merged, slice_lumi: Optional[Dict[str, float]]):
    """
    Recorded luminosity (fb^-1) of each unit of a group, or None:
      - with brilcalc values (configs/lumi_brilcalc or --lumi-csv): any group, summed over the unit's
        slices (units with a slice lacking a value are left out)
      - otherwise, groups of whole years (Run 2 / Run 3) from configs/lumi_fb.yaml, and only if every
        catalogued file of each of those years is in the merge (checks.json file_coverage): a test
        tag or a subset would divide part of a year's events by the whole year's luminosity.
        Slices within a year have no luminosity without brilcalc.
    """
    if slice_lumi:
        return {u: sum(slice_lumi[s.name] for s in sl) for u, sl in group.units.items()
                if all(s.name in slice_lumi for s in sl)} or None
    if not group.name.startswith('run') or not {int(u) for u in group.units} <= merged.full_years:
        return None
    from .style import LUMI_FB
    if not all(int(u) in LUMI_FB for u in group.units):
        return None
    return {u: LUMI_FB[int(u)] for u in group.units}


# ----------------------------------------------------------------------------- driver

def sections_of(keys: Sequence[str]) -> "OrderedDict[str, List[str]]":
    """Group histogram keys for the KS maps: event, one per SV / SV-track category, tracks."""
    out = OrderedDict()
    for k in keys:
        p = k.split('/')
        name = p[0] if p[0] in ('event', 'track') else f"{p[0]}_{p[1]}" + (f"_{p[2]}" if len(p) == 4 else '')
        out.setdefault(name, []).append(k)
    return out


def run(merged_dir: str, outdir: str, formats=('png', 'pdf'), all_vars: bool = False,
        years: Optional[Sequence[int]] = None, lumi_csv: Optional[Sequence[str]] = None,
        d_min: float = cmp.D_MIN, p_max: float = cmp.P_MAX, min_run_events: int = 100, log=print) -> str:
    t0 = time.time()
    m = inp.Merged(merged_dir)
    if years:
        m.slices = [s for s in m.slices if s.year in set(years)]
    groups = inp.make_groups(m.slices)
    keys_1d = [k for k in m.keys() if not m.is_2d(k)]
    plot_keys = keys_1d if all_vars else [k for k in plots.CURATED if k in keys_1d]
    log(f"{len(m.slices)} slices, {len(groups)} groups, {len(keys_1d)} 1D histograms "
        f"({len(plot_keys)} plotted); merge {'complete' if m.complete else 'PARTIAL'}")

    os.makedirs(os.path.join(outdir, 'stats'), exist_ok=True)
    painter = plots.Painter(m, outdir, formats)

    # ---- statistics on every 1D histogram
    results: List[cmp.Result] = []
    for g in groups:
        log(f"  comparing {g.name} ({len(g.units)} units)")
        results += cmp.compare_group(m, g, keys_1d, d_min, p_max, log=log)
    cmp.write_csv(os.path.join(outdir, 'stats', 'comparisons.csv'), results)
    by = {(r.group, r.unit, r.key): r for r in results}

    # ---- overlays
    for g in groups:
        log(f"  overlays {g.name}")
        for k in plot_keys:
            ks = {u: by[(g.name, u, k)].ks_d for u in g.units if (g.name, u, k) in by}
            painter.overlay(g, k, ks)
        for sec, ks_keys in sections_of(plot_keys).items():
            painter.ks_map(g, results, ks_keys, sec, d_max=max(0.05, 2.5 * d_min))

    # ---- 2D maps per year
    for year in sorted({s.year for s in m.slices}):
        ys = [s for s in m.slices if s.year == year]
        for k in plots.MAPS_2D + ['sv/badErr/xy', 'sv/badErr/rz']:
            painter.map2d(k, ys, str(year), f"maps/{year}")

    # ---- rates and trends
    lum, lum_info = slice_luminosity(m, lumi_csv)
    if lum_info:
        log(f"  luminosity from brilcalc for {len(lum)} of {len(m.slices)} slices")
    for g in groups:
        per_fb = unit_lumis(g, m, lum)
        for name in inp.RATES:
            painter.rates(g, name)
            if per_fb:
                painter.rates(g, name, lumi_by_unit=per_fb)
    for year in sorted({s.year for s in m.slices}):
        ys = [s for s in m.slices if s.year == year]
        for name in inp.RATES:
            painter.trend(year, ys, name, min_run_events)

    path = write_index(outdir, m, groups, results, painter.written, formats, d_min, p_max, lum, lum_info)
    log(f"done in {time.time() - t0:.0f}s: {sum(len(v) for v in painter.written.values())} plots, "
        f"{sum(r.flagged for r in results)} flagged comparisons of {len(results)}; {path}")
    return path


# ----------------------------------------------------------------------------- index.html

_CSS = """body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:24px;color:#222}
h1{font-size:22px} h2{font-size:18px;margin-top:32px;border-bottom:1px solid #ccc}
h3{font-size:15px;margin-top:20px} table{border-collapse:collapse;font-size:13px;margin:8px 0}
td,th{border:1px solid #ddd;padding:3px 8px;text-align:right} th{background:#f4f4f4}
td.l,th.l{text-align:left} .bad{background:#fde2e1} .ok{background:#e3f4e3}
.gal{display:flex;flex-wrap:wrap;gap:8px} .gal a{display:block;width:330px;font-size:11px;color:#333;
text-decoration:none} .gal img{width:330px;border:1px solid #ddd} details{margin:6px 0}
summary{cursor:pointer;font-weight:600} .note{color:#666;font-size:13px}"""


def _table(headers, rows, left=1):
    h = ''.join(f'<th class="{"l" if i < left else ""}">{html.escape(str(x))}</th>' for i, x in enumerate(headers))
    body = []
    for cls, r in rows:
        tds = ''.join(f'<td class="{"l" if i < left else ""}">{x}</td>' for i, x in enumerate(r))
        body.append(f'<tr class="{cls}">{tds}</tr>')
    return f'<table><tr>{h}</tr>{"".join(body)}</table>'


def write_index(outdir, m, groups, results, written, formats, d_min, p_max, lum, lum_info=None) -> str:
    img = 'png' if 'png' in formats else formats[0]
    vec = 'pdf' if 'pdf' in formats else img
    e = html.escape
    out = [f'<!doctype html><html><head><meta charset="utf-8"><title>SV data validation</title>'
           f'<style>{_CSS}</style></head><body>']
    out.append(f'<h1>SV data validation: {e(os.path.abspath(m.dir))}</h1>')
    out.append('<p class="note">Control regions only: no SV with leptonic d<sub>xy</sub>/&sigma; &ge; 500 or '
               'hadronic &ge; 800 is in any histogram, and events with such an analysis SV are vetoed before '
               'anything is counted. Made ' + time.strftime('%Y-%m-%d %H:%M') + '.</p>')

    # checks
    ch = m.checks
    dup = sum(ch.get('duplicate_lumis', {}).values())
    rows = [('ok' if not ch.get('missing') else 'bad', ['jobs merged', f"{ch.get('merged')} / {ch.get('jobs')}"]),
            ('ok' if dup == 0 else 'bad', ['duplicate lumis (per year and PD)', dup]),
            ('ok' if not ch.get('blinding_audit_failures') else 'bad',
             ['blinding audit failures', len(ch.get('blinding_audit_failures', []))]),
            ('ok' if not ch.get('slices_without_golden_json') else 'bad',
             ['slices without golden JSON', ', '.join(ch.get('slices_without_golden_json', [])) or 0])]
    out.append('<h2>Merge checks</h2>' + _table(['check', 'value'], rows))
    if not m.complete:
        out.append('<p class="note"><b>Partial merge</b>: luminosity labels are omitted and rates cover only '
                   'the merged jobs.</p>')

    # cutflow
    steps = list(next(iter(m.cutflow.values())).keys()) if m.cutflow else []
    rows = []
    for s in m.slices:
        c = m.cutflow.get(s.name, {})
        rows.append(('', [e(s.label)] + [f"{c.get(k, 0):,}" for k in steps]
                     + [f"{lum.get(s.name, 0):.3f}" if lum else '']))
    out.append('<h2>Cutflow per slice</h2><p class="note">Counts start after the blinding veto.</p>'
               + _table(['slice'] + steps + (['L [fb<sup>-1</sup>]'] if lum else ['']), rows))

    # outliers
    summ = cmp.unit_summary(results)
    out.append(f'<h2>Outlier summary</h2><p class="note">Each unit vs the rest of its group, on every 1D '
               f'histogram. Flagged: KS p &lt; {p_max:g} and Kolmogorov distance D &ge; {d_min:g}.</p>')
    for g in groups:
        rows = []
        for u in g.units:
            a = summ.get((g.name, u))
            if not a:
                continue
            rows.append(('bad' if a['flagged'] else '', [e(u), a['n'], a['flagged'], f"{a['max_d']:.3f}",
                                                           e(a['max_key'])]))
        out.append(f'<h3>{e(g.title)}</h3>' + _table(['unit', 'compared', 'flagged', 'max D', 'at'], rows))

    top = sorted((r for r in results if r.flagged), key=lambda r: -r.ks_d)[:60]
    rows = []
    for r in top:
        name = f"overlays/{r.group}/{plots._safe(r.key)}"
        link = f'<a href="{name}.{vec}">{e(r.key)}</a>' if os.path.exists(os.path.join(outdir, f"{name}.{img}")) \
            else e(r.key)
        rows.append(('', [e(r.group), e(r.unit), link, f"{r.ks_d:.3f}", cmp.fmt_p(r.ks_p), f"{r.chi2_ndf:.1f}",
                          f"{r.mean_shift:+.3f}", f"{r.n:,}"]))
    out.append('<h2>Largest flagged differences</h2>'
               + (_table(['group', 'unit', 'histogram', 'D', 'KS p', '&chi;&sup2;/ndf', 'mean shift / RMS', 'N'],
                         rows, left=3) if rows else '<p>none</p>')
               + '<p class="note">All comparisons: <a href="stats/comparisons.csv">stats/comparisons.csv</a>.</p>')

    # galleries
    order = ['stats', 'rates', 'trends', 'maps', 'overlays']
    out.append('<h2>Plots</h2>')
    for top_sec in order:
        secs = sorted(s for s in written if s.split('/')[0] == top_sec)
        for sec in secs:
            names = written[sec]
            items = ''.join(f'<a href="{sec}/{n}.{vec}"><img loading="lazy" src="{sec}/{n}.{img}">'
                            f'{e(n)}</a>' for n in names)
            out.append(f'<details{" open" if top_sec in ("stats", "rates") else ""}><summary>{e(sec)} '
                       f'({len(names)})</summary><div class="gal">{items}</div></details>')
    out.append('</body></html>')
    path = os.path.join(outdir, 'index.html')
    with open(path, 'w') as f:
        f.write('\n'.join(out))
    return path
