"""
Drawing: shape overlays with ratio pads, 2D maps, rates per unit, per-run trends, KS-distance maps.
All in the LLPStandardPlots style (style.py).
"""

from __future__ import annotations

import math
import os
from array import array
from typing import Dict, List, Optional, Sequence

import numpy as np
import ROOT

from . import inputs as inp
from .style import LUMI_FB, Style, energy_of

# --------------------------------------------------------------------------- curated variables

_SV_CURATED = ['nTracks', 'mass', 'mOverN', 'pt', 'eta', 'phi', 'pOverE', 'r', 'dxy', 'dxyError', 'dxySig',
               'normChi2', 'cosTheta', 'decayAngle', 'nSharedTracks', 'isUnique', 'negCovDiag']
CURATED = (
    [f'event/{v}' for v in ('PV_npvs', 'PV_x', 'PV_y', 'PV_z', 'BS_x', 'BS_y', 'PV_BS_dxy', 'fakePV', 'selCMet',
                            'nTrack', 'nSV', 'nSVbadErr', 'nSV_lep2_mu_base', 'nSV_lep2_e_base',
                            'nSV_had_id_base')]
    + [f'sv/{c}/{lvl}/{v}' for c, lvl in (('lep2_mu', 'base'), ('lep2_e', 'base'), ('lep2_noid', 'all'),
                                          ('trk3', 'all'), ('had_id', 'base'), ('had', 'all'))
       for v in _SV_CURATED]
    + [f'sv/badErr/{v}' for v in ('nTracks', 'r', 'normChi2', 'negCovDiag')]
    + [f'vtrk/{c}/{lvl}/{v}' for c, lvl in (('lep2', 'all'), ('had_id', 'base'))
       for v in ('cosThetaCM', 'compatibility', 'shift3D', 'hasSC', 'pOverESC', 'isShared')]
    + [f'track/{c}/{v}' for c in ('all', 'hyddraLike')
       for v in ('ptLog', 'eta', 'phi', 'ip2D', 'sip2D', 'ipValid', 'nValidHits', 'nLostHits', 'normChi2',
                 'relPtErr', 'highPurity')]
)
MAPS_2D = [f'sv/{c}/{lvl}/{v}' for c in ('lep2', 'had') for lvl in ('all', 'base') for v in ('xy', 'xyInner', 'rz')]
LOGY = ('dxySig', 'dxy', 'r', 'mass', 'pt', 'nTracks', 'chi2', 'normChi2', 'sip2D', 'ip2D', 'compatibility',
        'shift3D', 'pOverESC', 'PV_BS_dxy', 'selCMet', 'nSV', 'nSVbadErr', 'nTrack', 'mOverN', 'dxyError',
        'relPtErr', 'PV_npvs', 'nSV_lep2_mu_base', 'nSV_lep2_e_base', 'nSV_had_id_base')

CATEGORY_LABELS = {
    'lep2': '2-track SVs', 'lep2_mu': '#mu#mu SVs', 'lep2_e': 'ee SVs', 'lep2_both': '#mu+e ID SVs',
    'lep2_noid': '2-track no-ID SVs', 'trk3': '3-track SVs', 'had': '#geq4-track SVs',
    'had_id': 'hadronic SVs (m/N > 1)', 'had_noid': '#geq4-track SVs (m/N #leq 1)', 'badErr': 'SVs, invalid #sigma_{d_{xy}}',
}


def key_label(key: str) -> str:
    """Short region label of a histogram key, e.g. 'hadronic SVs (m/N > 1), baseline'."""
    p = key.split('/')
    if p[0] in ('sv', 'vtrk') and len(p) >= 3:
        cat = CATEGORY_LABELS.get(p[1], p[1])
        lvl = {'all': '', 'base': ', baseline'}.get(p[2], '') if len(p) == 4 else ''
        return ('tracks of ' if p[0] == 'vtrk' else '') + cat + lvl
    if p[0] == 'track':
        return 'Track_ (all)' if p[1] == 'all' else 'Track_ (HYDDRA-like)'
    return 'selected events'


def _safe(name: str) -> str:
    return name.replace('/', '__').replace(' ', '_')


def _lumi_for(years: Sequence[int], full_years) -> Optional[float]:
    """Header luminosity: only when every year shown is fully merged (else it would mislabel)."""
    if not set(years) <= set(full_years):
        return None
    try:
        return sum(LUMI_FB[y] for y in years)
    except KeyError:
        return None


class Painter:
    def __init__(self, merged: inp.Merged, outdir: str, formats=('png', 'pdf')):
        self.m = merged
        self.out = outdir
        self.formats = formats
        self.style = Style()
        self.written: Dict[str, List[str]] = {}

    def _save(self, canvas, section: str, name: str):
        d = os.path.join(self.out, section)
        os.makedirs(d, exist_ok=True)
        for fmt in self.formats:
            canvas.SaveAs(os.path.join(d, f"{name}.{fmt}"))
        self.written.setdefault(section, []).append(name)

    def _header(self, years, size_mult=1.0, cms_y=0.955, lumi_x=0.9, prelim_x=0.248, cms_x=0.16):
        return self.style.cms_labels(cms_x=cms_x, cms_y=cms_y, prelim_x=prelim_x, lumi_x=lumi_x,
                                     lumi_fb=_lumi_for(years, self.m.full_years), energy=energy_of(years),
                                     size_mult=size_mult)

    # ------------------------------------------------------------------ shape overlays
    def overlay(self, group: inp.Group, key: str, ks: Optional[Dict[str, float]] = None):
        hists = OrderedHists()
        for u, sl in group.units.items():
            h = self.m.hist(key, sl)
            if h is not None and h.Integral() > 0:      # entries only in under/overflow: nothing to show
                hists[u] = h
        if len(hists) < 2:
            return False
        total = None
        for h in hists.values():
            if total is None:
                total = h.Clone(); total.SetDirectory(0)
            else:
                total.Add(h)
        var = key.rsplit('/', 1)[-1]
        logy = var in LOGY
        c, p1, p2 = self.style.canvas_ratio(f"c_{group.name}_{_safe(key)}", logy=logy)
        p1.cd()
        leg = self.style.legend(0.765, 0.2, 0.995, 0.88, size=0.036 if len(hists) <= 6 else 0.03)
        ymax, ymin = 0.0, float('inf')
        norm = OrderedHists()
        for i, (u, h) in enumerate(hists.items()):
            hn = h.Clone(); hn.SetDirectory(0)
            hn.Scale(1.0 / hn.Integral())
            col = self.style.color(i)
            hn.SetLineColor(col); hn.SetMarkerColor(col); hn.SetMarkerStyle(self.style.marker(i))
            hn.SetMarkerSize(0.9); hn.SetLineWidth(2)
            norm[u] = hn
            ymax = max(ymax, hn.GetMaximum())
            for b in range(1, hn.GetNbinsX() + 1):
                if hn.GetBinContent(b) > 0:
                    ymin = min(ymin, hn.GetBinContent(b))
            extra = f"  D {ks[u]:.3f}" if ks and u in ks else ''
            leg.AddEntry(hn, f"{u}{extra}", "lp")
        first = True
        for hn in norm.values():
            hn.Draw("E1 X0" if first else "E1 X0 SAME")
            if first:
                self.style.axes(hn, xtitle='', ytitle='fraction of entries')
                hn.GetXaxis().SetLabelSize(0)
                if logy:
                    hn.SetMinimum(max(ymin * 0.5, 1e-7)); hn.SetMaximum(ymax * 30)
                else:
                    hn.SetMinimum(0); hn.SetMaximum(ymax * 1.35)
            first = False
        leg.Draw()
        hdr = self._header(group.years, size_mult=1.25, cms_y=0.92, lumi_x=0.75, prelim_x=0.235)
        lab = self.style.label(f"{group.title}; {key_label(key)}", 0.205, 0.83, size=0.045)

        p2.cd()
        tn = total.Clone(); tn.SetDirectory(0); tn.Scale(1.0 / tn.Integral())
        ratios = []
        for i, (u, hn) in enumerate(norm.items()):
            r = hn.Clone(); r.SetDirectory(0); r.Divide(tn)
            ratios.append(r)
            r.Draw("E1 X0" if i == 0 else "E1 X0 SAME")
            if i == 0:
                self.style.axes(r, xtitle=hists[u].GetXaxis().GetTitle(), ytitle='unit / total', ratio_pad=True)
                r.SetMinimum(0.5); r.SetMaximum(1.5)
        one = ROOT.TLine(tn.GetXaxis().GetXmin(), 1, tn.GetXaxis().GetXmax(), 1)
        one.SetLineStyle(2); one.Draw()
        c._keep = (hists, norm, total, tn, ratios, leg, hdr, lab, one)
        self._save(c, f"overlays/{group.name}", _safe(key))
        return True

    # ------------------------------------------------------------------ 2D maps
    def map2d(self, key: str, slices: Sequence[inp.Slice], title: str, section: str):
        h = self.m.hist(key, slices)
        if h is None:
            return False
        c = self.style.canvas_2d(f"c2_{_safe(title)}_{_safe(key)}")
        h.SetMinimum(0.5)
        h.Draw("COLZ")
        h.GetZaxis().SetTitle("SVs"); h.GetZaxis().SetTitleOffset(1.2)
        for ax in (h.GetXaxis(), h.GetYaxis()):
            ax.SetTitleSize(0.045); ax.SetLabelSize(0.04); ax.CenterTitle(True)
        years = sorted({s.year for s in slices})
        hdr = self._header(years, cms_x=0.13, cms_y=0.945, prelim_x=0.235, lumi_x=0.84)
        lab = self.style.label(f"{title}; {key_label(key)}", 0.15, 0.885, size=0.035)
        c._keep = (h, hdr, lab)
        self._save(c, section, _safe(f"{title}_{key}"))
        return True

    # ------------------------------------------------------------------ rates per unit
    def rates(self, group: inp.Group, name: str, lumi_by_unit: Optional[Dict[str, float]] = None):
        """One point per unit. With lumi_by_unit (unit label -> fb^-1), the denominator is the luminosity."""
        num, den, kind, label = inp.RATES[name]
        units = list(group.units.items())
        vals, errs = [], []
        for u, sl in units:
            rows = self.m.rows(sl)
            if lumi_by_unit is not None:
                if kind != 'count' or den != 'n_sel':
                    return False
                lum = lumi_by_unit.get(u, 0.0)
                n = rows[num].sum() if len(rows) else 0
                v, e = (n / lum, math.sqrt(n) / lum) if lum > 0 else (float('nan'), float('nan'))
            else:
                v, e = inp.rate(rows, name) if len(rows) else (float('nan'), float('nan'))
            vals.append(v); errs.append(e)
        ok = [i for i, v in enumerate(vals) if np.isfinite(v)]
        if len(ok) < 1:
            return False
        n = len(units)
        frame = ROOT.TH1D(f"fr_{group.name}_{name}_{lumi_by_unit is not None}", "", n, 0, n)
        frame.SetDirectory(0)
        for i, (u, _) in enumerate(units):
            frame.GetXaxis().SetBinLabel(i + 1, u)
        g = ROOT.TGraphErrors(len(ok), array('d', [i + 0.5 for i in ok]), array('d', [vals[i] for i in ok]),
                              array('d', [0.0] * len(ok)), array('d', [errs[i] for i in ok]))
        lo = min(vals[i] - errs[i] for i in ok)
        hi = max(vals[i] + errs[i] for i in ok)
        pad = (hi - lo) * 0.25 if hi > lo else abs(hi) * 0.1 + 1e-9
        c = self.style.canvas_1d(f"cr_{group.name}_{name}_{lumi_by_unit is not None}")
        c.SetBottomMargin(0.2 if n <= 4 else 0.3)
        ytitle = label.replace('per event', 'per fb^{-1}') if lumi_by_unit is not None else label
        frame.SetMinimum(max(lo - pad, 0) if lo >= 0 else lo - pad); frame.SetMaximum(hi + pad)
        frame.Draw("AXIS")
        self.style.axes(frame, xtitle='', ytitle=ytitle)
        frame.GetXaxis().SetLabelSize(0.045 if n <= 10 else 0.035)
        frame.GetXaxis().LabelsOption("v" if n > 4 else "h")
        g.SetMarkerStyle(20); g.SetMarkerSize(1.3); g.SetLineWidth(2)
        g.Draw("P SAME")
        hdr = self._header(group.years)
        lab = self.style.label(group.title, 0.2, 0.87, size=0.04)
        c._keep = (frame, g, hdr, lab)
        fname = name.replace('_per_event', '') + '_per_fb' if lumi_by_unit is not None else name
        self._save(c, f"rates/{group.name}", fname)
        return True

    # ------------------------------------------------------------------ per-run trends
    def trend(self, year: int, slices: Sequence[inp.Slice], name: str, min_events: int = 100):
        rows = self.m.rows(slices)
        if not len(rows):
            return False
        runs, vals, errs = inp.per_run(rows, name, min_events)
        if len(runs) < 2:
            return False
        _, _, _, label = inp.RATES[name]
        g = ROOT.TGraphErrors(len(runs), array('d', runs.astype(float)), array('d', vals),
                              array('d', [0.0] * len(runs)), array('d', errs))
        c = self.style.canvas_1d(f"ct_{year}_{name}", grid=True)
        c.SetCanvasSize(1100, 600)
        lo, hi = np.min(vals - errs), np.max(vals + errs)
        med = float(np.median(vals))
        # robust range: most runs, not the worst outlier
        q1, q9 = np.quantile(vals, [0.02, 0.98])
        span = max(q9 - q1, abs(med) * 0.05, 1e-9)
        ylo, yhi = q1 - span * 0.6, q9 + span * 0.9
        if lo >= 0:
            ylo = max(ylo, 0)
        x0, x1 = float(runs.min()) - 0.5, float(runs.max()) + 0.5
        frame = ROOT.TH1D(f"ft_{year}_{name}", "", 100, x0, x1); frame.SetDirectory(0)
        frame.SetMinimum(ylo); frame.SetMaximum(yhi)
        frame.Draw("AXIS")
        self.style.axes(frame, xtitle='run', ytitle=label)
        frame.GetXaxis().SetNoExponent(True); frame.GetXaxis().SetNdivisions(508)
        frame.GetYaxis().SetTitleOffset(1.0)
        # era / slice boundaries
        lines, labels = [], []
        for i_s, s in enumerate(sorted(slices, key=inp.Slice.sort_key)):
            r = rows[rows['slice'] == s.name]['run']
            if not len(r):
                continue
            a = float(r.min()) - 0.5
            ln = ROOT.TLine(a, ylo, a, yhi); ln.SetLineStyle(2); ln.SetLineColor(ROOT.kGray + 2); ln.Draw()
            lines.append(ln)
            t = ROOT.TLatex(a, yhi - (yhi - ylo) * (0.06 + 0.05 * (i_s % 2)),
                            f" {s.era}{(' ' + s.hem) if s.hem else ''}")
            t.SetTextSize(0.03); t.SetTextFont(42); t.SetTextColor(ROOT.kGray + 2); t.Draw()
            labels.append(t)
        g.SetMarkerStyle(20); g.SetMarkerSize(0.6); g.Draw("P SAME")
        n_out = int(np.sum((vals < ylo) | (vals > yhi)))
        hdr = self._header([year], cms_y=0.945, lumi_x=0.88)
        lab = self.style.label(f"{year}, runs with #geq {min_events} selected events"
                               + (f"; {n_out} runs outside the y range" if n_out else ''), 0.2, 0.87, size=0.035)
        c._keep = (g, frame, lines, labels, hdr, lab)
        self._save(c, f"trends/{year}", name)
        return True

    # ------------------------------------------------------------------ KS-distance map
    def ks_map(self, group: inp.Group, results, keys: Sequence[str], section_name: str, d_max: float = 0.1):
        res = {(r.unit, r.key): r for r in results if r.group == group.name}
        keys = [k for k in keys if any((u, k) in res for u in group.units)]
        units = list(group.units)
        if not keys or not units:
            return False
        h = ROOT.TH2D(f"ks_{group.name}_{section_name}", "", len(keys), 0, len(keys), len(units), 0, len(units))
        h.SetDirectory(0)
        prefix = os.path.commonprefix(keys)
        prefix = prefix[:prefix.rfind('/') + 1]
        for i, k in enumerate(keys):
            h.GetXaxis().SetBinLabel(i + 1, k[len(prefix):])
        for j, u in enumerate(units):
            h.GetYaxis().SetBinLabel(j + 1, u)
            for i, k in enumerate(keys):
                r = res.get((u, k))
                if r is not None:
                    h.SetBinContent(i + 1, j + 1, max(min(r.ks_d, d_max), 1e-6))
        w = max(900, 32 * len(keys) + 300)
        c = ROOT.TCanvas(f"cks_{group.name}_{section_name}", "", w, max(500, 45 * len(units) + 330))
        c.SetLeftMargin(0.16 * 900 / w + 0.02); c.SetRightMargin(0.13 * 900 / w); c.SetBottomMargin(0.25)
        c.SetTopMargin(0.13)
        h.SetMinimum(0); h.SetMaximum(d_max)
        h.GetXaxis().LabelsOption("v"); h.GetXaxis().SetLabelSize(0.035); h.GetYaxis().SetLabelSize(0.04)
        h.GetZaxis().SetTitle("Kolmogorov distance D (unit vs rest)"); h.GetZaxis().SetTitleOffset(1.0)
        h.GetZaxis().SetTitleSize(0.035); h.GetZaxis().SetLabelSize(0.03); h.GetZaxis().SetNdivisions(505)
        h.GetZaxis().SetDecimals(True)
        h.Draw("COLZ")
        marks = []
        for (u, k), r in res.items():
            if r.flagged and k in keys:
                m = ROOT.TMarker(keys.index(k) + 0.5, units.index(u) + 0.5, 29)
                m.SetMarkerColor(ROOT.kRed); m.SetMarkerSize(1.0); m.Draw(); marks.append(m)
        hdr = self.style.cms_labels(cms_x=c.GetLeftMargin(), cms_y=0.94, prelim_x=c.GetLeftMargin() + 0.075 * 900 / w + 0.02,
                                    lumi_x=1 - c.GetRightMargin(), lumi_fb=_lumi_for(group.years, self.m.full_years),
                                    energy=energy_of(group.years), size_mult=0.8)
        lab = self.style.label(f"{group.title}; {key_label(keys[0])}" + (f" [{prefix.rstrip('/')}]" if prefix else '')
                               + "; red stars: flagged", c.GetLeftMargin(), 0.885, size=0.03)
        c._keep = (h, marks, hdr, lab)
        self._save(c, f"stats/{group.name}", f"ksmap_{section_name}")
        return True


class OrderedHists(dict):
    """dict keeps insertion order on python >= 3.7; named for readability."""
