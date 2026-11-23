"""
Slim copy of the LLPStandardPlots style (src/style.py, src/plotter.py at 05cb711), so this package
has no dependency on it: the CMS style (the fallback of cmsstyle.setCMSStyle(), since the CMSSW
python has no cmsstyle), the CMS / Preliminary / luminosity labels, the colour palette, the
1D canvas and the main + ratio pad layout of the Data/MC plots.
"""

from __future__ import annotations

import os
from typing import List, Optional

import ROOT

# palette of LLPStandardPlots/rootlogon.C, then its StyleManager colours
HEX_COLORS = ["#5A4484", "#347889", "#F4B240", "#E54B26", "#C05780", "#7A68A6", "#2E8B57", "#8B4513"]
EXTRA_COLORS = [ROOT.kBlue - 4, ROOT.kGreen - 3, ROOT.kPink - 8, ROOT.kOrange + 7, ROOT.kMagenta - 2,
                ROOT.kAzure + 2, ROOT.kGray + 2, ROOT.kBlack]
MARKERS = [20, 21, 22, 23, 33, 34, 29, 47, 24, 25, 26, 32]

LUMI_FILE = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'configs', 'lumi_fb.yaml')


def load_lumi_fb(path: str = LUMI_FILE) -> dict:
    """Integrated luminosity (fb^-1) per year from configs/lumi_fb.yaml."""
    import yaml
    with open(path) as f:
        return {int(k): float(v) for k, v in yaml.safe_load(f).items()}


LUMI_FB = load_lumi_fb()
RUN2_YEARS = (2016, 2017, 2018)


def energy_of(years) -> str:
    years = set(years)
    if years <= set(RUN2_YEARS):
        return "13 TeV"
    if not years & set(RUN2_YEARS):
        return "13.6 TeV"
    return "13 + 13.6 TeV"


class Style:
    """StyleManager of LLPStandardPlots, reduced to what the validation plots use."""

    cms_text_size = 0.052
    prelim_text_size = 0.04
    lumi_text_size = 0.04
    label_size = 0.04
    axis_title_offset = 1.3
    axis_title_size = 0.045
    axis_label_size = 0.04
    canvas_width = 800
    canvas_height = 600
    margin_left = 0.12
    margin_right = 0.12
    margin_right_ratio = 0.25

    def __init__(self):
        self._colors = [ROOT.TColor.GetColor(h) for h in HEX_COLORS] + EXTRA_COLORS
        self.apply()

    def color(self, i: int) -> int:
        return self._colors[i % len(self._colors)]

    def marker(self, i: int) -> int:
        return MARKERS[i % len(MARKERS)]

    # ---------------------------------------------------------------- global style
    def apply(self):
        """setCMSStyle() of cmsstyle, as replicated in LLPStandardPlots' StyleManager fallback."""
        s = ROOT.TStyle("cmsStyle", "Style for P-CMS")
        s.SetCanvasBorderMode(0); s.SetCanvasColor(ROOT.kWhite)
        s.SetCanvasDefH(600); s.SetCanvasDefW(600); s.SetCanvasDefX(0); s.SetCanvasDefY(0)
        s.SetPadBorderMode(0); s.SetPadColor(ROOT.kWhite); s.SetPadGridX(False); s.SetPadGridY(False)
        s.SetGridColor(0); s.SetGridStyle(3); s.SetGridWidth(1)
        s.SetFrameBorderMode(0); s.SetFrameBorderSize(1); s.SetFrameFillColor(0); s.SetFrameFillStyle(0)
        s.SetFrameLineColor(1); s.SetFrameLineStyle(1); s.SetFrameLineWidth(1)
        s.SetHistLineColor(1); s.SetHistLineStyle(0); s.SetHistLineWidth(1)
        s.SetEndErrorSize(2); s.SetMarkerStyle(20); s.SetMarkerSize(1)
        s.SetOptFit(1); s.SetFitFormat("5.4g"); s.SetFuncColor(2); s.SetFuncStyle(1); s.SetFuncWidth(1)
        s.SetOptDate(0)
        s.SetLegendTextSize(0.04); s.SetLegendFont(42); s.SetLegendBorderSize(0); s.SetLegendFillColor(0)
        s.SetOptFile(0); s.SetOptStat(0); s.SetStatColor(ROOT.kWhite); s.SetStatFont(42)
        s.SetStatFontSize(0.025); s.SetStatTextColor(1); s.SetStatFormat("6.4g"); s.SetStatBorderSize(1)
        s.SetStatH(0.1); s.SetStatW(0.15)
        s.SetPadTopMargin(0.05); s.SetPadBottomMargin(0.13); s.SetPadLeftMargin(0.16); s.SetPadRightMargin(0.02)
        s.SetOptTitle(0); s.SetTitleFont(42); s.SetTitleColor(1); s.SetTitleTextColor(1)
        s.SetTitleFillColor(10); s.SetTitleFontSize(0.05)
        s.SetTitleColor(1, "XYZ"); s.SetTitleFont(42, "XYZ"); s.SetTitleSize(0.06, "XYZ")
        s.SetTitleXOffset(1.1); s.SetTitleYOffset(1.35)
        s.SetLabelColor(1, "XYZ"); s.SetLabelFont(42, "XYZ"); s.SetLabelOffset(0.012, "XYZ")
        s.SetLabelSize(0.05, "XYZ")
        s.SetAxisColor(1, "XYZ"); s.SetStripDecimals(True); s.SetTickLength(0.03, "XYZ")
        s.SetNdivisions(510, "XYZ"); s.SetPadTickX(1); s.SetPadTickY(1)
        s.SetOptLogx(0); s.SetOptLogy(0); s.SetOptLogz(0)
        s.SetPaperSize(20.0, 20.0); s.SetHatchesLineWidth(2); s.SetHatchesSpacing(1.3)
        s.SetPalette(ROOT.kViridis)
        s.cd()
        ROOT.gROOT.SetStyle(s.GetName())
        ROOT.gROOT.ForceStyle()
        ROOT.gROOT.SetBatch(True)
        ROOT.gErrorIgnoreLevel = ROOT.kWarning
        self._style = s

    # ---------------------------------------------------------------- labels
    def cms_labels(self, cms_x=0.16, cms_y=0.955, prelim="Preliminary", prelim_x=0.248, lumi_x=0.9,
                   lumi_fb: Optional[float] = None, energy: str = "13.6 TeV", size_mult=1.0):
        """StyleManager.draw_cms_labels: 'CMS' bold, 'Preliminary' italic, '<L> fb^{-1} (<E>)' right."""
        objs = []
        t = ROOT.TLatex(); t.SetNDC(); t.SetTextFont(61); t.SetTextSize(self.cms_text_size * size_mult)
        t.DrawLatex(cms_x, cms_y, "CMS"); objs.append(t)
        p = ROOT.TLatex(); p.SetNDC(); p.SetTextFont(52); p.SetTextSize(self.prelim_text_size * size_mult)
        p.DrawLatex(prelim_x, cms_y, prelim); objs.append(p)
        l = ROOT.TLatex(); l.SetNDC(); l.SetTextFont(42); l.SetTextSize(self.lumi_text_size * size_mult)
        l.SetTextAlign(31)
        if lumi_fb:
            txt = f"{lumi_fb:.1f} fb^{{-1}} ({energy})"
        else:
            txt = f"({energy})"
        l.DrawLatex(lumi_x, cms_y, txt); objs.append(l)
        return objs

    def label(self, text, x, y, size=None, align=11, font=42):
        t = ROOT.TLatex(); t.SetNDC(); t.SetTextFont(font); t.SetTextSize(size or self.label_size)
        t.SetTextAlign(align); t.DrawLatex(x, y, text)
        return t

    def legend(self, x1, y1, x2, y2, size=0.035, ncol=1):
        """cmsLeg: no border, transparent."""
        leg = ROOT.TLegend(x1, y1, x2, y2, "", "brNDC")
        leg.SetTextSize(size); leg.SetTextFont(42); leg.SetBorderSize(0)
        leg.SetFillStyle(0); leg.SetFillColor(0); leg.SetNColumns(ncol)
        return leg

    # ---------------------------------------------------------------- canvases
    def canvas_1d(self, name: str, logy=False, grid=True, right_margin=None):
        """Plotter1D._initialize_canvas."""
        c = ROOT.TCanvas(name, name, self.canvas_width, self.canvas_height)
        c.SetLeftMargin(self.margin_left + 0.04)
        c.SetRightMargin(self.margin_right - 0.02 if right_margin is None else right_margin)
        c.SetTopMargin(0.07); c.SetBottomMargin(0.14)
        c.SetGridx(grid); c.SetGridy(grid); c.SetLogy(logy)
        return c

    def canvas_ratio(self, name: str, logy=False, width=1100, height=800):
        """PlotterDataMC._setup_comparison_canvas + _setup_main_pad: main pad 0.3-1, ratio pad 0-0.3."""
        c = ROOT.TCanvas(name, name, width, height)
        p1 = ROOT.TPad(name + "_p1", "", 0, 0.3, 1, 1.0)
        p2 = ROOT.TPad(name + "_p2", "", 0, 0.0, 1, 0.3)
        p1.SetBottomMargin(0.01); p1.SetTopMargin(0.1)
        p2.SetTopMargin(0.02); p2.SetBottomMargin(0.4)
        for p in (p1, p2):
            p.SetLeftMargin(self.margin_left + 0.04)
            p.SetRightMargin(self.margin_right_ratio)
            p.SetGridx(True); p.SetGridy(True)
        p1.SetLogy(logy)
        c.cd(); p1.Draw(); p2.Draw()
        c._pads = (p1, p2)
        return c, p1, p2

    def canvas_2d(self, name: str):
        c = ROOT.TCanvas(name, name, 800, 700)
        c.SetLeftMargin(0.13); c.SetRightMargin(0.16); c.SetTopMargin(0.07); c.SetBottomMargin(0.12)
        c.SetLogz(True)
        return c

    def axes(self, h, xtitle=None, ytitle=None, ratio_pad=False):
        """Plotter1D.setup_axes: centred titles, LLPStandardPlots sizes; ratio pad scaled by 0.7/0.3."""
        k = 0.7 / 0.3 if ratio_pad else 1.0
        for ax, title in ((h.GetXaxis(), xtitle), (h.GetYaxis(), ytitle)):
            if title is not None:
                ax.SetTitle(title)
            ax.SetTitleSize(self.axis_title_size * k)
            ax.SetLabelSize(self.axis_label_size * k)
            ax.CenterTitle(True)
        h.GetXaxis().SetTitleOffset(self.axis_title_offset if not ratio_pad else 1.1)
        h.GetYaxis().SetTitleOffset(self.axis_title_offset if not ratio_pad else 0.45)
        if ratio_pad:
            h.GetYaxis().SetNdivisions(505)
