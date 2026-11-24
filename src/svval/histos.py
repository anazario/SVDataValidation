"""
Histogram definitions of the fill stage.

Every histogram lives in a *slice*: year + era (+ detector period, + 2018 HEM period), e.g.
'2016F_preVFP', '2018B_HEM', '2024G'. A job writes one ROOT directory per slice, so hadd of job
outputs adds identical slices and any coarser grouping (era, period, year, run) is a sum of slices.

Collections and categories
  event   one entry per selected event
  sv      one entry per *shown* SV (see selection.BLINDING), per SV category and level:
            categories  lep2 (2 tracks), lep2_mu / lep2_e / lep2_both / lep2_noid (loose-ID flavour),
                        trk3, had (>= 4 tracks), had_id / had_noid (mass/nTracks > 1)
            levels      all (valid dxyError), base (skimmer baseline)
          The analysis SVs are lep2_{mu,e,both}/base and had_id/base.
          Plus 'badErr': SVs with an invalid dxyError (no dxySig, no level).
  vtrk    one entry per track of a shown SV (VertexTrack_*), by the SV's category and level
  track   one entry per Track_* entry (an approximation of the HYDDRA input, which is not stored;
          'hyddraLike' adds pt > 1, normalizedChi2 < 5, |sip2D| >= 4)

dxySig histograms end at the CR threshold of the category (500 for 2-track, 800 otherwise);
the merge-stage audit requires their overflow to be empty.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import awkward as ak
import hist
import numpy as np

from . import selection as sel

PI = float(np.pi)


@dataclass(frozen=True)
class H1:
    name: str
    label: str
    bins: int
    lo: float
    hi: float
    fn: Callable          # record -> values (same structure as the collection)


@dataclass(frozen=True)
class H2:
    name: str
    xlabel: str
    xbins: Tuple[int, float, float]
    xfn: Callable
    ylabel: str
    ybins: Tuple[int, float, float]
    yfn: Callable


def _f(field):
    return lambda r: r[field]


# --------------------------------------------------------------------------- categories

SV_CATEGORIES = OrderedDict([
    ('lep2', lambda s: s.cls == sel.CLS_LEP),
    ('lep2_mu', lambda s: s.flavor == sel.FLAV_MU),
    ('lep2_e', lambda s: s.flavor == sel.FLAV_E),
    ('lep2_both', lambda s: s.flavor == sel.FLAV_BOTH),
    ('lep2_noid', lambda s: (s.cls == sel.CLS_LEP) & (s.flavor == sel.FLAV_NONE)),
    ('trk3', lambda s: s.cls == sel.CLS_TRK3),
    ('had', lambda s: s.cls == sel.CLS_HAD),
    ('had_id', lambda s: (s.cls == sel.CLS_HAD) & (s.mOverN > 1)),
    ('had_noid', lambda s: (s.cls == sel.CLS_HAD) & ~(s.mOverN > 1)),
])
SV_LEVELS = OrderedDict([
    ('all', lambda s: s.good & s.shown),
    ('base', lambda s: s.baseline & s.shown),
])


def dxysig_max(category: str) -> float:
    return sel.LEP_DXYSIG_MAX if category.startswith('lep2') else sel.HAD_DXYSIG_MAX


# --------------------------------------------------------------------------- SV variables

def sv_vars(category: str) -> List[H1]:
    thr = dxysig_max(category)
    return [
        H1('nTracks', 'N_{tracks}', 20, 1.5, 21.5, _f('Vertex_nTracks')),
        H1('mass', 'm [GeV]', 100, 0, 100, _f('Vertex_mass')),
        H1('massLog', 'log_{10}(m / GeV)', 100, 0, 3, lambda r: np.log10(r.Vertex_mass)),
        H1('mOverN', 'm/N_{tracks} [GeV]', 100, 0, 10, _f('mOverN')),
        H1('pt', 'p_{T} [GeV]', 100, 0, 200, _f('Vertex_pt')),
        H1('eta', '#eta', 60, -3, 3, _f('Vertex_eta')),
        H1('phi', '#phi', 64, -PI, PI, _f('Vertex_phi')),
        H1('pOverE', 'p/E', 100, 0.5, 1, _f('pOverE')),
        H1('x', 'x [cm]', 240, -60, 60, _f('Vertex_x')),
        H1('y', 'y [cm]', 240, -60, 60, _f('Vertex_y')),
        H1('z', 'z [cm]', 150, -150, 150, _f('Vertex_z')),
        H1('r', 'r = #sqrt{x^{2}+y^{2}} [cm]', 240, 0, 60, _f('r')),
        H1('dxy', 'd_{xy} [cm]', 240, 0, 60, _f('Vertex_dxy')),
        H1('dxyError', '#sigma_{d_{xy}} [cm]', 100, 0, 0.5, _f('Vertex_dxyError')),
        H1('dxySig', 'd_{xy}/#sigma_{d_{xy}}', 100, 0, thr, _f('dxySig')),
        H1('chi2', '#chi^{2}', 100, 0, 50, _f('Vertex_chi2')),
        H1('normChi2', '#chi^{2}/ndof', 100, 0, 5, _f('Vertex_normalizedChi2')),
        H1('ndof', 'ndof', 30, -0.5, 29.5, _f('Vertex_ndof')),
        H1('cosTheta', 'cos#theta', 100, -1, 1, _f('Vertex_cosTheta')),
        H1('cosThetaFwd', 'cos#theta', 100, 0.9, 1, _f('Vertex_cosTheta')),
        H1('decayAngle', 'cos#theta_{CM}^{*}', 100, -1, 1, _f('Vertex_decayAngle')),
        H1('sumCharge', '#sum q', 13, -6.5, 6.5, _f('Vertex_sumCharge')),
        H1('isUnique', 'isUnique', 2, -0.5, 1.5, _f('Vertex_isUnique')),
        H1('nSharedTracks', 'tracks shared with another SV', 8, -0.5, 7.5, _f('nShared')),
        H1('scMatchRatio', 'SC-matched track fraction', 21, -0.025, 1.025, _f('Vertex_scMatchRatio')),
        H1('cxx', 'C_{xx} [cm^{2}]', 120, -0.02, 0.1, _f('Vertex_cxx')),
        H1('cyy', 'C_{yy} [cm^{2}]', 120, -0.02, 0.1, _f('Vertex_cyy')),
        H1('czz', 'C_{zz} [cm^{2}]', 120, -0.1, 0.5, _f('Vertex_czz')),
        H1('negCovDiag', 'C_{xx}, C_{yy} or C_{zz} < 0', 2, -0.5, 1.5, _f('negCov')),
    ]


SV_2D = [
    H2('xy', 'x [cm]', (240, -60, 60), _f('Vertex_x'), 'y [cm]', (240, -60, 60), _f('Vertex_y')),
    H2('xyInner', 'x [cm]', (200, -10, 10), _f('Vertex_x'), 'y [cm]', (200, -10, 10), _f('Vertex_y')),
    H2('rz', 'z [cm]', (150, -150, 150), _f('Vertex_z'), 'r [cm]', (120, 0, 60), _f('r')),
    H2('dxyErrVsDxy', 'd_{xy} [cm]', (60, 0, 60), _f('Vertex_dxy'), '#sigma_{d_{xy}} [cm]', (50, 0, 0.5),
       _f('Vertex_dxyError')),
    H2('massVsNtracks', 'N_{tracks}', (20, 1.5, 21.5), _f('Vertex_nTracks'), 'm [GeV]', (50, 0, 100),
       _f('Vertex_mass')),
    H2('phiVsEta', '#eta', (30, -3, 3), _f('Vertex_eta'), '#phi', (32, -PI, PI), _f('Vertex_phi')),
]
SV_2D_CATEGORIES = ('lep2', 'trk3', 'had')     # 2D maps only for these, level 'all' and 'base'

BADERR_VARS = [
    H1('nTracks', 'N_{tracks}', 20, 1.5, 21.5, _f('Vertex_nTracks')),
    H1('mass', 'm [GeV]', 100, 0, 100, _f('Vertex_mass')),
    H1('r', 'r [cm]', 240, 0, 60, _f('r')),
    H1('dxy', 'd_{xy} [cm]', 240, 0, 60, _f('Vertex_dxy')),
    H1('z', 'z [cm]', 150, -150, 150, _f('Vertex_z')),
    H1('normChi2', '#chi^{2}/ndof', 100, 0, 5, _f('Vertex_normalizedChi2')),
    H1('negCovDiag', 'C_{xx}, C_{yy} or C_{zz} < 0', 2, -0.5, 1.5, _f('negCov')),
    H1('cxx', 'C_{xx} [cm^{2}]', 120, -0.02, 0.1, _f('Vertex_cxx')),
]
BADERR_2D = [SV_2D[0], SV_2D[2]]

# --------------------------------------------------------------------------- SV-track variables

# KUCMSDisplacedVertex_mini.hh: tracks whose assigned SC is within this dR count as electron tracks
SC_MATCH_DR = 0.04

VTRK_VARS = [
    H1('cosTheta', 'track cos#theta', 100, -1, 1, _f('VertexTrack_trackCosTheta')),
    H1('cosThetaCM', 'track cos#theta_{CM}', 100, -1, 1, _f('VertexTrack_trackCosThetaAtCM')),
    H1('compatibility', 'track compatibility', 100, 0, 10, _f('VertexTrack_trackCompatibility')),
    H1('shiftDz', '#Deltaz after track removal [cm]', 100, 0, 2, _f('VertexTrack_shiftDzAfterTrackRemoval')),
    H1('shift3D', '#Delta3D after track removal [cm]', 100, 0, 2, _f('VertexTrack_shift3DAfterTrackRemoval')),
    H1('hasShift', 'shift defined (SV with > 2 tracks)', 2, -0.5, 1.5, lambda r: r.VertexTrack_shift3DAfterTrackRemoval >= 0),
    # VertexTrack_SCDR >= 0 only means the Hungarian assignment gave the track *some* SC (there is no
    # dR cut), so its rate follows the number of stored SCs (~3x larger from 2024 on). The ntupleizer
    # calls a track electron-like at dR < SC_MATCH_DR; that is what Vertex_passLooseElectronID uses.
    H1('scMatch', f'SC matched (#DeltaR < {SC_MATCH_DR:g})', 2, -0.5, 1.5,
       lambda r: (r.VertexTrack_SCDR >= 0) & (r.VertexTrack_SCDR < SC_MATCH_DR)),
    H1('scDR', '#DeltaR(track, assigned SC)', 125, 0, 0.5, _f('VertexTrack_SCDR')),
    H1('energySC', 'E_{SC} [GeV]', 100, 0, 200, _f('VertexTrack_energySC')),
    H1('pOverESC', 'p/E_{SC}', 100, 0, 5, _f('VertexTrack_ratioPToEnergySC')),
    H1('pathLength', 'path length SV#rightarrowSC [cm]', 100, 0, 400, _f('VertexTrack_pathLength')),
    H1('isShared', 'track shared with another SV', 2, -0.5, 1.5, _f('shared')),
]

# --------------------------------------------------------------------------- Track_ variables

TRACK_VARS = [
    H1('pt', 'p_{T} [GeV]', 100, 0, 100, _f('Track_pt')),
    H1('ptLog', 'log_{10}(p_{T} / GeV)', 100, -1, 3, lambda r: np.log10(r.Track_pt)),
    H1('eta', '#eta', 70, -3.5, 3.5, _f('Track_eta')),
    H1('phi', '#phi', 64, -PI, PI, _f('Track_phi')),
    H1('ip2D', 'IP_{2D} [cm]', 200, -2, 2, _f('Track_ip2D')),
    H1('sip2D', 'S_{IP,2D}', 100, 0, 100, _f('Track_sip2D')),
    H1('ip3D', 'IP_{3D} [cm]', 100, 0, 2, _f('Track_ip3D')),
    H1('sip3D', 'S_{IP,3D}', 100, 0, 200, _f('Track_sip3D')),
    H1('ipValid', 'IP measurement valid', 2, -0.5, 1.5, lambda r: r.Track_sip2D > -900),
    H1('nValidHits', 'valid hits', 40, -0.5, 39.5, _f('Track_nValidHits')),
    H1('nLostHits', 'lost hits', 5, -0.5, 4.5, _f('Track_nLostHits')),
    H1('normChi2', '#chi^{2}/ndof', 100, 0, 10, _f('Track_normalizedChi2')),
    H1('relPtErr', '#sigma_{p_{T}}/p_{T}', 100, 0, 0.5, lambda r: r.Track_ptError / r.Track_pt),
    H1('charge', 'q', 3, -1.5, 1.5, _f('Track_charge')),
    H1('highPurity', 'highPurity', 2, -0.5, 1.5, lambda r: (r.Track_qualityMask & 4) > 0),
    H1('fromGeneral', 'general (1) / GSF (0) track', 2, -0.5, 1.5, _f('Track_isGenTrack')),
    H1('pdgId', '|PF pdgId| (0: none)', 4, -0.5, 3.5,
       lambda r: ak.where(abs(r.Track_pdgId) == 211, 1, ak.where(abs(r.Track_pdgId) == 11, 2,
                                                               ak.where(abs(r.Track_pdgId) == 13, 3, 0)))),
]
TRACK_CATEGORIES = OrderedDict([
    ('all', lambda t: ak.ones_like(t.Track_pt, dtype=bool)),
    ('hyddraLike', lambda t: (t.Track_pt > 1) & (t.Track_normalizedChi2 < 5) & (t.Track_sip2D >= 4)),
])

# --------------------------------------------------------------------------- event variables

def event_vars() -> List[H1]:
    out = [
        H1('PV_npvs', 'N_{PV}', 100, -0.5, 99.5, _f('PV_npvs')),
        H1('PV_x', 'PV x [cm]', 200, -0.5, 0.5, _f('PV_x')),
        H1('PV_y', 'PV y [cm]', 200, -0.5, 0.5, _f('PV_y')),
        H1('PV_z', 'PV z [cm]', 100, -25, 25, _f('PV_z')),
        H1('BS_x', 'beamspot x [cm]', 200, -0.5, 0.5, _f('BeamSpot_X')),
        H1('BS_y', 'beamspot y [cm]', 200, -0.5, 0.5, _f('BeamSpot_Y')),
        H1('BS_z', 'beamspot z [cm]', 100, -2, 2, _f('BeamSpot_Z')),
        H1('PV_BS_dxy', '|PV - beamspot|_{xy} [cm]', 100, 0, 0.1, _f('pvBsDxy')),
        H1('fakePV', 'N_{PV} = 1 (beamspot fallback)', 2, -0.5, 1.5, lambda e: e.PV_npvs == 1),
        H1('selCMet', 'p_{T}^{miss} (corrected) [GeV]', 100, 150, 1150, _f('selCMet')),
        H1('nTrack', 'N(Track_)', 100, -0.5, 199.5, lambda e: ak.num(e.Track_pt)),
        H1('nSV', 'N_{SV} (valid #sigma_{d_{xy}}, shown)', 21, -0.5, 20.5, _f('n_sv')),
        H1('nSVbadErr', 'N_{SV} with invalid #sigma_{d_{xy}}', 11, -0.5, 10.5, _f('n_badErr')),
    ]
    for cat in SV_CATEGORIES:
        for lvl in SV_LEVELS:
            out.append(H1(f'nSV_{cat}_{lvl}', f'N_{{SV}} ({cat}, {lvl})', 11, -0.5, 10.5, _f(f'n_{cat}_{lvl}')))
    return out


EVENT_2D = [
    H2('nSVvsNPV', 'N_{PV}', (50, -0.5, 99.5), _f('PV_npvs'), 'N_{SV}', (21, -0.5, 20.5), _f('n_sv')),
    H2('nSVlep2vsNPV', 'N_{PV}', (50, -0.5, 99.5), _f('PV_npvs'), 'N_{SV} (lep2, base)', (11, -0.5, 10.5),
       _f('n_lep2_base')),
    H2('nSVhadvsNPV', 'N_{PV}', (50, -0.5, 99.5), _f('PV_npvs'), 'N_{SV} (had, base)', (11, -0.5, 10.5),
       _f('n_had_base')),
]


# --------------------------------------------------------------------------- booking & filling

def _h1(h: H1):
    return hist.Hist(hist.axis.Regular(h.bins, h.lo, h.hi, name='x', label=h.label), storage=hist.storage.Double())


def _h2(h: H2):
    return hist.Hist(hist.axis.Regular(*h.xbins, name='x', label=h.xlabel),
                     hist.axis.Regular(*h.ybins, name='y', label=h.ylabel), storage=hist.storage.Double())


def _flat(x):
    x = ak.flatten(x, axis=None) if isinstance(x, ak.Array) else np.asarray(x)
    return ak.to_numpy(x).astype(np.float64) if isinstance(x, ak.Array) else x.astype(np.float64)


class HistSet:
    """All histograms of one slice, filled incrementally. Keys are ROOT paths inside the slice dir."""

    def __init__(self):
        self.h: "OrderedDict[str, hist.Hist]" = OrderedDict()
        for v in event_vars():
            self.h[f'event/{v.name}'] = _h1(v)
        for v in EVENT_2D:
            self.h[f'event/{v.name}'] = _h2(v)
        for cat in SV_CATEGORIES:
            for lvl in SV_LEVELS:
                for v in sv_vars(cat):
                    self.h[f'sv/{cat}/{lvl}/{v.name}'] = _h1(v)
                if cat in SV_2D_CATEGORIES:
                    for v in SV_2D:
                        self.h[f'sv/{cat}/{lvl}/{v.name}'] = _h2(v)
                for v in VTRK_VARS:
                    self.h[f'vtrk/{cat}/{lvl}/{v.name}'] = _h1(v)
        for v in BADERR_VARS:
            self.h[f'sv/badErr/{v.name}'] = _h1(v)
        for v in BADERR_2D:
            self.h[f'sv/badErr/{v.name}'] = _h2(v)
        for cat in TRACK_CATEGORIES:
            for v in TRACK_VARS:
                self.h[f'track/{cat}/{v.name}'] = _h1(v)

    def _fill1(self, key, spec: H1, rec):
        if len(rec) == 0:
            return
        vals = _flat(spec.fn(rec))
        if len(vals):
            self.h[key].fill(x=vals)

    def _fill2(self, key, spec: H2, rec):
        if len(rec) == 0:
            return
        x, y = _flat(spec.xfn(rec)), _flat(spec.yfn(rec))
        if len(x):
            self.h[key].fill(x=x, y=y)

    def fill(self, ev, sv, vt, tr):
        """ev: selected events; sv / vt / tr: their jagged SV, SV-track and Track_ records."""
        for v in event_vars():
            self._fill1(f'event/{v.name}', v, ev)
        for v in EVENT_2D:
            self._fill2(f'event/{v.name}', v, ev)
        for cat, cfn in SV_CATEGORIES.items():
            cmask = cfn(sv)
            for lvl, lfn in SV_LEVELS.items():
                m = cmask & lfn(sv)
                s = ak.flatten(sv[m])
                for v in sv_vars(cat):
                    self._fill1(f'sv/{cat}/{lvl}/{v.name}', v, s)
                if cat in SV_2D_CATEGORIES:
                    for v in SV_2D:
                        self._fill2(f'sv/{cat}/{lvl}/{v.name}', v, s)
                # tracks of these SVs: select by the parent SV's mask
                tm = ak.flatten(vt[_parent_mask(m, vt.VertexTrack_vertexIndex)])
                for v in VTRK_VARS:
                    self._fill1(f'vtrk/{cat}/{lvl}/{v.name}', v, tm)
        bad = ak.flatten(sv[~sv.good])
        for v in BADERR_VARS:
            self._fill1(f'sv/badErr/{v.name}', v, bad)
        for v in BADERR_2D:
            self._fill2(f'sv/badErr/{v.name}', v, bad)
        for cat, cfn in TRACK_CATEGORIES.items():
            t = ak.flatten(tr[cfn(tr)])
            for v in TRACK_VARS:
                self._fill1(f'track/{cat}/{v.name}', v, t)

    def add(self, other: "HistSet"):
        for k, h in other.h.items():
            self.h[k] += h


def _parent_mask(sv_mask, vertex_index):
    """Per-track boolean: the SV-level mask of the track's parent SV."""
    return sv_mask[ak.values_astype(vertex_index, np.int64)]


def audit_blinding(hists) -> List[str]:
    """
    Keys of dxySig histograms with entries at or above the CR threshold (the overflow bin, as the
    axis ends at the threshold). Must be empty. Accepts {key: hist.Hist}.
    """
    bad = []
    for key, h in hists.items():
        if key.endswith('/dxySig'):
            cat = key.split('/')[-3]
            if abs(h.axes[0].edges[-1] - dxysig_max(cat)) > 1e-9 or h[hist.overflow] > 0:
                bad.append(key)
    return bad
