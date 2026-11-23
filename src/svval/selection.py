"""
Event selection, SV classification and blinding.

Everything here is a port of KUCMSSkimmer (f9af717) so the validation sees the same objects as
the analysis; tests/test_selection_parity.py compares it event by event with kuSkimTree.

  event selection   golden JSON, MET filters (skimmer `metFlags`), MET trigger OR, HEM veto
                    (data, 2018 HEM runs), corrected MET (`selCMet`) > 150
  SV classification KUCMSAodSkimmer::processSV: dxyError > 0 and finite, the baseline
                    (dxy, z, eta-origin, cos theta), nTracks == 2 -> leptonic (ID: loose mu or e),
                    nTracks >= 4 -> hadronic (ID: mass/nTracks > 1), nTracks == 3 -> neither
  blinding          see BLINDING below; thresholds are constants, there is no unblind switch
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Optional, Tuple

import awkward as ak
import numpy as np

# --------------------------------------------------------------------------- blinding

LEP_DXYSIG_MAX = 500.0   # leptonic SV CR: dxySig < 500
HAD_DXYSIG_MAX = 800.0   # hadronic SV CR: dxySig < 800

BLINDING = """
1. Event veto: an event is dropped if it has an analysis SV above its CR threshold:
   a leptonic SV (baseline, 2 tracks, loose mu or e ID) with dxySig >= 500, or a hadronic SV
   (baseline, >= 4 tracks, mass/nTracks > 1) with dxySig >= 800. No mass requirement, so this is
   looser than (a superset of) every SR flag of the skimmer.
2. Per-SV cap: in the events kept, any SV at or above the threshold of its track multiplicity
   (500 for 2 tracks, 800 for >= 3 tracks) is removed from every histogram, whatever its ID or
   baseline status. So no histogram ever holds an SV beyond the CR.
SVs with an invalid dxyError (0 or non-finite) have no dxySig, are never analysis SVs, and are
only filled in their own 'badErr' category.
"""

# --------------------------------------------------------------------------- constants

MET_CUT = 150.0
TRIGGERS = (
    'HLT_PFMET120_PFMHT120_IDTight_v',
    'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v',
    'HLT_PFMET120_PFMHT120_IDTight_PFHT60_v',
    'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v',
)
MET_FILTERS = (   # the skimmer's metFlags (KUCMSAodSkimmer_EvntMetVars.cc)
    'Flag_BadChargedCandidateFilter', 'Flag_BadPFMuonFilter', 'Flag_EcalDeadCellTriggerPrimitiveFilter',
    'Flag_HBHENoiseFilter', 'Flag_HBHENoiseIsoFilter', 'Flag_ecalBadCalibFilter', 'Flag_eeBadScFilter',
    'Flag_goodVertices',
)
HEM_RUN_MIN, HEM_RUN_MAX = 319076, 326635   # hemRun: MIN < run < MAX (exclusive)
HEM_FIRST_RUN = HEM_RUN_MIN + 1             # 2018 'HEM' period starts here (event-level split)

EVENT_BRANCHES = (
    ['Evt_run', 'Evt_luminosityBlock', 'Evt_event', 'Met_px', 'Met_py',
     'Photon_pt', 'Photon_eta', 'Photon_phi', 'Photon_isOot', 'Photon_pixelSeed',
     'Electron_pt', 'Electron_eta', 'Electron_phi', 'Jet_pt', 'Jet_eta', 'Jet_phi',
     'Muon_pt', 'Muon_eta', 'Muon_phi']
    + list(TRIGGERS) + list(MET_FILTERS))

SV_BRANCHES = ['Vertex_nTracks', 'Vertex_mass', 'Vertex_dxy', 'Vertex_dxyError', 'Vertex_x', 'Vertex_y',
               'Vertex_z', 'Vertex_cosTheta', 'Vertex_passLooseMuonID', 'Vertex_passLooseElectronID']

# SV class codes
CLS_LEP, CLS_TRK3, CLS_HAD = 2, 3, 4
# leptonic flavour codes (2-track SVs)
FLAV_NONE, FLAV_MU, FLAV_E, FLAV_BOTH = 0, 1, 2, 3


# --------------------------------------------------------------------------- event level

def _dphi(a, b):
    """KUCMSHelperFunctions dPhi: a single +-2pi wrap."""
    dp = a - b
    dp = ak.where(dp > np.pi, dp - 2 * np.pi, dp)
    return ak.where(dp < -np.pi, dp + 2 * np.pi, dp)


def photon_excluded(ev) -> ak.Array:
    """The skimmer's correctedPhotonExcluded: a GED photon within dR < 0.3 of any OOT photon."""
    ged = ~ev.Photon_isOot
    pairs = ak.cartesian({'a': ak.zip({'eta': ev.Photon_eta, 'phi': ev.Photon_phi}),
                          'b': ak.zip({'eta': ev.Photon_eta, 'phi': ev.Photon_phi, 'oot': ev.Photon_isOot})},
                         nested=True)
    dr = np.hypot(pairs.a.eta - pairs.b.eta, _dphi(pairs.a.phi, pairs.b.phi))
    near_oot = ak.any((dr < 0.3) & pairs.b.oot, axis=2)
    return ged & near_oot


def corrected_met(ev, excluded: Optional[ak.Array] = None) -> np.ndarray:
    """selCMet: Met_p{x,y} minus OOT photons, plus GED photons that overlap an OOT photon."""
    if excluded is None:
        excluded = photon_excluded(ev)
    pt = ak.values_astype(ev.Photon_pt, np.float64)
    phi = ak.values_astype(ev.Photon_phi, np.float64)
    sign = ak.where(ev.Photon_isOot, -1.0, ak.where(excluded, 1.0, 0.0))
    px = ak.to_numpy(ev.Met_px).astype(np.float64) + ak.to_numpy(ak.sum(sign * pt * np.cos(phi), axis=1))
    py = ak.to_numpy(ev.Met_py).astype(np.float64) + ak.to_numpy(ak.sum(sign * pt * np.sin(phi), axis=1))
    return np.hypot(px, py).astype(np.float32)


def met_filters(ev) -> np.ndarray:
    out = np.ones(len(ev), dtype=bool)
    for f in MET_FILTERS:
        out &= ak.to_numpy(ev[f]).astype(bool)
    return out


def trigger_or(ev) -> np.ndarray:
    out = np.zeros(len(ev), dtype=bool)
    for t in TRIGGERS:
        out |= ak.to_numpy(ev[t]).astype(bool)
    return out


def _in_hem(eta, phi):
    return (eta > -3.2) & (eta < -1.2) & (phi > -1.77) & (phi < -0.67)


def hem_veto(ev, excluded: Optional[ak.Array] = None) -> np.ndarray:
    """Flag_hemVeto for data: (e1 || j1 || pho2 || mu1) && hemRun."""
    if excluded is None:
        excluded = photon_excluded(ev)
    run = ak.to_numpy(ev.Evt_run)
    hem_run = (run > HEM_RUN_MIN) & (run < HEM_RUN_MAX)
    e1 = ak.any((ev.Electron_pt > 20) & _in_hem(ev.Electron_eta, ev.Electron_phi), axis=1)
    j1 = ak.any((ev.Jet_pt > 10) & _in_hem(ev.Jet_eta, ev.Jet_phi), axis=1)
    p2 = ak.any((ev.Photon_pt > 30) & ~excluded & (abs(ev.Photon_eta) <= 2.6) & ~ev.Photon_pixelSeed
                & _in_hem(ev.Photon_eta, ev.Photon_phi), axis=1)
    m1 = ak.any((ev.Muon_pt > 20) & _in_hem(ev.Muon_eta, ev.Muon_phi), axis=1)
    return ak.to_numpy(e1 | j1 | p2 | m1) & hem_run


# --------------------------------------------------------------------------- SVs

def classify_svs(ev) -> ak.Array:
    """
    Per-SV record (same jagged structure as Vertex_*):
      good      dxyError finite and > 0 (skimmer skips the others)
      dxySig    float32 dxy/dxyError (NaN where not good)
      baseline  good and passes the skimmer baseline
      cls       2 / 3 / 4 = 2 tracks, 3 tracks, >= 4 tracks (CLS_*)
      flavor    2-track SVs: FLAV_NONE / MU / E / BOTH from the loose IDs
      lepAna    analysis leptonic SV: baseline, 2 tracks, mu or e ID
      hadAna    analysis hadronic SV: baseline, >= 4 tracks, mass/nTracks > 1
      mOverN    mass / nTracks
    """
    nt = ev.Vertex_nTracks
    dxy, err = ev.Vertex_dxy, ev.Vertex_dxyError
    good = (err > 0) & np.isfinite(err)
    with np.errstate(divide='ignore', invalid='ignore'):
        sig = ak.where(good, dxy / err, np.float32(np.nan))
    x, y, z = ev.Vertex_x, ev.Vertex_y, ev.Vertex_z
    # ROOT::Math::XYZVectorF(x, y, z).eta() in float: asinh(z / rho)
    rho = np.sqrt(x * x + y * y)
    with np.errstate(divide='ignore', invalid='ignore'):
        eta = np.arcsinh(z / rho)
    cos = ev.Vertex_cosTheta
    fail = (dxy < 1) | ((dxy < 15) & (abs(z) > 15)) | ((dxy > 15) & (abs(eta) > 1.)) | (cos > 0.995) | (cos <= -0.8)
    base = good & ~fail
    mu, el = ev.Vertex_passLooseMuonID, ev.Vertex_passLooseElectronID
    cls = ak.where(nt == 2, CLS_LEP, ak.where(nt == 3, CLS_TRK3, ak.where(nt >= 4, CLS_HAD, 0)))
    flavor = ak.where(nt == 2, ak.values_astype(mu, np.int8) + 2 * ak.values_astype(el, np.int8), FLAV_NONE)
    with np.errstate(divide='ignore', invalid='ignore'):
        m_over_n = ak.where(nt > 0, ev.Vertex_mass / nt, np.float32(0))
    return ak.zip({
        'good': good, 'dxySig': sig, 'baseline': base, 'cls': cls, 'flavor': flavor,
        'lepAna': base & (nt == 2) & (mu | el),
        'hadAna': base & (nt >= 4) & (m_over_n > 1),
        'mOverN': m_over_n,
    })


def blinding(sv: ak.Array) -> Tuple[np.ndarray, ak.Array]:
    """(event_veto, sv_shown). sv_shown is False for every SV that must not enter a histogram."""
    veto = ak.to_numpy(ak.any(sv.lepAna & (sv.dxySig >= LEP_DXYSIG_MAX), axis=1)
                       | ak.any(sv.hadAna & (sv.dxySig >= HAD_DXYSIG_MAX), axis=1))
    thr = ak.where(sv.cls == CLS_LEP, LEP_DXYSIG_MAX, HAD_DXYSIG_MAX)
    below = ak.fill_none(sv.dxySig < thr, False)       # NaN compares False -> not shown here
    shown = (sv.good & below) | ~sv.good                # bad-error SVs go to their own category
    return veto, shown


# --------------------------------------------------------------------------- full selection

def select(ev, lumimask, year: Optional[int] = None) -> Tuple[np.ndarray, ak.Array, ak.Array, Dict]:
    """
    Apply the event selection and the blinding.
    Returns (event_mask, sv_info, sv_shown, cutflow); cutflow counts are sequential.
    year is only used for the 2018 HEM-period label (see hem_period()).
    """
    n = len(ev)
    cut = OrderedDict()
    cut['all'] = np.ones(n, dtype=bool)
    cut['golden_json'] = lumimask(ak.to_numpy(ev.Evt_run), ak.to_numpy(ev.Evt_luminosityBlock))
    cut['met_filters'] = met_filters(ev)
    cut['trigger'] = trigger_or(ev)
    excl = photon_excluded(ev)
    cut['hem_veto'] = ~hem_veto(ev, excl)
    cut['met150'] = corrected_met(ev, excl) > MET_CUT
    sv = classify_svs(ev)
    veto, shown = blinding(sv)
    cut['blinding'] = ~veto
    mask = np.ones(n, dtype=bool)
    flow = OrderedDict()
    for name, m in cut.items():
        mask &= m
        flow[name] = int(mask.sum())
    return mask, sv, shown, flow


def hem_period(run: np.ndarray, year: int) -> np.ndarray:
    """2018 only: 'preHEM' / 'HEM' by run; '' for other years."""
    run = np.asarray(run)
    if year != 2018:
        return np.full(run.shape, '', dtype=object)
    return np.where(run >= HEM_FIRST_RUN, 'HEM', 'preHEM').astype(object)
