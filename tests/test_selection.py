import json

import awkward as ak
import numpy as np
import pytest

from svval import selection as sel
from svval.lumimask import LumiMask


# --------------------------------------------------------------------------- lumi mask

def test_lumimask(tmp_path):
    p = tmp_path / 'g.json'
    p.write_text(json.dumps({"100": [[1, 5], [10, 10]], "200": [[3, 4]]}))
    m = LumiMask([str(p)])
    run = [100, 100, 100, 100, 100, 100, 200, 200, 300, 99]
    lum = [1, 5, 6, 10, 11, 0, 3, 5, 1, 1]
    assert m(run, lum).tolist() == [True, True, False, True, False, False, True, False, False, False]


def test_lumimask_repo_jsons_load_and_cover_all_years():
    m = LumiMask.from_dir()
    assert len(m.files) == 8
    # one certified lumi from each year's JSON (first range of the first run)
    import glob, os
    for f in sorted(glob.glob(os.path.join(os.path.dirname(sel.__file__), '../../configs/golden_json/Cert*'))):
        d = json.load(open(f))
        run = min(d, key=int)
        assert m([int(run)], [d[run][0][0]])[0], f


# --------------------------------------------------------------------------- SVs and blinding

def _ev_sv(svs):
    """svs: list of events, each a list of dicts with SV fields (defaults fill the rest)."""
    default = dict(Vertex_nTracks=2, Vertex_mass=20.0, Vertex_dxy=5.0, Vertex_dxyError=0.05, Vertex_x=5.0,
                   Vertex_y=0.0, Vertex_z=1.0, Vertex_cosTheta=0.9, Vertex_passLooseMuonID=True,
                   Vertex_passLooseElectronID=False)
    cols = {k: [] for k in default}
    for ev in svs:
        for k in cols:
            cols[k].append([s.get(k, default[k]) for s in ev])
    arrs = {}
    for k, v in cols.items():
        t = np.uint32 if k == 'Vertex_nTracks' else (bool if 'pass' in k else np.float32)
        arrs[k] = ak.values_astype(ak.Array(v), t)
    return ak.zip(arrs, depth_limit=1)


def test_classification_and_flavor():
    ev = _ev_sv([[
        dict(),                                                        # mu-mu leptonic, analysis
        dict(Vertex_passLooseMuonID=False, Vertex_passLooseElectronID=True),
        dict(Vertex_passLooseElectronID=True),                         # both IDs
        dict(Vertex_passLooseMuonID=False),                            # 2 tracks, no ID
        dict(Vertex_nTracks=3),
        dict(Vertex_nTracks=5, Vertex_mass=10.0),                      # hadronic, m/n = 2
        dict(Vertex_nTracks=5, Vertex_mass=4.0),                       # hadronic, fails ID (m/n = 0.8)
        dict(Vertex_dxyError=0.0),                                     # bad error
        dict(Vertex_dxy=0.5),                                          # fails baseline (dxy < 1)
        dict(Vertex_cosTheta=0.999),                                   # fails baseline
        dict(Vertex_dxy=20.0, Vertex_x=20.0, Vertex_z=30.0),           # dxy > 15, |eta| > 1 -> fails
    ]])
    sv = sel.classify_svs(ev)[0]
    assert sv.cls.tolist() == [2, 2, 2, 2, 3, 4, 4, 2, 2, 2, 2]
    assert sv.flavor.tolist()[:5] == [sel.FLAV_MU, sel.FLAV_E, sel.FLAV_BOTH, sel.FLAV_NONE, sel.FLAV_NONE]
    assert sv.good.tolist() == [True] * 7 + [False] + [True] * 3
    assert sv.baseline.tolist() == [True] * 7 + [False] * 4
    assert sv.lepAna.tolist() == [True, True, True] + [False] * 8
    assert sv.hadAna.tolist() == [False] * 5 + [True, False] + [False] * 4


@pytest.mark.parametrize("svs,veto", [
    ([dict(Vertex_dxyError=5.0 / 499.0)], False),                       # lep analysis SV just below 500
    ([dict(Vertex_dxyError=5.0 / 500.0)], True),                        # at 500 -> SR side
    ([dict(Vertex_dxyError=5.0 / 900.0, Vertex_passLooseMuonID=False)], False),   # no ID -> not analysis
    ([dict(Vertex_nTracks=5, Vertex_mass=10.0, Vertex_dxyError=5.0 / 799.0)], False),
    ([dict(Vertex_nTracks=5, Vertex_mass=10.0, Vertex_dxyError=5.0 / 801.0)], True),
    ([dict(Vertex_nTracks=5, Vertex_mass=4.0, Vertex_dxyError=5.0 / 900.0)], False),  # had, fails ID
    ([dict(Vertex_dxyError=5.0 / 600.0, Vertex_cosTheta=0.999)], False),           # fails baseline
    ([dict(Vertex_mass=1.0, Vertex_dxyError=5.0 / 600.0)], True),       # no mass cut in the veto
])
def test_event_veto(svs, veto):
    v, _ = sel.blinding(sel.classify_svs(_ev_sv([svs])))
    assert bool(v[0]) is veto


def test_cap_never_shows_an_sv_beyond_its_cr():
    rng = np.random.default_rng(1)
    events = []
    for _ in range(300):
        events.append([dict(Vertex_nTracks=int(rng.integers(2, 8)),
                            Vertex_mass=float(rng.uniform(0, 50)),
                            # log-uniform sigma: dxySig from 5 to 5000, plus some invalid errors
                            Vertex_dxyError=float(0.0 if rng.random() < 0.1 else 10 ** rng.uniform(-3, 0)),
                            Vertex_cosTheta=float(rng.uniform(-1, 1)),
                            Vertex_passLooseMuonID=bool(rng.integers(0, 2)))
                       for _ in range(int(rng.integers(0, 6)))])
    sv = sel.classify_svs(_ev_sv(events))
    veto, shown = sel.blinding(sv)
    kept = sv[~veto][shown[~veto]]
    good = kept[kept.good]
    sig, cls = ak.flatten(good.dxySig), ak.flatten(good.cls)
    assert ak.all(sig[cls == sel.CLS_LEP] < sel.LEP_DXYSIG_MAX)
    assert ak.all(sig[cls != sel.CLS_LEP] < sel.HAD_DXYSIG_MAX)
    # the test must actually exercise the cap: some SVs above threshold existed in kept events
    assert ak.sum(~shown[~veto]) > 0
    assert ak.sum(~kept.good) > 0            # bad-error SVs stay (own category)


def test_thresholds_are_the_analysis_ones():
    assert (sel.LEP_DXYSIG_MAX, sel.HAD_DXYSIG_MAX) == (500.0, 800.0)


# --------------------------------------------------------------------------- MET / HEM

def _ev_pho(phos, met=(100.0, 0.0), run=320000, ele=(), jet=(), mu=()):
    """One event; jagged columns are typed (as from a ROOT file) even when empty."""
    def col(xs, i, dtype=np.float32):
        return ak.unflatten(np.array([x[i] for x in xs], dtype=dtype), [len(xs)])
    out = {'Met_px': np.array([met[0]], np.float32), 'Met_py': np.array([met[1]], np.float32),
           'Evt_run': np.array([run], np.int32),
           'Photon_isOot': col(phos, 3, bool), 'Photon_pixelSeed': col(phos, 4, bool)}
    for name, xs in (('Photon', phos), ('Electron', ele), ('Jet', jet), ('Muon', mu)):
        for i, var in enumerate(('pt', 'eta', 'phi')):
            out[f'{name}_{var}'] = col(xs, i)
    return ak.zip(out, depth_limit=1)


def test_corrected_met():
    # OOT photon (pt 50, phi 0) is subtracted; GED photon on top of it (dR 0.1) is added back;
    # an isolated GED photon is ignored
    ev = _ev_pho([(50, 0.0, 0.0, 1, 0), (40, 0.1, 0.0, 0, 0), (30, 1.0, 2.0, 0, 0)])
    assert sel.photon_excluded(ev)[0].tolist() == [False, True, False]
    assert sel.corrected_met(ev)[0] == pytest.approx(100 - 50 + 40)


def test_hem_veto():
    hem = (-2.0, -1.0)
    assert sel.hem_veto(_ev_pho([], ele=[(25, *hem)]))[0]
    assert not sel.hem_veto(_ev_pho([], ele=[(15, *hem)]))[0]
    assert not sel.hem_veto(_ev_pho([], ele=[(25, *hem)], run=319076))[0]   # run outside hemRun
    assert sel.hem_veto(_ev_pho([], jet=[(11, *hem)]))[0]
    assert sel.hem_veto(_ev_pho([], mu=[(21, *hem)]))[0]
    assert sel.hem_veto(_ev_pho([(31, *hem, 0, 0)]))[0]
    assert not sel.hem_veto(_ev_pho([(31, *hem, 0, 1)]))[0]                 # pixel seed
    assert not sel.hem_veto(_ev_pho([(25, *hem, 0, 0)]))[0]                 # photon needs > 30
    assert not sel.hem_veto(_ev_pho([], jet=[(50, 1.0, -1.0)]))[0]          # wrong eta
