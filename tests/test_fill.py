import awkward as ak
import hist
import numpy as np
import pytest

from svval import fill, histos
from svval import selection as sel


def _events():
    """2 events. Event 0: SV0 (2 trk: tracks 5, 7), SV1 (4 trk: 7, 8, 9, 10) -> track 7 shared.
    Event 1: one 2-track SV (tracks 1, 2), nothing shared."""
    f32 = lambda x: ak.values_astype(ak.Array(x), np.float32)
    ev = {
        'Vertex_nTracks': ak.values_astype(ak.Array([[2, 4], [2]]), np.uint32),
        'VertexTrack_vertexIndex': ak.values_astype(ak.Array([[0, 0, 1, 1, 1, 1], [0, 0]]), np.uint32),
        'VertexTrack_trackIndex': ak.values_astype(ak.Array([[5, 7, 7, 8, 9, 10], [1, 2]]), np.uint32),
    }
    for f in fill.VTRK_FIELDS:
        ev.setdefault(f, f32([[0.5] * 6, [0.5] * 2]))
    for f in fill.VERTEX_FIELDS:
        if f not in ev:
            ev[f] = f32([[1.0, 2.0], [3.0]]) if 'pass' not in f and f != 'Vertex_isUnique' else \
                ak.Array([[True, False], [True]])
    for f in fill.TRACK_FIELDS:
        ev[f] = f32([[1.0], []])
    return ak.zip(ev, depth_limit=1)


def test_shared_tracks():
    ev = _events()
    info = ak.zip({'good': ak.Array([[True, True], [True]])})
    sv, vt, _ = fill.build_records(ev, info, ak.Array([[True, True], [True]]))
    assert vt.shared.tolist() == [[False, True, True, False, False, False], [False, False]]
    assert sv.nShared.tolist() == [[1, 1], [0]]


def test_build_records_rejects_noncontiguous_vertextracks():
    ev = _events()
    ev = ak.with_field(ev, ak.values_astype(ak.Array([[3, 4], [2]]), np.uint32), 'Vertex_nTracks')
    with pytest.raises(fill.JobFailure):
        fill.build_records(ev, ak.zip({'good': ak.Array([[True, True], [True]])}), ak.Array([[True, True], [True]]))


def test_slice_names():
    run = np.array([319076, 319077, 320000])
    assert fill.slice_names(2018, 'B', '', run).tolist() == ['2018B_preHEM', '2018B_HEM', '2018B_HEM']
    assert fill.slice_names(2016, 'F', 'preVFP', run[:1]).tolist() == ['2016F_preVFP']
    assert fill.slice_names(2024, 'G', '', run[:1]).tolist() == ['2024G']


def test_lumi_table_presence_and_counters():
    lt = fill.LumiTable()
    n = 4
    er = ak.zip({c: np.arange(n) for c in fill.LUMI_COUNTERS if c not in ('n_sel', 'n_fakePV', 'sum_npv')}
                | {'PV_npvs': np.array([1, 20, 30, 40])}, depth_limit=1)
    # events: lumis (1,1) (1,1) (1,2) (2,5); only events 1 and 3 selected
    lt.add(0, [1, 1, 1, 2], [1, 1, 2, 5], np.array([False, True, False, True]), er)
    lt.add(0, [1], [2], np.array([True]), er[:1])          # a second chunk of the same file
    t = lt.table()
    assert [(r['run'], r['lumi']) for r in t] == [(1, 1), (1, 2), (2, 5)]   # every lumi read is present
    assert t['n_sel'].tolist() == [1, 1, 1]
    assert t['sum_npv'].tolist() == [20, 1, 40]
    assert t['n_fakePV'].tolist() == [0, 1, 0]
    assert t['n_sv'].tolist() == [1, 0, 3]


def test_audit_blinding():
    h = {}
    for cat in ('lep2_mu', 'had_id'):
        spec = next(v for v in histos.sv_vars(cat) if v.name == 'dxySig')
        h[f'sv/{cat}/base/dxySig'] = histos._h1(spec)
    h['sv/lep2_mu/base/dxySig'].fill(x=[10.0, 499.9])
    h['sv/had_id/base/dxySig'].fill(x=[799.0])
    assert histos.audit_blinding(h) == []
    h['sv/had_id/base/dxySig'].fill(x=[800.0])
    assert histos.audit_blinding(h) == ['sv/had_id/base/dxySig']
    # a dxySig axis that does not end at the threshold is also a failure
    h2 = {'sv/lep2/all/dxySig': hist.Hist(hist.axis.Regular(10, 0, 800))}
    assert histos.audit_blinding(h2) == ['sv/lep2/all/dxySig']


def test_dxysig_axes_end_at_thresholds():
    hs = histos.HistSet()
    for key, h in hs.h.items():
        if key.endswith('/dxySig'):
            assert h.axes[0].edges[-1] == histos.dxysig_max(key.split('/')[-3]), key
    assert histos.dxysig_max('lep2_noid') == sel.LEP_DXYSIG_MAX
    assert histos.dxysig_max('trk3') == sel.HAD_DXYSIG_MAX
