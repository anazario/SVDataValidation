import json
import os

import numpy as np
import pytest

from svval.plotting import inputs as inp

ROOT = pytest.importorskip("ROOT")


def test_parse_slice():
    s = inp.parse_slice('2016F_preVFP')
    assert (s.year, s.era, s.period, s.hem, s.label, s.run) == (2016, 'F', 'preVFP', '', '2016F preVFP', 'Run2')
    s = inp.parse_slice('2018B_HEM')
    assert (s.period, s.hem) == ('', 'HEM')
    s = inp.parse_slice('2024G')
    assert (s.period, s.hem, s.run) == ('', '', 'Run3')
    for bad in ('2024', 'G2024', '2016F_pre_VFP_HEM'):
        with pytest.raises(ValueError):
            inp.parse_slice(bad)


def test_sort_and_groups():
    names = ['2018B_HEM', '2016F_postVFP', '2018B_preHEM', '2016F_preVFP', '2024G', '2016B_preVFP', '2017C']
    sl = sorted((inp.parse_slice(n) for n in names), key=inp.Slice.sort_key)
    assert [s.name for s in sl] == ['2016B_preVFP', '2016F_preVFP', '2016F_postVFP', '2017C',
                                    '2018B_preHEM', '2018B_HEM', '2024G']
    groups = {g.name: g for g in inp.make_groups(sl)}
    assert set(groups) == {'y2016', 'y2018', 'run2'}          # 2017 / 2024 / Run 3 have one unit only
    assert list(groups['y2016'].units) == ['2016B preVFP', '2016F preVFP', '2016F postVFP']
    assert list(groups['run2'].units) == ['2016', '2017', '2018']
    assert [s.name for s in groups['run2'].units['2018']] == ['2018B_preHEM', '2018B_HEM']


def _table(rows):
    dt = [('slice', 'U24'), ('run', np.uint32), ('lumi', np.uint32), ('n_sel', np.int64), ('n_sv', np.int64),
          ('n_badErr', np.int64), ('n_fakePV', np.int64), ('sum_npv', np.int64)]
    return np.array(rows, dtype=dt)


def test_rates_and_per_run():
    t = _table([('a', 1, 1, 100, 50, 10, 1, 2000), ('a', 1, 2, 100, 150, 0, 3, 2000),
                ('a', 2, 1, 50, 0, 0, 0, 500), ('a', 3, 1, 0, 0, 0, 0, 0)])
    v, e = inp.rate(t, 'sv_per_event')
    assert v == pytest.approx(200 / 250) and e == pytest.approx(np.sqrt(200) / 250)
    v, e = inp.rate(t, 'badErr_frac')
    assert v == pytest.approx(10 / 210) and e == pytest.approx(np.sqrt(v * (1 - v) / 210))
    assert inp.rate(t, 'mean_npv')[0] == pytest.approx(18.0)
    assert np.isnan(inp.rate(t[3:], 'sv_per_event')[0])
    runs, vals, _ = inp.per_run(t, 'sv_per_event', min_events=60)
    assert runs.tolist() == [1] and vals.tolist() == [1.0]           # run 2 (50 events) and 3 dropped


def test_compare_hists_flags_large_significant_differences():
    from svval.plotting.compare import compare_hists
    rng = np.random.default_rng(3)
    def h(name, mu, n):
        x = ROOT.TH1D(name, '', 50, -5, 5); x.SetDirectory(0)
        for v in rng.normal(mu, 1, n):
            x.Fill(v)
        return x
    ref = h('ref', 0.0, 200000)
    same = compare_hists(h('same', 0.0, 50000), ref)
    shifted = compare_hists(h('shift', 0.1, 50000), ref)          # ~4% KS distance
    tiny = compare_hists(h('tiny', 0.02, 50000), ref)             # significant but small: not flagged
    assert not same['flagged'] and same['ks_d'] < 0.01
    assert shifted['flagged'] and shifted['ks_d'] > 0.02 and shifted['mean_shift'] == pytest.approx(0.1, abs=0.02)
    assert not tiny['flagged']
    assert compare_hists(h('few', 0.0, 10), ref) is None


class _FakeMerged:
    """Merged stand-in: {(key, slice name): TH1}."""
    def __init__(self, hists):
        self.h = hists

    def is_2d(self, key):
        return False

    def hist(self, key, slices):
        out = None
        for s in slices:
            x = self.h.get((key, s.name))
            if x is None:
                continue
            if out is None:
                out = x.Clone(); out.SetDirectory(0)
            else:
                out.Add(x)
        return out


def _fill(name, values, nbins=4, lo=-0.5, hi=3.5):
    x = ROOT.TH1D(name, '', nbins, lo, hi); x.SetDirectory(0)
    for v in values:
        x.Fill(v)
    return x


def test_constant_keys_and_unflagged_bs_pv():
    from svval.plotting import compare as cmp
    sl = [inp.parse_slice(n) for n in ('2016B_preVFP', '2016C_preVFP')]
    g = next(x for x in inp.make_groups(sl) if x.name == 'y2016')
    hists = {}
    for i, s in enumerate(sl):
        hists[('sv/lep2/all/nTracks', s.name)] = _fill(f'nt{i}', [2] * 100)                  # constant
        hists[('sv/lep2/all/ndof', s.name)] = _fill(f'nd{i}', [9] * 100)                     # overflow only
        hists[('event/PV_npvs', s.name)] = _fill(f'pv{i}', [i] * 100 + [3] * 100)            # very different
        hists[('event/nSV', s.name)] = _fill(f'ns{i}', [i] * 100 + [3] * 100)                # same difference
    hists[('sv/lep2/all/mass', '2016B_preVFP')] = _fill('m0', [2] * 100)                    # one bin here,
    hists[('sv/lep2/all/mass', '2016C_preVFP')] = _fill('m1', [1] * 100)                    # another there
    m = _FakeMerged(hists)
    keys = sorted({k for k, _ in hists})
    const = cmp.constant_keys(m, sl, keys)
    assert const == {'sv/lep2/all/nTracks': '1.5 to 2.5', 'sv/lep2/all/ndof': 'overflow'}
    res = {(r.unit, r.key): r for r in cmp.compare_group(m, g, [k for k in keys if k not in const])}
    assert res[('2016B preVFP', 'event/nSV')].flagged
    assert not res[('2016B preVFP', 'event/PV_npvs')].flagged
    assert res[('2016B preVFP', 'event/PV_npvs')].ks_d == pytest.approx(res[('2016B preVFP', 'event/nSV')].ks_d)
    assert not cmp.flaggable('event/BS_x') and not cmp.flaggable('event/PV_BS_dxy')
    assert cmp.flaggable('event/nSV') and cmp.flaggable('sv/had/all/r')


def test_brilcalc_formats_and_summary_block(tmp_path):
    from svval.plotting.lumi import read_brilcalc
    p = tmp_path / 'byls.csv'
    p.write_text("#Data tag : 24v1 , Norm tag: None\n"
                 "#run:fill,ls,time,beamstatus,E(GeV),delivered(/ub),recorded(/ub),avgpu,source\n"
                 "355100:7983,1:1,07/04/22 15:22:19,STABLE BEAMS,6800,2000000.0,1000000.0,10.0,HFOC\n"
                 "355100:7983,2:2,07/04/22 15:22:42,STABLE BEAMS,6800,2000000.0,3000000.0,10.0,HFOC\n"
                 "#Summary:\n#nfill,nrun,nls,ncms,totdelivered(/ub),totrecorded(/ub)\n1,1,2,2,4000000.0,4000000.0\n")
    t = read_brilcalc([str(p)])
    assert t.by_ls == {(355100, 1): pytest.approx(1e-3), (355100, 2): pytest.approx(3e-3)}  # /ub -> /fb
    assert t.by_run == {}
    q = tmp_path / 'byrun.csv'
    q.write_text("#Data tag : 24v1 , Norm tag: x\n#run:fill,time,nls,ncms,delivered(/pb),recorded(/pb)\n"
                 "355100:7983,07/04/22 15:22:19,100,100,250.0,200.0\n355101:7983,07/04/22 17:00:00,50,50,60.0,50.0\n"
                 "#Summary:\n#nfill,nrun,nls,ncms,totdelivered(/pb),totrecorded(/pb)\n1,2,150,150,310.0,250.0\n")
    t = read_brilcalc([str(q)])
    assert t.by_run == {355100: pytest.approx(0.2), 355101: pytest.approx(0.05)}             # /pb -> /fb; no summary row
    assert t.nls == {355100: 100, 355101: 50}
    assert t.by_year == {}                                            # file name has no year
    q2 = tmp_path / 'lumi_2022.csv'
    q2.write_text(q.read_text())
    assert read_brilcalc([str(q2)]).by_year == {2022: pytest.approx(0.25)}


def test_slice_lumis_scale_by_golden_ls_coverage():
    from svval.plotting.lumi import LumiTable, slice_lumis
    rows = [('2022C', 1, 1), ('2022C', 1, 2), ('2022C', 1, 2),        # LS 2 read twice (JetMET0 and 1)
            ('2022C', 1, 9),                                          # not golden
            ('2022D', 2, 1), ('2022D', 3, 1)]
    t = np.array(rows, dtype=[('slice', 'U24'), ('run', np.uint32), ('lumi', np.uint32)])
    golden = lambda r, l: ~((np.asarray(r) == 1) & (np.asarray(l) == 9))
    counts = {1: 4, 2: 1, 3: 2}                                       # golden LS per run
    vals, info = slice_lumis(t, LumiTable(by_run={1: 8.0, 2: 1.0}), golden, counts)
    assert vals['2022C'] == pytest.approx(8.0 * 2 / 4)                # no nls: 2 of 4 golden LS present
    assert info['2022C'] == {'ls_present': 2, 'ls_golden': 4, 'runs_without_lumi': [], 'runs_short_normtag': []}
    assert info['2022D']['runs_without_lumi'] == [3]
    # the normtag covers only 1 of run 1's 4 golden LS: 8.0 is the lumi of that one LS
    vals, info = slice_lumis(t, LumiTable(by_run={1: 8.0, 2: 1.0}, nls={1: 1, 2: 1}), golden, counts)
    assert vals['2022C'] == pytest.approx(8.0 / 1 * 2)                # average per LS x 2 present
    assert info['2022C']['runs_short_normtag'] == [1]
    vals, _ = slice_lumis(t, LumiTable(by_ls={(1, 1): 0.5, (1, 2): 0.25, (2, 1): 1.0, (3, 1): 2.0}), golden, counts)
    assert vals == {'2022C': pytest.approx(0.75), '2022D': pytest.approx(3.0)}


def test_sections_of():
    from svval.plotting.run import sections_of
    s = sections_of(['event/nSV', 'sv/had/all/mass', 'sv/had/base/mass', 'sv/badErr/r', 'vtrk/lep2/all/scMatch',
                     'track/all/pt'])
    assert list(s) == ['event', 'sv_had_all', 'sv_had_base', 'sv_badErr', 'vtrk_lep2_all', 'track']


def test_unit_lumis():
    from svval.plotting.run import unit_lumis
    sl = [inp.parse_slice(n) for n in ('2016B_preVFP', '2016F_postVFP', '2017C', '2018A_preHEM')]
    groups = {g.name: g for g in inp.make_groups(sl)}
    per_slice = {'2016B_preVFP': 10.0, '2016F_postVFP': 5.0, '2017C': 2.0}
    assert unit_lumis(groups['y2016'], None, per_slice) == {'2016B preVFP': 10.0, '2016F postVFP': 5.0}
    assert unit_lumis(groups['run2'], None, per_slice) == {'2016': 15.0, '2017': 2.0}   # 2018: no value
    assert unit_lumis(groups['run2'], None, {}) is None


def test_header_luminosity_from_csv_totals():
    from svval.plotting.plots import _lumi_for
    from svval.plotting.style import lumi_label
    year_lumi = {2016: 36.313753344, 2018: 59.561262519}
    assert lumi_label(_lumi_for([2016, 2018], {2016, 2018}, year_lumi)) == '96'
    assert lumi_label(36.5) == '37' and lumi_label(36.49) == '36'
    assert _lumi_for([2016, 2018], {2016}, year_lumi) is None         # 2018 not fully merged
    assert _lumi_for([2017], {2017}, year_lumi) is None               # no CSV for 2017
