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


def test_brilcalc_csv_and_lumi_per_slice(tmp_path):
    from svval.plotting.run import read_brilcalc_byls
    p = tmp_path / 'lumi.csv'
    p.write_text("#Data tag : 24v1 , Norm tag: None\n"
                 "#run:fill,ls,time,beamstatus,E(GeV),delivered(/ub),recorded(/ub),avgpu,source\n"
                 "355100:7983,1:1,07/04/22 15:22:19,STABLE BEAMS,6800,2000000.0,1000000.0,10.0,HFOC\n"
                 "355100:7983,2:2,07/04/22 15:22:42,STABLE BEAMS,6800,2000000.0,3000000.0,10.0,HFOC\n"
                 "#Summary:\n")
    lum = read_brilcalc_byls([str(p)])
    assert lum == {(355100, 1): pytest.approx(1e-3), (355100, 2): pytest.approx(3e-3)}  # /ub -> /fb


def test_sections_of():
    from svval.plotting.run import sections_of
    s = sections_of(['event/nSV', 'sv/had/all/mass', 'sv/had/base/mass', 'sv/badErr/r', 'vtrk/lep2/all/hasSC',
                     'track/all/pt'])
    assert list(s) == ['event', 'sv_had_all', 'sv_had_base', 'sv_badErr', 'vtrk_lep2_all', 'track']


def test_unit_lumis():
    from svval.plotting.run import unit_lumis
    from svval.plotting.style import LUMI_FB
    assert LUMI_FB[2024] == pytest.approx(108.9527975) and LUMI_FB[2016] == pytest.approx(36.31375334)
    sl = [inp.parse_slice(n) for n in ('2016B_preVFP', '2016F_postVFP', '2017C', '2018A_preHEM')]
    groups = {g.name: g for g in inp.make_groups(sl)}

    class M:
        full_years = {2016, 2017, 2018}
    assert unit_lumis(groups['run2'], M, None) == {'2016': pytest.approx(36.31375334),
                                                   '2017': pytest.approx(41.47984914),
                                                   '2018': pytest.approx(59.8324224)}
    assert unit_lumis(groups['y2016'], M, None) is None              # slices: needs brilcalc
    M.full_years = {2016, 2018}
    assert unit_lumis(groups['run2'], M, None) is None               # 2017 not fully merged
    per_slice = {'2016B_preVFP': 10.0, '2016F_postVFP': 5.0}
    assert unit_lumis(groups['y2016'], M, per_slice) == {'2016B preVFP': 10.0, '2016F postVFP': 5.0}
