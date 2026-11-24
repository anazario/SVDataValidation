import json
import os
import shutil

import numpy as np
import pytest

hist = pytest.importorskip("hist")
uproot = pytest.importorskip("uproot")
if shutil.which('hadd') is None:
    pytest.skip("hadd not available", allow_module_level=True)

from svval import condor, merge as M  # noqa: E402
from svval.fill import LUMI_COUNTERS, LumiTable  # noqa: E402

STORE = '/store/user/x/svval/v34/t'


def _job(eos_root, i, year, era, files, lumis, n_sel):
    """Write one fake job output (.root with one slice, _lumis.npz, .json) under <eos_root>/<STORE>."""
    d = os.path.join(eos_root, STORE.lstrip('/'))
    os.makedirs(d, exist_ok=True)
    stem = os.path.join(d, f'job_{i:05d}')
    sl = f'{year}{era}'
    h = hist.Hist(hist.axis.Regular(100, 0, 800, name='x'))
    h.fill(x=np.full(n_sel, 100.0))
    n = hist.Hist(hist.axis.Regular(21, -0.5, 20.5, name='x'))
    n.fill(x=np.ones(n_sel))
    with uproot.recreate(stem + '.root') as f:
        f[f'{sl}/sv/had_id/base/dxySig'] = h
        f[f'{sl}/event/nSV'] = n
    t = np.zeros(len(lumis), dtype=LumiTable.DTYPE)
    for k, (fi, run, lumi) in enumerate(lumis):
        t[k]['file'], t[k]['run'], t[k]['lumi'], t[k]['n_sel'] = fi, run, lumi, 1
    np.savez_compressed(stem + '_lumis.npz', lumis=t, files=np.array(files))
    with open(stem + '.json', 'w') as f:
        json.dump({'job': i, 'year': year, 'era': era, 'period': '', 'files': files,
                   'cutflow': {sl: {'unvetoed': 10, 'golden_json': 10, 'met150': n_sel}}}, f)


@pytest.fixture
def fake_eos(tmp_path, monkeypatch):
    eos = tmp_path / 'eos'
    man = tmp_path / 'man.json'
    files = ['/store/a/f0.root', '/store/a/f1.root', '/store/a/f2.root']
    man.write_text(json.dumps({'files': [{'path': f, 'pd': 'JetMET0', 'year': 2024} for f in files]}))
    _job(str(eos), 0, 2024, 'G', files[:2], [(0, 1, 1), (1, 1, 2)], 5)
    _job(str(eos), 1, 2024, 'H', files[2:], [(0, 2, 1)], 3)
    meta = {'tag': 't', 'eos_out': STORE, 'manifest': str(man), 'jobs': {'0': {}, '1': {}}}
    monkeypatch.setattr(condor, 'status', lambda tag, base=None: {
        'meta': meta, 'ids': [0, 1], 'done': [0, 1], 'active': [], 'held': [], 'pending': [], 'failed': []})
    uploaded = []

    def fake_xrdcp(src, dst, force=False):
        def local(u):
            return str(eos) + u.split('cmseos.fnal.gov/', 1)[1] if u.startswith('root://') else u
        if dst.startswith('root://'):
            uploaded.append(dst.split('cmseos.fnal.gov/', 1)[1])
        os.makedirs(os.path.dirname(local(dst)), exist_ok=True)
        shutil.copy(local(src), local(dst))
    monkeypatch.setattr(M, 'xrdcp', fake_xrdcp)
    monkeypatch.setattr(M, 'eos_exists', lambda p: os.path.exists(str(eos) + p))
    monkeypatch.setattr(M, 'eos_mkdir', lambda p: os.makedirs(str(eos) + p, exist_ok=True))
    return eos, uploaded


def test_merge_stages_hadds_and_uploads(tmp_path, fake_eos):
    eos, uploaded = fake_eos
    scratch, out = tmp_path / 'scratch', tmp_path / 'local'
    checks = M.merge('t', str(out), scratch=str(scratch), workers=3, era_workers=2, hadd_jobs=1, log=lambda m: None)
    assert checks['merged'] == 2 and checks['destination'] == STORE + '/merged'
    assert checks['duplicate_lumis'] == {'2024 JetMET0': 0} and checks['file_coverage'] == {'2024': 1.0}
    dest = str(eos) + STORE + '/merged'
    for d in (dest, str(out)):
        assert sorted(os.listdir(d)) == ['all.root', 'checks.json', 'cutflow.json', 'eras', 'lumis.npz']
        assert sorted(os.listdir(os.path.join(d, 'eras'))) == ['2024G.root', '2024H.root']
    with uproot.open(os.path.join(dest, 'all.root')) as f:
        assert sorted(k.split(';')[0] for k in f.keys() if k.count('/') == 0) == ['2024G', '2024H']
        assert f['2024G/event/nSV'].values().sum() == 5
    assert uploaded[-1] == STORE + '/merged/checks.json'          # the completion marker goes last
    assert not os.path.exists(scratch / 'svval_merge_t')        # scratch cleaned
    z = np.load(os.path.join(out, 'lumis.npz'))
    assert sorted(set(z['lumis']['slice'])) == ['2024G', '2024H'] and z['lumis']['n_sel'].sum() == 3
    with pytest.raises(M.MergeError, match='--force'):           # an existing merge is not overwritten
        M.merge('t', str(out), scratch=str(scratch), log=lambda m: None)
    M.merge('t', str(out), scratch=str(scratch), force=True, log=lambda m: None)


def test_merge_without_upload_and_failed_staging(tmp_path, fake_eos, monkeypatch):
    eos, uploaded = fake_eos
    checks = M.merge('t', str(tmp_path / 'local'), scratch=str(tmp_path / 's'), upload=False, log=lambda m: None)
    assert checks['destination'] is None and uploaded == []
    os.remove(str(eos) + STORE + '/job_00001.root')               # a staging failure stops everything
    monkeypatch.setattr(M, 'xrdcp', lambda s, d, force=False: (_ for _ in ()).throw(M.MergeError('xrdcp failed'))
                        if s.endswith('job_00001.root') else shutil.copy(str(eos) + s.split('cmseos.fnal.gov/', 1)[1], d))
    with pytest.raises(M.MergeError, match='staging'):
        M.merge('t', str(tmp_path / 'local2'), scratch=str(tmp_path / 's2'), upload=False, log=lambda m: None)
