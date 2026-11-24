import json
import os
import tarfile

import pytest

from svval import condor


def test_parse_ids():
    assert condor.parse_ids(None, [3, 1, 2]) == [1, 2, 3]
    assert condor.parse_ids('1,3-4', range(10)) == [1, 3, 4]
    with pytest.raises(condor.CondorError):
        condor.parse_ids('1,99', range(10))
    with pytest.raises(condor.CondorError):
        condor.parse_ids('a-b', range(10))


@pytest.fixture
def joblist(tmp_path):
    man = tmp_path / 'man.json'
    man.write_text(json.dumps({'version': 'v34', 'xrootd_prefix': 'root://cmseos.fnal.gov/',
                               'tree': 'tree/llpgtree', 'files': []}))
    jobs = tmp_path / 'jobs.json'
    jobs.write_text(json.dumps({'manifest': str(man), 'jobs': [
        {'job': i, 'year': 2024, 'era': 'G', 'period': '', 'bytes': 10, 'files': [f'/store/f{i}_{k}.root' for k in range(4)]}
        for i in range(3)]}))
    return str(jobs)


def test_prepare_and_dry_run(tmp_path, joblist):
    base = str(tmp_path / 'condor')
    meta = condor.prepare('t1', joblist, [0, 2], max_files=2, eos_out='/store/user/x/y', base=base)
    d = condor.tag_dir('t1', base)
    assert sorted(meta['jobs']) == ['0', '2']
    spec = json.load(open(os.path.join(d, 'specs', 'job_00002.json')))
    assert spec['files'] == ['/store/f2_0.root', '/store/f2_1.root']          # max_files applied
    assert spec['prefix'] == 'root://cmseos.fnal.gov/' and spec['tree'] == 'tree/llpgtree'
    with tarfile.open(os.path.join(d, 'svval_code.tgz')) as t:
        names = t.getnames()
    assert 'src/svval/fill.py' in names
    assert any(n.startswith('configs/golden_json/') and n.endswith('.json') for n in names)
    assert not any('__pycache__' in n for n in names)
    wrapper = open(os.path.join(d, 'run_job.sh')).read()
    assert '@' not in wrapper and condor.CMSSW in wrapper
    # the .json must be the last thing copied
    assert wrapper.index('for ext in .root _lumis.npz .json') > 0

    assert condor.submit('t1', [0, 2], base=base, dry_run=True) is None
    q = open(os.path.join(d, 'queue_000.txt')).read().split()
    assert q == ['00000', '00002']
    jdl = open(os.path.join(d, 'submit_000.jdl')).read()
    assert jdl.rstrip().endswith('queue job from queue_000.txt')
    assert 'arguments               = $(job) /store/user/x/y' in jdl
    assert 'specs/job_$(job).json' in jdl and 'use_x509userproxy       = true' in jdl

    assert 'max_materialize' not in jdl
    condor.submit('t1', [0, 2], base=base, dry_run=True, max_materialize=25)
    jdl1 = open(os.path.join(d, 'submit_001.jdl')).read().splitlines()
    assert jdl1[-2:] == ['max_materialize         = 25', 'queue job from queue_001.txt']
    with pytest.raises(condor.CondorError):
        condor.submit('t1', [0], base=base, dry_run=True, max_materialize=0)

    with pytest.raises(condor.CondorError):     # job 1 not prepared
        condor.submit('t1', [1], base=base, dry_run=True)
    with pytest.raises(condor.CondorError):     # same tag, other max_files
        condor.prepare('t1', joblist, [1], max_files=3, base=base)
    condor.prepare('t1', joblist, [1], max_files=2, base=base)                   # extending is fine
    assert sorted(json.load(open(os.path.join(d, 'tag.json')))['jobs']) == ['0', '1', '2']

    # refresh: tarball repacked, code version appended
    before = len(json.load(open(os.path.join(d, 'tag.json')))['code'])
    v = condor.refresh_code('t1', base)
    meta = json.load(open(os.path.join(d, 'tag.json')))
    assert len(meta['code']) == before + 1 and meta['code'][-1]['version'] == v and v


def test_merge_flags_slices_without_golden_json():
    from svval.merge import no_golden_json_slices
    cf = {'2024G': {'unvetoed': 10, 'golden_json': 9}, '2027B': {'unvetoed': 5, 'golden_json': 0},
          '2025E': {'unvetoed': 0, 'golden_json': 0}}
    assert no_golden_json_slices(cf) == ['2027B']


def test_status_counts_unmaterialized_jobs_as_pending(tmp_path, joblist, monkeypatch):
    base = str(tmp_path / 'condor')
    condor.prepare('t2', joblist, [0, 1, 2], eos_out='/store/user/x/t2', base=base)
    d = condor.tag_dir('t2', base)
    condor.submit('t2', [0, 1, 2], base=base, dry_run=True, max_materialize=1)
    with open(os.path.join(d, 'clusters.txt'), 'w') as f:
        f.write('777 queue_000.txt 3 now\n')
    monkeypatch.setattr(condor, 'done_jobs', lambda eos: {0})
    monkeypatch.setattr(condor, 'queued_jobs', lambda tag: {1: 'running'})
    monkeypatch.setattr(condor, 'live_factories', lambda cl: {777})
    st = condor.status('t2', base)
    assert (st['done'], st['active'], st['pending'], st['failed']) == ([0], [1], [2], [])
    monkeypatch.setattr(condor, 'live_factories', lambda cl: set())      # factory gone: job 2 failed
    st = condor.status('t2', base)
    assert (st['pending'], st['failed']) == ([], [2])
