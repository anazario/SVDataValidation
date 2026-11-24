import json
import os

import pytest

from svval import clean as C

TAG = 'old_tag'
STORE = f'/store/user/me/SVDataValidation/v34/{TAG}'


@pytest.fixture
def area(tmp_path):
    """A tag with condor bookkeeping, local outputs and EOS job outputs + merge (FUSE under tmp)."""
    fuse, base, outputs = tmp_path / 'eos', tmp_path / 'condor', tmp_path / 'outputs'
    (base / TAG).mkdir(parents=True)
    (base / TAG / 'tag.json').write_text(json.dumps({'eos_out': STORE}))
    (outputs / TAG / 'merged').mkdir(parents=True)
    (outputs / TAG / 'merged' / 'all.root').write_bytes(b'x' * 10)
    eos = fuse / STORE.lstrip('/')
    (eos / 'merged').mkdir(parents=True)
    (eos / 'merged' / 'all.root').write_bytes(b'x' * 10)
    for i in range(3):
        (eos / f'job_{i:05d}.root').write_bytes(b'x' * 100)
        (eos / f'job_{i:05d}.json').write_text('{}')
        (eos / f'job_{i:05d}_lumis.npz').write_bytes(b'x')
    removed = []

    def rm_dir(store):                     # stands in for `eos rm -r`
        import shutil
        removed.append(store)
        shutil.rmtree(str(fuse) + store)

    kw = dict(base=str(base), outputs=str(outputs), fuse=str(fuse), user='me', rm_dir=rm_dir,
              queued=lambda tag: {}, log=lambda m: None)
    return kw, eos, base, outputs, removed


def test_full_clean_after_typing_the_tag(area):
    kw, eos, base, outputs, removed = area
    assert C.clean(TAG, ask=lambda q: TAG, **kw)
    assert removed == [STORE] and not eos.exists()
    assert not (outputs / TAG).exists() and not (base / TAG).exists()


def test_keep_merged_removes_only_job_files(area):
    kw, eos, base, outputs, removed = area
    t = C.plan(TAG, keep_merged=True, base=kw['base'], outputs=kw['outputs'], fuse=kw['fuse'], user='me')
    assert len(t) == 1 and t[0].n_files == 9 and t[0].n_bytes == 300 + 6 + 3
    assert C.clean(TAG, keep_merged=True, yes=True, **kw)
    assert sorted(os.listdir(eos)) == ['merged'] and removed == []
    assert (outputs / TAG / 'merged' / 'all.root').exists() and (base / TAG / 'tag.json').exists()


def test_dry_run_wrong_answer_and_queued_jobs_remove_nothing(area):
    kw, eos, base, outputs, removed = area
    assert not C.clean(TAG, dry_run=True, ask=lambda q: pytest.fail('dry run must not ask'), **kw)
    assert not C.clean(TAG, ask=lambda q: 'yes', **kw)                 # must type the tag itself
    kw['queued'] = lambda tag: {1: 'running'}
    with pytest.raises(C.CleanError, match='still in the condor queue'):
        C.clean(TAG, yes=True, **kw)
    assert len(os.listdir(eos)) == 10 and (outputs / TAG).exists() and (base / TAG).exists()


@pytest.mark.parametrize('path', ['/store/user/me', '/store/user/me/SVDataValidation/v34',
                                  f'/store/group/lpcsusylep/{TAG}', f'/store/user/other/{TAG}',
                                  f'/store/user/me/x/../{TAG}/..'])
def test_refuses_eos_paths_that_are_not_the_tag_dir(path):
    with pytest.raises(C.CleanError):
        C.check_eos_path(path, TAG, user='me')
    C.check_eos_path(STORE, TAG, user='me')


def test_cli_clean_dry_run_runs(capsys, monkeypatch):
    from svval import cli
    monkeypatch.setattr(C, 'eos_out_of', lambda tag, base=None: None)
    assert cli.main(['clean', '--tag', 'no_such_tag_xyz', '--dry-run']) == 0
    assert 'nothing to clean' in capsys.readouterr().out
