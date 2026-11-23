import os

import numpy as np
import pytest

from svval import catalog as cat
from svval.lumis import find_duplicate_lumis

CFG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'datasets_v34.yaml')

# Every request-name shape seen on EOS for v34 (2026-11-23): name, year, era, HIPM
REAL_REQUESTS = [
    ("kucmsntuple_MET_R16_SVHPM100_MiniAODv2_v34_MET_MINIAOD_Run2016C-HIPM_UL2016_MiniAODv2-v2", 2016, 'C', True),
    ("kucmsntuple_MET_R16_SVHPM100_MiniAODv2_v34_MET_MINIAOD_Run2016F-HIPM_UL2016_MiniAODv2-v2", 2016, 'F', True),
    ("kucmsntuple_MET_R16_SVHPM100_MiniAODv2_v34_MET_MINIAOD_Run2016F-UL2016_MiniAODv2-v2", 2016, 'F', False),
    ("kucmsntuple_MET_R16_SVHPM100_MiniAODv2_v34_MET_MINIAOD_Run2016H-UL2016_MiniAODv2-v2", 2016, 'H', False),
    ("kucmsntuple_MET_R16_SVHPM100_MiniAODv2_v34_MET_Run2016B-ver1_HIPM_UL2016_MiniAODv2-v2", 2016, 'B', True),
    ("kucmsntuple_MET_R16_SVHPM100_MiniAODv2_v34_MET_Run2016B-ver2_HIPM_UL2016_MiniAODv2-v2", 2016, 'B', True),
    ("kucmsntuple_MET_R17_SVHPM100_MiniAODv2_v34_MET_MINIAOD_Run2017F-UL2017_MiniAODv2-v1", 2017, 'F', False),
    ("kucmsntuple_MET_R18_SVHPM100_MiniAODv2_v34_MET_MINIAOD_Run2018D-UL2018_MiniAODv2_GT36-v1", 2018, 'D', False),
    ("kucmsntuple_JetMET_R22_SVHPM100_MiniAOD_v34_JetMET_MINIAOD_Run2022G-19Dec2023-v1", 2022, 'G', False),
    ("kucmsntuple_JetMET_R23_SVHPM100_MiniAOD_v34_PixSeedFix_JetMET1_MINIAOD_Run2023B-19Dec2023-v1", 2023, 'B', False),
    ("kucmsntuple_JetMET_R24_SVHPM100_MiniAOD_v34_JetMET0_MINIAOD_Run2024I-MINIv6NANOv15_v2-v1", 2024, 'I', False),
    ("kucmsntuple_JetMET_R25_SVHPM100_MiniAOD_v34_JetMET1_MINIAOD_Run2025D-PromptReco-v1", 2025, 'D', False),
    ("kucmsntuple_JetMET_R25_SVHPM100_v34_JetMET0_MINIAOD_Run2025Cv1", 2025, 'C', False),
    ("kucmsntuple_JetMET_R25_SVHPM100_v34_JetMET0_MINIAOD_Run2025E", 2025, 'E', False),
    ("kucmsntuple_JetMET_R25_SVHPM100_v34_recovery_JetMET0_MINIAOD_Run2025G", 2025, 'G', False),
    ("kucmsntuple_JetMET_R26_SVHPM100_MiniAOD_v34_JetMET1_MINIAOD_Run2026D-PromptReco-v1", 2026, 'D', False),
]


@pytest.fixture(scope='module')
def cfg():
    return cat.load_config(CFG_PATH)


@pytest.mark.parametrize("name,year,era,hipm", REAL_REQUESTS)
def test_parse_real_request_names(name, year, era, hipm):
    assert cat.parse_request(name, year) == (era, hipm)


def test_parse_rejects_wrong_year_and_missing_token():
    with pytest.raises(cat.CatalogError):
        cat.parse_request(REAL_REQUESTS[0][0], 2017)
    with pytest.raises(cat.CatalogError):
        cat.parse_request("kucmsntuple_something_without_era", 2016)


def test_periods(cfg):
    assert cat.period_of(cfg, 2016, 'F', True) == 'preVFP'
    assert cat.period_of(cfg, 2016, 'F', False) == 'postVFP'
    assert cat.period_of(cfg, 2022, 'D', False) == 'preEE'
    assert cat.period_of(cfg, 2022, 'E', False) == 'postEE'
    assert cat.period_of(cfg, 2023, 'C', False) == 'preBPix'
    assert cat.period_of(cfg, 2023, 'D', False) == 'postBPix'
    assert cat.period_of(cfg, 2024, 'C', False) == ''
    with pytest.raises(cat.CatalogError):
        cat.period_of(cfg, 2022, 'Z', False)


def test_aug31_r18d_not_configured(cfg):
    assert not any('R18D' in d['dir'] for d in cfg['datasets'])


# ---------------------------------------------------------------- scanning a fake EOS tree

def _touch(path, nbytes=10):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(b'x' * nbytes)


def _fake_cfg(tmp_path):
    return {
        'eos_base': '/store/group/test', 'fuse_prefix': str(tmp_path), 'xrootd_prefix': 'root://h/',
        'datasets': [{'dir': 'kucmsntuple_JetMET_R25_X', 'year': 2025}],
        'exclude': [{'pattern': 'backup/', 'reason': 'dup'}], 'periods': {},
    }


def test_scan_layout_and_exclusion(tmp_path):
    cfg = _fake_cfg(tmp_path)
    top = tmp_path / 'store/group/test/kucmsntuple_JetMET_R25_X'
    req = 'kucmsntuple_JetMET_R25_SVHPM100_v34_recovery_JetMET0_MINIAOD_Run2025G'
    _touch(str(top / 'JetMET0' / req / '260520_143517/0000/a_1.root'), 100)
    _touch(str(top / 'JetMET0' / req / '260520_143517/0009/a_9001.root'), 50)
    _touch(str(top / 'JetMET0' / req / '260520_143517/0000/log/cmsRun_1.log.tar.gz'))
    os.makedirs(str(top / 'JetMET0' / req / '260520_143517/0000/log'), exist_ok=True)
    _touch(str(top / 'backup' / req / '260615_213601/0000/a_1.root'), 100)

    # CRAB 'log/' dirs inside a block are not descended into; only *.root directly in a block count
    recs, excl = cat.scan(cfg)
    assert sorted(r.path for r in recs) == sorted([
        f'/store/group/test/kucmsntuple_JetMET_R25_X/JetMET0/{req}/260520_143517/0000/a_1.root',
        f'/store/group/test/kucmsntuple_JetMET_R25_X/JetMET0/{req}/260520_143517/0009/a_9001.root'])
    assert excl == {'backup/': 1}
    assert {(r.year, r.era, r.pd, r.size) for r in recs} == {(2025, 'G', 'JetMET0', 100), (2025, 'G', 'JetMET0', 50)}
    assert cat.url_for('root://h/', recs[0].path).startswith('root://h//store/')
    assert cat.url_for(str(tmp_path), recs[0].path).startswith(str(tmp_path) + '/store/')


def test_scan_rejects_unexpected_layout(tmp_path):
    cfg = _fake_cfg(tmp_path)
    top = tmp_path / 'store/group/test/kucmsntuple_JetMET_R25_X'
    _touch(str(top / 'JetMET0/kucmsntuple_x_Run2025E/not_a_stamp/0000/a.root'))
    with pytest.raises(cat.CatalogError):
        cat.scan(cfg)


def test_task_report_flags_double_stamp():
    def rec(req, stamp, era='C'):
        return cat.FileRecord(path=f'/s/{req}/{stamp}/0000/f.root', size=1, dataset='d', pd='JetMET0',
                              request=req, stamp=stamp, year=2025, era=era, period='')
    rep = cat.task_report([rec('r_Run2025Cv1', '1'), rec('r_Run2025Cv1', '2'), rec('r_Run2025Cv2', '3')])
    assert list(rep['duplicate_requests']) == ['d/JetMET0/r_Run2025Cv1']
    assert rep['multi_request_eras'] == {'2025C JetMET0': ['r_Run2025Cv1', 'r_Run2025Cv2']}


# ---------------------------------------------------------------- selection & chunks

def _recs():
    out = []
    for year, era, period, n in [(2016, 'F', 'preVFP', 3), (2016, 'F', 'postVFP', 2), (2024, 'C', '', 5),
                                 (2024, 'D', '', 1)]:
        for i in range(n):
            out.append(cat.FileRecord(path=f'/s/{year}{era}{period}/{i}.root', size=4, dataset='d', pd='P',
                                      request='r', stamp='s', year=year, era=era, period=period))
    return out


def test_selection():
    r = _recs()
    assert len(cat.select(r, None)) == len(r) == len(cat.select(r, 'all'))
    assert len(cat.select(r, 'run2')) == 5
    assert len(cat.select(r, '2016:preVFP')) == 3
    assert len(cat.select(r, '24C,2024D')) == 6
    assert len(cat.select(r, '2024c')) == 5
    with pytest.raises(cat.CatalogError):
        cat.parse_selection('2024CC')


def test_chunks_never_mix_eras_and_cover_all():
    r = _recs()
    ch = cat.make_chunks(r, target_bytes=9)
    files = [f for c in ch for f in c['files']]
    assert sorted(files) == sorted(x.path for x in r)
    for c in ch:
        assert len({p.split('/')[2] for p in c['files']}) == 1
        assert c['bytes'] <= 9 or len(c['files']) == 1
    assert [(c['year'], c['era'], c['period'], len(c['files'])) for c in ch] == [
        (2016, 'F', 'preVFP', 2), (2016, 'F', 'preVFP', 1), (2016, 'F', 'postVFP', 2),
        (2024, 'C', '', 2), (2024, 'C', '', 2), (2024, 'C', '', 1), (2024, 'D', '', 1)]


# ---------------------------------------------------------------- duplicate lumis

def _k(run, lumi):
    return (np.uint64(run) << np.uint64(32)) | np.uint64(lumi)


def test_duplicate_lumis():
    keys = {
        'a': np.array([_k(1, 1), _k(1, 2), _k(2, 7)], dtype=np.uint64),
        'b': np.array([_k(1, 3)], dtype=np.uint64),
        'c': np.array([_k(1, 2), _k(2, 7)], dtype=np.uint64),
        'e': np.zeros(0, dtype=np.uint64),
    }
    d = find_duplicate_lumis(keys)
    assert d['n_pairs'] == 2
    assert d['by_file_pair'] == {'a | c': 2}
    assert d['examples'] == [[1, 2], [2, 7]]
    assert find_duplicate_lumis({'a': keys['a'], 'b': keys['b']})['n_pairs'] == 0
