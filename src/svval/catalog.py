"""
File catalog of the data ntuples: which file belongs to which year / era / period.

EOS layout of a dataset (one level per CRAB convention):

    <dataset dir>/<primary dataset>/<CRAB request>/<task timestamp>/<NNNN>/<file>.root
    kucmsntuple_JetMET_R24_.../JetMET0/kucmsntuple_..._JetMET0_MINIAOD_Run2024C-MINIv6NANOv15-v1/260708_214317/0000/x.root

The era comes from the request name (`Run20YYX`). The names are not uniform across years
(`Run2016B-ver2_HIPM`, `Run2025Cv1`, `Run2025E`, `..._recovery_..._Run2025G`), so the
parser only relies on the `Run20YY<letter>` token and on `HIPM` for 2016.

Python 3.9 compatible (the LPC CMSSW_15 python).
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

RUN2_YEARS = (2016, 2017, 2018)
RUN3_YEARS = (2022, 2023, 2024, 2025, 2026)

_ERA_RE = re.compile(r'Run20(\d\d)([A-Z])')
_STAMP_RE = re.compile(r'^\d{6}_\d{6}$')
_BLOCK_RE = re.compile(r'^\d{4}$')


class CatalogError(RuntimeError):
    pass


@dataclass(frozen=True)
class FileRecord:
    path: str      # /store/... (no prefix)
    size: int      # bytes
    dataset: str   # dataset dir name
    pd: str        # primary dataset (MET, JetMET, JetMET0, JetMET1)
    request: str   # CRAB request name
    stamp: str     # CRAB task timestamp
    year: int
    era: str       # single letter
    period: str    # detector period, '' if the year has none

    @property
    def era_label(self) -> str:
        return f"{self.year}{self.era}"


# ---------------------------------------------------------------------------- config

def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ('eos_base', 'fuse_prefix', 'xrootd_prefix', 'datasets'):
        if key not in cfg:
            raise CatalogError(f"{path}: missing '{key}'")
    cfg.setdefault('exclude', [])
    cfg.setdefault('periods', {})
    cfg['periods'] = {int(k): v for k, v in cfg['periods'].items()}
    return cfg


# ---------------------------------------------------------------------------- parsing

def parse_request(request: str, expected_year: int) -> Tuple[str, bool]:
    """CRAB request name -> (era letter, is_HIPM). The year must match the dataset's."""
    m = _ERA_RE.search(request)
    if not m:
        raise CatalogError(f"no Run20YYX token in request name '{request}'")
    year = 2000 + int(m.group(1))
    if year != expected_year:
        raise CatalogError(f"request '{request}' is {year}, but its dataset is {expected_year}")
    return m.group(2), ('HIPM' in request)


def period_of(cfg: dict, year: int, era: str, hipm: bool) -> str:
    rule = cfg['periods'].get(year)
    if not rule:
        return ''
    if rule['rule'] == 'hipm':
        return rule['hipm'] if hipm else rule['nohipm']
    if rule['rule'] == 'era':
        for name, eras in rule.items():
            if name != 'rule' and era in eras:
                return name
        raise CatalogError(f"{year}{era}: era not assigned to any period in the config")
    raise CatalogError(f"{year}: unknown period rule '{rule['rule']}'")


def is_excluded(cfg: dict, relpath: str) -> Optional[str]:
    for ex in cfg['exclude']:
        if ex['pattern'] in relpath:
            return ex['pattern']
    return None


# ---------------------------------------------------------------------------- scanning

def _scandir_sorted(path: str):
    with os.scandir(path) as it:
        return sorted(it, key=lambda e: e.name)


def scan(cfg: dict, root: Optional[str] = None) -> Tuple[List[FileRecord], Dict[str, int]]:
    """
    Walk every configured dataset dir under <root><eos_base> (root defaults to the FUSE
    prefix). Returns (records, excluded counts per pattern). Anything that does not fit the
    CRAB layout raises, so a new directory shape is noticed instead of silently skipped.
    """
    root = cfg['fuse_prefix'] if root is None else root
    base = root.rstrip('/') + cfg['eos_base']
    records: List[FileRecord] = []
    excluded: Dict[str, int] = defaultdict(int)

    for ds in cfg['datasets']:
        ds_dir, year = ds['dir'], int(ds['year'])
        top = os.path.join(base, ds_dir)
        if not os.path.isdir(top):
            raise CatalogError(f"dataset dir not found: {top}")
        for pd in _scandir_sorted(top):
            if not pd.is_dir():
                continue
            if is_excluded(cfg, pd.name + '/'):
                excluded[is_excluded(cfg, pd.name + '/')] += sum(
                    len(fs) for _, _, fs in os.walk(pd.path))
                continue
            for req in _scandir_sorted(pd.path):
                if not req.is_dir():
                    continue
                era, hipm = parse_request(req.name, year)
                period = period_of(cfg, year, era, hipm)
                for stamp in _scandir_sorted(req.path):
                    if not (stamp.is_dir() and _STAMP_RE.match(stamp.name)):
                        raise CatalogError(f"unexpected entry (not a CRAB task stamp): {stamp.path}")
                    for block in _scandir_sorted(stamp.path):
                        if not (block.is_dir() and _BLOCK_RE.match(block.name)):
                            raise CatalogError(f"unexpected entry (not a NNNN block): {block.path}")
                        for f in _scandir_sorted(block.path):
                            if not f.name.endswith('.root'):
                                continue      # CRAB log tarballs etc.
                            rel = os.path.relpath(f.path, top)
                            pat = is_excluded(cfg, rel)
                            if pat:
                                excluded[pat] += 1
                                continue
                            records.append(FileRecord(
                                path=f.path[len(root.rstrip('/')):], size=f.stat().st_size,
                                dataset=ds_dir, pd=pd.name, request=req.name, stamp=stamp.name,
                                year=year, era=era, period=period))
    return records, dict(excluded)


# ---------------------------------------------------------------------------- checks

def task_report(records: Sequence[FileRecord]) -> dict:
    """
    Structural duplicate checks (no file is opened):
      duplicate_requests : one request name with more than one task stamp -> almost surely a
                           double processing (error).
      multi_request_eras : one (year, era, PD) made by several requests. Often legitimate
                           (2024I / 2025C / 2025F have v1 and v2 datasets), but listed so it is
                           reviewed; the lumi check settles overlaps.
      empty_files        : 0-byte files.
    """
    stamps = defaultdict(set)
    reqs = defaultdict(set)
    for r in records:
        stamps[(r.dataset, r.pd, r.request)].add(r.stamp)
        reqs[(r.year, r.era, r.pd)].add(r.request)
    return {
        'duplicate_requests': {'/'.join(k): sorted(v) for k, v in stamps.items() if len(v) > 1},
        'multi_request_eras': {f"{y}{e} {pd}": sorted(v)
                               for (y, e, pd), v in sorted(reqs.items()) if len(v) > 1},
        'empty_files': [r.path for r in records if r.size == 0],
    }


# ---------------------------------------------------------------------------- manifest I/O

def write_manifest(path: str, cfg: dict, records: Sequence[FileRecord],
                   excluded: Dict[str, int], report: dict) -> None:
    out = {
        'version': cfg.get('version'),
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'xrootd_prefix': cfg['xrootd_prefix'],
        'fuse_prefix': cfg['fuse_prefix'],
        'tree': cfg.get('tree', 'tree/llpgtree'),
        'excluded': excluded,
        'report': report,
        'files': [asdict(r) for r in records],
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(out, f, indent=0)
    os.replace(tmp, path)


def url_for(prefix: str, store_path: str) -> str:
    """'/eos/uscms' + '/store/x' -> '/eos/uscms/store/x';  'root://h/' + '/store/x' -> 'root://h//store/x'."""
    if prefix.startswith('root://'):
        return prefix.rstrip('/') + '/' + store_path
    return prefix.rstrip('/') + store_path


def read_manifest(path: str) -> Tuple[dict, List[FileRecord]]:
    with open(path) as f:
        man = json.load(f)
    return man, [FileRecord(**d) for d in man.pop('files')]


# ---------------------------------------------------------------------------- selection

def parse_selection(spec: Optional[str]) -> Optional[List[Tuple[int, Optional[str], Optional[str]]]]:
    """
    'all' | 'run2' | 'run3' | comma list of '2024', '2024C', '2016:preVFP'
    -> list of (year, era or None, period or None); None means everything.
    """
    if spec is None or spec.strip().lower() == 'all':
        return None
    out = []
    for tok in (t.strip() for t in spec.split(',') if t.strip()):
        low = tok.lower()
        if low in ('run2', 'run3'):
            out += [(y, None, None) for y in (RUN2_YEARS if low == 'run2' else RUN3_YEARS)]
            continue
        m = re.fullmatch(r'(?:20)?(\d\d)([A-Za-z])?(?::(\w+))?', tok)
        if not m:
            raise CatalogError(f"bad selection token '{tok}' (e.g. 2024, 2024C, 2016:preVFP, run3)")
        out.append((2000 + int(m.group(1)), m.group(2).upper() if m.group(2) else None, m.group(3)))
    return out


def select(records: Iterable[FileRecord], spec: Optional[str]) -> List[FileRecord]:
    sel = parse_selection(spec)
    if sel is None:
        return list(records)
    keep = []
    for r in records:
        for year, era, period in sel:
            if r.year == year and (era is None or r.era == era) and (period is None or r.period == period):
                keep.append(r)
                break
    return keep


# ---------------------------------------------------------------------------- summaries & chunks

def group_key(r: FileRecord) -> Tuple[int, str, str]:
    """A job never mixes these, so every job output belongs to exactly one era and period."""
    return (r.year, r.era, r.period)


def summarize(records: Sequence[FileRecord]) -> "OrderedDict[Tuple[int, str, str, str], Tuple[int, int]]":
    acc = defaultdict(lambda: [0, 0])
    for r in records:
        a = acc[(r.year, r.era, r.period, r.pd)]
        a[0] += 1
        a[1] += r.size
    return OrderedDict((k, tuple(v)) for k, v in sorted(acc.items()))


def format_summary(records: Sequence[FileRecord]) -> str:
    lines = [f"{'era':<7}{'period':<10}{'PD':<9}{'files':>8}{'GB':>9}"]
    yr_files = defaultdict(int)
    yr_bytes = defaultdict(int)
    for (year, era, period, pd), (n, b) in summarize(records).items():
        lines.append(f"{year}{era:<3}{period:<10}{pd:<9}{n:>8}{b / 1e9:>9.1f}")
        yr_files[year] += n
        yr_bytes[year] += b
    lines.append('-' * 43)
    for y in sorted(yr_files):
        lines.append(f"{y:<26}{yr_files[y]:>8}{yr_bytes[y] / 1e9:>9.1f}")
    lines.append(f"{'total':<26}{sum(yr_files.values()):>8}{sum(yr_bytes.values()) / 1e9:>9.1f}")
    return '\n'.join(lines)


def make_chunks(records: Sequence[FileRecord], target_bytes: int) -> List[dict]:
    """
    Split into jobs of about target_bytes each, never across (year, era, period).
    Files keep manifest order, so a job reads neighbouring files of one CRAB task.
    """
    if target_bytes <= 0:
        raise CatalogError("target size must be positive")
    groups = OrderedDict()
    for r in records:
        groups.setdefault(group_key(r), []).append(r)
    chunks = []
    for (year, era, period), files in groups.items():
        cur, size = [], 0
        for r in files:
            if cur and size + r.size > target_bytes:
                chunks.append((year, era, period, cur, size))
                cur, size = [], 0
            cur.append(r)
            size += r.size
        if cur:
            chunks.append((year, era, period, cur, size))
    out = []
    for i, (year, era, period, files, size) in enumerate(chunks):
        out.append({'job': i, 'year': year, 'era': era, 'period': period,
                    'bytes': size, 'files': [r.path for r in files]})
    return out
