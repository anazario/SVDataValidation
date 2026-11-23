"""
Lumi-section bookkeeping of the catalog: reads only Evt_run / Evt_luminosityBlock.

CRAB data splitting never splits a lumi section across jobs, so a (run, lumi) pair present in
two files means the same data was processed twice (e.g. a recovery task overlapping its
parent). The scan also finds unreadable / empty files and records the events per file.

Per-file results are cached (one small .npz each), so rescans only open new files.
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .catalog import FileRecord, url_for

RUN_BRANCH = 'Evt_run'
LUMI_BRANCH = 'Evt_luminosityBlock'


def _cache_file(cache_dir: str, path: str, size: int) -> str:
    h = hashlib.sha1(f"{path}:{size}".encode()).hexdigest()
    return os.path.join(cache_dir, h[:2], h + '.npz')


def read_file_lumis(url: str, tree: str) -> Tuple[np.ndarray, int]:
    """Unique (run, lumi) pairs as one uint64 key array (run << 32 | lumi), and the entries."""
    import uproot
    with uproot.open(url) as f:
        t = f[tree]
        n = t.num_entries
        if n == 0:
            return np.zeros(0, dtype=np.uint64), 0
        a = t.arrays([RUN_BRANCH, LUMI_BRANCH], library='np')
    keys = (a[RUN_BRANCH].astype(np.uint64) << np.uint64(32)) | a[LUMI_BRANCH].astype(np.uint64)
    return np.unique(keys), n


def _work(args):
    path, size, url, tree, cache_dir = args
    cf = _cache_file(cache_dir, path, size)
    if os.path.exists(cf):
        z = np.load(cf)
        return path, z['keys'], int(z['n']), None
    try:
        keys, n = read_file_lumis(url, tree)
    except Exception as e:  # noqa: BLE001 - every failure is a finding here
        return path, None, 0, f"{type(e).__name__}: {e}"
    os.makedirs(os.path.dirname(cf), exist_ok=True)
    tmp = cf + f'.{os.getpid()}.tmp.npz'
    np.savez(tmp, keys=keys, n=n)
    os.replace(tmp, cf)
    return path, keys, n, None


def scan_lumis(records: Sequence[FileRecord], prefix: str, tree: str, cache_dir: str,
               workers: int = 8, progress=None) -> dict:
    """
    prefix: what to prepend to /store/... (FUSE '/eos/uscms' interactively, or the xrootd URL).
    Returns a report dict: files, events, bad_files, duplicate lumis (with the files involved).
    """
    tasks = [(r.path, r.size, url_for(prefix, r.path), tree, cache_dir) for r in records]
    by_path = {r.path: r for r in records}
    keys_of: Dict[str, np.ndarray] = {}
    events: Dict[str, int] = {}
    bad: Dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_work, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs)):
            path, keys, n, err = fut.result()
            if err:
                bad[path] = err
            else:
                keys_of[path] = keys
                events[path] = n
            if progress and (i + 1) % 500 == 0:
                progress(i + 1, len(tasks))

    dup = find_duplicate_lumis(keys_of)
    per_era = defaultdict(lambda: [0, 0, 0])   # files, events, lumis
    for p, n in events.items():
        r = by_path[p]
        e = per_era[f"{r.year}{r.era}"]
        e[0] += 1
        e[1] += n
        e[2] += len(keys_of[p])
    return {
        'n_files': len(records),
        'n_read': len(events),
        'bad_files': bad,
        'empty_files': sorted(p for p, n in events.items() if n == 0),
        'per_era': {k: {'files': v[0], 'events': v[1], 'lumis': v[2]} for k, v in sorted(per_era.items())},
        'duplicate_lumis': dup,
    }


def find_duplicate_lumis(keys_of: Dict[str, np.ndarray]) -> dict:
    """
    {'n_pairs': int, 'by_file_pair': {"fileA | fileB": n_shared}, 'examples': [[run, lumi], ...]}
    Sort-based, so it scales to the ~10^7 lumis of the full dataset.
    """
    paths = sorted(keys_of)
    if not paths:
        return {'n_pairs': 0, 'by_file_pair': {}, 'examples': []}
    lens = np.array([len(keys_of[p]) for p in paths])
    allk = np.concatenate([keys_of[p] for p in paths]) if lens.sum() else np.zeros(0, np.uint64)
    owner = np.repeat(np.arange(len(paths)), lens)
    order = np.argsort(allk, kind='stable')
    allk, owner = allk[order], owner[order]
    same = allk[1:] == allk[:-1]
    if not same.any():
        return {'n_pairs': 0, 'by_file_pair': {}, 'examples': []}
    dup_keys = np.unique(allk[1:][same])
    pairs = defaultdict(int)
    idx = np.searchsorted(allk, dup_keys)
    for k, start in zip(dup_keys, idx):
        stop = start
        while stop < len(allk) and allk[stop] == k:
            stop += 1
        files = sorted({paths[o] for o in owner[start:stop]})
        for a in range(len(files)):
            for b in range(a + 1, len(files)):
                pairs[f"{files[a]} | {files[b]}"] += 1
    ex = [[int(k >> np.uint64(32)), int(k & np.uint64(0xFFFFFFFF))] for k in dup_keys[:20]]
    return {'n_pairs': int(len(dup_keys)), 'by_file_pair': dict(sorted(pairs.items())), 'examples': ex}
