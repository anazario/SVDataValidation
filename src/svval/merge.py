"""
Merge stage: job outputs of a tag (on EOS) -> merged histograms, lumi table and checks.

  <outdir>/eras/<year><era>.root   hadd of the era's jobs (slice directories kept)
  <outdir>/all.root                hadd of the eras: every slice of the tag in one file
  <outdir>/lumis.npz               every (file, run, lumi) row of every job, with year / era / slice / PD
  <outdir>/cutflow.json            cutflow per slice, summed over jobs
  <outdir>/checks.json             completeness, duplicate lumis (per year and PD), blinding audit

Fails (exit != 0) if jobs are missing (unless allowed), lumis are duplicated, or the audit finds
anything.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional

import numpy as np

from . import condor
from . import histos
from .lumis import find_duplicate_lumis


class MergeError(RuntimeError):
    pass


def _hadd(out: str, inputs: List[str], jobs: int = 4):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + '.tmp.root'
    cmd = ['hadd', '-f', '-j', str(jobs), tmp] + inputs
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if p.returncode != 0:
        raise MergeError(f"hadd failed for {out}:\n{p.stdout[-2000:]}")
    os.replace(tmp, out)


def audit_file(path: str) -> List[str]:
    """Blinding audit of a (merged) ROOT file: every dxySig histogram of every slice."""
    import uproot
    bad = []
    with uproot.open(path) as f:
        for key, cls in f.classnames().items():
            name = key.split(';')[0]
            if not (cls.startswith('TH1') and name.endswith('/dxySig')):
                continue
            h = f[key]
            cat = name.split('/')[-3]
            edges = h.axis().edges()
            if abs(edges[-1] - histos.dxysig_max(cat)) > 1e-9 or h.values(flow=True)[-1] > 0:
                bad.append(name)
    return bad


def merge(tag: str, outdir: str, allow_missing: bool = False, base: Optional[str] = None,
          hadd_jobs: int = 4, log=print) -> dict:
    st = condor.status(tag, base)
    meta = st['meta']
    ids, done = st['ids'], set(st['done'])
    missing = sorted(set(ids) - done)
    if missing and not allow_missing:
        raise MergeError(f"{len(missing)} of {len(ids)} jobs are not done (e.g. {missing[:10]}); "
                         "resubmit, or merge with --allow-missing")
    src = condor.FUSE + meta['eos_out']
    use = sorted(done)
    log(f"merging {len(use)} jobs of tag {tag} from {src}")

    # ---- per-job metadata, cutflow, lumi tables
    by_era: Dict[str, List[str]] = defaultdict(list)
    cutflow: Dict[str, Dict[str, int]] = OrderedDict()
    lumi_parts, paths = [], []
    keys_by_group: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
    pd_of = _pd_lookup(meta['manifest'])
    for i in use:
        stem = os.path.join(src, f'job_{i:05d}')
        with open(stem + '.json') as f:
            jm = json.load(f)
        by_era[f"{jm['year']}{jm['era']}"].append(stem + '.root')
        for sl, fl in jm['cutflow'].items():
            c = cutflow.setdefault(sl, OrderedDict())
            for k, v in fl.items():
                c[k] = c.get(k, 0) + v
        z = np.load(stem + '_lumis.npz')
        t, files = z['lumis'], [str(x) for x in z['files']]
        if len(files) != len(jm['files']):
            raise MergeError(f"job {i}: lumi table lists {len(files)} files, metadata {len(jm['files'])}")
        offset = len(paths)
        paths.extend(files)
        lumi_parts.append((t, offset, jm['year'], jm['era'], i))
        for fi, path in enumerate(files):
            rows = t[t['file'] == fi]
            k = (rows['run'].astype(np.uint64) << np.uint64(32)) | rows['lumi'].astype(np.uint64)
            keys_by_group[f"{jm['year']} {pd_of.get(path, '?')}"][path] = np.unique(k)

    # ---- duplicate lumis across all jobs (per year and PD)
    dups = {g: find_duplicate_lumis(ks) for g, ks in sorted(keys_by_group.items())}
    n_dup = sum(d['n_pairs'] for d in dups.values())
    log(f"duplicate lumis (per year and PD): {n_dup}")

    # ---- hadd: eras, then everything
    era_files = []
    for era in sorted(by_era):
        out = os.path.join(outdir, 'eras', f'{era}.root')
        log(f"  hadd {era}: {len(by_era[era])} jobs")
        _hadd(out, by_era[era], hadd_jobs)
        era_files.append(out)
    _hadd(os.path.join(outdir, 'all.root'), era_files, hadd_jobs)

    # ---- merged lumi table
    rows = []
    for t, offset, year, era, job in lumi_parts:
        r = np.zeros(len(t), dtype=[('path', np.int32), ('job', np.int32), ('year', np.int16),
                                    ('era', 'U1')] + [(n, t.dtype[n]) for n in t.dtype.names if n != 'file'])
        r['path'] = t['file'] + offset
        r['job'], r['year'], r['era'] = job, year, era
        for n in t.dtype.names:
            if n != 'file':
                r[n] = t[n]
        rows.append(r)
    lumis = np.concatenate(rows) if rows else np.zeros(0)
    np.savez_compressed(os.path.join(outdir, 'lumis.npz'), lumis=lumis, paths=np.array(paths))

    # ---- audit and summary
    bad = audit_file(os.path.join(outdir, 'all.root'))
    no_json = no_golden_json_slices(cutflow)
    checks = {'tag': tag, 'jobs': len(ids), 'merged': len(use), 'missing': missing,
              'duplicate_lumis': {g: d['n_pairs'] for g, d in dups.items()},
              'duplicate_examples': {g: d['by_file_pair'] for g, d in dups.items() if d['n_pairs']},
              'blinding_audit_failures': bad, 'slices_without_golden_json': no_json}
    with open(os.path.join(outdir, 'cutflow.json'), 'w') as f:
        json.dump(cutflow, f, indent=1)
    with open(os.path.join(outdir, 'checks.json'), 'w') as f:
        json.dump(checks, f, indent=1)
    if bad:
        raise MergeError(f"BLINDING AUDIT FAILED on the merged file: {bad[:10]}")
    if n_dup:
        raise MergeError(f"{n_dup} duplicate lumis, see {outdir}/checks.json")
    if no_json:
        raise MergeError(f"slices with events but none in the golden JSON: {no_json}. Add the year's "
                         "certification JSON to configs/golden_json/ and rerun the jobs")
    return checks


def no_golden_json_slices(cutflow: Dict[str, Dict[str, int]]) -> List[str]:
    """Slices whose events all fail the golden JSON: almost surely a missing certification file
    (the lumi mask treats a run absent from every JSON as bad), not bad data."""
    return sorted(sl for sl, c in cutflow.items() if c.get('unvetoed', 0) > 0 and c.get('golden_json', 0) == 0)


def _pd_lookup(manifest: str) -> Dict[str, str]:
    with open(manifest) as f:
        man = json.load(f)
    return {r['path']: r['pd'] for r in man['files']}
