"""
Merge stage, on the pattern of KUCMSSkimmer's skim_submit.py --transfer: the per-job outputs of a
tag are staged from EOS to local scratch with xrdcp (several streams at once), merged there with
hadd (independent eras at the same time; no -k, so a corrupt input fails the merge), checked, and
the merged products are copied with xrdcp to their EOS destination. A local copy is kept for
`svval plot`, and the scratch area is removed.

Products (EOS: <eos_out>/merged/ by default; local: outputs/<tag>/merged/):
  eras/<year><era>.root   hadd of the era's jobs (slice directories kept)
  all.root                hadd of the eras: every slice of the tag in one file
  lumis.npz               every (file, run, lumi) row of every job, with year / era / slice / PD
  cutflow.json            cutflow per slice, summed over jobs
  checks.json             completeness, duplicate lumis (per year and PD), blinding audit, golden
                          JSON, per-year file coverage. Copied last: it marks a complete upload.

Fails (exit != 0, nothing uploaded) if jobs are missing (unless allowed), a staging or hadd fails,
lumis are duplicated, the blinding audit finds anything, or a slice has no golden-JSON event.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np

from . import condor
from . import histos
from .catalog import url_for
from .fill import slice_names
from .lumis import find_duplicate_lumis

XRD = 'root://cmseos.fnal.gov/'
DEFAULT_SCRATCH = '/uscmst1b_scratch/lpc1/3DayLifetime/$USER'      # as skim_submit.py


class MergeError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- transport (xrootd)

def xrdcp(src: str, dst: str, force: bool = False):
    """xrdcp src dst (root:// URL or local path on either side); raises MergeError on failure."""
    cmd = ['xrdcp', '-s', '--retry', '3'] + (['-f'] if force else []) + [src, dst]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if p.returncode != 0:
        raise MergeError(f"xrdcp {src} -> {dst} failed:\n{p.stdout[-1000:]}")


def eos_exists(store_path: str) -> bool:
    p = subprocess.run(['xrdfs', XRD.rstrip('/'), 'stat', store_path], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    return p.returncode == 0


def eos_mkdir(store_path: str):
    p = subprocess.run(['xrdfs', XRD.rstrip('/'), 'mkdir', '-p', store_path], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True)
    if p.returncode != 0:
        raise MergeError(f"cannot create {store_path} on EOS:\n{p.stdout}")


def _parallel(fn, items, workers: int, what: str, log=None):
    """Run fn(item) for every item on `workers` threads (the work is in subprocesses); re-raise the
    first failure after the others have finished."""
    errors, done = [], 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(fn, x) for x in items]
        for f in futures:
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                errors.append(e)
            done += 1
            if log and done % 500 == 0:
                log(f"    {what}: {done}/{len(items)}")
    if errors:
        raise MergeError(f"{len(errors)} {what} failures, e.g. {errors[0]}")


def _hadd(out: str, inputs: List[str], jobs: int, scratch: str):
    """hadd -f -j N -d scratch, written to a temporary name first; no -k (corrupt input = failure)."""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + '.tmp.root'
    cmd = ['hadd', '-f', '-j', str(jobs), '-v', '0', '-d', scratch, tmp] + inputs
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
          scratch: str = DEFAULT_SCRATCH, dest: Optional[str] = None, upload: bool = True,
          force: bool = False, workers: int = 8, era_workers: int = 4, hadd_jobs: int = 2,
          keep_scratch: bool = False, dry_run: bool = False, log=print) -> dict:
    """
    dest: EOS directory (/store/...) of the merged products, default <eos_out>/merged.
    workers: parallel xrdcp streams for staging; era_workers x hadd_jobs: concurrent hadd processes.
    """
    t0 = time.time()
    st = condor.status(tag, base)
    meta = st['meta']
    ids, done = st['ids'], set(st['done'])
    missing = sorted(set(ids) - done)
    if missing and not allow_missing:
        raise MergeError(f"{len(missing)} of {len(ids)} jobs are not done (e.g. {missing[:10]}); "
                         "resubmit, or merge with --allow-missing")
    use = sorted(done)
    dest = dest or meta['eos_out'].rstrip('/') + '/merged'
    work = os.path.join(os.path.expandvars(scratch), f"svval_merge_{tag}")
    log(f"merging {len(use)} jobs of tag {tag}: stage {meta['eos_out']} -> {work}, "
        f"merged -> {dest if upload else '(no upload)'} and {outdir}")
    if upload and not force and eos_exists(dest + '/checks.json'):
        raise MergeError(f"{dest} already holds a merge (checks.json); use --force to overwrite")
    if dry_run:
        log(f"  [dry-run] stage {3 * len(use)} files with {workers} xrdcp streams, hadd per era "
            f"({era_workers} at once, -j {hadd_jobs}), upload {'to ' + dest if upload else 'skipped'}")
        return {'tag': tag, 'jobs': len(ids), 'merged': len(use), 'missing': missing, 'dry_run': True}

    # ---- stage the job outputs (always from scratch: leftovers of an interrupted run are not trusted)
    if os.path.exists(work):
        shutil.rmtree(work)
    jobs_dir, mdir = os.path.join(work, 'jobs'), os.path.join(work, 'merged')
    os.makedirs(jobs_dir)
    os.makedirs(os.path.join(mdir, 'eras'))
    transfers = [(url_for(XRD, f"{meta['eos_out']}/job_{i:05d}{ext}"), os.path.join(jobs_dir, f"job_{i:05d}{ext}"))
                 for i in use for ext in ('.json', '_lumis.npz', '.root')]
    log(f"  staging {len(transfers)} files ({workers} xrdcp streams)")
    _parallel(lambda t: xrdcp(t[0], t[1], force=True), transfers, workers, 'staging', log)

    # ---- per-job metadata, cutflow, lumi tables
    by_era: Dict[str, List[str]] = defaultdict(list)
    cutflow: Dict[str, Dict[str, int]] = OrderedDict()
    lumi_parts, paths = [], []
    keys_by_group: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
    pd_of = _pd_lookup(meta['manifest'])
    files_per_year = _files_per_year(meta['manifest'])
    merged_files = defaultdict(set)
    for i in use:
        stem = os.path.join(jobs_dir, f'job_{i:05d}')
        with open(stem + '.json') as f:
            jm = json.load(f)
        by_era[f"{jm['year']}{jm['era']}"].append(stem + '.root')
        for sl, fl in jm['cutflow'].items():
            c = cutflow.setdefault(sl, OrderedDict())
            for k, v in fl.items():
                c[k] = c.get(k, 0) + v
        with np.load(stem + '_lumis.npz') as z:        # closed: open handles block deleting scratch on NFS
            t, files = z['lumis'], [str(x) for x in z['files']]
        if len(files) != len(jm['files']):
            raise MergeError(f"job {i}: lumi table lists {len(files)} files, metadata {len(jm['files'])}")
        offset = len(paths)
        paths.extend(files)
        merged_files[int(jm['year'])].update(files)
        lumi_parts.append((t, offset, jm['year'], jm['era'], jm['period'], i))
        for fi, path in enumerate(files):
            rows = t[t['file'] == fi]
            k = (rows['run'].astype(np.uint64) << np.uint64(32)) | rows['lumi'].astype(np.uint64)
            keys_by_group[f"{jm['year']} {pd_of.get(path, '?')}"][path] = np.unique(k)

    # ---- duplicate lumis across all jobs (per year and PD)
    dups = {g: find_duplicate_lumis(ks) for g, ks in sorted(keys_by_group.items())}
    n_dup = sum(d['n_pairs'] for d in dups.values())
    log(f"  duplicate lumis (per year and PD): {n_dup}")

    # ---- hadd: eras at the same time, then everything
    eras = sorted(by_era)
    era_files = [os.path.join(mdir, 'eras', f'{e}.root') for e in eras]
    log(f"  hadd of {len(eras)} eras ({era_workers} at once, -j {hadd_jobs})")
    _parallel(lambda k: _hadd(era_files[k], by_era[eras[k]], hadd_jobs, work), list(range(len(eras))),
              era_workers, 'hadd')
    _hadd(os.path.join(mdir, 'all.root'), era_files, max(hadd_jobs, 1) * max(era_workers, 1), work)

    # ---- merged lumi table
    rows = []
    for t, offset, year, era, period, job in lumi_parts:
        r = np.zeros(len(t), dtype=[('path', np.int32), ('job', np.int32), ('year', np.int16),
                                    ('era', 'U1'), ('slice', 'U24')]
                     + [(n, t.dtype[n]) for n in t.dtype.names if n != 'file'])
        r['path'] = t['file'] + offset
        r['job'], r['year'], r['era'] = job, year, era
        r['slice'] = slice_names(year, era, period, t['run'])   # same names as the histogram slices
        for n in t.dtype.names:
            if n != 'file':
                r[n] = t[n]
        rows.append(r)
    lumis = np.concatenate(rows) if rows else np.zeros(0)
    np.savez_compressed(os.path.join(mdir, 'lumis.npz'), lumis=lumis, paths=np.array(paths))

    # ---- audit and summary
    bad = audit_file(os.path.join(mdir, 'all.root'))
    no_json = no_golden_json_slices(cutflow)
    checks = {'tag': tag, 'jobs': len(ids), 'merged': len(use), 'missing': missing,
              'duplicate_lumis': {g: d['n_pairs'] for g, d in dups.items()},
              'duplicate_examples': {g: d['by_file_pair'] for g, d in dups.items() if d['n_pairs']},
              'blinding_audit_failures': bad, 'slices_without_golden_json': no_json,
              # fraction of each year's catalogued ntuple files in this merge (1 = the whole year):
              # luminosity labels need 1
              'file_coverage': {str(y): len(f) / files_per_year[y] if files_per_year.get(y) else 0.0
                                for y, f in sorted(merged_files.items())},
              'destination': dest if upload else None}
    with open(os.path.join(mdir, 'cutflow.json'), 'w') as f:
        json.dump(cutflow, f, indent=1)
    with open(os.path.join(mdir, 'checks.json'), 'w') as f:
        json.dump(checks, f, indent=1)

    # ---- local copy (for svval plot) before any failure, so a failing merge can be inspected
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    shutil.copytree(mdir, outdir)
    if bad:
        raise MergeError(f"BLINDING AUDIT FAILED on the merged file: {bad[:10]} (nothing uploaded; "
                         f"scratch kept in {work})")
    if n_dup:
        raise MergeError(f"{n_dup} duplicate lumis, see {outdir}/checks.json (nothing uploaded)")
    if no_json:
        raise MergeError(f"slices with events but none in the golden JSON: {no_json}. Add the year's "
                         "certification JSON to configs/golden_json/ and rerun the jobs (nothing uploaded)")

    # ---- upload: products first, checks.json last (marks a complete merge on EOS)
    if upload:
        eos_mkdir(dest + '/eras')
        uploads = [(os.path.join(mdir, 'eras', f'{e}.root'), f"{dest}/eras/{e}.root") for e in eras]
        uploads += [(os.path.join(mdir, n), f"{dest}/{n}") for n in ('all.root', 'lumis.npz', 'cutflow.json')]
        log(f"  xrdcp {len(uploads) + 1} merged files -> {dest}")
        _parallel(lambda u: xrdcp(u[0], url_for(XRD, u[1]), force=True), uploads, workers, 'upload')
        xrdcp(os.path.join(mdir, 'checks.json'), url_for(XRD, f"{dest}/checks.json"), force=True)
    if not keep_scratch:
        shutil.rmtree(work)
    log(f"  done in {time.time() - t0:.0f}s")
    return checks


def no_golden_json_slices(cutflow: Dict[str, Dict[str, int]]) -> List[str]:
    """Slices whose events all fail the golden JSON: almost surely a missing certification file
    (the lumi mask treats a run absent from every JSON as bad), not bad data."""
    return sorted(sl for sl, c in cutflow.items() if c.get('unvetoed', 0) > 0 and c.get('golden_json', 0) == 0)


def _files_per_year(manifest: str) -> Dict[int, int]:
    with open(manifest) as f:
        man = json.load(f)
    out = defaultdict(int)
    for r in man['files']:
        out[int(r['year'])] += 1
    return dict(out)


def _pd_lookup(manifest: str) -> Dict[str, str]:
    with open(manifest) as f:
        man = json.load(f)
    return {r['path']: r['pd'] for r in man['files']}
