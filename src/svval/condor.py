"""
HTCondor submission on the LPC.

A submission ("tag") lives in condor/<tag>/ on the submit node:
  tag.json          what was submitted: job list, EOS output dir, max_files
  specs/            one self-contained job spec per job (job_NNNNN.json): the only per-job input
  svval_code.tgz    src/ + configs/ (the golden JSONs) at submission time
  run_job.sh        worker wrapper: CMSSW env from cvmfs, `svval fill`, xrdcp to EOS (.json last)
  submit.jdl        condor description; one `queue job from` file per submission round
  clusters.txt      cluster ids of every round
  logs/             stdout / stderr / condor logs

Outputs go to <eos_out>/job_NNNNN{.root,_lumis.npz,.json}; a job is done when its .json is on EOS.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tarfile
import time
from typing import Dict, Iterable, List, Optional, Set

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CMSSW = 'CMSSW_15_1_0_patch5'
SCRAM_ARCH = 'el9_amd64_gcc12'
XRD = 'root://cmseos.fnal.gov/'
FUSE = '/eos/uscms'

WRAPPER = r'''#!/bin/bash
# svval condor wrapper: run_job.sh <job id, 5 digits> <EOS output dir (/store/...)>
JOB=$1
EOSOUT=$2
echo "svval job ${JOB} on $(hostname) at $(date)"
source /cvmfs/cms.cern.ch/cmsset_default.sh || exit 90
export SCRAM_ARCH=@SCRAM_ARCH@
scram project -n cmssw CMSSW @CMSSW@ > /dev/null || exit 91
pushd cmssw/src > /dev/null || exit 92
eval `scramv1 runtime -sh` || exit 92
popd > /dev/null
tar xzf svval_code.tgz || exit 93
export PYTHONPATH=$PWD/src
python3 -m svval fill --job-file job_${JOB}.json -o out || exit 94
# the .json goes last: its presence on EOS marks a complete job
for ext in .root _lumis.npz .json; do
  xrdcp -f --retry 3 out/job_${JOB}${ext} @XRD@${EOSOUT}/job_${JOB}${ext} || exit 95
done
echo "svval job ${JOB} done at $(date)"
'''

JDL = '''universe                = vanilla
executable              = run_job.sh
arguments               = $(job) {eos_out}
transfer_input_files    = svval_code.tgz, specs/job_$(job).json
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_output_files   = ""
output                  = logs/job_$(job).$(Cluster).out
error                   = logs/job_$(job).$(Cluster).err
log                     = logs/cluster_$(Cluster).log
request_cpus            = 1
request_memory          = 2000
request_disk            = 2000000
use_x509userproxy       = true
+SVValTag               = "{tag}"
+SVValJob               = "$(job)"
'''


class CondorError(RuntimeError):
    pass


def tag_dir(tag: str, base: Optional[str] = None) -> str:
    return os.path.join(base or os.path.join(REPO, 'condor'), tag)


def default_eos_out(version: str, tag: str) -> str:
    user = os.environ.get('USER', 'anazario')
    return f"/store/user/{user}/SVDataValidation/{version}/{tag}"


def parse_ids(spec: Optional[str], all_ids: Iterable[int]) -> List[int]:
    """'1,4,10-12' -> [1, 4, 10, 11, 12]; None -> all."""
    all_ids = sorted(all_ids)
    if not spec:
        return all_ids
    out = set()
    for tok in spec.split(','):
        tok = tok.strip()
        m = re.fullmatch(r'(\d+)(?:-(\d+))?', tok)
        if not m:
            raise CondorError(f"bad job id token '{tok}'")
        a = int(m.group(1))
        b = int(m.group(2) or a)
        out.update(range(a, b + 1))
    unknown = out - set(all_ids)
    if unknown:
        raise CondorError(f"job ids not in the job list: {sorted(unknown)[:10]}")
    return sorted(out)


def _code_tarball(path: str):
    with tarfile.open(path, 'w:gz') as tar:
        for sub in ('src/svval', 'configs'):
            tar.add(os.path.join(REPO, sub), arcname=sub,
                    filter=lambda ti: None if '__pycache__' in ti.name or ti.name.endswith('.pyc') else ti)


def prepare(tag: str, jobs_json: str, ids: List[int], max_files: Optional[int] = None,
            eos_out: Optional[str] = None, base: Optional[str] = None) -> dict:
    """Create (or extend) condor/<tag>/ for the given job ids. Returns the tag metadata."""
    d = tag_dir(tag, base)
    with open(jobs_json) as f:
        spec = json.load(f)
    with open(spec['manifest']) as f:
        man_head = json.load(f)
    by_id = {j['job']: j for j in spec['jobs']}
    meta_path = os.path.join(d, 'tag.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta['jobs_json'] != os.path.abspath(jobs_json) or meta['max_files'] != max_files:
            raise CondorError(f"tag {tag} exists with another job list or max_files; use a new tag")
    else:
        meta = {'tag': tag, 'jobs_json': os.path.abspath(jobs_json), 'manifest': spec['manifest'],
                'version': man_head.get('version'), 'max_files': max_files,
                'eos_out': eos_out or default_eos_out(man_head.get('version') or 'vX', tag),
                'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'jobs': {}}
    os.makedirs(os.path.join(d, 'specs'), exist_ok=True)
    os.makedirs(os.path.join(d, 'logs'), exist_ok=True)
    for i in ids:
        j = dict(by_id[i])
        if max_files:
            j['files'] = j['files'][:max_files]
        j['prefix'] = man_head['xrootd_prefix']
        j['tree'] = man_head.get('tree', 'tree/llpgtree')
        with open(os.path.join(d, 'specs', f'job_{i:05d}.json'), 'w') as f:
            json.dump(j, f)
        meta['jobs'][str(i)] = {'year': j['year'], 'era': j['era'], 'period': j['period'],
                                'n_files': len(j['files']), 'bytes': j['bytes']}
    _code_tarball(os.path.join(d, 'svval_code.tgz'))
    with open(os.path.join(d, 'run_job.sh'), 'w') as f:
        f.write(WRAPPER.replace('@SCRAM_ARCH@', SCRAM_ARCH).replace('@CMSSW@', CMSSW).replace('@XRD@', XRD))
    os.chmod(os.path.join(d, 'run_job.sh'), 0o755)
    with open(os.path.join(d, 'submit.jdl'), 'w') as f:
        f.write(JDL.format(eos_out=meta['eos_out'], tag=tag))
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=1)
    return meta


def _run(cmd: List[str], cwd: Optional[str] = None) -> str:
    """Run through /bin/sh: the LPC condor_* wrappers are scripts without a shebang (their first
    line is a shell `exec` into python), which only a shell can start."""
    line = ' '.join(shlex.quote(c) for c in cmd)
    p = subprocess.run(['/bin/sh', '-c', line], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True)
    if p.returncode != 0:
        raise CondorError(f"{' '.join(cmd)} failed ({p.returncode}):\n{p.stdout}")
    return p.stdout


def check_proxy(min_hours: int = 24):
    try:
        _run(['voms-proxy-info', '-exists', '-valid', f'{min_hours}:00'])
    except CondorError:
        raise CondorError(f"no grid proxy valid for {min_hours} h (jobs read and write EOS with it): "
                          "run  voms-proxy-init -voms cms -valid 192:00")


def make_eos_dir(eos_out: str):
    _run(['eos', XRD.rstrip('/'), 'mkdir', '-p', eos_out])


def submit(tag: str, ids: List[int], base: Optional[str] = None, dry_run: bool = False,
           max_materialize: Optional[int] = None) -> Optional[int]:
    """
    One submission round (one cluster). max_materialize: at most this many jobs of the cluster in
    the queue at once (HTCondor late materialization); the rest are added as jobs leave the queue.
    """
    if max_materialize is not None and max_materialize < 1:
        raise CondorError("max_materialize must be >= 1")
    d = tag_dir(tag, base)
    with open(os.path.join(d, 'tag.json')) as f:
        meta = json.load(f)
    missing = [i for i in ids if str(i) not in meta['jobs']]
    if missing:
        raise CondorError(f"jobs not prepared for tag {tag}: {missing[:10]}")
    rnd = len([x for x in os.listdir(d) if x.startswith('queue_')])
    qfile = f'queue_{rnd:03d}.txt'
    with open(os.path.join(d, qfile), 'w') as f:
        f.write(''.join(f'{i:05d}\n' for i in ids))
    jdl = os.path.join(d, f'submit_{rnd:03d}.jdl')
    with open(os.path.join(d, 'submit.jdl')) as f:
        body = f.read()
    with open(jdl, 'w') as f:
        f.write(body + (f'max_materialize         = {max_materialize}\n' if max_materialize else '')
                + f'queue job from {qfile}\n')
    if dry_run:
        return None
    check_proxy()
    make_eos_dir(meta['eos_out'])
    out = _run(['condor_submit', os.path.basename(jdl)], cwd=d)
    m = re.search(r'submitted to cluster (\d+)', out)
    if not m:
        raise CondorError(f"cannot parse condor_submit output:\n{out}")
    cluster = int(m.group(1))
    with open(os.path.join(d, 'clusters.txt'), 'a') as f:
        f.write(f"{cluster} {qfile} {len(ids)} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return cluster


# --------------------------------------------------------------------------- status

CONDOR_STATUS = {1: 'idle', 2: 'running', 3: 'removed', 4: 'completed', 5: 'held', 6: 'transferring', 7: 'suspended'}


def done_jobs(eos_out: str) -> Set[int]:
    d = FUSE + eos_out
    if not os.path.isdir(d):
        return set()
    return {int(m.group(1)) for m in (re.fullmatch(r'job_(\d{5})\.json', n) for n in os.listdir(d)) if m}


def queued_jobs(tag: str) -> Dict[int, str]:
    """{job: condor state} for this tag's jobs still in the queue."""
    try:
        out = _run(['condor_q', '-constraint', f'SVValTag == "{tag}"', '-af', 'SVValJob', 'JobStatus'])
    except CondorError:
        return {}
    q = {}
    for line in out.split('\n'):
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit():
            q[int(parts[0])] = CONDOR_STATUS.get(int(parts[1]), parts[1])
    return q


def rounds(tag: str, base: Optional[str] = None) -> List[dict]:
    """Submission rounds recorded in clusters.txt: cluster, queue file, job ids."""
    d = tag_dir(tag, base)
    path = os.path.join(d, 'clusters.txt')
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            with open(os.path.join(d, parts[1])) as q:
                out.append({'cluster': int(parts[0]), 'queue': parts[1], 'ids': [int(x) for x in q.read().split()]})
    return out


def live_factories(clusters: Iterable[int]) -> Set[int]:
    """Clusters that still have a late-materialization factory (jobs left to materialize)."""
    clusters = set(clusters)
    if not clusters:
        return set()
    try:
        out = _run(['condor_q', '-factory', '-af', 'ClusterId'])
    except CondorError:
        return set()
    return {int(x) for x in out.split() if x.isdigit()} & clusters


def last_error(tag: str, job: int, base: Optional[str] = None, n: int = 3) -> str:
    """Tail of the newest stderr of a job and the wrapper exit code, if any."""
    logs = os.path.join(tag_dir(tag, base), 'logs')
    errs = sorted((f for f in os.listdir(logs) if f.startswith(f'job_{job:05d}.') and f.endswith('.err')),
                  key=lambda f: os.path.getmtime(os.path.join(logs, f)))
    if not errs:
        return 'no log yet'
    with open(os.path.join(logs, errs[-1])) as f:
        lines = [x.rstrip() for x in f if x.strip()]
    return ' | '.join(lines[-n:]) if lines else 'empty stderr'


def status(tag: str, base: Optional[str] = None) -> dict:
    with open(os.path.join(tag_dir(tag, base), 'tag.json')) as f:
        meta = json.load(f)
    ids = sorted(int(i) for i in meta['jobs'])
    done = done_jobs(meta['eos_out']) & set(ids)
    queue = queued_jobs(tag)
    held = sorted(i for i, s in queue.items() if s == 'held')
    active = {i for i, s in queue.items() if s in ('idle', 'running', 'transferring', 'suspended')}
    # late materialization: jobs of a cluster whose factory is alive but which are not in the
    # queue yet are pending, not failed (otherwise resubmit would submit them twice)
    rnds = rounds(tag, base)
    live = live_factories(r['cluster'] for r in rnds)
    pending = set()
    for r in rnds:
        if r['cluster'] in live:
            pending.update(set(r['ids']) - done - set(queue))
    failed = sorted(set(ids) - done - active - set(held) - pending)
    return {'meta': meta, 'ids': ids, 'done': sorted(done), 'active': sorted(active), 'held': held,
            'pending': sorted(pending), 'failed': failed, 'queue': queue}
