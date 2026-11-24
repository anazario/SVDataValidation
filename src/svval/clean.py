"""
Remove what a tag produced, so old productions don't use up EOS quota and disk.

  svval clean --tag all_v34                  everything: EOS job outputs + merge, outputs/<tag>/, condor/<tag>/
  svval clean --tag all_v34 --keep-merged    only the per-job outputs on EOS (the bulk); the merge,
                                             plots and condor bookkeeping stay
  svval clean --tag all_v34 --dry-run        list what would be removed, remove nothing

It lists every location with its size first and deletes only after you type the tag (or with --yes).
It refuses while jobs of the tag are still in the condor queue, and refuses an EOS path that is not a
<tag> directory under /store/user/$USER/.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from . import condor

JOB_FILE = re.compile(r'job_\d{5}[._].+')     # job_00012.root, .json, _lumis.npz


class CleanError(RuntimeError):
    pass


@dataclass
class Target:
    what: str              # human description
    path: str              # local path, or /store/... for EOS
    eos: bool
    files: List[str]       # for a partial EOS clean: the file names to remove; empty = the whole path
    n_files: int
    n_bytes: int


def _du(path: str) -> Tuple[int, int]:
    n = b = 0
    for root, _, names in os.walk(path):
        for x in names:
            try:
                b += os.lstat(os.path.join(root, x)).st_size
                n += 1
            except OSError:
                pass
    return n, b


def human(n: int) -> str:
    for unit in ('B', 'kB', 'MB', 'GB', 'TB'):
        if n < 1000 or unit == 'TB':
            return f"{n:.0f} {unit}" if unit == 'B' else f"{n:.1f} {unit}"
        n /= 1000.0
    return str(n)


def check_eos_path(store_path: str, tag: str, user: Optional[str] = None):
    """Only a <tag> directory under /store/user/<user>/ may be removed."""
    user = user or os.environ.get('USER', '')
    p = os.path.normpath(store_path)
    if not user or not p.startswith(f"/store/user/{user}/") or os.path.basename(p) != tag:
        raise CleanError(f"refusing to remove {store_path}: not a '{tag}' directory under /store/user/{user}/")


def eos_out_of(tag: str, base: Optional[str] = None) -> Optional[str]:
    meta = os.path.join(condor.tag_dir(tag, base), 'tag.json')
    if not os.path.exists(meta):
        return None
    with open(meta) as f:
        return json.load(f)['eos_out']


def plan(tag: str, keep_merged: bool = False, eos_out: Optional[str] = None, base: Optional[str] = None,
         outputs: Optional[str] = None, fuse: str = condor.FUSE, user: Optional[str] = None) -> List[Target]:
    """What `clean` would remove, existing locations only."""
    out: List[Target] = []
    eos_out = eos_out or eos_out_of(tag, base)
    if eos_out:
        check_eos_path(eos_out, tag, user)
        d = fuse + eos_out
        if os.path.isdir(d):
            if keep_merged:
                names = sorted(x for x in os.listdir(d) if JOB_FILE.fullmatch(x))
                if names:
                    b = sum(os.lstat(os.path.join(d, x)).st_size for x in names)
                    out.append(Target('per-job outputs on EOS', eos_out, True, names, len(names), b))
            else:
                n, b = _du(d)
                out.append(Target('job outputs and merge on EOS', eos_out, True, [], n, b))
    if not keep_merged:
        for what, p in (('local merge and plots', os.path.join(outputs or os.path.join(condor.REPO, 'outputs'), tag)),
                        ('condor specs and logs', condor.tag_dir(tag, base))):
            if os.path.isdir(p):
                n, b = _du(p)
                out.append(Target(what, p, False, [], n, b))
    return out


def eos_rm_dir(store_path: str):
    p = subprocess.run(['eos', condor.XRD.rstrip('/'), 'rm', '-r', store_path], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True)
    if p.returncode != 0:
        raise CleanError(f"eos rm -r {store_path} failed:\n{p.stdout[-1000:]}")


def remove(targets: List[Target], fuse: str = condor.FUSE, rm_dir: Callable[[str], None] = eos_rm_dir,
           workers: int = 16, log=print):
    for t in targets:
        log(f"removing {t.what}: {t.path}")
        if t.eos and t.files:
            d = fuse + t.path
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(lambda x: os.remove(os.path.join(d, x)), t.files))
        elif t.eos:
            rm_dir(t.path)
        else:
            shutil.rmtree(t.path)


def clean(tag: str, keep_merged: bool = False, eos_out: Optional[str] = None, dry_run: bool = False,
          yes: bool = False, base: Optional[str] = None, ask: Callable[[str], str] = input,
          queued: Callable[[str], dict] = condor.queued_jobs, outputs: Optional[str] = None,
          fuse: str = condor.FUSE, user: Optional[str] = None, rm_dir: Callable[[str], None] = eos_rm_dir,
          log=print) -> bool:
    """Returns True if something was removed."""
    targets = plan(tag, keep_merged, eos_out, base, outputs=outputs, fuse=fuse, user=user)
    if not targets:
        log(f"nothing to clean for tag {tag}"
            + ('' if eos_out or eos_out_of(tag, base) else
               " (no condor/<tag>/tag.json, so its EOS directory is unknown: pass --eos-out)"))
        return False
    log(f"tag {tag}: " + ('would remove' if dry_run else 'will remove'))
    for t in targets:
        log(f"  {t.what:30} {t.n_files:7d} files {human(t.n_bytes):>9}   {t.path}")
    log(f"  {'total':30} {sum(t.n_files for t in targets):7d} files "
        f"{human(sum(t.n_bytes for t in targets)):>9}")
    if dry_run:
        return False
    q = queued(tag)
    if q:
        raise CleanError(f"{len(q)} jobs of tag {tag} are still in the condor queue; condor_rm them first")
    if not yes and ask(f"type the tag ({tag}) to delete this: ").strip() != tag:
        log("not confirmed, nothing removed")
        return False
    remove(targets, fuse=fuse, rm_dir=rm_dir, log=log)
    log(f"done: tag {tag} cleaned")
    return True
