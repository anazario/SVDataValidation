"""
svval command line.

  svval catalog  -c configs/datasets_v34.yaml -o manifests/v34.json    scan EOS (LPC interactive)
  svval summary  -m manifests/v34.json [-s 2024]                      files / GB per era
  svval lumicheck -m manifests/v34.json [-s 2016] [-j 8]              duplicate lumis, bad files
  svval chunks   -m manifests/v34.json -s run3 --target-gb 5 -o jobs/run3.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import catalog as cat


def _cmd_catalog(a):
    cfg = cat.load_config(a.config)
    records, excluded = cat.scan(cfg, root=a.root)
    report = cat.task_report(records)
    cat.write_manifest(a.output, cfg, records, excluded, report)
    print(cat.format_summary(records))
    print(f"\nexcluded: {excluded or 'none'}")
    _print_report(report)
    print(f"\nwrote {a.output} ({len(records)} files)")
    return 1 if report['duplicate_requests'] else 0


def _print_report(report):
    if report['duplicate_requests']:
        print("\nERROR: request names with more than one CRAB task stamp:")
        for k, v in report['duplicate_requests'].items():
            print(f"  {k}: {v}")
    if report['multi_request_eras']:
        print("\neras made by several CRAB requests (review; lumicheck tests overlaps):")
        for k, v in report['multi_request_eras'].items():
            print(f"  {k}:")
            for r in v:
                print(f"      {r}")
    if report['empty_files']:
        print(f"\n{len(report['empty_files'])} zero-byte files, e.g. {report['empty_files'][0]}")


def _cmd_summary(a):
    man, records = cat.read_manifest(a.manifest)
    records = cat.select(records, a.select)
    print(f"manifest {a.manifest} (created {man['created']}), selection '{a.select or 'all'}'\n")
    print(cat.format_summary(records))
    if not a.select:
        _print_report(man['report'])
    return 0


def _cmd_lumicheck(a):
    from .lumis import scan_lumis
    man, records = cat.read_manifest(a.manifest)
    records = cat.select(records, a.select)
    prefix = a.prefix or man['fuse_prefix']
    print(f"reading run/lumi of {len(records)} files via {prefix} with {a.workers} workers")
    rep = scan_lumis(records, prefix, man['tree'], a.cache, workers=a.workers,
                     progress=lambda i, n: print(f"  {i}/{n}", flush=True))
    out = a.output or os.path.splitext(a.manifest)[0] + f"_lumicheck_{(a.select or 'all').replace(',', '_')}.json"
    with open(out, 'w') as f:
        json.dump(rep, f, indent=1)
    print(f"\n{'era':<8}{'files':>7}{'events':>14}{'lumis':>10}")
    for era, v in rep['per_era'].items():
        print(f"{era:<8}{v['files']:>7}{v['events']:>14}{v['lumis']:>10}")
    print(f"\nunreadable files : {len(rep['bad_files'])}")
    for p, e in list(rep['bad_files'].items())[:10]:
        print(f"  {p}\n     {e}")
    print(f"empty trees      : {len(rep['empty_files'])}")
    d = rep['duplicate_lumis']
    print(f"duplicate lumis  : {d['n_pairs']} (run, lumi) pairs in more than one file of the same year and PD")
    for g, n in d['by_group'].items():
        print(f"  {g:<14} {n}")
    for k, n in list(d['by_file_pair'].items())[:10]:
        print(f"  {n:6d}  {k}")
    print(f"\nwrote {out}")
    return 1 if (rep['bad_files'] or d['n_pairs']) else 0


def _cmd_chunks(a):
    _, records = cat.read_manifest(a.manifest)
    records = cat.select(records, a.select)
    chunks = cat.make_chunks(records, int(a.target_gb * 1e9))
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    with open(a.output, 'w') as f:
        json.dump({'manifest': os.path.abspath(a.manifest), 'selection': a.select or 'all',
                   'target_gb': a.target_gb, 'jobs': chunks}, f, indent=0)
    sizes = [c['bytes'] / 1e9 for c in chunks]
    print(f"{len(chunks)} jobs from {len(records)} files; GB/job min {min(sizes):.2f} "
          f"median {sorted(sizes)[len(sizes) // 2]:.2f} max {max(sizes):.2f}\nwrote {a.output}")
    return 0


def _cmd_fill(a):
    from .fill import run_job
    if a.job_file:
        with open(a.job_file) as f:
            job = json.load(f)              # self-contained spec written by `svval submit`
        prefix, tree = a.prefix or job['prefix'], job['tree']
    else:
        if a.job is None:
            raise SystemExit("--jobs needs --job")
        with open(a.jobs) as f:
            spec = json.load(f)
        job = spec['jobs'][a.job]
        if job['job'] != a.job:
            raise SystemExit(f"job list is not indexed by job number ({job['job']} != {a.job})")
        man = cat.read_manifest(spec['manifest'])[0] if os.path.exists(spec['manifest']) else {}
        prefix = a.prefix or man.get('xrootd_prefix', 'root://cmseos.fnal.gov/')
        tree = man.get('tree', 'tree/llpgtree')
    if a.max_files:
        job = dict(job, files=job['files'][:a.max_files])
    out = os.path.join(a.outdir, f"job_{job['job']:05d}")
    print(f"job {job['job']}: {job['year']}{job['era']} {job['period']} {len(job['files'])} files "
          f"{job['bytes'] / 1e9:.2f} GB via {prefix}", flush=True)
    res = run_job(job, prefix, tree, out, log=lambda m: print(m, flush=True))
    print(f"done: {res['entries']} entries in {res['seconds']:.0f}s, slices {res['slices']}; wrote {out}.*")
    return 0


def _cmd_submit(a):
    from . import condor
    with open(a.jobs) as f:
        spec = json.load(f)
    jobs = spec['jobs']
    if a.select:
        keep = {(r.year, r.era, r.period) for r in cat.select(cat.read_manifest(spec['manifest'])[1], a.select)}
        jobs = [j for j in jobs if (j['year'], j['era'], j['period']) in keep]
    ids = condor.parse_ids(a.ids, [j['job'] for j in jobs])
    meta = condor.prepare(a.tag, a.jobs, ids, max_files=a.max_files, eos_out=a.eos_out)
    gb = sum(meta['jobs'][str(i)]['bytes'] for i in ids) / 1e9
    print(f"tag {a.tag}: {len(ids)} jobs ({gb:.0f} GB{' before max-files' if a.max_files else ''}) "
          f"-> {meta['eos_out']}")
    cl = condor.submit(a.tag, ids, dry_run=a.dry_run, max_materialize=a.max_mat)
    lim = f", at most {a.max_mat} in the queue at once" if a.max_mat else ''
    print(("dry run: prepared " + condor.tag_dir(a.tag)) if a.dry_run else f"submitted cluster {cl}{lim}")
    return 0


def _print_status(st, verbose):
    from . import condor
    n = len(st['ids'])
    print(f"tag {st['meta']['tag']}: {n} jobs | done {len(st['done'])} | queued {len(st['active'])} "
          f"| waiting to materialize {len(st['pending'])} | held {len(st['held'])} "
          f"| failed / not submitted {len(st['failed'])}")
    for label in ('held', 'failed'):
        for i in st[label][: (None if verbose else 10)]:
            print(f"  {label:6s} {i:5d}  {condor.last_error(st['meta']['tag'], i)}")


def _cmd_status(a):
    from . import condor
    st = condor.status(a.tag)
    _print_status(st, a.verbose)
    return 0 if len(st['done']) == len(st['ids']) else 3


def _cmd_resubmit(a):
    from . import condor
    st = condor.status(a.tag)
    ids = st['failed'] + (st['held'] if a.include_held else [])
    if not ids:
        print("nothing to resubmit")
        _print_status(st, False)
        return 0
    if a.include_held and st['held']:
        condor._run(['condor_rm', '-constraint', f'SVValTag == "{a.tag}" && JobStatus == 5'])
    cl = condor.submit(a.tag, sorted(ids), dry_run=a.dry_run, max_materialize=a.max_mat)
    print(f"resubmitted {len(ids)} jobs" + ('' if a.dry_run else f" as cluster {cl}"))
    return 0


def _cmd_merge(a):
    from .merge import merge
    outdir = a.outdir or os.path.join('outputs', a.tag, 'merged')
    checks = merge(a.tag, outdir, allow_missing=a.allow_missing, hadd_jobs=a.hadd_jobs)
    print(f"merged {checks['merged']}/{checks['jobs']} jobs into {outdir}; duplicate lumis 0; "
          f"blinding audit ok")
    return 0


def _cmd_plot(a):
    from .plotting.run import run
    merged = a.merged or os.path.join('outputs', a.tag, 'merged')
    outdir = a.outdir or os.path.join(os.path.dirname(os.path.abspath(merged)), 'plots')
    years = [y for y, _, _ in cat.parse_selection(a.select)] if a.select else None
    run(merged, outdir, formats=tuple(a.formats.split(',')), all_vars=a.all_vars, years=years,
        lumi_csv=a.lumi_csv, d_min=a.d_min, p_max=a.p_max, min_run_events=a.min_run_events,
        log=lambda m: print(m, flush=True))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog='svval', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    sel_help = "all | run2 | run3 | comma list of 2024, 2024C, 2016:preVFP"

    c = sub.add_parser('catalog', help='scan EOS and write the file manifest')
    c.add_argument('-c', '--config', required=True)
    c.add_argument('-o', '--output', required=True)
    c.add_argument('--root', help='filesystem prefix of /store (default: fuse_prefix of the config)')
    c.set_defaults(fn=_cmd_catalog)

    s = sub.add_parser('summary', help='files / GB per era of a manifest')
    s.add_argument('-m', '--manifest', required=True)
    s.add_argument('-s', '--select', help=sel_help)
    s.set_defaults(fn=_cmd_summary)

    lc = sub.add_parser('lumicheck', help='duplicate lumis, unreadable and empty files')
    lc.add_argument('-m', '--manifest', required=True)
    lc.add_argument('-s', '--select', help=sel_help)
    lc.add_argument('-j', '--workers', type=int, default=8)
    lc.add_argument('--prefix', help='file prefix (default: the FUSE prefix; root://cmseos.fnal.gov/ for xrootd)')
    lc.add_argument('--cache', default='manifests/lumicache')
    lc.add_argument('-o', '--output')
    lc.set_defaults(fn=_cmd_lumicheck)

    ch = sub.add_parser('chunks', help='split a selection into condor job file lists')
    ch.add_argument('-m', '--manifest', required=True)
    ch.add_argument('-s', '--select', help=sel_help)
    ch.add_argument('--target-gb', type=float, default=5.0)
    ch.add_argument('-o', '--output', required=True)
    ch.set_defaults(fn=_cmd_chunks)

    fl = sub.add_parser('fill', help='run one fill job of a job list')
    src = fl.add_mutually_exclusive_group(required=True)
    src.add_argument('--jobs', help='job list written by `svval chunks` (with --job)')
    src.add_argument('--job-file', help='single self-contained job spec (as shipped to condor)')
    fl.add_argument('--job', type=int)
    fl.add_argument('-o', '--outdir', required=True)
    fl.add_argument('--prefix', help="file prefix (default: the manifest's xrootd prefix)")
    fl.add_argument('--max-files', type=int, help='only the first N files of the job (testing)')
    fl.set_defaults(fn=_cmd_fill)

    sb = sub.add_parser('submit', help='prepare and submit fill jobs to condor (LPC)')
    sb.add_argument('--tag', required=True, help='submission name; outputs go to .../<version>/<tag>/')
    sb.add_argument('--jobs', required=True, help='job list written by `svval chunks`')
    sb.add_argument('-s', '--select', help=sel_help + ' (subset of the job list)')
    sb.add_argument('--ids', help="job ids, e.g. '3,10-12' (default: all, after --select)")
    sb.add_argument('--max-files', type=int, help='only the first N files of each job (tests)')
    sb.add_argument('--eos-out', help='EOS output dir /store/... (default /store/user/$USER/SVDataValidation/<version>/<tag>)')
    sb.add_argument('--dry-run', action='store_true', help='write condor/<tag>/ but do not submit')
    sb.add_argument('--max-mat', type=int, metavar='N',
                    help='at most N jobs in the queue at once (condor max_materialize); the rest follow as jobs finish')
    sb.set_defaults(fn=_cmd_submit)

    stp = sub.add_parser('status', help='done / queued / held / failed jobs of a tag')
    stp.add_argument('--tag', required=True)
    stp.add_argument('-v', '--verbose', action='store_true')
    stp.set_defaults(fn=_cmd_status)

    rs = sub.add_parser('resubmit', help='resubmit the jobs of a tag that are neither done nor queued')
    rs.add_argument('--tag', required=True)
    rs.add_argument('--include-held', action='store_true', help='also remove and resubmit held jobs')
    rs.add_argument('--dry-run', action='store_true')
    rs.add_argument('--max-mat', type=int, metavar='N', help='as for submit')
    rs.set_defaults(fn=_cmd_resubmit)

    mg = sub.add_parser('merge', help='hadd job outputs, merge lumi tables, duplicate-lumi and blinding checks')
    mg.add_argument('--tag', required=True)
    mg.add_argument('-o', '--outdir', help='default outputs/<tag>/merged')
    mg.add_argument('--allow-missing', action='store_true')
    mg.add_argument('--hadd-jobs', type=int, default=4)
    mg.set_defaults(fn=_cmd_merge)

    pl = sub.add_parser('plot', help='plots, era/year comparisons and index.html of a merged tag')
    g = pl.add_mutually_exclusive_group(required=True)
    g.add_argument('--tag', help='use outputs/<tag>/merged')
    g.add_argument('--merged', help='merged directory (svval merge output)')
    pl.add_argument('-o', '--outdir', help='default: <merged>/../plots')
    pl.add_argument('-s', '--select', help='years only, e.g. 2016 or run3 (eras are compared within years)')
    pl.add_argument('--formats', default='png,pdf')
    pl.add_argument('--all-vars', action='store_true', help='overlay every 1D histogram (default: curated list)')
    pl.add_argument('--lumi-csv', nargs='+', help='brilcalc --byls csv file(s): adds per-pb^-1 rates')
    pl.add_argument('--d-min', type=float, default=0.02, help='flag threshold on the Kolmogorov distance')
    pl.add_argument('--p-max', type=float, default=1e-3, help='flag threshold on the KS p-value')
    pl.add_argument('--min-run-events', type=int, default=100, help='minimum selected events of a run in trends')
    pl.set_defaults(fn=_cmd_plot)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == '__main__':
    sys.exit(main())
