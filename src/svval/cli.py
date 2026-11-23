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
    print(f"duplicate lumis  : {d['n_pairs']} (run, lumi) pairs in more than one file")
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

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == '__main__':
    sys.exit(main())
