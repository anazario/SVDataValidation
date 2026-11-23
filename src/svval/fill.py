"""
Fill stage: one job = one chunk of ntuple files of a single (year, era, period).

Outputs, next to each other (<out> = <outdir>/job_NNNNN):
  <out>.root        histograms, one directory per slice (histos.py)
  <out>_lumis.npz   per (file, run, lumi): presence of every lumi read (for the duplicate-lumi check)
                    and counters over the *selected* events (per-run / per-lumi trends)
  <out>.json        job metadata: files, entries, cutflow per slice, timing, failures

Blinding (selection.BLINDING) is applied before anything is counted or filled.
A file that cannot be read after retries fails the job (exit code != 0): a partial job output
must never be merged.
"""

from __future__ import annotations

import json
import os
import socket
import time
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional

import awkward as ak
import numpy as np
import uproot

from . import catalog as cat
from . import histos
from . import selection as sel
from .lumimask import LumiMask

VERTEX_FIELDS = ['Vertex_nTracks', 'Vertex_mass', 'Vertex_pt', 'Vertex_eta', 'Vertex_phi', 'Vertex_p',
                 'Vertex_x', 'Vertex_y', 'Vertex_z', 'Vertex_dxy', 'Vertex_dxyError', 'Vertex_chi2',
                 'Vertex_normalizedChi2', 'Vertex_ndof', 'Vertex_cosTheta', 'Vertex_decayAngle',
                 'Vertex_sumCharge', 'Vertex_isUnique', 'Vertex_scMatchRatio', 'Vertex_cxx', 'Vertex_cyy',
                 'Vertex_czz', 'Vertex_passLooseMuonID', 'Vertex_passLooseElectronID']
VTRK_FIELDS = ['VertexTrack_vertexIndex', 'VertexTrack_trackIndex', 'VertexTrack_trackCosTheta',
               'VertexTrack_trackCosThetaAtCM', 'VertexTrack_trackCompatibility',
               'VertexTrack_shiftDzAfterTrackRemoval', 'VertexTrack_shift3DAfterTrackRemoval',
               'VertexTrack_SCDR', 'VertexTrack_energySC', 'VertexTrack_ratioPToEnergySC',
               'VertexTrack_pathLength']
TRACK_FIELDS = ['Track_pt', 'Track_eta', 'Track_phi', 'Track_ip2D', 'Track_sip2D', 'Track_ip3D', 'Track_sip3D',
                'Track_nValidHits', 'Track_nLostHits', 'Track_normalizedChi2', 'Track_ptError', 'Track_charge',
                'Track_qualityMask', 'Track_isGenTrack', 'Track_pdgId']
EXTRA_EVENT = ['PV_npvs', 'PV_x', 'PV_y', 'PV_z', 'BeamSpot_X', 'BeamSpot_Y', 'BeamSpot_Z']

BRANCHES = sorted(set(sel.EVENT_BRANCHES + sel.SV_BRANCHES + VERTEX_FIELDS + VTRK_FIELDS + TRACK_FIELDS
                      + EXTRA_EVENT))

# per-lumi counters over selected events (all SV counts: shown SVs only)
LUMI_COUNTERS = ['n_sel', 'n_fakePV', 'sum_npv', 'n_sv', 'n_badErr', 'n_negCov',
                 'n_lep2_mu_base', 'n_lep2_e_base', 'n_lep2_both_base', 'n_lep2_noid_base',
                 'n_trk3_base', 'n_had_id_base', 'n_had_noid_base', 'n_lep2_all', 'n_had_all',
                 'n_vtrk_shared']


class JobFailure(RuntimeError):
    pass


# --------------------------------------------------------------------------- building records

def build_records(ev, sv_info, shown):
    """Jagged SV, SV-track and Track_ records with the derived quantities the histograms use."""
    nt = ev.Vertex_nTracks
    vt = ak.zip({f: ev[f] for f in VTRK_FIELDS})
    # contiguity of VertexTrack by vertex is relied on below; checked on real data, enforced here
    if not ak.all(ak.sum(nt, axis=1) == ak.num(vt)):
        raise JobFailure("VertexTrack entries do not match sum(Vertex_nTracks)")
    pairs = ak.cartesian({'a': vt[['VertexTrack_trackIndex', 'VertexTrack_vertexIndex']],
                          'b': vt[['VertexTrack_trackIndex', 'VertexTrack_vertexIndex']]}, nested=True)
    shared = ak.any((pairs.a.VertexTrack_trackIndex == pairs.b.VertexTrack_trackIndex)
                    & (pairs.a.VertexTrack_vertexIndex != pairs.b.VertexTrack_vertexIndex), axis=2)
    vt = ak.with_field(vt, shared, 'shared')
    n_shared = ak.sum(ak.unflatten(shared, ak.flatten(nt), axis=1), axis=2)

    fields = {f: ev[f] for f in VERTEX_FIELDS}
    for f in sv_info.fields:
        fields[f] = sv_info[f]
    fields['shown'] = shown
    fields['r'] = np.hypot(ev.Vertex_x, ev.Vertex_y)
    p, m = ev.Vertex_p, ev.Vertex_mass
    fields['pOverE'] = p / np.sqrt(p * p + m * m)
    fields['negCov'] = (ev.Vertex_cxx < 0) | (ev.Vertex_cyy < 0) | (ev.Vertex_czz < 0)
    fields['nShared'] = n_shared
    sv = ak.zip(fields)
    tr = ak.zip({f: ev[f] for f in TRACK_FIELDS})
    return sv, vt, tr


def event_record(ev, sv, cmet):
    """Selected events with the per-event derived counters (SV counts over shown SVs)."""
    out = {f: ev[f] for f in EXTRA_EVENT}
    out['Evt_run'] = ev.Evt_run
    out['Evt_luminosityBlock'] = ev.Evt_luminosityBlock
    out['selCMet'] = cmet
    out['pvBsDxy'] = np.hypot(ev.PV_x - ev.BeamSpot_X, ev.PV_y - ev.BeamSpot_Y)
    out['Track_pt'] = ev.Track_pt
    good = sv.good & sv.shown
    out['n_sv'] = ak.sum(good, axis=1)
    out['n_badErr'] = ak.sum(~sv.good, axis=1)
    out['n_negCov'] = ak.sum(sv.negCov & sv.shown, axis=1)
    for c, cfn in histos.SV_CATEGORIES.items():
        cm = cfn(sv)
        for lvl, lfn in histos.SV_LEVELS.items():
            out[f'n_{c}_{lvl}'] = ak.sum(cm & lfn(sv), axis=1)
    return ak.zip(out, depth_limit=1)


def slice_names(year: int, era: str, period: str, run: np.ndarray) -> np.ndarray:
    base = f"{year}{era}" + (f"_{period}" if period else '')
    hem = sel.hem_period(run, year)
    return np.array([base + (f"_{h}" if h else '') for h in hem], dtype=object)


# --------------------------------------------------------------------------- lumi table

class LumiTable:
    def __init__(self):
        self.rows: List[np.ndarray] = []

    def add(self, file_index: int, run, lumi, sel_mask, er):
        """
        run / lumi / er: every event of the chunk. Every lumi read gets a row (presence, for the
        duplicate-lumi check); the counters only sum events with sel_mask (selected, so unvetoed).
        """
        key = (np.asarray(run, np.uint64) << np.uint64(32)) | np.asarray(lumi, np.uint64)
        uk, inv = np.unique(key, return_inverse=True)
        tab = np.zeros(len(uk), dtype=self.DTYPE)
        tab['file'] = file_index
        tab['run'] = (uk >> np.uint64(32)).astype(np.uint32)
        tab['lumi'] = (uk & np.uint64(0xFFFFFFFF)).astype(np.uint32)
        if sel_mask.any():
            s = er[sel_mask]
            idx = inv[sel_mask]
            vals = {
                'n_sel': np.ones(len(s), np.int64),
                'n_fakePV': ak.to_numpy(s.PV_npvs == 1).astype(np.int64),
                'sum_npv': ak.to_numpy(s.PV_npvs).astype(np.int64),
                'n_sv': ak.to_numpy(s.n_sv), 'n_badErr': ak.to_numpy(s.n_badErr),
                'n_negCov': ak.to_numpy(s.n_negCov),
                'n_vtrk_shared': ak.to_numpy(s.n_vtrk_shared),
            }
            for c in LUMI_COUNTERS:
                if c not in vals:
                    vals[c] = ak.to_numpy(s[c])
            for c, v in vals.items():
                np.add.at(tab[c], idx, v.astype(np.int64))
        self.rows.append(tab)

    DTYPE = np.dtype([('file', np.int32), ('run', np.uint32), ('lumi', np.uint32)]
                     + [(c, np.int64) for c in LUMI_COUNTERS])

    def table(self) -> np.ndarray:
        if not self.rows:
            return np.zeros(0, dtype=self.DTYPE)
        t = np.concatenate(self.rows)
        # merge chunks of the same file / lumi
        order = np.lexsort((t['lumi'], t['run'], t['file']))
        t = t[order]
        k = np.stack([t['file'].astype(np.int64), t['run'].astype(np.int64), t['lumi'].astype(np.int64)], 1)
        new = np.ones(len(t), bool)
        new[1:] = np.any(k[1:] != k[:-1], axis=1)
        starts = np.flatnonzero(new)
        out = t[starts].copy()
        for c in LUMI_COUNTERS:
            out[c] = np.add.reduceat(t[c], starts)
        return out


# --------------------------------------------------------------------------- the job

def open_kwargs(url: str) -> dict:
    """root:// is read with the XRootD bindings directly: the CMSSW python has no fsspec-xrootd
    (and the direct source is also the safe choice when PyROOT shares the process)."""
    if url.startswith('root://'):
        return {'handler': uproot.source.xrootd.XRootDSource}
    return {}


def _open_tree(url: str, tree: str, retries: int = 3):
    last = None
    for attempt in range(retries):
        try:
            return uproot.open(url, **open_kwargs(url))[tree]
        except (OSError, TimeoutError, ValueError) as e:     # I/O-like: retry; anything else is a bug
            last = e
            if attempt + 1 < retries:
                time.sleep(10 * (attempt + 1))
    raise JobFailure(f"cannot open {url}: {type(last).__name__}: {last}")


def run_job(job: dict, prefix: str, tree: str, out: str, lumimask: Optional[LumiMask] = None,
            step_size: str = '150 MB', log=print) -> dict:
    t0 = time.time()
    lumimask = lumimask or LumiMask.from_dir()
    year, era, period = int(job['year']), job['era'], job['period']
    hsets: Dict[str, histos.HistSet] = OrderedDict()
    flows: Dict[str, Dict[str, int]] = defaultdict(lambda: OrderedDict())
    lumis = LumiTable()
    entries = []
    n_read = 0
    for fi, path in enumerate(job['files']):
        url = cat.url_for(prefix, path)
        t = _open_tree(url, tree)
        entries.append(int(t.num_entries))
        if t.num_entries == 0:
            continue
        for ev in t.iterate(BRANCHES, step_size=step_size):
            n_read += len(ev)
            # mask includes the blinding veto as its first step: vetoed events are never
            # selected, counted or filled (selection.BLINDING)
            mask, sv_info, shown, steps, cmet = sel.select(ev, lumimask)
            sv, vt, tr = build_records(ev, sv_info, shown)
            er = event_record(ev, sv, cmet)
            er = ak.with_field(er, ak.sum(vt.shared & sv.shown[ak.values_astype(vt.VertexTrack_vertexIndex, np.int64)],
                                          axis=1), 'n_vtrk_shared')
            run = ak.to_numpy(ev.Evt_run)
            lumis.add(fi, run, ak.to_numpy(ev.Evt_luminosityBlock), mask, er)
            names = slice_names(year, era, period, run)
            for name in np.unique(names):
                sm = names == name
                fl = flows[name]
                for k, m in steps.items():
                    fl[k] = fl.get(k, 0) + int(np.sum(m & sm))
                fm = sm & mask
                if fm.any():
                    hsets.setdefault(name, histos.HistSet()).fill(er[fm], sv[fm], vt[fm], tr[fm])
        log(f"  [{fi + 1}/{len(job['files'])}] {path.rsplit('/', 1)[-1]}  entries {entries[-1]}  "
            f"{time.time() - t0:.0f}s")

    _write(out, job, hsets, flows, lumis, entries, n_read, prefix, time.time() - t0)
    return {'slices': list(hsets), 'entries': sum(entries), 'seconds': time.time() - t0}


def _write(out, job, hsets, flows, lumis, entries, n_read, prefix, seconds):
    for name, hs in hsets.items():
        bad = histos.audit_blinding(hs.h)
        if bad:
            raise JobFailure(f"blinding audit failed in slice {name}: {bad[:5]}")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + '.tmp.root'
    with uproot.recreate(tmp) as f:
        for name, hs in hsets.items():
            for key, h in hs.h.items():
                f[f"{name}/{key}"] = h
    np.savez_compressed(out + '_lumis.tmp.npz', lumis=lumis.table(), files=np.array(job['files']))
    meta = {
        'job': job.get('job'), 'year': job['year'], 'era': job['era'], 'period': job['period'],
        'files': job['files'], 'entries': entries, 'n_read': n_read, 'prefix': prefix,
        'cutflow': flows, 'slices': list(hsets), 'seconds': round(seconds, 1),
        'host': socket.gethostname(), 'blinding': {'lep_dxysig_max': sel.LEP_DXYSIG_MAX,
                                                    'had_dxysig_max': sel.HAD_DXYSIG_MAX},
    }
    with open(out + '.tmp.json', 'w') as fj:
        json.dump(meta, fj, indent=1)
    # the .json is written last: its presence marks a complete job output
    os.replace(tmp, out + '.root')
    os.replace(out + '_lumis.tmp.npz', out + '_lumis.npz')
    os.replace(out + '.tmp.json', out + '.json')
