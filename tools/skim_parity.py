#!/usr/bin/env python3
"""
Event-by-event parity of svval.selection with a KUCMSSkimmer kuSkimTree made from the same ntuples.

The skim has no run/lumi/event branches, so events are matched by the PV position plus the raw
MET px/py (PV_x/y/z, Met_px/py in the ntuple; PVx/y/z, selMetPx/Py in the skim). The PV alone is
not unique: events without a reconstructed vertex get the beamspot-fallback PV, at one fixed
position per run (seen as PV_npvs == 1 with identical PV coordinates).

  python3 tools/skim_parity.py -m manifests/v34.json -s 2016F:postVFP \
      --skim /eos/uscms/store/group/lpcsusylep/malazaro/KUCMSSkims/skims_v54/MET_Run2016F-UL2016_...root
  (--max-files N to use only the first N ntuple files; skim events from other files are ignored)

Checks, on the matched events:
  - every skim event (from these files) is matched exactly once, and passes our golden JSON
  - selCMet, Flag_MetFilters, Flag_hemVeto and the 4 triggers agree
  - per event, the sorted (dxySig, mass) lists of Leptonic/Hadronic SVs agree, and SV_nLeptonic/nHadronic
  - our event veto == (passNLepGe1SelectionHighDxySigSR or any hadronic SV with ID and dxySig >= 800),
    and every skim *HighDxySigSR flag implies our veto
"""

import argparse
import sys
from collections import Counter

import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, __file__.rsplit('/tools/', 1)[0] + '/src')
from svval import catalog as cat               # noqa: E402
from svval import selection as sel             # noqa: E402
from svval.lumimask import LumiMask            # noqa: E402

SKIM_BRANCHES = ['PVx', 'PVy', 'PVz', 'selMetPx', 'selMetPy', 'selCMet', 'Flag_MetFilters', 'Flag_hemVeto',
                 'Trigger_PFMET120_PFMHT120_IDTight', 'Trigger_PFMETNoMu120_PFMHTNoMu120_IDTight',
                 'Trigger_PFMET120_PFMHT120_IDTight_PFHT60', 'Trigger_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60',
                 'LeptonicSV_dxySig', 'LeptonicSV_mass', 'HadronicSV_dxySig', 'HadronicSV_mass',
                 'HadronicSV_passHadronicID', 'SV_nLeptonic', 'SV_nHadronic',
                 'passNLepGe1SelectionHighDxySigSR', 'passNHadGe1SelectionHighDxySigSR',
                 'passNSVGe1SelectionHighDxySigSR']
TRIG_MAP = {'Trigger_' + t[4:-2]: t for t in sel.TRIGGERS}


def event_key(*cols):
    a = np.stack([np.asarray(c, np.float32) for c in cols], axis=1)
    return a.view(np.dtype((np.void, 4 * len(cols)))).ravel()


def sorted_pairs(sig, mass):
    return [sorted(zip(np.round(s, 3).tolist(), np.round(m, 3).tolist())) for s, m in zip(sig, mass)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-m', '--manifest', required=True)
    p.add_argument('-s', '--select', required=True)
    p.add_argument('--skim', required=True)
    p.add_argument('--max-files', type=int)
    p.add_argument('--prefix', help='default: the FUSE prefix of the manifest')
    a = p.parse_args()

    man, recs = cat.read_manifest(a.manifest)
    recs = cat.select(recs, a.select)[:a.max_files]
    prefix = a.prefix or man['fuse_prefix']
    print(f"{len(recs)} ntuple files ({a.select}); skim {a.skim.rsplit('/', 1)[-1]}")

    br = sel.EVENT_BRANCHES + sel.SV_BRANCHES + ['PV_x', 'PV_y', 'PV_z', 'PV_npvs']
    ev = ak.concatenate([uproot.open(cat.url_for(prefix, r.path))[man['tree']].arrays(br) for r in recs])
    lm = LumiMask.from_dir()
    good_lumi = lm(ak.to_numpy(ev.Evt_run), ak.to_numpy(ev.Evt_luminosityBlock))
    excl = sel.photon_excluded(ev)
    ours = {
        'selCMet': sel.corrected_met(ev, excl), 'Flag_MetFilters': sel.met_filters(ev),
        'Flag_hemVeto': sel.hem_veto(ev, excl),
    }
    for sk, nt in TRIG_MAP.items():
        ours[sk] = ak.to_numpy(ev[nt]).astype(bool)
    sv = sel.classify_svs(ev)
    veto, _ = sel.blinding(sv)
    lep = sv.baseline & (sv.cls == sel.CLS_LEP)
    had = sv.baseline & (sv.cls == sel.CLS_HAD)
    key_nt = event_key(ev.PV_x, ev.PV_y, ev.PV_z, ev.Met_px, ev.Met_py)
    dup_nt = [k for k, c in Counter(key_nt.tolist()).items() if c > 1]
    pv_dup = [k for k, c in Counter(event_key(ev.PV_x, ev.PV_y, ev.PV_z).tolist()).items() if c > 1]
    print(f"ntuple events {len(ev)}, passing golden JSON {good_lumi.sum()}; shared PV positions {len(pv_dup)} "
          f"(PV_npvs==1 events: {int(np.sum(ak.to_numpy(ev.PV_npvs) == 1))}); duplicate PV+MET keys {len(dup_nt)}")

    sk = uproot.open(a.skim)['kuSkimTree'].arrays(SKIM_BRANCHES)
    key_sk = event_key(sk.PVx, sk.PVy, sk.PVz, sk.selMetPx, sk.selMetPy)
    index = {k: i for i, k in enumerate(key_nt.tolist())}
    idx = np.array([index.get(k, -1) for k in key_sk.tolist()])
    m = idx >= 0
    print(f"skim events {len(sk)}, matched to these ntuple files {m.sum()}")
    if a.max_files is None and not m.all():
        print(f"FAIL: {np.sum(~m)} skim events not found in the ntuple files")
    sk, idx = sk[m], idx[m]
    if len(np.unique(idx)) != len(idx):
        print("FAIL: several skim events matched the same ntuple event")
    fails = 0

    def check(name, ok):
        nonlocal fails
        nbad = int(np.sum(~np.asarray(ok)))
        print(f"  {'ok  ' if nbad == 0 else 'FAIL'} {name:<52} mismatches {nbad}/{len(ok)}")
        fails += nbad > 0

    check('golden JSON (skim events must pass)', good_lumi[idx])
    check('selCMet (rel 1e-5)', np.isclose(ours['selCMet'][idx], ak.to_numpy(sk.selCMet), rtol=1e-5, atol=1e-3))
    for b in ['Flag_MetFilters', 'Flag_hemVeto'] + list(TRIG_MAP):
        check(b, ours[b][idx] == ak.to_numpy(sk[b]).astype(bool))
    s = sv[idx]
    l, h = lep[idx], had[idx]
    check('SV_nLeptonic', ak.to_numpy(ak.sum(s.lepAna, axis=1)) == ak.to_numpy(sk.SV_nLeptonic))
    check('SV_nHadronic', ak.to_numpy(ak.sum(h, axis=1)) == ak.to_numpy(sk.SV_nHadronic))
    evm = ev[idx]
    check('LeptonicSV (dxySig, mass) lists',
          np.array([x == y for x, y in zip(sorted_pairs(ak.to_list(s.dxySig[l]), ak.to_list(evm.Vertex_mass[l])),
                                           sorted_pairs(ak.to_list(sk.LeptonicSV_dxySig), ak.to_list(sk.LeptonicSV_mass)))]))
    check('HadronicSV (dxySig, mass) lists',
          np.array([x == y for x, y in zip(sorted_pairs(ak.to_list(s.dxySig[h]), ak.to_list(evm.Vertex_mass[h])),
                                           sorted_pairs(ak.to_list(sk.HadronicSV_dxySig), ak.to_list(sk.HadronicSV_mass)))]))
    v = veto[idx]
    sk_veto = (ak.to_numpy(sk.passNLepGe1SelectionHighDxySigSR)
               | ak.to_numpy(ak.any(sk.HadronicSV_passHadronicID & (sk.HadronicSV_dxySig >= 800), axis=1)))
    check('event veto == skim lep SR | any had ID & dxySig>=800', v == sk_veto)
    for f in ('passNLepGe1SelectionHighDxySigSR', 'passNHadGe1SelectionHighDxySigSR', 'passNSVGe1SelectionHighDxySigSR'):
        check(f'{f} => veto', ~ak.to_numpy(sk[f]) | v)
    print(f"\nskim SR-flag events among matched: {int(np.sum(sk_veto))}; events we veto: {int(v.sum())}")
    print("PARITY OK" if fails == 0 else f"PARITY FAILED ({fails} checks)")
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
