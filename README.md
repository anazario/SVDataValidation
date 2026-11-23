# SVDataValidation

Per-year and per-era health checks of the HYDDRA secondary vertices (SVs) in the KUCMS
miniAOD **data** ntuples (v34): the SVs, their tracks and the analysis observables, compared
across eras and years with statistical tests. Only the SV control regions are ever filled
(leptonic dxySig < 500, hadronic dxySig < 800); the blinding is applied in the fill stage.

Status: **step 1 (file catalog)** and **step 2 (selection + blinding)** done. Fill, condor,
merge and plotting are still to come.

## Environment

Everything runs in the LPC CMSSW_15_1_0_patch5 python (3.9: uproot, awkward, hist, ROOT),
so the code stays 3.9-compatible and condor jobs need no tarball.

    cd .../CMSSW_15_1_0_patch5/src && cmsenv && cd -
    export PYTHONPATH=$PWD/src

## Catalog

Input datasets, exclusions and detector periods are in `configs/datasets_v34.yaml`.

    python3 -m svval catalog   -c configs/datasets_v34.yaml -o manifests/v34.json   # ~1 min over FUSE
    python3 -m svval summary   -m manifests/v34.json -s 2024
    python3 -m svval lumicheck -m manifests/v34.json -s 2025G -j 8                 # duplicate lumis, bad files
    python3 -m svval chunks    -m manifests/v34.json -s run3 --target-gb 5 -o jobs/run3.json

Selections (`-s`): `all`, `run2`, `run3`, or a comma list of `2024`, `2024C`, `2016:preVFP`.

* The era comes from the `Run20YYX` token of the CRAB request name; 2016 HIPM/non-HIPM from `HIPM`.
* Periods: 2016 preVFP/postVFP, 2022 preEE (CD) / postEE (EFG), 2023 preBPix (BC) / postBPix (D).
  2018 HEM is run-based (from run 319077) and is split per event in the fill stage.
* Excluded: `JetMET_R25/backup/` (copy of the JetMET0 Run2025G recovery task). The separate
  `kucmsntuple_MET_R18D_SVHPM100_MiniAODv2_v34` (2026-08-31 reprocessing of 2018D) is not configured.
* `catalog` fails on any directory that does not follow the CRAB layout, and on a request with
  two task stamps. `lumicheck` relies on CRAB never splitting a lumi section across jobs: a
  (run, lumi) pair in two files is double-processed data.
* Job chunks never mix (year, era, period), so each job output belongs to exactly one era.

## Selection and blinding (`svval.selection`)

A port of KUCMSSkimmer (f9af717), checked event by event against kuSkimTree with
`tools/skim_parity.py` (2016F postVFP all 22543 skim events, 2018D and 2024H 40 files each:
every check agrees).

* Events: golden JSON (`configs/golden_json`, copied from the skimmer), the skimmer's MET filters,
  the 4-path PFMET(NoMu)120 trigger OR, the data HEM veto (2018 runs 319077-326634; removes ~47%
  of 2018D events), corrected MET `selCMet` > 150 (OOT photons subtracted, GED photons within
  dR 0.3 of an OOT photon added back).
* SVs: invalid `dxyError` (0 or non-finite) -> 'badErr'; skimmer baseline; 2 tracks = leptonic
  (flavour from the loose mu / e IDs: mu, e, both), 3 tracks, >= 4 tracks = hadronic (ID m/N > 1).
* Blinding (constants, no unblind switch): the event veto drops events with an analysis SV beyond
  the CR (leptonic dxySig >= 500, hadronic >= 800; looser than every SR flag, equal to the skimmer's
  `anyLepDxySigGeq500 || anyHadDxySigGeq800`); the per-SV cap removes every other SV beyond the
  threshold of its track multiplicity from all histograms.

Observations from the parity runs, to be followed up by the fill stage:
* Events without a reconstructed PV get the beamspot-fallback PV (PV_npvs == 1, one fixed position
  per run): ~1.7% of events in 2016F/2018D, 0.06% in 2024H. SV dxy is measured from this PV.
* ~2% of Run 2 SVs (2016F, 2018D) have an invalid dxyError; none in 2024H/G. They are not in
  fake-PV events.

## Tests

    python3 -m pytest -q
