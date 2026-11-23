# SVDataValidation

Per-year and per-era health checks of the HYDDRA secondary vertices (SVs) in the KUCMS
miniAOD **data** ntuples (v34): the SVs, their tracks and the analysis observables, compared
across eras and years with statistical tests. Only the SV control regions are ever filled
(leptonic dxySig < 500, hadronic dxySig < 800); the blinding is applied in the fill stage.

Status: steps 1-4 done (catalog, selection + blinding, fill stage, condor + merge). Plotting and
the statistical comparisons are still to come.

## Environment

Everything runs in the LPC CMSSW_15_1_0_patch5 python (3.9: uproot, awkward, hist, ROOT),
so the code stays 3.9-compatible and condor jobs need no tarball.

    cd .../CMSSW_15_1_0_patch5/src && cmsenv && cd -
    export PYTHONPATH=$PWD/src

On the LPC the framework lives in `/uscms_data/d3/anazario/SVDataValidation` (outside the CMSSW
release area; only `cmsenv` is taken from a release). Manifests, job lists, condor submission dirs
and merged outputs are created there (all git-ignored); job outputs go to EOS.

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
  fake-PV events. Run 2 SVs also have negative diagonal covariance entries (Vertex_cxx/cyy/czz < 0,
  down to -78 in 2018D; 0.4% of 2-track SVs in 2018B after selection), Run 3 none: the likely origin
  of the invalid errors. Both are histogrammed (`negCovDiag`, `cxx`, `sv/badErr`).

## Fill stage (`svval.fill`, `svval.histos`)

    python3 -m svval chunks -m manifests/v34.json --target-gb 5 -o jobs/all_5gb.json   # 881 jobs
    python3 -m svval fill --jobs jobs/all_5gb.json --job 353 -o outputs/test [--max-files 5]

Reads over xrootd (uproot `XRootDSource`; the CMSSW python has no fsspec-xrootd), about 2.5 s per
file and ~360 MB RSS, so a 5 GB job takes ~7 min. Per job, in `<outdir>/job_NNNNN`:

* `.root`: one directory per slice (`2018B_HEM`, `2016F_preVFP`, `2024G`, ...) with `event/`,
  `sv/<category>/<level>/`, `sv/badErr/`, `vtrk/<category>/<level>/`, `track/{all,hyddraLike}/`
  (852 histograms per slice; categories and variables in `histos.py`). hadd-able.
* `_lumis.npz`: a row per (file, run, lumi) read, with counters over the selected events
  (N events, fake PVs, sum N_PV, SVs per class, invalid errors, negative covariances, shared tracks).
* `.json`: cutflow per slice, entries per file, timing. Written last: it marks a complete job.

Blinding in the fill stage: the veto is the first cutflow step, so nothing stored counts a vetoed
event (a pre-veto count at full selection would be the inclusive SR yield); every SV-derived number
uses shown SVs only; every job runs the dxySig-overflow audit and fails instead of writing if it
finds anything. A file that cannot be opened after 3 tries fails the job.

## Condor and merge (`svval.condor`, `svval.merge`)

From the LPC working area, with a grid proxy (`voms-proxy-init -voms cms -valid 192:00`):

    python3 -m svval submit   --tag all_v34 --jobs jobs/all_5gb.json            # 881 jobs
    python3 -m svval submit   --tag t --jobs jobs/all_5gb.json -s 2024G --ids 353 --max-files 3   # tests
    python3 -m svval status   --tag all_v34 [-v]
    python3 -m svval resubmit --tag all_v34 [--include-held]
    python3 -m svval merge    --tag all_v34                                    # -> outputs/all_v34/merged/

* `condor/<tag>/` holds the per-job specs (each job ships only its own file list), a code tarball
  (`src/` + `configs/`), the wrapper (`scram project` of CMSSW_15_1_0_patch5 from cvmfs,
  `svval fill`, `xrdcp` to EOS), the jdl and logs. Re-running `submit` with the same tag adds jobs.
* Outputs: `/store/user/$USER/SVDataValidation/<version>/<tag>/job_NNNNN{.root,_lumis.npz,.json}`.
  A job is done when its `.json` is on EOS (copied last). `status` combines that with condor_q
  (jobs carry the `SVValTag` / `SVValJob` ClassAds); `resubmit` sends every job neither done nor
  queued and shows the last stderr lines of failures.
* `merge` requires every job (or `--allow-missing`), hadds per era (`eras/<year><era>.root`) and
  then everything (`all.root`, all slices), merges the lumi tables (`lumis.npz`), sums the cutflow
  (`cutflow.json`), and fails on duplicate lumis (per year and PD, across all jobs) or a blinding
  audit failure on the merged file (`checks.json`).
* The LPC `condor_*` commands are scripts without a shebang, so they are run through `/bin/sh`.

## Tests

    python3 -m pytest -q
