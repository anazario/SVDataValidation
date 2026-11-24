# SVDataValidation

Per-year and per-era health checks of the HYDDRA secondary vertices (SVs) in the KUCMS
miniAOD **data** ntuples (v34): the SVs, their tracks and the analysis observables, compared
across eras and years with statistical tests. Only the SV control regions are ever filled
(leptonic dxySig < 500, hadronic dxySig < 800); the blinding is applied in the fill stage.

Status: all five stages exist (catalog, selection + blinding, fill, condor + merge, plots and
statistical comparisons).

## Environment

Everything runs in the LPC CMSSW_15_1_0_patch5 python (3.9: uproot, awkward, hist, ROOT),
so the code stays 3.9-compatible and condor jobs need no tarball.

    cd /uscms_data/d3/anazario/SVDataValidation
    source setup.csh        # tcsh (LPC login shell); bash: source setup.sh

It sets up the CMSSW_15_1_0_patch5 environment (from the release in YOURWORKINGAREA) and puts
`src/` on PYTHONPATH; without it `python3 -m svval` fails with "No module named svval".

On the LPC the framework lives in `/uscms_data/d3/anazario/SVDataValidation` (outside the CMSSW
release area; only `cmsenv` is taken from a release). Manifests, job lists, condor submission dirs
and merged outputs are created there (all git-ignored); job outputs go to EOS.

## Quick start: validating a new production

Example: new ntuples for one full year (here 2026, version v35) have finished on EOS.
Everything runs on the LPC, from `/uscms_data/d3/anazario/SVDataValidation`.

1. **Update and set up**

       cd /uscms_data/d3/anazario/SVDataValidation && git pull
       source setup.csh
       voms-proxy-init -voms cms -valid 192:00

2. **Golden JSON.** Make sure the year's certification JSON is in `configs/golden_json/`
   (a run missing from every JSON fails the lumi mask, so without it every event is rejected;
   `merge` stops with an error if a slice has events but none pass). If a newer JSON replaces an
   old one, remove the old file (overlapping ranges are an error).

3. **Dataset config.** Copy `configs/datasets_v34.yaml` to `configs/datasets_v35.yaml` and list
   only the new directories, e.g.

       version: v35
       datasets:
         - {dir: kucmsntuple_JetMET_R26_SVHPM100_MiniAOD_v35, year: 2026}

   Add `exclude:` patterns for backup / duplicate directories, and a `periods:` entry if the year
   has detector periods (optional).

4. **Catalog, checks, job list**

       python3 -m svval catalog   -c configs/datasets_v35.yaml -o manifests/v35.json
       python3 -m svval summary   -m manifests/v35.json
       python3 -m svval lumicheck -m manifests/v35.json -j 8        # optional; merge checks lumis too
       python3 -m svval chunks    -m manifests/v35.json --target-gb 5 -o jobs/v35_2026.json

   `catalog` fails on directories that do not follow the CRAB layout and lists eras made by
   several CRAB requests for review. With a manifest of several years, `-s 2026` (or `-s 2026C`,
   `-s run3`) restricts `summary`, `lumicheck` and `chunks`.

5. **Submit, follow, resubmit**

       python3 -m svval submit   --tag v35_2026 --jobs jobs/v35_2026.json [--max-mat 200]
       python3 -m svval status   --tag v35_2026          # repeat until everything is done
       python3 -m svval resubmit --tag v35_2026          # failed jobs (shows their last stderr)

   `--max-mat N` keeps at most N jobs in the queue at once (condor `max_materialize`); the rest
   are added as jobs finish, and `status` counts them as "waiting to materialize". Outputs go to
   `/store/user/$USER/SVDataValidation/v35/v35_2026/`. A quick test first:
   `submit --tag v35_test --jobs jobs/v35_2026.json --ids 0 --max-files 3`.

6. **Merge** (all jobs must be done, or `--allow-missing`)

       python3 -m svval merge --tag v35_2026     # -> EOS .../v35/v35_2026/merged/ and outputs/v35_2026/merged/

   Look at `checks.json` (duplicate lumis, blinding audit, slices without golden JSON) and
   `cutflow.json`. The histograms are in `all.root` (one directory per slice) and `eras/`.

7. **Plots and era comparisons**

       python3 -m svval plot --tag v35_2026        # -> outputs/v35_2026/plots/index.html (~1 min)

   The plots are in `outputs/v35_2026/plots/` on the LPC; open `index.html` there. To browse them on
   a laptop, copy the directory first, e.g.
   `rsync -a lpc:/uscms_data/d3/anazario/SVDataValidation/outputs/v35_2026/plots/ ~/data/svval_plots/`
   and open `~/data/svval_plots/index.html`. With one
   year only, the eras of that year are compared with each other; nothing is compared across runs.

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

    python3 -m svval submit   --tag all_v34 --jobs jobs/all_5gb.json [--max-mat 200]   # 881 jobs
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
* Jobs run the code packed into `condor/<tag>/svval_code.tgz` at the first submission. After fixing
  a bug, `resubmit --refresh-code` repacks it from the current checkout first; every packing is
  recorded with its git commit in `tag.json` (`code`), so it stays known which code made which job.
* `--max-mat N` (submit and resubmit) sets condor `max_materialize`: at most N jobs of the round in
  the queue at once. Jobs not materialized yet are not in condor_q; `status` reports them as
  "waiting to materialize" while the cluster's factory is alive, so `resubmit` never sends them twice.
* `merge` follows skim_submit.py `--transfer`: it stages every job output from EOS to local scratch
  with `xrdcp` (`--workers` streams, default 8; scratch `--scratch`, default
  `/uscmst1b_scratch/lpc1/3DayLifetime/$USER`), hadds there (eras at the same time: `--era-workers`
  4 x `hadd -j` `--hadd-jobs` 2; no `-k`, so a corrupt input fails the merge), and copies the products
  with `xrdcp` to EOS (`--dest`, default `<job outputs>/merged/`; `checks.json` last, as the
  completion marker). A local copy goes to `outputs/<tag>/merged/` (`-o`) for `plot`, and scratch is
  removed (`--keep-scratch` keeps it). An existing merge on EOS is only overwritten with `--force`;
  `--no-upload` keeps everything local, `--dry-run` only shows the plan.
  Products: `eras/<year><era>.root`, `all.root` (all slices), `lumis.npz` (lumi tables), `cutflow.json`,
  `checks.json`. Requires every job (or `--allow-missing`); fails, uploading nothing, on a staging or
  hadd error, duplicate lumis (per year and PD, across all jobs), a blinding-audit failure on the
  merged file, or a slice with events but none in the golden JSON. The full v34 production is about
  2,600 files to stage; the merged products are a few tens of MB.
* The LPC `condor_*` commands are scripts without a shebang, so they are run through `/bin/sh`.

## Plots and comparisons (`svval.plotting`)

    python3 -m svval plot --tag all_v34 [-s 2016] [--all-vars] [--lumi-csv brilcalc_*.csv] [--formats png,pdf]

Units and groups: a *slice* is an era, split further by detector period and the 2018 HEM period
(`2016F_preVFP`, `2018B_HEM`, `2024G`). The slices of a year form one group, the years of a Run
another. Output in `outputs/<tag>/plots/` (or `-o`):

* `stats/comparisons.csv`: every 1D histogram (811), every unit vs the *rest* of its group
  (never vs a total that contains it): Kolmogorov distance D and p-value, chi2/ndf and p-value of
  the normalized shapes, mean shift in RMS of the reference. A comparison is **flagged** when
  KS p < 1e-3 *and* D >= 0.02 (`--p-max`, `--d-min`): with these statistics any drift is
  significant, so the size of the difference decides. Beamspot and PV histograms (`event/BS_*`,
  `event/PV_*`, including `PV_npvs`) are in the CSV but never flagged and not in the KS maps:
  beam conditions and pileup change by design. Histograms with one populated bin in the whole
  merge (e.g. `nTracks` of 2-track SVs) are constant by construction, so they are skipped and
  listed in `index.html`. `stats/<group>/ksmap_*`: D per unit and variable for each section
  (event, SV category and level, SV tracks, Track_), flagged cells starred.
* SV-track SC match: `vtrk/*/scMatch` is `0 <= VertexTrack_SCDR < 0.04`, the ntupleizer's own
  electron-track threshold (what `Vertex_passLooseElectronID` uses). `SCDR >= 0` alone only means
  the Hungarian assignment gave the track some SC, at any dR, so it follows the number of stored
  SCs, which is ~3x larger from 2024 on. Merges filled before this change have the old `hasSC`.
* `overlays/<group>/`: shape overlays of the units (unit area) with the ratio to the group total,
  and each unit's D in the legend; a curated list of ~150 histograms (`plots.CURATED`), or all
  with `--all-vars`.
* `rates/<group>/`: per unit, SVs per event by class (mu mu, ee, mu+e, no-ID, 3-track, hadronic
  with/without ID), shared tracks per SV, fractions of SVs with invalid sigma_dxy or negative
  covariance, fraction of fake-PV events, mean N_PV; also per fb^-1 (see Luminosity below).
* `trends/<year>/`: the same quantities run by run (runs with >= 100 selected events), with the
  slice boundaries.
* `maps/<year>/`: SV x-y (full and inner 10 cm) and r-z maps, also for the invalid-error SVs.
* `index.html`: merge checks, cutflow per slice, outlier summary per unit, the 60 largest flagged
  differences (linked to their overlays) and galleries of everything.

Luminosity: everything comes from brilcalc, computed **once** and committed in
`configs/lumi_brilcalc/lumi_<year>.csv` (recorded luminosity of every golden run).
`tools/make_lumi_csv.sh` makes them (run on lxplus from the repository top; brilcalc comes from the
BRIL container on cvmfs, nothing to install; normtag_PHYSICS up to 2024, normtag_BRIL from 2025).
Rerun it when a golden JSON changes or a newer normtag covers the recent runs.

* Plot headers: the sum of the CSV year totals of the years shown, rounded to the nearest integer
  (e.g. 96 fb^-1 for 2016 + 2018). Only drawn when every catalogued ntuple file of those years is in
  the merge (`checks.json` `file_coverage`); a test tag, a subset or a partial merge shows only the
  energy.
* Per-fb^-1 rates (slices and years): per run, recorded / nls of the brilcalc row (the average per
  lumi section) times the golden lumi sections present in the merge. Missing jobs or CRAB failures
  therefore do not bias the rates, and neither do runs where the normtag covers fewer lumi sections
  than the golden JSON (brilcalc reports only the overlap; these are listed in the index).
* `index.html` cutflow: L per slice, golden-LS coverage, runs without a value (their slice gets no
  per-fb^-1 point), runs whose L was extrapolated. `--lumi-csv` replaces the committed CSVs
  (per-run or per-LS brilcalc output).

## Tests

    python3 -m pytest -q
