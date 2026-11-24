#!/bin/bash
# One-time: recorded luminosity per run of every golden run, for the per-fb^-1 rates of svval plot.
# Writes configs/lumi_brilcalc/lumi_<year>.csv (commit them; svval reads them automatically).
#
# Nothing to install: brilcalc comes from the BRIL container on cvmfs (brilws-env), sourced below
# when brilcalc is not already available. Run where the luminosity DB is reachable, e.g. on lxplus,
# from the repository top:
#     bash tools/make_lumi_csv.sh                 # all years; YEARS="2024 2025" for a subset
#     BRIL_OPTS="-c web" bash tools/make_lumi_csv.sh   # outside CERN (untested from the LPC)
#
# Normtag (luminometer calibration), per year: normtag_PHYSICS.json (the recommended one) up to 2024,
# whose runs it covers (to 386951); normtag_BRIL.json (best available calibration per run, covers the
# newest runs) from 2025. Override with NORMTAG=<file> for every year, or NORMTAG_2025=<file> etc.
# Rerun a subset with YEARS="2025 2026"; rerun everything when a golden JSON or normtag changes.
# brilcalc warnings "... is not a superset of ..." are lumi-section edges where the golden JSON and
# the normtag differ by a few sections: harmless, brilcalc uses the overlap.
NT=/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags
OUT=configs/lumi_brilcalc
mkdir -p $OUT
shopt -s expand_aliases            # brilws-env defines brilcalc as an alias
if ! type brilcalc > /dev/null 2>&1; then
  source /cvmfs/cms-bril.cern.ch/cms-lumi-pog/brilws-docker/brilws-env > /dev/null
fi
type brilcalc > /dev/null 2>&1 || { echo "brilcalc not available (no /cvmfs/cms-bril.cern.ch?)"; exit 1; }
[ -d configs/golden_json ] || { echo "run from the SVDataValidation directory"; exit 1; }

json_of() {   # golden JSON of a year (the files of configs/golden_json)
  case $1 in
    2016) echo Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt ;;
    2017) echo Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt ;;
    2018) echo Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.json ;;
    2022) echo Cert_Collisions2022_355100_362760_Golden.json ;;
    2023) echo Cert_Collisions2023_366442_370790_Golden.json ;;
    2024) echo Cert_Collisions2024_378981_386951_Golden.json ;;
    2025) echo Cert_Collisions2025_391658_398903_Golden.json ;;
    2026) echo Cert_Collisions2026_401624_403937_golden.json ;;
    *) echo "no golden JSON configured for $1" >&2; return 1 ;;
  esac
}
normtag_of() {
  local v="NORMTAG_$1"
  if [ -n "${!v}" ]; then echo "${!v}"
  elif [ -n "$NORMTAG" ]; then echo "$NORMTAG"
  elif [ "$1" -le 2024 ]; then echo $NT/normtag_PHYSICS.json
  else echo $NT/normtag_BRIL.json
  fi
}

failed=""
for year in ${YEARS:-2016 2017 2018 2022 2023 2024 2025 2026}; do
  json=$(json_of $year) || { failed="$failed $year"; continue; }
  nt=$(normtag_of $year)
  csv=$OUT/lumi_${year}.csv
  echo "== $year: $json, normtag $(basename $nt)"
  if ! brilcalc lumi $BRIL_OPTS -u /fb --output-style csv --normtag $nt -i configs/golden_json/$json -o $csv \
       || ! grep -q '^[0-9]' $csv; then
    echo "   FAILED (no rows); $csv removed"
    rm -f $csv; failed="$failed $year"; continue
  fi
  # summary block: '#nfill,nrun,...,totrecorded(/fb)' then '#<values>' (brilcalc prefixes it with '#')
  total=$(awk -F, '/^#nfill/{getline; print $NF; exit}' $csv)
  echo "   recorded ${total} fb^-1"
done
if [ -n "$failed" ]; then echo "FAILED years:$failed  (fix and rerun with YEARS=\"...\")"; exit 1; fi
echo "done: commit $OUT/"
