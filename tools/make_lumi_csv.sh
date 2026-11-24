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
# NORMTAG: the luminometer calibration; normtag_PHYSICS.json is the recommended one. If it does not
# cover the newest runs yet (prompt data), those runs are missing from the CSV and their slices get
# no per-fb^-1 rate (svval reports them as 'runs without L'): rerun when it does.
set -e
NORMTAG=${NORMTAG:-/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json}
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
for year in ${YEARS:-2016 2017 2018 2022 2023 2024 2025 2026}; do
  json=$(json_of $year)
  echo "== $year: $json"
  brilcalc lumi $BRIL_OPTS -u /fb --output-style csv --normtag $NORMTAG \
      -i configs/golden_json/$json -o $OUT/lumi_${year}.csv
  total=$(awk -F, '/^#Summary/{s=1; next} s && !/^#/{print $NF; exit}' $OUT/lumi_${year}.csv)
  ref=$(awk -v y=$year -F': *' '$1==y{print $2}' configs/lumi_fb.yaml)
  echo "   recorded ${total} fb^-1   (configs/lumi_fb.yaml: ${ref})"
done
echo "done: commit $OUT/"
