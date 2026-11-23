# source setup.sh   (bash): CMSSW python + svval on PYTHONPATH
_svval_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source /cvmfs/cms.cern.ch/cmsset_default.sh
pushd /uscms_data/d3/anazario/YOURWORKINGAREA/CMSSW_15_1_0_patch5/src > /dev/null
eval "$(scramv1 runtime -sh)"
popd > /dev/null
export PYTHONPATH=${_svval_dir}/src${PYTHONPATH:+:$PYTHONPATH}
echo "svval: CMSSW_15_1_0_patch5 python, PYTHONPATH += ${_svval_dir}/src"
unset _svval_dir
