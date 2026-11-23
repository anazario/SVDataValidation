# source setup.csh   (tcsh, the LPC login shell), from the SVDataValidation directory:
# CMSSW python + svval on PYTHONPATH. (tcsh cannot tell a sourced script its own path, so the
# current directory is used.)
if ( ! -d src/svval ) then
    echo "setup.csh: run it from the SVDataValidation directory (no src/svval in `pwd`)"
else
    set _svval_dir = `pwd`
    source /cvmfs/cms.cern.ch/cmsset_default.csh
    pushd /uscms_data/d3/anazario/YOURWORKINGAREA/CMSSW_15_1_0_patch5/src > /dev/null
    eval `scramv1 runtime -csh`
    popd > /dev/null
    if ( $?PYTHONPATH ) then
        setenv PYTHONPATH ${_svval_dir}/src:${PYTHONPATH}
    else
        setenv PYTHONPATH ${_svval_dir}/src
    endif
    echo "svval: CMSSW_15_1_0_patch5 python, PYTHONPATH += ${_svval_dir}/src"
    unset _svval_dir
endif
