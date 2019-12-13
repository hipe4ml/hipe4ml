#!/bin/bash -e
# standing on the shoulders of [giants](https://github.com/alidock/alidock/blob/master/alidock-installer.sh)

TEST=$1
set -o pipefail
cd "$(dirname "$0")"/..

function pinfo() { echo -e "\033[32m${1}\033[m" >&2; }
function pwarn() { echo -e "\033[33m${1}\033[m" >&2; }
function perr() { echo -e "\033[31m${1}\033[m" >&2; }

setup-hipe4ml() {
    pinfo "installing: hipe4ml"
    pip3 install --upgrade --force-reinstall --no-deps -e .
}

test-pylint() {
    pinfo "running test: pylint"
    type pylint
    find . -name '*.py' -a -not -path './dist/*' \
        -a -not -path './build/*' | xargs pylint
}

test-flake8() {
    pinfo "running test: flake8"
    type flake8
    # stop the build if there are Python syntax errors or undefined names
    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    # exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
    flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
}

test-pytest() {
    setup-hipe4ml
    pinfo "running test: pytest"
    type pytest
    pytest tests
}

test-all() {
    test-pylint
    test-flake8
    test-pytest
}

# Check parameters
[[ $# == 0 ]] && test-all
while [[ $# -gt 0 ]]; do
    case "$1" in

    all) test-all ;;
    pylint) test-pylint ;;
    flake8) test-flake8 ;;
    pytest) test-pytest ;;

    --quiet)
        function pinfo() { :; }
        function pwarn() { :; }
        ;;
    --help)
        pinfo "run_tests.sh: entrypoint to launch tests locally or on CI"
        pinfo ""
        pinfo "Normal usage:"
        pinfo "    run_tests.sh [parameters] [test|all]   # no arguments: test all!"
        pinfo ""
        pwarn "Specific tests:"
        pwarn "    run_tests.sh pylint                    # test with pylint"
        pwarn "    run_tests.sh flake8                    # test with flake8"
        pwarn "    run_tests.sh test-pytest               # test with pytest"
        pwarn ""
        pwarn "Parameters:"
        pwarn "    --help                                 # print this help"
        pwarn "    --quiet                                # suppress messages (except errors)"
        exit 1
        ;;
    *)
        perr "Unknown option: $1"
        exit 2
        ;;
    esac
    shift
done
