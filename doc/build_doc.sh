#!/bin/bash -e
# standing on the shoulders of [giants](https://github.com/alidock/alidock/blob/master/alidock-installer.sh)

DOC_REPO=hipe4ml.github.io.git
TMP_DIR="$(dirname "$0")"/builddoc
DATE=$(date +%h-%d-%Y-%H:%M)

set -o pipefail

function pinfo() { echo -e "\033[32m${1}\033[m" >&2; }
function pwarn() { echo -e "\033[33m${1}\033[m" >&2; }
function perr() { echo -e "\033[31m${1}\033[m" >&2; }

function run-clone-doc() {
    pinfo "cloning documentation repository $DOC_REPO"
    type git
    run-clean
    mkdir -p $TMP_DIR
    git clone git@github.com:hipe4ml/$DOC_REPO $TMP_DIR/$DOC_REPO
}

function run-pdoc() {
    pinfo "producing documentation: pdoc3"
    type pdoc3 || pip3 install pdoc3
    pdoc3 --html ../hipe4ml -o $TMP_DIR/$DOC_REPO
}

function run-push-doc() {
    pinfo "pushing documentation to $DOC_REPO"
    cd $TMP_DIR/$DOC_REPO
    git add -A
    git commit -m "Update documentation $DATE"
    git push -f
}

function run-clean() {
    pwarn "cleaning $DOC_REPO"
    [[ -d $TMP_DIR ]] && rm -rf $TMP_DIR || true
}

function run-all() {
    run-clone-doc
    run-pdoc
    run-push-doc
    run-clean
}

# Check parameters
[[ $# == 0 ]] && run-all
while [[ $# -gt 0 ]]; do
    case "$1" in

    all) run-all ;;
    pdoc) run-pdoc ;;
    clonedoc) run-clone-doc ;;
    pushdoc) run-push-doc ;;
    clean) run-clean ;;

    --quiet)
        function pinfo() { :; }
        function pwarn() { :; }
        ;;
    --help)
        pinfo "build_doc.sh: entrypoint to build documentation"
        pinfo ""
        pinfo "Normal usage:"
        pinfo "    build_doc.sh [parameters] [target|all]   # no arguments: test all!"
        pinfo ""
        pwarn "Specific target:"
        pwarn "    build_doc.sh pdoc                      # run pdoc on hipe4ml"
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
