#!/usr/bin/env bash
# Hardware-free checks: Python unit tests, the progress-predictor test, and that the
# C++ host tools compile.  Needs python3 (>= 3.11, or `pip install tomli`), g++, make.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
echo "== python unit tests"
python3 -m unittest discover -s tests -v
echo "== progress predictor (monitor/tests)"
make -s -C monitor test
echo "== sampler builds"
make -s -C monitor
if [ -f llama.cpp/ggml/src/edgepace/abi.h ]; then
    echo "== client and policy DSO build (llama.cpp patched)"
    make -s -C client
    tmp=$(mktemp -d)
    g++ -O2 -std=c++17 -fPIC -shared -pthread -Illama.cpp/external/edgepace_policy -Illama.cpp/external/edgepace_policy/include \
        -o "$tmp/libedgepace_policy.so" llama.cpp/external/edgepace_policy/edgepace_policy.cpp
    rm -rf "$tmp"
else
    echo "== skipping the client and policy builds: run scripts/fetch_llama.sh first"
fi
echo "OFFLINE_TESTS_OK"
