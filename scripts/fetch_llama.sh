#!/usr/bin/env bash
# Fetch upstream llama.cpp at the EdgePace base and apply the EdgePace patches
# (patches/llama.cpp: the EdgePace sliding-window scheduler, the graph cutting, the
# external policy) -> ./llama.cpp on branch edgepace.
#
#   scripts/fetch_llama.sh                 # clone github.com/ggml-org/llama.cpp
#   LLAMA_REPO=/path/to/llama.cpp scripts/fetch_llama.sh   # or a local clone / mirror
#
# Re-running on an existing ./llama.cpp only checks that it carries the patches.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
base=790cf51aabd61763486050dec7451d9147cb7c61     # upstream llama.cpp the patches apply to
repo=${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp.git}
dst="$root/llama.cpp"
patches=("$root"/patches/llama.cpp/*.patch)

if [ -d "$dst/.git" ]; then
    if git -C "$dst" merge-base --is-ancestor "$base" HEAD 2>/dev/null &&
       [ -f "$dst/ggml/src/ggml-edgepace.h" ]; then
        echo "llama.cpp already patched: $(git -C "$dst" log --oneline -1)"
        exit 0
    fi
    echo "$dst exists but is not a patched llama.cpp; move it away first" >&2
    exit 2
fi
git clone --filter=blob:none --no-checkout "$repo" "$dst"
git -C "$dst" checkout -q -b edgepace "$base"
# the patches carry their authors; commit as whoever runs this
git -C "$dst" -c user.name="${GIT_AUTHOR_NAME:-edgepace}" -c user.email="${GIT_AUTHOR_EMAIL:-edgepace@localhost}" \
    am -q --committer-date-is-author-date "${patches[@]}"
echo "llama.cpp ready: $(git -C "$dst" log --oneline -1) (${#patches[@]} patches on $base)"
