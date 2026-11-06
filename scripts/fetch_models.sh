#!/usr/bin/env bash
# Download the example models (GGUF, Hugging Face) into models/.
#
#   scripts/fetch_models.sh                  # the Jetson examples' models
#   scripts/fetch_models.sh qwen25-1.5b qwen3-vl-2b   # by name
#   scripts/fetch_models.sh --list
#
# MODELS_DIR (default models/) is where they go; downloads resume and are checked
# by size.  HF_TOKEN is passed on when set.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
dir=${MODELS_DIR:-"$root/models"}
hf=https://huggingface.co

# name | repository | file | bytes
catalog="
qwen25-1.5b|Qwen/Qwen2.5-1.5B-Instruct-GGUF|qwen2.5-1.5b-instruct-fp16.gguf|3560416288
qwen25-vl-3b|ggml-org/Qwen2.5-VL-3B-Instruct-GGUF|Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf|1929901056
qwen25-vl-3b|ggml-org/Qwen2.5-VL-3B-Instruct-GGUF|mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf|1338428128
qwen3-vl-2b|Qwen/Qwen3-VL-2B-Instruct-GGUF|Qwen3VL-2B-Instruct-Q8_0.gguf|1834427424
qwen3-vl-2b|Qwen/Qwen3-VL-2B-Instruct-GGUF|mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf|445053216
"

if [ "${1:-}" = "--list" ]; then
    echo "$catalog" | awk -F'|' 'NF == 4 { printf "%-14s %-40s %6.2f GB\n", $1, $3, $4 / 1e9 }'
    exit 0
fi
if [ "${1:-}" = "--files" ]; then          # the file names of a model (android/setup_android.sh)
    echo "$catalog" | awk -F'|' -v n="${2:?--files NAME}" '$1 == n { print $3 }'
    exit 0
fi
names=("$@")
[ ${#names[@]} -gt 0 ] || names=(qwen25-1.5b qwen25-vl-3b)
mkdir -p "$dir"
auth=()
[ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
progress=(-sS)
[ -t 2 ] && progress=(--progress-bar)

for name in "${names[@]}"; do
    rows=$(echo "$catalog" | awk -F'|' -v n="$name" '$1 == n')
    [ -n "$rows" ] || { echo "unknown model $name (--list)" >&2; exit 2; }
    while IFS='|' read -r _ repo file bytes; do
        out="$dir/$file"
        if [ -f "$out" ] && [ "$(stat -c %s "$out")" = "$bytes" ]; then
            echo "have $out"
            continue
        fi
        echo "get  $repo/$file ($(awk -v b="$bytes" 'BEGIN { printf "%.2f GB", b / 1e9 }'))"
        df_free=$(df -B1 --output=avail "$dir" | tail -1)
        [ "$df_free" -gt "$bytes" ] || { echo "not enough free space in $dir for $file" >&2; exit 1; }
        curl -fL --retry 5 -C - "${progress[@]}" "${auth[@]}" -o "$out.part" "$hf/$repo/resolve/main/$file"
        [ "$(stat -c %s "$out.part")" = "$bytes" ] || { echo "size mismatch: $out.part" >&2; exit 1; }
        mv "$out.part" "$out"
    done <<< "$rows"
done
echo "models in $dir"
