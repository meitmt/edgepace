#!/usr/bin/env bash
# Install the Android build on the phone (DEVICE_ROOT, default /data/local/tmp/edgepace):
# build/android/{bin,lib}, the clock lock script, the example request bodies and,
# with --media / --model, videos and models.
#
#   android/install.sh                                    # runtime only
#   android/install.sh --media media/test_1080p30_hevc.mp4 --model models/qwen2.5-1.5b-instruct-fp16.gguf
#   ADB=/mnt/c/.../adb.exe android/install.sh ...         # WSL with the Windows adb
#
# Files are sent with `adb push` (from WSL with the Windows adb: through their
# \\wsl.localhost path) to a staging file, moved into place as root, sent only when
# their sha256 differs, and every sent file is checked on the phone afterwards.
# Videos land in $DEVICE_ROOT/media/ and models in $DEVICE_ROOT/models/ under their
# own names, the paths the examples/workloads/oneplus13_*.toml use.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${ANDROID_OUT:-$ROOT/build/android}
DEV=${DEVICE_ROOT:-/data/local/tmp/edgepace}
ADB=${ADB:-adb}
adb_=("$ADB" ${ADB_SERIAL:+-s "$ADB_SERIAL"})
STAGE=/data/local/tmp/.edgepace_push
media=() models=()
while [ $# -gt 0 ]; do
    case "$1" in
        --media) media+=("$2"); shift 2 ;;
        --model) models+=("$2"); shift 2 ;;
        *) echo "usage: $0 [--media VIDEO.mp4] [--model MODEL.gguf ...]" >&2; exit 2 ;;
    esac
done
[ -f "$OUT/SHA256SUMS" ] || { echo "no Android build in $OUT (android/build_android.sh)" >&2; exit 2; }
sh_() { "${adb_[@]}" shell "su -c '$1'" < /dev/null | tr -d '\r'; }
host_path() {   # a path the adb executable can read
    if [[ "$ADB" == *.exe ]] && command -v wslpath > /dev/null; then wslpath -w "$1"; else echo "$1"; fi
}
sh_ "mkdir -p $DEV/bin $DEV/lib $DEV/runs $DEV/requests $DEV/media $DEV/models"
remote_sums=$(sh_ "cd $DEV && find bin lib media models requests locked_run_android.sh -type f 2>/dev/null | xargs sha256sum 2>/dev/null" || true)

# (local file, device path relative to $DEV, executable)
files=()
while read -r f; do files+=("$OUT/$f|$f|x"); done < <(cd "$OUT" && find bin lib -type f | sort)
files+=("$ROOT/android/locked_run_android.sh|locked_run_android.sh|x")
for r in "$ROOT"/requests/*.json; do files+=("$r|requests/$(basename "$r")|"); done
for m in "${media[@]}"; do files+=("$m|media/$(basename "$m")|"); done
for m in "${models[@]}"; do files+=("$m|models/$(basename "$m")|"); done

expected=""
for entry in "${files[@]}"; do
    IFS='|' read -r src dst exe <<< "$entry"
    [ -f "$src" ] || { echo "missing $src" >&2; exit 2; }
    sum=$(sha256sum "$src" | cut -d' ' -f1)
    expected+="$sum  $dst"$'\n'
    grep -q "^$sum  $dst\$" <<< "$remote_sums" && continue
    echo "send $dst ($(du -h "$src" | cut -f1))"
    "${adb_[@]}" push -q "$(host_path "$src")" "$STAGE" > /dev/null
    sh_ "mv $STAGE $DEV/$dst${exe:+ && chmod 755 $DEV/$dst}"
done
actual=$(sh_ "cd $DEV && sha256sum $(cut -d' ' -f3 <<< "$expected" | tr '\n' ' ')")
if diff <(sort -k2 <<< "${expected%$'\n'}") <(sort -k2 <<< "$actual") > /dev/null; then
    echo "INSTALL_OK $DEV ($(wc -l <<< "${expected%$'\n'}") files)"
else
    echo "files on the phone differ from the host:" >&2
    diff <(sort -k2 <<< "${expected%$'\n'}") <(sort -k2 <<< "$actual") >&2 || true
    exit 1
fi
