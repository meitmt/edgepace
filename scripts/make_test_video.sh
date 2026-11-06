#!/usr/bin/env bash
# The test video of the example applications: 1920x1080, 30 fps, HEVC at 750 kbit/s
# (the Jetson application decodes with h265parse ! nvv4l2decoder, the phone's with
# MediaCodec), 125 s — the format of the paper's clip (an OpenEQA scene, not
# redistributable).
#
#   scripts/make_test_video.sh [output]               # default media/test_1080p30_hevc.mp4
#   scripts/make_test_video.sh --synthetic [output]   # offline: ffmpeg's testsrc2 pattern
#
# Default source: "Big Buck Bunny" (c) Blender Foundation, www.bigbuckbunny.org,
# CC BY 3.0 — downloaded (275 MB, cached in media/.cache), 125 s from 1:00 re-encoded.
# Content matters: the profile measures how far the LLM slows the application's frames,
# and a synthetic pattern decodes and processes more lightly than camera-like footage
# (it gave the PMU stall weight 0 and a lower throughput operating point on the Orin).
# Needs ffmpeg with libx265, curl, python3.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
synthetic=0
[ "${1:-}" = "--synthetic" ] && { synthetic=1; shift; }
out=${1:-"$root/media/test_1080p30_hevc.mp4"}
mkdir -p "$(dirname "$out")"
encode=(-c:v libx265 -preset medium -x265-params log-level=error -b:v 750k -pix_fmt yuv420p -tag:v hvc1 -an)
if [ "$synthetic" = 1 ]; then
    ffmpeg -hide_banner -loglevel error -y -f lavfi -i "testsrc2=size=1920x1080:rate=30" -t 125 "${encode[@]}" "$out"
else
    cache="$root/media/.cache"
    src="$cache/bbb_sunflower_1080p_30fps_normal.mp4"
    mkdir -p "$cache"
    if [ ! -f "$src" ]; then
        url=https://download.blender.org/demo/movies/BBB/bbb_sunflower_1080p_30fps_normal.mp4.zip
        curl -fL --retry 5 -C - -o "$src.zip" "$url"
        python3 -m zipfile -e "$src.zip" "$cache/"
        rm -f "$src.zip"
    fi
    ffmpeg -hide_banner -loglevel error -y -ss 60 -i "$src" -t 125 -r 30 "${encode[@]}" "$out"
fi
ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate,bit_rate -show_entries format=duration \
    -of compact "$out"
echo "$out"
