#!/usr/bin/env bash
# Run a command on a Jetson Orin with MAXN_SUPER and CPU/GPU/EMC locked, then
# always restore the default platform configuration and verify it.
#
#   scripts/locked_run.sh <output-dir> -- <command> [args...]
#
# Root access: uses `sudo`; set SUDO_PW in the environment for non-interactive
# use.  Default mode on this board is nvpmodel -m 1 (25W); MAXN_SUPER is -m 2.
# Override with DEFAULT_MODE / MAXN_MODE.  EMC must lock at EXPECT_EMC_HZ
# (default 3199000000, the Orin Nano Super maximum).
#
# Evidence: platform_locked.txt (before the command), tegrastats.log (root,
# 200 ms, CPU/GPU/EMC frequencies throughout), platform_end.txt (after);
# LOCK_VERIFIED, LOCK_CONTINUOUS and LOCK_HELD summarize them.
#
# Video clocks: VIDEO_CLOCKS=dynamic (default) leaves NVDEC/VIC under
# their devfreq governor, as in the paper-era runs; VIDEO_CLOCKS=locked pins
# them at their maximum (as batch.py's VideoClocks does) and restores them.
# Both record the state every second in video_clocks.log; VIDEO_CLOCKS=...
# and, when locked, VIDEO_LOCK_CONTINUOUS summarize it.
set -u -o pipefail

out=${1:?usage: locked_run.sh <output-dir> -- <command...>}
shift
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "missing command" >&2; exit 2; }
mkdir -p "$out"
out=$(cd "$out" && pwd)
default_mode=${DEFAULT_MODE:-1}
maxn_mode=${MAXN_MODE:-2}
expect_emc_hz=${EXPECT_EMC_HZ:-3199000000}
video_clocks=${VIDEO_CLOCKS:-dynamic}
case "$video_clocks" in dynamic|locked) ;; *) echo "VIDEO_CLOCKS must be dynamic or locked" >&2; exit 2;; esac
video_devices="/sys/class/devfreq/15480000.nvdec /sys/class/devfreq/15340000.vic"
video_settings="min_freq max_freq governor device/power/control"
video_saved=0
video_sampler_pid=
telemetry_started=0
log="$out/platform_commands.log"
: > "$log"

as_root() {
    if [ -n "${SUDO_PW:-}" ]; then
        printf '%s\n' "$SUDO_PW" | sudo -S -p '' "$@"
    else
        sudo "$@"
    fi
}

logged() {
    printf 'COMMAND=%s\n' "$*" >> "$log"
    as_root "$@" >> "$log" 2>&1
    local rc=$?
    printf 'EXIT_CODE=%s\n' "$rc" >> "$log"
    return "$rc"
}

show() { as_root /usr/bin/jetson_clocks --show > "$1" 2>&1; }

# every CPU/GPU line pinned (min == max == current); EMC overridden, at its
# maximum, and that maximum is the expected EMC_HZ
verify_locked() {
    local file=$1
    grep -q 'NV Power Mode: MAXN_SUPER' "$file" || { echo "not MAXN_SUPER"; return 1; }
    awk -v want_emc="$expect_emc_hz" '
        /^cpu[0-9]+:/ || /^GPU MinFreq=/ {
            n++
            for (i = 1; i <= NF; i++) { split($i, kv, "="); v[kv[1]] = kv[2] }
            if (v["MinFreq"] != v["MaxFreq"] || v["MaxFreq"] != v["CurrentFreq"]) {
                print "unlocked: " $0; bad = 1
            }
        }
        /^EMC MinFreq=/ {
            emc = 1
            for (i = 1; i <= NF; i++) { split($i, kv, "="); e[kv[1]] = kv[2] }
            if (e["FreqOverride"] != 1 || e["CurrentFreq"] != e["MaxFreq"] ||
                e["CurrentFreq"] != want_emc) {
                print "EMC not locked at " want_emc ": " $0; bad = 1
            }
        }
        END { if (n < 2 || !emc) { print "missing CPU/GPU/EMC lines"; bad = 1 } exit bad }
    ' "$file"
}

# every root tegrastats line must show the locked CPU/GPU/EMC frequencies
# (tegrastats truncates to whole MHz, e.g. a 1020000000 Hz GPU prints 1019)
verify_continuous() {
    local file=$1 cpu_mhz=$2 gpu_mhz=$3 emc_mhz=$4
    awk -v cpu="$cpu_mhz" -v gpu="$gpu_mhz" -v emc="$emc_mhz" '
        function off(a, b) { d = a - b; return d > 1 || d < -1 }
        /EMC_FREQ/ {
            n++
            if (match($0, /CPU \[[^]]*\]/)) {
                k = split(substr($0, RSTART + 5, RLENGTH - 6), cores, ",")
                for (i = 1; i <= k; i++) {
                    split(cores[i], f, "@")
                    if (f[2] != "off" && off(f[2] + 0, cpu)) { cpu_bad++; break }
                }
            }
            if (match($0, /EMC_FREQ [0-9]+%@[0-9]+/)) {
                split(substr($0, RSTART, RLENGTH), f, "@"); if (off(f[2] + 0, emc)) emc_bad++
            } else emc_bad++
            if (match($0, /GR3D_FREQ [0-9]+%@\[[0-9]+/)) {
                split(substr($0, RSTART, RLENGTH), f, "["); if (off(f[2] + 0, gpu)) gpu_bad++
            } else gpu_bad++
        }
        END {
            printf "samples=%d cpu_off_lock=%d gpu_off_lock=%d emc_off_lock=%d\n",
                   n, cpu_bad, gpu_bad, emc_bad
            exit (n == 0 || cpu_bad || gpu_bad || emc_bad)
        }
    ' "$file"
}

stop_telemetry() {
    if [ "$telemetry_started" = 1 ]; then
        as_root tegrastats --stop >/dev/null 2>&1
        telemetry_started=0
    fi
    if [ -n "$video_sampler_pid" ]; then
        kill "$video_sampler_pid" 2>/dev/null
        wait "$video_sampler_pid" 2>/dev/null
        video_sampler_pid=
    fi
}

# --- video clocks (NVDEC/VIC devfreq) ---------------------------------------
# sysfs writes go through `sh -c` arguments: as_root's stdin carries the password
write_sysfs() {
    printf 'COMMAND=write %s=%s\n' "$1" "$2" >> "$log"
    as_root sh -c 'printf "%s\n" "$2" > "$1"' _ "$1" "$2" >> "$log" 2>&1
    local rc=$?
    [ "$rc" -eq 0 ] && [ "$(cat "$1")" = "$2" ] || { printf 'WRITE_FAILED=%s\n' "$1" >> "$log"; return 1; }
}

video_state() {   # one line: device key=value ...
    local dev key
    for dev in $video_devices; do
        printf '%s' "$(basename "$dev")"
        for key in $video_settings cur_freq; do printf ' %s=%s' "$key" "$(cat "$dev/$key")"; done
        printf '\n'
    done
}

save_video() {
    video_state > "$out/video_clocks_before.txt" && video_saved=1
}

lock_video() {
    local dev maximum
    for dev in $video_devices; do
        maximum=$(tr ' ' '\n' < "$dev/available_frequencies" | sort -n | tail -1)
        write_sysfs "$dev/device/power/control" on || return 1
        write_sysfs "$dev/max_freq" "$maximum" || return 1
        write_sysfs "$dev/min_freq" "$maximum" || return 1
        write_sysfs "$dev/governor" performance || return 1
        echo "$(basename "$dev") max=$maximum" >> "$out/video_clocks_target.txt"
    done
    sleep 1
    video_state > "$out/video_clocks_locked.txt"
    awk '{ for (i = 2; i <= NF; i++) { split($i, kv, "="); v[kv[1]] = kv[2] }
           if (v["governor"] != "performance" || v["device/power/control"] != "on" ||
               v["min_freq"] != v["max_freq"] || v["cur_freq"] != v["max_freq"]) { print "unlocked: " $0; bad = 1 } }
         END { exit bad }' "$out/video_clocks_locked.txt"
}

restore_video() {
    [ "$video_saved" = 1 ] || return 0
    local status=0 dev name line key value
    for dev in $video_devices; do
        name=$(basename "$dev")
        line=$(grep "^$name " "$out/video_clocks_before.txt")
        # min before max: the saved minimum is below any maximum
        for key in min_freq max_freq governor device/power/control; do
            value=$(printf '%s\n' "$line" | tr ' ' '\n' | awk -F= -v k="$key" '$1 == k { print $2 }')
            [ "$(cat "$dev/$key")" = "$value" ] || write_sysfs "$dev/$key" "$value" || status=1
        done
    done
    video_state > "$out/video_clocks_after.txt"
    # settings (not cur_freq) must match the saved state
    if ! diff <(sed 's/ cur_freq=[0-9]*//' "$out/video_clocks_before.txt") \
              <(sed 's/ cur_freq=[0-9]*//' "$out/video_clocks_after.txt") >> "$log" 2>&1; then
        status=1
    fi
    return "$status"
}

start_video_sampler() {
    ( while true; do printf '%s ' "$(date +%s.%N)"; video_state | tr '\n' ' '; printf '\n'; sleep 1; done ) \
        > "$out/video_clocks.log" 2>&1 &
    video_sampler_pid=$!
}

verify_default() {
    local file=$1
    grep -q 'EMC .*FreqOverride=0' "$file" || { echo "EMC override still set"; return 1; }
    grep -q 'Dynamic Speed Control=nvfancontrol' "$file" || { echo "fan not dynamic"; return 1; }
    as_root /usr/sbin/nvpmodel -q 2>&1 | tail -1 | grep -qx "$default_mode" \
        || { echo "nvpmodel not mode $default_mode"; return 1; }
}

restore() {
    local status=0
    stop_telemetry
    : > "$out/platform_restore.log"
    log="$out/platform_restore.log"
    logged /usr/bin/jetson_clocks --restore "$out/before.conf" || status=1
    logged /usr/sbin/nvpmodel -m "$default_mode" || status=1
    logged systemctl start nvfancontrol || status=1
    if restore_video; then
        [ "$video_saved" = 1 ] && echo "VIDEO_RESTORE_VERIFIED=1" | tee -a "$out/platform_restore.log"
    else
        echo "VIDEO_RESTORE_VERIFIED=0" | tee -a "$out/platform_restore.log"
        status=1
    fi
    sleep 2
    show "$out/platform_after.txt"
    as_root /usr/sbin/nvpmodel -q > "$out/nvpmodel_after.txt" 2>&1
    if verify_default "$out/platform_after.txt" >> "$out/platform_restore.log" 2>&1; then
        echo "RESTORE_VERIFIED=1" | tee -a "$out/platform_restore.log"
    else
        echo "RESTORE_VERIFIED=0" | tee -a "$out/platform_restore.log"
        status=1
    fi
    return "$status"
}

# every platform query below needs root; fail clearly instead of misreading
# a sudo error as an abnormal platform state
if ! as_root true 2>/dev/null; then
    echo "sudo authentication failed; platform untouched" >&2
    exit 9
fi

# --- before: never record an already-locked state as the default ----------
# If an earlier run left the board partly locked, pass DEFAULT_CONF: a
# `jetson_clocks --store` file captured in a verified default state.  It then
# becomes the restore reference instead of the current (abnormal) state.
show "$out/platform_before.txt"
as_root /usr/sbin/nvpmodel -q > "$out/nvpmodel_before.txt" 2>&1
if verify_default "$out/platform_before.txt" > "$out/platform_before_check.txt" 2>&1; then
    if [ -n "${DEFAULT_CONF:-}" ]; then
        cp "$DEFAULT_CONF" "$out/before.conf" || exit 4
        echo "DEFAULT_CONF_SOURCE=$DEFAULT_CONF" >> "$log"
    else
        logged /usr/bin/jetson_clocks --store "$out/before.conf" || exit 4
    fi
elif [ -n "${DEFAULT_CONF:-}" ]; then
    echo "PRE_RUN_STATE_ABNORMAL=1" | tee -a "$log"
    cat "$out/platform_before_check.txt" | tee -a "$log"
    cp "$DEFAULT_CONF" "$out/before.conf" || exit 4
    echo "DEFAULT_CONF_SOURCE=$DEFAULT_CONF" | tee -a "$log"
else
    echo "platform is not in its default state; refusing to store it as default" >&2
    echo "(pass DEFAULT_CONF=<verified default jetson_clocks --store file>):" >&2
    cat "$out/platform_before_check.txt" >&2
    exit 3
fi

# restore runs on every exit path; a failed restore turns success into exit 7
trap 'exit 130' INT TERM
trap 'rc=$?; trap - EXIT; if ! restore && [ "$rc" -eq 0 ]; then rc=7; fi; exit "$rc"' EXIT

save_video || { echo "cannot read video clocks" >&2; exit 4; }
logged /usr/sbin/nvpmodel -m "$maxn_mode" || exit 5
logged /usr/bin/jetson_clocks --fan || exit 5
sleep 2
show "$out/platform_locked.txt"
if ! verify_locked "$out/platform_locked.txt" > "$out/platform_locked_check.txt" 2>&1; then
    echo "lock verification failed:" >&2; cat "$out/platform_locked_check.txt" >&2
    exit 6
fi
echo "LOCK_VERIFIED=1"
if [ "$video_clocks" = locked ]; then
    if ! lock_video > "$out/video_clocks_lock_check.txt" 2>&1; then
        echo "video clock lock failed:" >&2; cat "$out/video_clocks_lock_check.txt" >&2
        exit 6
    fi
fi
echo "VIDEO_CLOCKS=$video_clocks ($(video_state | awk '{ printf "%s %s %s; ", $1, $4, $6 }'))"

# --- during: continuous frequency/temperature record -----------------------
# root tegrastats is the only one that reports EMC and GPU frequencies
as_root tegrastats --stop >/dev/null 2>&1
rm -f "$out/tegrastats.log"
if as_root tegrastats --interval 200 --logfile "$out/tegrastats.log" --start; then
    telemetry_started=1
else
    echo "cannot start root tegrastats; no continuous frequency record" >&2
    exit 8
fi
start_video_sampler

"$@" > "$out/command.log" 2>&1
command_rc=$?
echo "COMMAND_EXIT_CODE=$command_rc" | tee "$out/command_exit_code.txt"
stop_telemetry

# --- after the command, before restore: was the lock held throughout? ------
cpu_mhz=$(awk '/^cpu0:/ { for (i = 1; i <= NF; i++) if ($i ~ /^CurrentFreq=/) { split($i, kv, "="); print kv[2] / 1000 } }' "$out/platform_locked.txt")
gpu_mhz=$(awk '/^GPU MinFreq=/ { for (i = 1; i <= NF; i++) if ($i ~ /^CurrentFreq=/) { split($i, kv, "="); print kv[2] / 1000000 } }' "$out/platform_locked.txt")
emc_mhz=$((expect_emc_hz / 1000000))
if verify_continuous "$out/tegrastats.log" "$cpu_mhz" "$gpu_mhz" "$emc_mhz" > "$out/platform_continuous_check.txt" 2>&1; then
    echo "LOCK_CONTINUOUS=1 ($(cat "$out/platform_continuous_check.txt"))"
else
    echo "LOCK_CONTINUOUS=0 (platform condition abnormal: $(cat "$out/platform_continuous_check.txt"))"
fi
# video clocks over the run: NVDEC cur_freq spread, and when locked, every sample at target
awk '{ for (i = 2; i <= NF; i++) if ($i ~ /^15480000.nvdec$/) { split($(i + 5), kv, "="); print kv[2] / 1e6 } }' \
    "$out/video_clocks.log" | sort -n | awk '{ v[NR] = $1 } END {
        if (NR) printf "nvdec_mhz_samples=%d min=%g p50=%g max=%g\n", NR, v[1], v[int((NR + 1) / 2)], v[NR] }' \
    > "$out/video_clocks_summary.txt"
if [ "$video_clocks" = locked ]; then
    if awk '{ for (i = 2; i <= NF; i++) { split($i, kv, "="); if (kv[1] == "governor" && kv[2] != "performance") bad++
              if (kv[1] == "min_freq") lo = kv[2]; if (kv[1] == "max_freq") hi = kv[2]
              if (kv[1] == "cur_freq" && (kv[2] != hi || lo != hi)) bad++ } }
            END { exit (NR == 0 || bad) }' "$out/video_clocks.log"; then
        echo "VIDEO_LOCK_CONTINUOUS=1 ($(cat "$out/video_clocks_summary.txt"))"
    else
        echo "VIDEO_LOCK_CONTINUOUS=0 ($(cat "$out/video_clocks_summary.txt"))"
    fi
else
    echo "VIDEO_CLOCKS_OBSERVED ($(cat "$out/video_clocks_summary.txt"))"
fi
show "$out/platform_end.txt"
if verify_locked "$out/platform_end.txt" > "$out/platform_end_check.txt" 2>&1; then
    echo "LOCK_HELD=1"
else
    echo "LOCK_HELD=0 (platform condition abnormal)"; cat "$out/platform_end_check.txt"
fi
exit "$command_rc"
