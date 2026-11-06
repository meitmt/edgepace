#!/system/bin/sh
# Phone counterpart of scripts/locked_run.sh (runs ON the device, as root):
# fixed-performance mode, every CPU policy and the Adreno GPU at their maximum
# frequency, DDR and LLCC bus DCVS boosted to their hardware maximum; run the
# command; always restore the saved state and verify it.  The lock settings are
# those of the paper's phone runs.
#
#   locked_run_android.sh OUT_DIR -- COMMAND [ARGS...]
#
# Evidence in OUT_DIR: platform_before.txt, platform_locked.txt,
# platform_restored.txt, platform_samples.txt (every second while the command
# runs); summary lines LOCK_VERIFIED, LOCK_CONTINUOUS, RESTORE_VERIFIED,
# COMMAND_EXIT_CODE and the thermal status before/after.
set -u
out=${1:?usage: locked_run_android.sh OUT_DIR -- COMMAND...}
shift
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "missing command" >&2; exit 2; }
[ "$(id -u)" = 0 ] || { echo "needs root" >&2; exit 2; }
mkdir -p "$out"
GPU=/sys/class/kgsl/kgsl-3d0/devfreq
BUS=/sys/devices/system/cpu/bus_dcvs

snapshot() {
    for p in /sys/devices/system/cpu/cpufreq/policy*; do
        echo "cpu $(basename $p) governor=$(cat $p/scaling_governor) min=$(cat $p/scaling_min_freq) max=$(cat $p/scaling_max_freq) cur=$(cat $p/scaling_cur_freq)"
    done
    echo "gpu governor=$(cat $GPU/governor) min=$(cat $GPU/min_freq) max=$(cat $GPU/max_freq) cur=$(cat $GPU/cur_freq)"
    for b in DDR LLCC; do echo "bus $b boost=$(cat $BUS/$b/boost_freq) cur=$(cat $BUS/$b/cur_freq) hw_max=$(cat $BUS/$b/hw_max_freq)"; done
}
thermal() { dumpsys thermalservice 2>/dev/null | awk -F: '/Thermal Status:/ { gsub(/[[:space:]]/, "", $2); print $2; exit }'; }

# save
state="$out/state"
: > "$state.cpu"
for p in /sys/devices/system/cpu/cpufreq/policy*; do
    echo "$p|$(cat $p/scaling_governor)|$(cat $p/scaling_min_freq)|$(cat $p/scaling_max_freq)" >> "$state.cpu"
done
echo "$(cat $GPU/governor)|$(cat $GPU/min_freq)|$(cat $GPU/max_freq)" > "$state.gpu"
echo "$(cat $BUS/DDR/boost_freq)|$(cat $BUS/LLCC/boost_freq)" > "$state.bus"
snapshot > "$out/platform_before.txt"
echo "THERMAL_BEFORE=$(thermal)"

restore() {
    while IFS='|' read -r p governor min max; do
        echo "$max" > "$p/scaling_max_freq" 2>/dev/null
        echo "$min" > "$p/scaling_min_freq" 2>/dev/null
        echo "$governor" > "$p/scaling_governor" 2>/dev/null
    done < "$state.cpu"
    IFS='|' read -r g gmin gmax < "$state.gpu"
    echo "$gmax" > $GPU/max_freq 2>/dev/null; echo "$gmin" > $GPU/min_freq 2>/dev/null; echo "$g" > $GPU/governor 2>/dev/null
    IFS='|' read -r ddr llcc < "$state.bus"
    echo "$ddr" > $BUS/DDR/boost_freq 2>/dev/null; echo "$llcc" > $BUS/LLCC/boost_freq 2>/dev/null
    cmd power set-fixed-performance-mode-enabled false >/dev/null 2>&1
    snapshot > "$out/platform_restored.txt"
    ok=1
    while IFS='|' read -r p governor min max; do
        [ "$(cat $p/scaling_governor)" = "$governor" ] && [ "$(cat $p/scaling_min_freq)" = "$min" ] && \
            [ "$(cat $p/scaling_max_freq)" = "$max" ] || ok=0
    done < "$state.cpu"
    [ "$(cat $GPU/governor)|$(cat $GPU/min_freq)|$(cat $GPU/max_freq)" = "$g|$gmin|$gmax" ] || ok=0
    [ "$(cat $BUS/DDR/boost_freq)|$(cat $BUS/LLCC/boost_freq)" = "$ddr|$llcc" ] || ok=0
    echo "RESTORE_VERIFIED=$ok"
    echo "THERMAL_AFTER=$(thermal)"
}
trap restore EXIT
trap 'exit 130' INT TERM

# lock
cmd power set-fixed-performance-mode-enabled true >/dev/null 2>&1
# Fixed-performance mode caps the CPU policies (on the OnePlus 13 at 1.9968 /
# 2.6496 GHz instead of 3.5328 / 4.32 GHz, as in the 2026-09 runs): request the
# maximum, then lock to what the policy accepted, read back.
expect=""
for p in /sys/devices/system/cpu/cpufreq/policy*; do
    m=$(tr ' ' '\n' < "$p/scaling_available_frequencies" | sort -n | tail -n 1)
    echo "$m" > "$p/scaling_max_freq"; echo "$m" > "$p/scaling_min_freq"; echo performance > "$p/scaling_governor"
    expect="$expect $p=$(cat $p/scaling_max_freq)"
done
gmax=$(tr ' ' '\n' < $GPU/available_frequencies | sort -n | tail -n 1)
echo "$gmax" > $GPU/max_freq; echo "$gmax" > $GPU/min_freq
cat $BUS/DDR/hw_max_freq > $BUS/DDR/boost_freq
cat $BUS/LLCC/hw_max_freq > $BUS/LLCC/boost_freq
snapshot > "$out/platform_locked.txt"
locked() {   # 1 when every CPU policy and the GPU run at the locked maximum
    for kv in $expect; do p=${kv%=*}; m=${kv#*=}; [ "$(cat $p/scaling_cur_freq)" = "$m" ] || return 1; done
    [ "$(cat $GPU/cur_freq)" = "$gmax" ]
}
echo "LOCKED_CLOCKS cpu=$(for kv in $expect; do printf '%s ' "${kv##*/}"; done)gpu=$gmax ddr=$(cat $BUS/DDR/boost_freq) llcc=$(cat $BUS/LLCC/boost_freq)"
if locked; then echo LOCK_VERIFIED=1; else echo LOCK_VERIFIED=0; fi

# run, sampling the platform every second; the DDR / LLCC bus clocks go to
# platform_bus.txt (they can drop under load while the CPU and GPU
# clocks hold): time, DDR boost and current, LLCC boost and current, and the
# shell (skin) sensors in millidegrees
shells=$(for z in /sys/class/thermal/thermal_zone*; do case $(cat $z/type) in shell_*) echo $z/temp;; esac; done)
: > "$out/platform_samples.txt"; : > "$out/platform_bus.txt"
( while :; do
      now=$(date +%s)
      echo "$now $(locked && echo locked || echo off) $(cat $GPU/cur_freq) $(for kv in $expect; do cat ${kv%=*}/scaling_cur_freq; done | tr '\n' ' ')" >> "$out/platform_samples.txt"
      echo "$now $(cat $BUS/DDR/boost_freq) $(cat $BUS/DDR/cur_freq) $(cat $BUS/LLCC/boost_freq) $(cat $BUS/LLCC/cur_freq) $(cat $shells | tr '\n' ' ')" >> "$out/platform_bus.txt"
      sleep 1
  done ) 2>/dev/null &
sampler=$!
"$@"
code=$?
kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
total=$(wc -l < "$out/platform_samples.txt"); off=$(grep -c ' off ' "$out/platform_samples.txt")
echo "COMMAND_EXIT_CODE=$code"
echo "LOCK_CONTINUOUS=$([ "$off" = 0 ] && echo 1 || echo 0) (samples=$total off_lock=$off)"
exit $code
