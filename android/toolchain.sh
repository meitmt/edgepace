# Android cross toolchain for android/setup_base.sh and android/build_android.sh
# (sourced; ROOT must be set).
#
#   ANDROID_NDK_HOME   Android NDK (r27c tested); required
#   ANDROID_HOST_CLANG 1: compile with the host's clang (an aarch64 host, e.g. a Jetson:
#                      the NDK ships x86_64 host binaries only), else the NDK's clang
#                      (default on x86_64 hosts)
#
# Defines ANDROID_CXX (array: compile + link C++ for API 28), ANDROID_CMAKE_ARGS (array)
# and ANDROID_SYSROOT.
NDK=${ANDROID_NDK_HOME:?set ANDROID_NDK_HOME to an Android NDK (r27c tested)}
[ -d "$NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot" ] || { echo "not an NDK: $NDK" >&2; exit 2; }
ANDROID_SYSROOT=$NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot
ANDROID_TOOLCHAIN_DIR=${ANDROID_TOOLCHAIN_DIR:-$ROOT/build/android-toolchain}
if [ -z "${ANDROID_HOST_CLANG:-}" ]; then
    [ "$(uname -m)" = x86_64 ] && ANDROID_HOST_CLANG=0 || ANDROID_HOST_CLANG=1
fi
ARCH_FLAGS=(-march=armv8.7a)

if [ "$ANDROID_HOST_CLANG" = 0 ]; then
    NDK_BIN=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin
    ANDROID_CXX=("$NDK_BIN/clang++" --target=aarch64-linux-android28 "${ARCH_FLAGS[@]}" -static-libstdc++)
    ANDROID_CLANGXX=("$NDK_BIN/clang++")          # foreground: own target / stdlib flags
    ANDROID_CMAKE_ARGS=(-DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake"
                        -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_STL=c++_static
                        -DCMAKE_C_FLAGS="${ARCH_FLAGS[*]}" -DCMAKE_CXX_FLAGS="${ARCH_FLAGS[*]}")
    ANDROID_UNWIND=()
else
    # Host clang (14 tested) with the NDK's sysroot, libc++ headers and compiler-rt.
    # Its resource directory = the host clang's headers + the NDK's Android builtins;
    # LLVM 14 cannot read the NDK's zstd-compressed debug sections, so the system
    # libraries it links against are stripped copies.
    CLANG=${ANDROID_CLANG:-$(command -v clang)}
    CLANGXX=${ANDROID_CLANGXX:-$(command -v clang++)}
    [ -n "$CLANG" ] && [ -n "$CLANGXX" ] || { echo "ANDROID_HOST_CLANG=1 needs clang/clang++" >&2; exit 2; }
    builtins=$(ls "$NDK"/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/*/lib/linux/libclang_rt.builtins-aarch64-android.a | head -1)
    ANDROID_UNWIND=("$(dirname "$builtins")/aarch64/libunwind.a")
    resource=$ANDROID_TOOLCHAIN_DIR/clang-resource
    links=$ANDROID_TOOLCHAIN_DIR/link-libs
    if [ ! -f "$resource/lib/linux/libclang_rt.builtins-aarch64-android.a" ]; then
        mkdir -p "$resource/lib/linux"
        cp -r "$("$CLANG" -print-resource-dir)/include" "$resource/"
        cp "$builtins" "$resource/lib/linux/"
    fi
    for api in 28 35; do
        mkdir -p "$links/$api"
        for lib in libc.so libm.so libdl.so libandroid.so liblog.so libmediandk.so libz.so; do
            src=$ANDROID_SYSROOT/usr/lib/aarch64-linux-android/$api/$lib
            if [ -f "$src" ] && [ ! -f "$links/$api/$lib" ]; then
                cp "$src" "$links/$api/$lib"
                objcopy --strip-debug "$links/$api/$lib" 2>/dev/null || true
            fi
        done
    done
    common=(-resource-dir="$resource" -rtlib=compiler-rt)
    cxxlib=(-isystem "$ANDROID_SYSROOT/usr/include/c++/v1" -D_LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS -stdlib=libc++)
    ANDROID_CXX=("$CLANGXX" --target=aarch64-linux-android28 --sysroot="$ANDROID_SYSROOT" "${ARCH_FLAGS[@]}"
                 "${common[@]}" "${cxxlib[@]}" -static-libstdc++ -L"$links/28")
    ANDROID_CLANGXX=("$CLANGXX" --sysroot="$ANDROID_SYSROOT" "${common[@]}")
    ANDROID_FG_LINKS=$links/35
    ANDROID_CMAKE_ARGS=(-DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64
                        -DCMAKE_C_COMPILER="$CLANG" -DCMAKE_CXX_COMPILER="$CLANGXX"
                        -DCMAKE_C_COMPILER_TARGET=aarch64-linux-android28 -DCMAKE_CXX_COMPILER_TARGET=aarch64-linux-android28
                        -DCMAKE_SYSROOT="$ANDROID_SYSROOT"
                        -DCMAKE_C_FLAGS="${ARCH_FLAGS[*]} ${common[*]}"
                        -DCMAKE_CXX_FLAGS="${ARCH_FLAGS[*]} ${common[*]} ${cxxlib[*]} -static-libstdc++"
                        -DCMAKE_CXX_STANDARD_LIBRARIES="${ANDROID_UNWIND[*]}"
                        -DCMAKE_EXE_LINKER_FLAGS="-L$links/28" -DCMAKE_SHARED_LINKER_FLAGS="-L$links/28"
                        -DCMAKE_MODULE_LINKER_FLAGS="-L$links/28")
fi
