// Loads geniex's prebuilt Hexagon backend (libggml-hexagon.so, linked as a
// regular library, not a GGML_BACKEND_DL module) through the dynamic-backend
// entry points, so the EdgePace libggml-base can discover it next to the binary.
#include <dlfcn.h>

#include <cstdlib>

using backend_reg_fn = void * (*)();

extern "C" int ggml_backend_score() {
    return 1;
}

extern "C" void * ggml_backend_init() {
    const char * implementation = std::getenv("GGML_HEXAGON_IMPL");
    if (implementation == nullptr || *implementation == '\0') {
        implementation = "libggml-hexagon.so";
    }

    static void * handle = dlopen(implementation, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        return nullptr;
    }

    auto registry = reinterpret_cast<backend_reg_fn>(
        dlsym(handle, "ggml_backend_hexagon_reg"));
    return registry == nullptr ? nullptr : registry();
}
