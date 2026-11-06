// Loads geniex's prebuilt OpenCL (Adreno) backend (libggml-opencl.so, linked as a
// regular library, not a GGML_BACKEND_DL module) through the dynamic-backend entry
// points, as ggml-hexagon-dl-shim.cpp does for Hexagon (a VLM's vision encoder
// runs on the GPU, MTMD_BACKEND_DEVICE=GPUOpenCL).
#include <dlfcn.h>

#include <cstdlib>

using backend_reg_fn = void * (*)();

extern "C" int ggml_backend_score() {
    return 1;
}

extern "C" void * ggml_backend_init() {
    const char * implementation = std::getenv("GGML_OPENCL_IMPL");
    if (implementation == nullptr || *implementation == '\0') {
        implementation = "libggml-opencl.so";
    }

    static void * handle = dlopen(implementation, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        return nullptr;
    }

    auto registry = reinterpret_cast<backend_reg_fn>(
        dlsym(handle, "ggml_backend_opencl_reg"));
    return registry == nullptr ? nullptr : registry();
}
