// Thin launcher for unmodified ThunderKittens B300 NVFP4 (BF16 output).
// Requires K padded to a multiple of 768 and standard 128x4 swizzled E4M3 scales.
// See docs/benchmarks.md for build flags and the pinned upstream revision.
#define main thunderkittens_benchmark_main_unused
#include "kernels/gemm/nvfp4_b300/nvfp4_b300_gemm.cu"
#undef main

using C = nvfp4_gemm::config<fp8e4m3, true>;
using G = nvfp4_gemm::globals<C>;
extern "C" void* tk_create(void* a, void* b, void* sa, void* sb, void* d, void* one, int m, int n, int k) {
    if (m % 256 || n % 256 || k % 768) return nullptr;
    G g{
        G::A_fp4x2_gl{static_cast<fp4e2m1_2*>(a), nullptr, nullptr, m, k/2},
        G::A_sc_gl{static_cast<fp8e4m3*>(sa), m/128, k/768, nullptr, nullptr},
        G::A_sc_global_gl{static_cast<float*>(one), nullptr, nullptr, nullptr, nullptr},
        G::B_fp4x2_gl{static_cast<fp4e2m1_2*>(b), nullptr, nullptr, n, k/2},
        G::B_sc_gl{static_cast<fp8e4m3*>(sb), n/128, k/768, nullptr, nullptr},
        G::B_sc_global_gl{static_cast<float*>(one), nullptr, nullptr, nullptr, nullptr},
        G::D_gl{static_cast<bf16*>(d), nullptr, nullptr, m, n}
    };
    if (cudaFuncSetAttribute(kernel_entrypoint<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory()) != cudaSuccess) return nullptr;
    return new G(g);
}
extern "C" int tk_run(void* context, void* stream) {
    auto& g = *static_cast<G*>(context);
    LaunchConfig<true, true> config(g.grid(), g.block(), g.dynamic_shared_memory(), static_cast<cudaStream_t>(stream), C::CLUSTER_SIZE);
    return cudaLaunchKernelEx(config, kernel_entrypoint<C>, g);
}
extern "C" void tk_destroy(void* p) { delete static_cast<G*>(p); }
