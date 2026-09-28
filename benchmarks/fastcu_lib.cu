// Python-callable wrapper around fast.cu r9 gb300/nvfp4 (https://github.com/pranjalssh/fast.cu).
// Build: nvcc -std=c++17 -O3 -DNDEBUG -gencode arch=compute_103a,code=sm_103a -Xcompiler -fPIC -shared \
//        -I<fast.cu>/gb300/nvfp4 fastcu_lib.cu -o libfastcu.so -lcublasLt -lcuda   (CUDA >= 13.1)
// Python-callable wrapper around fast.cu's r9 NVFP4 GEMM (gb300/nvfp4).
// Reuses main.cu's device setup and per-shape L2-side schedule; main() is renamed away.
#define main fastcu_harness_main_unused
#include "main.cu"
#undef main

namespace {
int g_M = -1, g_N = -1;
struct TmapKey { const void *a, *b, *sfa, *sfb; int M, N, K; };
TmapKey g_key{};
CUtensorMap g_A, g_B, g_SFA, g_SFB;
}

extern "C" int fastcu_init() {
    bench::setup_device();
    return bench::g.sms;
}

// Build and upload the per-shape L2-ownership schedule (host-side, once per shape).
extern "C" void fastcu_prepare(int M, int N, int K) {
    const size_t a = size_t(M) * (K / 2), b = size_t(N) * (K / 2);
    const size_t sfa = size_t((M + 127) / 128 * 128) * ((K + 63) / 64 * 4);
    const size_t sfb = size_t((N + 127) / 128 * 128) * ((K + 63) / 64 * 4);
    sched::ScheduleMode mode = sched::ScheduleMode::AUTO;
    if (const char* e = std::getenv("FASTCU_SCHEDULE")) {
        if (!sched::parse_schedule_mode(e, &mode)) { std::fprintf(stderr, "bad FASTCU_SCHEDULE %s\n", e); std::abort(); }
    }
    bench::setup_schedule(M, N, a + b + sfa + sfb, mode, false, sched::ScheduleMode::AUTO);
    g_M = M; g_N = N;
}

extern "C" int fastcu_gemm(void* A, void* B, void* SFA, void* SFB, void* C,
                           int M, int N, int K, void* stream) {
    if (M != g_M || N != g_N) return 1;  // schedule not prepared for this shape
    if (K % 32 != 0) return 2;
    TmapKey k{A, B, SFA, SFB, M, N, K};
    if (std::memcmp(&k, &g_key, sizeof(k)) != 0) {
        const int stride = K / 2;
        g_A = host::make_ab_tmap(A, M, K, stride);
        g_B = host::make_ab_tmap(B, N, K, stride);
        g_SFA = host::make_sf_tmap(static_cast<uint8_t*>(SFA), M, K);
        g_SFB = host::make_sf_tmap(static_cast<uint8_t*>(SFB), N, K);
        g_key = k;
    }
    nvfp4::nvfp4_gemm_kernel<<<bench::g.grid, nvfp4::TB_SIZE, sizeof(nvfp4::SmemCD),
                               static_cast<cudaStream_t>(stream)>>>(
        g_A, g_B, g_SFA, g_SFB, static_cast<__half*>(C), M, N, K,
        bench::MAIN_TABLE * nvfp4::L2A_ROUTE_WORK_CAP);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

extern "C" unsigned fastcu_placement_errors() {
    unsigned v = 0;
    cudaMemcpyFromSymbol(&v, nvfp4::l2a_placement_errors, sizeof(v));
    return v;
}
