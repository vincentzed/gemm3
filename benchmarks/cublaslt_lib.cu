// NVFP4 cuBLASLt adapter, matching fast.cu's first-heuristic/64 MiB policy.
// nvcc -std=c++17 -O3 -shared -Xcompiler -fPIC cublaslt_lib.cu -o libcublaslt_bench.so -lcublasLt
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

#define LT(call) do { if ((call) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(#call); } while (0)
struct Plan {
    cublasLtHandle_t handle{};
    cublasLtMatmulDesc_t op{};
    cublasLtMatrixLayout_t al{}, bl{}, cl{};
    cublasLtMatmulPreference_t pref{};
    cublasLtMatmulAlgo_t algo{};
    void *a, *b, *sf_a, *sf_b, *out, *workspace{};
    size_t workspace_bytes{};
    Plan(void* aa, void* bb, void* sa, void* sb, void* cc, int m, int n, int k)
        : a(aa), b(bb), sf_a(sa), sf_b(sb), out(cc) {
        LT(cublasLtCreate(&handle));
        LT(cublasLtMatrixLayoutCreate(&al, CUDA_R_4F_E2M1, k, m, k));
        LT(cublasLtMatrixLayoutCreate(&bl, CUDA_R_4F_E2M1, k, n, k));
        LT(cublasLtMatrixLayoutCreate(&cl, CUDA_R_16F, m, n, n));
        cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
        LT(cublasLtMatrixLayoutSetAttribute(cl, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
        LT(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
        int32_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        int8_t fast_accum = 0;
        LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
        LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
        LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum, sizeof(fast_accum)));
        LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
        LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
        LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sf_a, sizeof(sf_a)));
        LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sf_b, sizeof(sf_b)));
        LT(cublasLtMatmulPreferenceCreate(&pref));
        size_t limit = 64ULL << 20;
        LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit, sizeof(limit)));
        cublasLtMatmulHeuristicResult_t result{};
        int count = 0;
        LT(cublasLtMatmulAlgoGetHeuristic(handle, op, al, bl, cl, cl, pref, 1, &result, &count));
        if (count != 1 || result.state != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("No cuBLASLt heuristic");
        algo = result.algo;
        workspace_bytes = result.workspaceSize;
        if (workspace_bytes && cudaMalloc(&workspace, workspace_bytes) != cudaSuccess) throw std::runtime_error("Workspace allocation failed");
    }
    ~Plan() {
        if (workspace) cudaFree(workspace);
        if (pref) cublasLtMatmulPreferenceDestroy(pref);
        if (op) cublasLtMatmulDescDestroy(op);
        if (al) cublasLtMatrixLayoutDestroy(al);
        if (bl) cublasLtMatrixLayoutDestroy(bl);
        if (cl) cublasLtMatrixLayoutDestroy(cl);
        if (handle) cublasLtDestroy(handle);
    }
};
extern "C" void* cublas_create(void* a, void* b, void* sa, void* sb, void* c, int m, int n, int k) {
    try { return new Plan(a, b, sa, sb, c, m, n, k); }
    catch (const std::exception& e) { std::fprintf(stderr, "cuBLASLt: %s\n", e.what()); return nullptr; }
}
extern "C" int cublas_run(void* context, void* stream) {
    auto& p = *static_cast<Plan*>(context);
    float alpha = 1.f, beta = 0.f;
    return cublasLtMatmul(p.handle, p.op, &alpha, p.a, p.al, p.b, p.bl, &beta, p.out, p.cl,
                          p.out, p.cl, &p.algo, p.workspace, p.workspace_bytes, static_cast<cudaStream_t>(stream));
}
extern "C" void cublas_destroy(void* p) { delete static_cast<Plan*>(p); }
extern "C" size_t cublas_workspace(void* p) { return static_cast<Plan*>(p)->workspace_bytes; }
extern "C" size_t cublas_version() { return cublasLtGetVersion(); }
