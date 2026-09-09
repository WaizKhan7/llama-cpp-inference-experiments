#include <../common.cuh>
#include <../fattn.cuh>
#include <../fattn-llama32-fa-decode.cuh>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {
constexpr int HQ=32, HKV=8, D=64, GROUP=4, QS=68, KS=80, VS=96, OS=72;
constexpr float SCALE=0.125f, SENTINEL=-777.0f;
void check(cudaError_t s, const char * w) { if (s != cudaSuccess) { std::fprintf(stderr, "CUDA %s: %s\n", w, cudaGetErrorString(s)); std::exit(1); } }
template<typename T> struct device_buffer {
    T * p = nullptr;
    explicit device_buffer(size_t n) { check(cudaMalloc(reinterpret_cast<void **>(&p), n*sizeof(T)), "malloc"); }
    ~device_buffer() { if (p) cudaFree(p); }
    device_buffer(const device_buffer &) = delete;
};
void tensor(ggml_tensor & t, enum ggml_type type, int64_t a, int64_t b, int64_t c, int64_t d, size_t e0, size_t e1, size_t e2, size_t e3) {
    std::memset(&t, 0, sizeof(t)); t.type=type; t.ne[0]=a; t.ne[1]=b; t.ne[2]=c; t.ne[3]=d; t.nb[0]=e0; t.nb[1]=e1; t.nb[2]=e2; t.nb[3]=e3;
}
bool guard_test() {
    ggml_tensor q,k,v,m,d;
    tensor(q,GGML_TYPE_F32,64,1,32,1,4,256,256,8192);
    tensor(k,GGML_TYPE_F16,64,256,8,1,2,128,32768,262144);
    tensor(v,GGML_TYPE_F16,64,256,8,1,2,128,32768,262144);
    tensor(m,GGML_TYPE_F16,256,16,1,1,2,512,8192,8192);
    tensor(d,GGML_TYPE_F32,64,32,1,1,4,256,8192,8192);
    d.op=GGML_OP_FLASH_ATTN_EXT; d.src[0]=&q; d.src[1]=&k; d.src[2]=&v; d.src[3]=&m; std::memcpy(d.op_params,&SCALE,sizeof(SCALE));
    bool ok=ggml_cuda_llama32_fa_decode_supported(&d);
    q.ne[1]=2; ok=ok && !ggml_cuda_llama32_fa_decode_supported(&d);
    q.ne[1]=1; q.nb[0]=2; ok=ok && !ggml_cuda_llama32_fa_decode_supported(&d);
    q.nb[0]=4; d.ne[1]=31; ok=ok && !ggml_cuda_llama32_fa_decode_supported(&d);
    d.ne[1]=32;
    q.nb[0]=4; v.type=GGML_TYPE_F32; ok=ok && !ggml_cuda_llama32_fa_decode_supported(&d);
    std::printf("guard: %s\n",ok?"PASS":"FAIL"); return ok;
}
bool case_test(int visible) {
    const int padded=((visible+255)/256)*256;
    std::vector<float> q(HQ*QS), out(HQ*OS,SENTINEL), ref(HQ*D);
    const half masked_value = __float2half(-std::numeric_limits<float>::infinity());
    std::vector<half> k(HKV*padded*KS),v(HKV*padded*VS),m(16*padded,masked_value);
    for(int h=0;h<HQ;++h) for(int x=0;x<D;++x) q[h*QS+x]=.03f*float((h+3)*(x+5)%29-14);
    for(int h=0;h<HKV;++h) for(int p=0;p<visible;++p) {
        for(int x=0;x<D;++x) {
            k[(h*padded+p)*KS+x]=__float2half(.02f*float((h+2)*(p+1)*(x+3)%31-15));
            v[(h*padded+p)*VS+x]=__float2half(.025f*float((h+5)*(p+7)*(x+1)%37-18));
        }
        m[p]=__float2half(p%23==7 ? -.125f : 0.f);
    }
    for(int h=0;h<HQ;++h) {
        std::vector<float> score(visible); float mx=-std::numeric_limits<float>::infinity();
        for(int p=0;p<visible;++p) { float dot=0; for(int x=0;x<D;++x) dot+=q[h*QS+x]*__half2float(k[((h/GROUP)*padded+p)*KS+x]); score[p]=dot*SCALE+__half2float(m[p]); mx=std::fmax(mx,score[p]); }
        float den=0; for(int p=0;p<visible;++p) den+=std::exp(score[p]-mx);
        for(int x=0;x<D;++x) { float num=0; for(int p=0;p<visible;++p) num+=std::exp(score[p]-mx)*__half2float(v[((h/GROUP)*padded+p)*VS+x]); ref[h*D+x]=num/den; }
    }
    device_buffer<float> dq(q.size()),do_(out.size()); device_buffer<half> dk(k.size()),dv(v.size()),dm(m.size());
    check(cudaMemcpy(dq.p,q.data(),q.size()*sizeof(float),cudaMemcpyHostToDevice),"copy Q");
    check(cudaMemcpy(dk.p,k.data(),k.size()*sizeof(half),cudaMemcpyHostToDevice),"copy K");
    check(cudaMemcpy(dv.p,v.data(),v.size()*sizeof(half),cudaMemcpyHostToDevice),"copy V");
    check(cudaMemcpy(dm.p,m.data(),m.size()*sizeof(half),cudaMemcpyHostToDevice),"copy mask");
    check(cudaMemcpy(do_.p,out.data(),out.size()*sizeof(float),cudaMemcpyHostToDevice),"copy output");
    ggml_tensor tq, tk, tv, tm, td;
    tensor(tq, GGML_TYPE_F32, 64, 1, 32, 1, sizeof(float), D*sizeof(float), QS*sizeof(float), HQ*QS*sizeof(float));
    tensor(tk, GGML_TYPE_F16, 64, padded, 8, 1, sizeof(half), KS*sizeof(half), padded*KS*sizeof(half), 8*padded*KS*sizeof(half));
    tensor(tv, GGML_TYPE_F16, 64, padded, 8, 1, sizeof(half), VS*sizeof(half), padded*VS*sizeof(half), 8*padded*VS*sizeof(half));
    tensor(tm, GGML_TYPE_F16, padded, 16, 1, 1, sizeof(half), padded*sizeof(half), 16*padded*sizeof(half), 16*padded*sizeof(half));
    tensor(td, GGML_TYPE_F32, 64, 32, 1, 1, sizeof(float), OS*sizeof(float), HQ*OS*sizeof(float), HQ*OS*sizeof(float));
    tq.data=dq.p; tk.data=dk.p; tv.data=dv.p; tm.data=dm.p; td.data=do_.p;
    td.op=GGML_OP_FLASH_ATTN_EXT;
    td.src[0]=&tq; td.src[1]=&tk; td.src[2]=&tv; td.src[3]=&tm;
    std::memcpy(td.op_params, &SCALE, sizeof(SCALE));

    ggml_cuda_llama32_fa_decode_test_dispatch_count=0;
    ggml_cuda_llama32_fa_decode_test_route=0;
    ggml_backend_cuda_context context(0);
    ggml_cuda_flash_attn_ext(context, &td);
    check(cudaGetLastError(),"dispatcher launch");
    check(cudaDeviceSynchronize(),"dispatcher synchronize");
    check(cudaMemcpy(out.data(),do_.p,out.size()*sizeof(float),cudaMemcpyDeviceToHost),"copy output back");
    float maxe=0; bool ok=true;
    ok=ok && ggml_cuda_llama32_fa_decode_test_dispatch_count==1;
    ok=ok && ggml_cuda_llama32_fa_decode_test_route==2;
    for(int h=0;h<HQ;++h) { for(int x=0;x<D;++x) { float a=out[h*OS+x],e=std::fabs(a-ref[h*D+x]); maxe=std::fmax(maxe,e); ok=ok&&std::isfinite(a)&&e<=1e-4f+1e-4f*std::fabs(ref[h*D+x]); } for(int x=D;x<OS;++x) ok=ok&&out[h*OS+x]==SENTINEL; }
    std::printf("route_count=%d route_kind=%d\n", ggml_cuda_llama32_fa_decode_test_dispatch_count, ggml_cuda_llama32_fa_decode_test_route);
    std::printf("visible=%d padded=%d max_abs_error=%.8f: %s\n",visible,padded,maxe,ok?"PASS":"FAIL"); return ok;
}
}
int main() {
#if defined(_WIN32)
    if (_putenv_s("GGML_CUDA_LLAMA32_FA_DECODE_ENABLED", "1") != 0 ||
        _putenv_s("GGML_CUDA_LLAMA32_FD_SPLITK_ENABLED", "1") != 0) {
        std::perror("_putenv_s");
        return 1;
    }
#else
    if (setenv("GGML_CUDA_LLAMA32_FA_DECODE_ENABLED", "1", 1) != 0 ||
        setenv("GGML_CUDA_LLAMA32_FD_SPLITK_ENABLED", "1", 1) != 0) {
        std::perror("setenv");
        return 1;
    }
#endif
    cudaDeviceProp p; check(cudaGetDeviceProperties(&p,0),"get properties"); std::printf("GGML-boundary Split-K FD validation on %s\n",p.name);
    bool ok=guard_test(); for(int visible : {1,127,128,129,255,256,257,511}) ok=case_test(visible)&&ok;
    std::printf("GGML-boundary correctness: %s\n",ok?"PASS":"FAIL"); return ok?0:1;
}

