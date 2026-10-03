// Board-only synchronous C API inference. No model conversion or training.
#include "rknn_api.h"
#include <arm_neon.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string error_message;
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b-a).count();
}
void check(int ret, const char* op) {
  if (ret) throw std::runtime_error(std::string(op)+" returned "+std::to_string(ret));
}
std::string quote(const char* s) {
  std::string value="\"";
  for (; *s; ++s) { if (*s=='"'||*s=='\\') value+='\\'; if (*s>=32) value+=*s; }
  return value+'"';
}
std::string attr_json(const rknn_tensor_attr& a) {
  std::ostringstream s;
  s << "{\"index\":" << a.index << ",\"name\":" << quote(a.name)
    << ",\"dims\":[";
  for (unsigned j=0;j<a.n_dims;++j) { if(j) s << ','; s << a.dims[j]; }
  s << "],\"n_elems\":" << a.n_elems << ",\"size\":" << a.size << ",\"size_with_stride\":" << a.size_with_stride << ",\"w_stride\":" << a.w_stride << ",\"fmt\":" << a.fmt
    << ",\"type\":" << a.type << ",\"qnt_type\":" << a.qnt_type
    << ",\"zp\":" << a.zp << ",\"scale\":" << a.scale << '}';
  return s.str();
}
struct Session {
  rknn_context ctx=0;
  std::vector<rknn_tensor_attr> inputs, outputs;
  bool perf;
  std::string metadata;
  Session(const char* path, int core_mask, bool query_perf, bool profiling): perf(query_perf) {
    if (core_mask<=0 || core_mask>7) throw std::runtime_error("explicit NPU core mask must be 1..7");
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if (!f) throw std::runtime_error(std::string("cannot read model: ")+path);
    auto length=f.tellg();
    if (length<=0 || static_cast<uint64_t>(length)>UINT32_MAX) throw std::runtime_error("invalid model length");
    std::vector<char> model(static_cast<size_t>(length)); f.seekg(0); f.read(model.data(),length);
    if (!f) throw std::runtime_error("short model read");
    check(rknn_init(&ctx,model.data(),model.size(),profiling?RKNN_FLAG_COLLECT_PERF_MASK:0,nullptr),"rknn_init");
    try {
      check(rknn_set_core_mask(ctx,static_cast<rknn_core_mask>(core_mask)),"rknn_set_core_mask");
      rknn_input_output_num io{}; check(rknn_query(ctx,RKNN_QUERY_IN_OUT_NUM,&io,sizeof(io)),"query IN_OUT_NUM");
      inputs.resize(io.n_input); outputs.resize(io.n_output);
      for(unsigned i=0;i<io.n_input;++i) { inputs[i].index=i; check(rknn_query(ctx,RKNN_QUERY_INPUT_ATTR,&inputs[i],sizeof(inputs[i])),"query INPUT_ATTR"); }
      for(unsigned i=0;i<io.n_output;++i) { outputs[i].index=i; check(rknn_query(ctx,RKNN_QUERY_OUTPUT_ATTR,&outputs[i],sizeof(outputs[i])),"query OUTPUT_ATTR"); }
      rknn_sdk_version v{}; check(rknn_query(ctx,RKNN_QUERY_SDK_VERSION,&v,sizeof(v)),"query SDK_VERSION");
      std::ostringstream s; s << "{\"core_mask\":" << core_mask << ",\"profiling\":" << (profiling?"true":"false")
        << ",\"api_version\":" << quote(v.api_version) << ",\"driver_version\":" << quote(v.drv_version) << ",\"inputs\":[";
      for(unsigned i=0;i<inputs.size();++i) { if(i)s << ','; s << attr_json(inputs[i]); }
      s << "],\"outputs\":["; for(unsigned i=0;i<outputs.size();++i) { if(i)s << ','; s << attr_json(outputs[i]); }
      s << "],\"native_nhwc_inputs\":[";
      for(unsigned i=0;i<inputs.size();++i){if(i)s << ',';rknn_tensor_attr a{};a.index=i;int result=rknn_query(ctx,RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR,&a,sizeof(a));if(!result)s << attr_json(a);else s << "null";}
      s << "]}"; metadata=s.str();
    } catch (...) { rknn_destroy(ctx);ctx=0;throw; }
  }
  ~Session() { if(ctx) rknn_destroy(ctx); }
  Session(const Session&)=delete;
};
struct Result {
  Session& s;
  std::vector<rknn_output> out;
  bool held=false;
  double input_ms=0,run_get_ms=0,perf_ms=-1,query_ms=0;
  Result(Session& session, std::vector<rknn_input>& in):s(session),out(s.outputs.size()) {
    auto a=Clock::now(); check(rknn_inputs_set(s.ctx,in.size(),in.data()),"rknn_inputs_set");
    auto b=Clock::now(); check(rknn_run(s.ctx,nullptr),"rknn_run");
    for(unsigned i=0;i<out.size();++i) { out[i].index=i; out[i].want_float=0; }
    check(rknn_outputs_get(s.ctx,out.size(),out.data(),nullptr),"rknn_outputs_get"); held=true;
    auto c=Clock::now(); input_ms=ms(a,b);run_get_ms=ms(b,c);
    if(s.perf) { rknn_perf_run p{}; if(!rknn_query(s.ctx,RKNN_QUERY_PERF_RUN,&p,sizeof(p))&&p.run_duration>0)perf_ms=p.run_duration/1000.; }
    query_ms=ms(c,Clock::now());
  }
  ~Result() { if(held)rknn_outputs_release(s.ctx,out.size(),out.data()); }
  void timings(double* t, double convert_ms, double release_ms, double total_ms) const {
    // input, run/get(native), output convert, release, query, PERF_RUN, complete C call
    t[0]=input_ms;t[1]=run_get_ms;t[2]=convert_ms;t[3]=release_ms;t[4]=query_ms;t[5]=perf_ms;t[6]=total_ms;
  }
  double release() {auto a=Clock::now();check(rknn_outputs_release(s.ctx,out.size(),out.data()),"rknn_outputs_release");held=false;return ms(a,Clock::now());}
};
void convert_float(const rknn_output& out,const rknn_tensor_attr& a,float* dst) {
  const unsigned n=a.n_elems;
  if(!out.buf)throw std::runtime_error("null RKNN output");
  unsigned i=0;
  if(a.type==RKNN_TENSOR_FLOAT32) { if(out.size<n*4)throw std::runtime_error("short FP32 output");std::memcpy(dst,out.buf,n*4);return; }
  if(a.type==RKNN_TENSOR_FLOAT16) {
    if(out.size<n*2)throw std::runtime_error("short FP16 output");
    const __fp16* src=static_cast<const __fp16*>(out.buf);
    for(;i+8<=n;i+=8) {auto h=vld1q_f16(src+i);vst1q_f32(dst+i,vcvt_f32_f16(vget_low_f16(h)));vst1q_f32(dst+i+4,vcvt_f32_f16(vget_high_f16(h)));}
    for(;i<n;++i)dst[i]=src[i];
    return;
  }
  const float scale=a.qnt_type==RKNN_TENSOR_QNT_DFP?std::ldexp(1.f,-a.fl):a.scale;
  const float zp=a.qnt_type==RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC?a.zp:0;
  if(a.type==RKNN_TENSOR_INT8) {
    if(out.size<n)throw std::runtime_error("short INT8 output");
    const int8_t* src=static_cast<const int8_t*>(out.buf);
    for(;i+8<=n;i+=8) {auto w=vmovl_s8(vld1_s8(src+i));auto l=vcvtq_f32_s32(vmovl_s16(vget_low_s16(w)));auto h=vcvtq_f32_s32(vmovl_s16(vget_high_s16(w)));vst1q_f32(dst+i,vmulq_n_f32(vsubq_f32(l,vdupq_n_f32(zp)),scale));vst1q_f32(dst+i+4,vmulq_n_f32(vsubq_f32(h,vdupq_n_f32(zp)),scale));}
    for(;i<n;++i)dst[i]=(src[i]-zp)*scale;
    return;
  }
  if(a.type==RKNN_TENSOR_UINT8) { if(out.size<n)throw std::runtime_error("short UINT8 output");auto src=static_cast<const uint8_t*>(out.buf);for(;i<n;++i)dst[i]=(src[i]-zp)*scale;return; }
  throw std::runtime_error("unsupported output type "+std::to_string(a.type));
}
void write_nchw(const rknn_output& out,const rknn_tensor_attr& a,float* dst) {
  if(a.fmt==RKNN_TENSOR_NCHW) {convert_float(out,a,dst);return;}
  if(a.fmt!=RKNN_TENSOR_NHWC||a.n_dims!=4)throw std::runtime_error("unsupported output format");
  std::vector<float> tmp(a.n_elems);convert_float(out,a,tmp.data());
  const unsigned batch=a.dims[0],height=a.dims[1],width=a.dims[2],channels=a.dims[3];
  for(unsigned b=0;b<batch;++b)for(unsigned c=0;c<channels;++c)for(unsigned p=0;p<height*width;++p)
    dst[(b*channels+c)*height*width+p]=tmp[(b*height*width+p)*channels+c];
}
void image_contract(const Session& s,unsigned index,unsigned side) {
  if(index>=s.inputs.size())throw std::runtime_error("missing image input");
  const auto& a=s.inputs[index];
  if(a.n_dims!=4||a.n_elems!=side*side*3||a.fmt!=RKNN_TENSOR_NHWC||a.dims[0]!=1||a.dims[1]!=side||a.dims[2]!=side||a.dims[3]!=3)
    throw std::runtime_error("expected NHWC RGB image input contract at index "+std::to_string(index));
}
rknn_input rgb_input(const uint8_t* rgb,unsigned bytes) {
  if(!rgb)throw std::runtime_error("null RGB input");
  rknn_input i{};i.index=0;i.buf=const_cast<uint8_t*>(rgb);i.size=bytes;i.type=RKNN_TENSOR_UINT8;i.fmt=RKNN_TENSOR_NHWC;i.pass_through=0;return i;
}
struct Fear {
  Session templ, search;
  std::vector<uint16_t> feature_half;
  std::vector<float> feature_float;
  bool ready=false,half=false;
  std::string metadata;
  Fear(const char*t,const char*s,int core,bool perf,bool profile):templ(t,core,perf,profile),search(s,core,perf,profile) {
    if(templ.inputs.size()!=1||templ.outputs.size()!=1||search.inputs.size()!=2||search.outputs.size()!=2)throw std::runtime_error("unexpected FEAR I/O count");
    image_contract(templ,0,128);image_contract(search,0,256);
    auto& ti=templ.outputs[0];auto& si=search.inputs[1];
    if(ti.n_dims!=4||ti.n_elems!=16384||si.n_dims!=4||si.fmt!=RKNN_TENSOR_NHWC||si.dims[0]!=1||si.dims[1]!=8||si.dims[2]!=8||si.dims[3]!=256)
      throw std::runtime_error("FEAR template feature layout incompatible");
    if(search.outputs[0].n_elems!=1024||search.outputs[1].n_elems!=256)throw std::runtime_error("FEAR requires bbox then class outputs");
    half=ti.type==RKNN_TENSOR_FLOAT16;
    if(half)feature_half.resize(16384);else feature_float.resize(16384);
    metadata="{\"template\":"+templ.metadata+",\"search\":"+search.metadata+",\"template_feature_input\":\"NHWC "+std::string(half?"FP16":"FP32")+"\"}";
  }
};
struct Yolo {Session s;Yolo(const char*p,int core,bool perf,bool profile):s(p,core,perf,profile){if(s.inputs.size()!=1||s.outputs.size()!=9)throw std::runtime_error("YOLO requires one image/nine outputs");image_contract(s,0,640);} };
template<class F>int protect(F fn){try{fn();error_message.clear();return 0;}catch(const std::exception&e){error_message=e.what();return -1;}}
} // namespace

extern "C" {
const char* au_infer_error(){return error_message.c_str();}
void* au_fear_create(const char*t,const char*s,int core,int perf,int profile){try{return new Fear(t,s,core,perf,profile);}catch(const std::exception&e){error_message=e.what();return nullptr;}}
void au_fear_close(void*p){delete static_cast<Fear*>(p);}
const char* au_fear_metadata(void*p){return static_cast<Fear*>(p)->metadata.c_str();}
int au_fear_template(void*p,const uint8_t*rgb,unsigned bytes,float*feature_nchw,double*timing){return protect([&]{
  auto& f=*static_cast<Fear*>(p);if(bytes!=128*128*3)throw std::runtime_error("template byte count mismatch");
  auto begin=Clock::now();std::vector<rknn_input> in{rgb_input(rgb,bytes)};Result r(f.templ,in);auto a=Clock::now();
  const auto& attr=f.templ.outputs[0];write_nchw(r.out[0],attr,feature_nchw);
  // Standard outputs_get follows OUTPUT_ATTR layout. Transpose template output
  // once into the *queried search input* NHWC layout; never label NCHW as NHWC.
  for(unsigned p2=0;p2<64;++p2)for(unsigned c=0;c<256;++c){
    unsigned out_index=attr.fmt==RKNN_TENSOR_NCHW?c*64+p2:p2*256+c;
    if(f.half)f.feature_half[p2*256+c]=static_cast<const uint16_t*>(r.out[0].buf)[out_index];
    else f.feature_float[p2*256+c]=feature_nchw[c*64+p2];
  }
  f.ready=true;double conv=ms(a,Clock::now()),rel=r.release();r.timings(timing,conv,rel,ms(begin,Clock::now()));
});}
int au_fear_search(void*p,const uint8_t*rgb,unsigned bytes,float*bbox,float*cls,double*timing){return protect([&]{
  auto& f=*static_cast<Fear*>(p);if(!f.ready)throw std::runtime_error("initialize template before search");if(bytes!=256*256*3)throw std::runtime_error("search byte count mismatch");
  auto begin=Clock::now();rknn_input feat{};feat.index=1;feat.buf=f.half?static_cast<void*>(f.feature_half.data()):static_cast<void*>(f.feature_float.data());feat.size=16384*(f.half?2:4);feat.type=f.half?RKNN_TENSOR_FLOAT16:RKNN_TENSOR_FLOAT32;feat.fmt=RKNN_TENSOR_NHWC;feat.pass_through=0;
  std::vector<rknn_input> in{rgb_input(rgb,bytes),feat};Result r(f.search,in);auto a=Clock::now();write_nchw(r.out[0],f.search.outputs[0],bbox);write_nchw(r.out[1],f.search.outputs[1],cls);
  double conv=ms(a,Clock::now()),rel=r.release();r.timings(timing,conv,rel,ms(begin,Clock::now()));
});}
void* au_yolo_create(const char*p,int core,int perf,int profile){try{return new Yolo(p,core,perf,profile);}catch(const std::exception&e){error_message=e.what();return nullptr;}}
void au_yolo_close(void*p){delete static_cast<Yolo*>(p);}
const char* au_yolo_metadata(void*p){return static_cast<Yolo*>(p)->s.metadata.c_str();}
int au_yolo_infer(void*p,const uint8_t*rgb,unsigned bytes,float**outputs,double*timing){return protect([&]{
  auto& s=static_cast<Yolo*>(p)->s;if(bytes!=640*640*3)throw std::runtime_error("YOLO byte count mismatch");auto begin=Clock::now();std::vector<rknn_input>in{rgb_input(rgb,bytes)};Result r(s,in);auto a=Clock::now();
  for(unsigned i=0;i<9;++i)write_nchw(r.out[i],s.outputs[i],outputs[i]);
  double conv=ms(a,Clock::now()),rel=r.release();r.timings(timing,conv,rel,ms(begin,Clock::now()));
});}
}
