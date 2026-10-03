// MPP DMA decoder/encoder with RGA-only pixel conversion, crop, pad and resize.
// Adapted from the board's proven relate_anything_rknn/cpp_v8/video_mpp_rga.cpp.
#include "native_io.h"
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/allocators/gstdmabuf.h>
#include <gst/video/video.h>
#include <rga/im2d.h>
#include <rga/RgaApi.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now()-t).count();
}
int align16(int v) { return (v+15)&~15; }
void check(IM_STATUS s, const char* op) {
  if (s != IM_STATUS_SUCCESS) throw std::runtime_error(std::string(op)+": "+imStrError(s));
}
IM_STATUS process_mpp_nv12(rga_buffer_t src,rga_buffer_t dst,im_rect from,im_rect to) {
  // MPP may allocate pages above 4 GiB. RGA2 cannot map those DMA buffers;
  // RGA3 can. Destination and all other scratch buffers use system-dma32.
  im_opt_t options{};options.core=IM_SCHEDULER_RGA3_CORE0|IM_SCHEDULER_RGA3_CORE1;
  rga_buffer_t empty{};im_rect z{};
  return improcess(src,dst,empty,from,to,z,0,nullptr,&options,0);
}
void init_gst() { static std::once_flag once; std::call_once(once, []{gst_init(nullptr,nullptr);}); }
std::string quote(const char* path) {
  gchar* e=g_strescape(path,nullptr); std::string r="\""+std::string(e)+"\"";g_free(e);return r;
}
std::string bus_error(GstElement* pipeline) {
  GstBus* bus=gst_element_get_bus(pipeline);
  GstMessage* m=gst_bus_pop_filtered(bus,GST_MESSAGE_ERROR);
  std::string s="pipeline did not produce a sample";
  if(m) { GError* e=nullptr;gchar* d=nullptr;gst_message_parse_error(m,&e,&d);
    if(e) s=e->message;if(d)s+=" ("+std::string(d)+")";
    if(e)g_error_free(e);g_free(d);gst_message_unref(m); }
  gst_object_unref(bus);return s;
}
GstElement* launch(const std::string& spec) {
  init_gst();GError* e=nullptr;GstElement* p=gst_parse_launch(spec.c_str(),&e);
  if(e||!p){std::string s=e?e->message:"parse failed";if(e)g_error_free(e);
    if(p)gst_object_unref(p);throw std::runtime_error(s);}
  return p;
}
struct Dma {
  int fd=-1;size_t size;void* ptr=nullptr;
  explicit Dma(size_t bytes):size(bytes) {
    int heap=open("/dev/dma_heap/system-dma32",O_RDONLY|O_CLOEXEC);
    if(heap<0)throw std::runtime_error("open system-dma32: "+std::string(strerror(errno)));
    dma_heap_allocation_data a{};a.len=bytes;a.fd_flags=O_RDWR|O_CLOEXEC;
    int r=ioctl(heap,DMA_HEAP_IOCTL_ALLOC,&a);int e=errno;close(heap);
    if(r<0)throw std::runtime_error("DMA alloc: "+std::string(strerror(e)));
    fd=a.fd;ptr=mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    if(ptr==MAP_FAILED){close(fd);throw std::runtime_error("DMA mmap");}
  }
  ~Dma(){if(ptr&&ptr!=MAP_FAILED)munmap(ptr,size);if(fd>=0)close(fd);}
  void sync(bool start, bool write=false) {
    dma_buf_sync s{};s.flags=(write?DMA_BUF_SYNC_RW:DMA_BUF_SYNC_READ)|
      (start?DMA_BUF_SYNC_START:DMA_BUF_SYNC_END);
    if(ioctl(fd,DMA_BUF_IOCTL_SYNC,&s)<0)throw std::runtime_error("DMA sync");
  }
};
// A shared pool outlives Decoder/Encoder while downstream GstBuffer/Frame holds
// a buffer. Reuse does not occur until every consumer releases its shared_ptr.
struct Pool {
  std::mutex mutex;std::vector<std::shared_ptr<Dma>> buffers;
  std::shared_ptr<Dma> get(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex);
    for(auto& b:buffers)if(b.use_count()==1&&b->size>=bytes)return b;
    auto b=std::make_shared<Dma>(bytes);buffers.push_back(b);return b;
  }
};
struct Frame {
  GstSample* sample=nullptr;int width=0,height=0,stride=0,hstride=0,fd=-1;
  GstClockTime pts=GST_CLOCK_TIME_NONE;int csc=IM_YUV_TO_RGB_BT601_LIMIT;
  std::shared_ptr<Pool> pool;std::shared_ptr<Dma> rgb;
  std::mutex mutex;bool cpu_sync=false;double format_ms=0;
  explicit Frame(GstSample* s,std::shared_ptr<Pool> p):sample(s),pool(p) {
    GstCaps* caps=gst_sample_get_caps(s);
    GstVideoInfo info;if(!gst_video_info_from_caps(&info,caps)||
       GST_VIDEO_INFO_FORMAT(&info)!=GST_VIDEO_FORMAT_NV12)
      throw std::runtime_error("MPP output must be NV12");
    width=GST_VIDEO_INFO_WIDTH(&info);height=GST_VIDEO_INFO_HEIGHT(&info);
    // GstVideoInfo invents BT709 for untagged HD video. FFmpeg/OpenCV's
    // untagged MPEG4 convention is BT601; preserve v1 model input colors.
    // Use explicit metadata when the decoder actually provides it.
    if(gst_structure_has_field(gst_caps_get_structure(caps,0),"colorimetry")) {
      if(info.colorimetry.matrix==GST_VIDEO_COLOR_MATRIX_BT709)csc=IM_YUV_TO_RGB_BT709_LIMIT;
      else if(info.colorimetry.range==GST_VIDEO_COLOR_RANGE_0_255)csc=IM_YUV_TO_RGB_BT601_FULL;
    }
    GstBuffer* b=gst_sample_get_buffer(s);
    if(gst_buffer_n_memory(b)!=1||!gst_is_dmabuf_memory(gst_buffer_peek_memory(b,0)))
      throw std::runtime_error("MPP output is not single DMA-BUF memory");
    fd=gst_dmabuf_memory_get_fd(gst_buffer_peek_memory(b,0));
    GstVideoMeta* meta=gst_buffer_get_video_meta(b);
    stride=meta?meta->stride[0]:GST_VIDEO_INFO_PLANE_STRIDE(&info,0);
    size_t uv=meta?meta->offset[1]:GST_VIDEO_INFO_PLANE_OFFSET(&info,1);
    if(fd<0||stride<width||uv%stride)throw std::runtime_error("unsupported DMA stride");
    hstride=uv/stride;if(hstride<height)throw std::runtime_error("short DMA height stride");
    pts=GST_BUFFER_PTS(b);
  }
  ~Frame(){if(cpu_sync)try{rgb->sync(false,true);}catch(...){}if(sample)gst_sample_unref(sample);}
  rga_buffer_t nv12(){auto r=wrapbuffer_fd(fd,width,height,RK_FORMAT_YCbCr_420_SP,stride,hstride);
    r.color_space_mode=csc;return r;}
  rga_buffer_t rgb_buffer() {
    if(cpu_sync){rgb->sync(false,true);cpu_sync=false;}
    if(!rgb){auto t=Clock::now();rgb=pool->get(size_t(align16(width))*height*3);
      auto d=wrapbuffer_fd(rgb->fd,width,height,RK_FORMAT_RGB_888,align16(width),height);
      check(process_mpp_nv12(nv12(),d,{0,0,width,height},{0,0,width,height}),"RGA3 NV12->RGB");
      format_ms=elapsed(t);}
    return wrapbuffer_fd(rgb->fd,width,height,RK_FORMAT_RGB_888,align16(width),height);
  }
  void* rgb_ptr(){rgb_buffer();rgb->sync(true,true);cpu_sync=true;return rgb->ptr;}
};
struct Decoder {
  GstElement* pipeline=nullptr;GstElement* sink=nullptr;
  std::shared_ptr<Pool> pool=std::make_shared<Pool>();
  explicit Decoder(const char* path) {
    pipeline=launch("filesrc location="+quote(path)+" ! qtdemux name=d d.video_0 ! queue ! parsebin ! "
      "mppvideodec dma-feature=true arm-afbc=false ! video/x-raw(memory:DMABuf),format=NV12 ! "
      "appsink name=framesink sync=false max-buffers=8 drop=false");
    sink=gst_bin_get_by_name(GST_BIN(pipeline),"framesink");
    if(!sink||gst_element_set_state(pipeline,GST_STATE_PLAYING)==GST_STATE_CHANGE_FAILURE)
      throw std::runtime_error(bus_error(pipeline));
  }
  ~Decoder(){if(pipeline)gst_element_set_state(pipeline,GST_STATE_NULL);
    if(sink)gst_object_unref(sink);if(pipeline)gst_object_unref(pipeline);}
  Frame* read(bool& eos) {
    GstSample* s=gst_app_sink_try_pull_sample(GST_APP_SINK(sink),5*GST_SECOND);
    if(!s){eos=gst_app_sink_is_eos(GST_APP_SINK(sink));if(eos)return nullptr;
      throw std::runtime_error(bus_error(pipeline));}
    try{return new Frame(s,pool);}catch(...){gst_sample_unref(s);throw;}
  }
};
struct Scratch {
  std::shared_ptr<Dma> pad,out,middle[2];
  Dma& allocate(std::shared_ptr<Dma>& b,size_t size){if(!b||b->size<size)b=std::make_shared<Dma>(size);return *b;}
};
thread_local Scratch scratch;
void resize_rgb_safe(rga_buffer_t source,rga_buffer_t destination) {
  // RK3588 RGA engines have per-operation scale limits. Rectangular FEAR
  // contexts can exceed those limits on only one axis; preserve their geometry
  // with hardware-only intermediate resizes instead of clamping the context.
  int stage=0;
  auto unsafe=[](int a,int b){return double(b)/a>8.0||double(b)/a<.125;};
  while(unsafe(source.width,destination.width)||unsafe(source.height,destination.height)) {
    auto next_side=[](int from,int to){
      int midpoint=std::lround(std::sqrt(double(from)*to));
      return std::clamp(midpoint,std::max(2,(from+7)/8),std::min(8192,from*8));
    };
    int w=next_side(source.width,destination.width),h=next_side(source.height,destination.height);
    int stride=align16(w);
    Dma& middle=scratch.allocate(scratch.middle[stage%2],size_t(stride)*h*3);
    auto target=wrapbuffer_fd(middle.fd,w,h,RK_FORMAT_RGB_888,stride,h);
    check(imresize(source,target),"RGA FEAR intermediate resize");
    source=target;++stage;
    if(stage>8)throw std::runtime_error("RGA multipass resize failed to converge");
  }
  check(imresize(source,destination),"RGA FEAR resize");
}
void rgb_blit(rga_buffer_t src,rga_buffer_t dst,im_rect from,im_rect to) {
  // This board's im2d validator rejects rectangle coordinates equal to one,
  // even for RGB where odd x/y are legal. The legacy API submits the same
  // DMA fd rectangles directly without changing any crop coordinates.
  if(from.x==1||from.y==1||to.x==1||to.y==1) {
    rga_info_t s{},d{};s.fd=src.fd;d.fd=dst.fd;s.mmuFlag=d.mmuFlag=1;
    s.rect={from.x,from.y,from.width,from.height,src.wstride,src.hstride,src.format};
    d.rect={to.x,to.y,to.width,to.height,dst.wstride,dst.hstride,dst.format};
    const int r=c_RkRgaBlit(&s,&d,nullptr);
    if(r<0)throw std::runtime_error("RGA legacy exact RGB crop failed: "+std::to_string(r));
  } else {
    rga_buffer_t empty{};im_rect z{};
    check(improcess(src,dst,empty,from,to,z,0,nullptr,nullptr,0),"RGA crop/pad");
  }
}
void copy_output(Dma& d,void* ptr,int side) {
  d.sync(true);std::memcpy(ptr,d.ptr,size_t(side)*side*3);d.sync(false);
}
struct Encoder {
  GstElement* pipeline=nullptr;GstElement* src=nullptr;
  std::shared_ptr<Pool> pool=std::make_shared<Pool>();int width,height,ws,hs;double fps;uint64_t count=0;
  explicit Encoder(const char* path,int w,int h,double f,int bps):width(w),height(h),ws(align16(w)),hs(align16(h)),fps(f) {
    int numerator=std::lround(f*1000);int denominator=1000;
    pipeline=launch("appsrc name=framesrc format=time is-live=false block=true max-bytes="+
      std::to_string(size_t(ws)*hs*3/2*6)+" caps=video/x-raw,format=NV12,width="+
      std::to_string(w)+",height="+std::to_string(h)+",framerate="+std::to_string(numerator)+"/"+
      std::to_string(denominator)+" ! mpph264enc bps="+std::to_string(bps)+
      " max-pending=4 ! h264parse ! mp4mux ! filesink sync=false location="+quote(path));
    src=gst_bin_get_by_name(GST_BIN(pipeline),"framesrc");
    if(!src||gst_element_set_state(pipeline,GST_STATE_PLAYING)==GST_STATE_CHANGE_FAILURE)
      throw std::runtime_error(bus_error(pipeline));
  }
  ~Encoder(){if(pipeline)gst_element_set_state(pipeline,GST_STATE_NULL);
    if(src)gst_object_unref(src);if(pipeline)gst_object_unref(pipeline);}
  void write(Frame& f,double& convert,double& submit) {
    if(f.width!=width||f.height!=height)throw std::runtime_error("encoder frame dimensions");
    auto t=Clock::now();std::shared_ptr<Dma> d=pool->get(size_t(ws)*hs*3/2);
    auto dst=wrapbuffer_fd(d->fd,width,height,RK_FORMAT_YCbCr_420_SP,ws,hs);
    {std::lock_guard<std::mutex> lock(f.mutex);
      auto source=f.rgb_buffer();
      // NumPy callers may retain a view across a crop that already ended CPU
      // access. Flush those writes as well, before handing RGB back to RGA.
      f.rgb->sync(true,true);f.rgb->sync(false,true);f.cpu_sync=false;
      check(imcvtcolor(source,dst,RK_FORMAT_RGB_888,RK_FORMAT_YCbCr_420_SP,
            IM_RGB_TO_YUV_BT709_LIMIT),"RGA RGB->NV12");}
    convert=elapsed(t);t=Clock::now();
    GstAllocator* allocator=gst_dmabuf_allocator_new();
    GstMemory* memory=gst_dmabuf_allocator_alloc(allocator,dup(d->fd),d->size);gst_object_unref(allocator);
    GstBuffer* b=gst_buffer_new();gst_buffer_append_memory(b,memory);
    auto* owner=new std::shared_ptr<Dma>(d);
    gst_mini_object_set_qdata(GST_MINI_OBJECT(memory),g_quark_from_static_string("auv-dma-owner"),owner,
      [](gpointer p){delete static_cast<std::shared_ptr<Dma>*>(p);});
    gsize offsets[GST_VIDEO_MAX_PLANES]={0,size_t(ws)*hs,0,0};
    gint strides[GST_VIDEO_MAX_PLANES]={ws,ws,0,0};
    gst_buffer_add_video_meta_full(b,GST_VIDEO_FRAME_FLAG_NONE,GST_VIDEO_FORMAT_NV12,
      width,height,2,offsets,strides);
    GST_BUFFER_PTS(b)=gst_util_uint64_scale(count,1000000000000ULL,std::lround(fps*1000));
    GST_BUFFER_DTS(b)=GST_BUFFER_PTS(b);
    GST_BUFFER_DURATION(b)=gst_util_uint64_scale(1,1000000000000ULL,std::lround(fps*1000));
    ++count;GstFlowReturn r=gst_app_src_push_buffer(GST_APP_SRC(src),b);
    if(r!=GST_FLOW_OK)throw std::runtime_error("MPP encoder push failed: "+bus_error(pipeline));
    submit=elapsed(t);
  }
  void finish(){if(gst_app_src_end_of_stream(GST_APP_SRC(src))!=GST_FLOW_OK)
      throw std::runtime_error("encoder EOS rejected");
    GstBus* bus=gst_element_get_bus(pipeline);
    GstMessage* m=gst_bus_timed_pop_filtered(bus,30*GST_SECOND,
      GstMessageType(GST_MESSAGE_EOS|GST_MESSAGE_ERROR));gst_object_unref(bus);
    if(!m)throw std::runtime_error("encoder flush timed out");
    bool error=GST_MESSAGE_TYPE(m)==GST_MESSAGE_ERROR;gst_message_unref(m);
    if(error)throw std::runtime_error(bus_error(pipeline));
  }
};
}

extern "C" {
const char* auv_last_error(){return last_error.c_str();}
void* auv_decoder_open(const char* p){try{return new Decoder(p);}catch(const std::exception& e){last_error=e.what();return nullptr;}}
void* auv_decoder_read(void* d,int* eos){try{bool end=false;auto* f=static_cast<Decoder*>(d)->read(end);*eos=end;return f;}
 catch(const std::exception& e){last_error=e.what();*eos=-1;return nullptr;}}
void auv_decoder_close(void* d){delete static_cast<Decoder*>(d);}
void auv_frame_close(void* f){delete static_cast<Frame*>(f);}
void* auv_frame_clone(void* handle,double* copy_ms) {
  try {
    auto t=Clock::now();auto& source=*static_cast<Frame*>(handle);
    std::lock_guard<std::mutex> lock(source.mutex);
    auto source_rgb=source.rgb_buffer();
    std::unique_ptr<Frame> clone(new Frame(gst_sample_ref(source.sample),source.pool));
    clone->rgb=source.pool->get(size_t(align16(source.width))*source.height*3);
    auto target=wrapbuffer_fd(clone->rgb->fd,clone->width,clone->height,RK_FORMAT_RGB_888,
                             align16(clone->width),clone->height);
    check(imcopy(source_rgb,target),"RGA immutable-frame draw copy");
    *copy_ms=elapsed(t);return clone.release();
  } catch(const std::exception& e){last_error=e.what();return nullptr;}
}
int auv_frame_width(void* f){return static_cast<Frame*>(f)->width;}
int auv_frame_height(void* f){return static_cast<Frame*>(f)->height;}
int64_t auv_frame_pts(void* f){return static_cast<Frame*>(f)->pts;}
int auv_frame_stride(void* f){return align16(static_cast<Frame*>(f)->width)*3;}
double auv_frame_format_ms(void* f){return static_cast<Frame*>(f)->format_ms;}
void* auv_frame_rgb(void* f){try{auto& v=*static_cast<Frame*>(f);std::lock_guard<std::mutex> l(v.mutex);return v.rgb_ptr();}
 catch(const std::exception& e){last_error=e.what();return nullptr;}}
int auv_frame_crop(void* frame,int x,int y,int w,int h,int side,int color,void* output,double* ms) {
  try {auto t=Clock::now();auto& f=*static_cast<Frame*>(frame);std::lock_guard<std::mutex> lock(f.mutex);
    if(w<2||h<2||w>8192||h>8192||side<2||side>8192)throw std::runtime_error("RGA crop dimensions outside 2..8192");
    auto src=f.rgb_buffer();int stride=align16(w);
    Dma& p=scratch.allocate(scratch.pad,size_t(stride)*h*3);
    auto pad=wrapbuffer_fd(p.fd,w,h,RK_FORMAT_RGB_888,stride,h);
    int sx=std::max(0,x),sy=std::max(0,y),ex=std::min(f.width,x+w),ey=std::min(f.height,y+h);
    if(ex<=sx||ey<=sy)throw std::runtime_error("crop has no source pixels");
    if(sx!=x||sy!=y||ex!=x+w||ey!=y+h)check(imfill(pad,{0,0,w,h},color),"RGA crop fill");
    rgb_blit(src,pad,{sx,sy,ex-sx,ey-sy},{sx-x,sy-y,ex-sx,ey-sy});
    Dma& o=scratch.allocate(scratch.out,size_t(side)*side*3);
    auto dst=wrapbuffer_fd(o.fd,side,side,RK_FORMAT_RGB_888);
    resize_rgb_safe(pad,dst);copy_output(o,output,side);*ms=elapsed(t);return 0;
  }catch(const std::exception& e){last_error=e.what();return -1;}
}
int auv_frame_letterbox(void* frame,int side,int color,void* output,double* ratio,int* px,int* py,double* ms) {
  try {auto t=Clock::now();auto& f=*static_cast<Frame*>(frame);std::lock_guard<std::mutex> lock(f.mutex);
    double scale=std::min(double(side)/f.width,double(side)/f.height);
    int w=std::lround(f.width*scale),h=std::lround(f.height*scale);
    int x=(side-w)/2,y=(side-h)/2;
    Dma& o=scratch.allocate(scratch.out,size_t(side)*side*3);
    auto dst=wrapbuffer_fd(o.fd,side,side,RK_FORMAT_RGB_888);
    check(imfill(dst,{0,0,side,side},color),"RGA letterbox fill");
    check(process_mpp_nv12(f.nv12(),dst,{0,0,f.width,f.height},{x,y,w,h}),"RGA3 letterbox NV12->RGB resize");
    copy_output(o,output,side);*ratio=scale;*px=x;*py=y;*ms=elapsed(t);return 0;
  }catch(const std::exception& e){last_error=e.what();return -1;}
}
void* auv_encoder_open(const char* p,int w,int h,double fps,int bps){try{return new Encoder(p,w,h,fps,bps);}
 catch(const std::exception& e){last_error=e.what();return nullptr;}}
int auv_encoder_write(void* e,void* f,double* c,double* s){try{static_cast<Encoder*>(e)->write(*static_cast<Frame*>(f),*c,*s);return 0;}
 catch(const std::exception& x){last_error=x.what();return -1;}}
int auv_encoder_close(void* p,double* ms){auto* e=static_cast<Encoder*>(p);try{auto t=Clock::now();e->finish();*ms=elapsed(t);delete e;return 0;}
 catch(const std::exception& x){last_error=x.what();delete e;return -1;}}
}
