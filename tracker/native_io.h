#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
const char* auv_last_error(void);
void* auv_decoder_open(const char* path);
void* auv_decoder_read(void* decoder, int* eos);
void auv_decoder_close(void* decoder);
void auv_frame_close(void* frame);
void* auv_frame_clone(void* frame, double* copy_ms);
int auv_frame_width(void* frame);
int auv_frame_height(void* frame);
int64_t auv_frame_pts(void* frame);
int auv_frame_stride(void* frame);
void* auv_frame_rgb(void* frame);
double auv_frame_format_ms(void* frame);
int auv_frame_crop(void* frame, int x, int y, int width, int height,
                   int side, int pad_rgb, void* output_rgb, double* milliseconds);
int auv_frame_letterbox(void* frame, int side, int pad_rgb, void* output_rgb,
                        double* scale, int* pad_x, int* pad_y, double* milliseconds);
void* auv_encoder_open(const char* path, int width, int height, double fps, int bps);
int auv_encoder_write(void* encoder, void* frame, double* convert_ms, double* submit_ms);
int auv_encoder_close(void* encoder, double* flush_ms);
#ifdef __cplusplus
}
#endif
