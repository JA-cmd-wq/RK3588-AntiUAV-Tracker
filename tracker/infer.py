"""Board C API wrapper: uint8 RGB NHWC images, original FEAR CPU Exp.

This removes Python full-image float32 NCHW packing. inputs_set still performs
the model's required uint8 -> FP16 conversion; this is not input zero-copy.
Template feature FP16 values are transposed once, retained in C++, and passed
as NHWC according to the queried search-input attributes. Native output dtype
conversion is measured separately from run/outputs_get.
"""
from __future__ import annotations
import ctypes as C
import json
import time
from pathlib import Path
import numpy as np

_PTR = C.POINTER(C.c_float)
_BYTE = C.POINTER(C.c_uint8)
_TIMINGS = ('input_set_ms', 'run_get_ms', 'output_convert_ms', 'output_release_ms',
            'perf_query_ms', 'device_run_ms', 'native_total_ms')
_LIB = None

def core_mask(value):
    masks = {'0': 1, '1': 2, '2': 4, '01': 3, '02': 5, '12': 6, '012': 7}
    if str(value) not in masks:
        raise ValueError('Explicit core must be 0, 1, 2, 01, 02, 12 or 012')
    return masks[str(value)]

def validate_cores(fear='1', yolo='02'):
    if core_mask(fear) & core_mask(yolo):
        raise ValueError('FEAR and YOLO NPU core masks overlap')

def _library():
    global _LIB
    if _LIB is not None:
        return _LIB
    path = Path(__file__).with_name('libanti_uav_infer.so')
    if not path.is_file():
        raise RuntimeError(f'Build native inference on RK3588 first: {path.with_name("build_infer.sh")}')
    lib = C.CDLL(str(path))  # ctypes releases the GIL while the C API runs.
    lib.au_infer_error.argtypes=[]; lib.au_infer_error.restype=C.c_char_p
    lib.au_fear_create.argtypes=[C.c_char_p,C.c_char_p,C.c_int,C.c_int,C.c_int];lib.au_fear_create.restype=C.c_void_p
    lib.au_yolo_create.argtypes=[C.c_char_p,C.c_int,C.c_int,C.c_int];lib.au_yolo_create.restype=C.c_void_p
    for name in ('fear','yolo'):
        getattr(lib,f'au_{name}_close').argtypes=[C.c_void_p]
        getattr(lib,f'au_{name}_close').restype=None
        getattr(lib,f'au_{name}_metadata').argtypes=[C.c_void_p]
        getattr(lib,f'au_{name}_metadata').restype=C.c_char_p
    lib.au_fear_template.argtypes=[C.c_void_p,_BYTE,C.c_uint,_PTR,C.POINTER(C.c_double)];lib.au_fear_template.restype=C.c_int
    lib.au_fear_search.argtypes=[C.c_void_p,_BYTE,C.c_uint,_PTR,_PTR,C.POINTER(C.c_double)];lib.au_fear_search.restype=C.c_int
    lib.au_yolo_infer.argtypes=[C.c_void_p,_BYTE,C.c_uint,C.POINTER(_PTR),C.POINTER(C.c_double)];lib.au_yolo_infer.restype=C.c_int
    _LIB=lib
    return lib

def _check(ret, lib):
    if ret:
        raise RuntimeError(lib.au_infer_error().decode('utf-8','replace'))

def _rgb(image, side):
    if not isinstance(image,np.ndarray) or image.dtype != np.uint8 or image.shape != (side,side,3):
        raise ValueError(f'Expected uint8 RGB image of shape {(side,side,3)}')
    if not image.flags.c_contiguous:
        raise ValueError('RGB input must be contiguous; do not hide an input copy')
    return image.ctypes.data_as(_BYTE)

def _time_dict(native, start):
    result=dict(zip(_TIMINGS,map(float,native)))
    result['inference_ms']=(time.perf_counter()-start)*1000
    result.update(inputs_ms=result['input_set_ms'], outputs_ms=result['output_convert_ms'],
                  npu_ms=result['device_run_ms'] if result['device_run_ms'] > 0 else None,
                  total_ms=result['inference_ms'], cpu_exp_ms=0.)
    result['profiling_enabled']=False
    return result

def _shape_nchw(attr):
    dims=attr['dims']
    if len(dims)!=4:
        raise ValueError(f'Only four-dimensional model outputs supported: {attr}')
    if attr['fmt']==0:
        return tuple(dims)
    if attr['fmt']==1:
        return (dims[0],dims[3],dims[1],dims[2])
    raise ValueError(f'Unsupported output format: {attr}')

class NativeFear:
    def __init__(self, template_model, search_model, core='1', *, query_perf=True, profiling=False):
        self.lib=_library();self.handle=None;self.core=core;self.profiling=profiling
        self.handle=self.lib.au_fear_create(str(template_model).encode(),str(search_model).encode(),core_mask(core),int(query_perf),int(profiling))
        if not self.handle:
            _check(-1,self.lib)
        self.description=json.loads(self.lib.au_fear_metadata(self.handle))
        self.description.update(backend='native-rknn-c-api',image_input='RGB uint8 NHWC; pass_through=0',
                                bbox_output='log_distances_restored_with_cpu_exp',input_zero_copy=False)
        self.feature=None

    def template(self, rgb):
        if not self.handle:
            raise RuntimeError('NativeFear is closed')
        ptr=_rgb(rgb,128);feature=np.empty((1,256,8,8),np.float32);native=(C.c_double*7)()
        start=time.perf_counter()
        _check(self.lib.au_fear_template(self.handle,ptr,rgb.nbytes,feature.ctypes.data_as(_PTR),native),self.lib)
        self.feature=feature
        result=_time_dict(native,start);result['profiling_enabled']=self.profiling
        return result

    def search(self, rgb):
        if not self.handle:
            raise RuntimeError('NativeFear is closed')
        ptr=_rgb(rgb,256);bbox=np.empty((1,4,16,16),np.float32);cls=np.empty((1,1,16,16),np.float32);native=(C.c_double*7)()
        start=time.perf_counter()
        _check(self.lib.au_fear_search(self.handle,ptr,rgb.nbytes,bbox.ctypes.data_as(_PTR),cls.ctypes.data_as(_PTR),native),self.lib)
        result=_time_dict(native,start)
        t=time.perf_counter();np.exp(bbox,out=bbox);result['cpu_exp_ms']=(time.perf_counter()-t)*1000
        result['inference_ms']=(time.perf_counter()-start)*1000;result['total_ms']=result['inference_ms'];result['profiling_enabled']=self.profiling
        return bbox,cls,result

    def close(self):
        if self.handle:
            self.lib.au_fear_close(self.handle);self.handle=None

    def __enter__(self):return self
    def __exit__(self,*_):self.close()

class NativeYolo:
    def __init__(self, model, core='0', *, query_perf=True, profiling=False):
        self.lib=_library();self.handle=None;self.core=core;self.profiling=profiling
        self.handle=self.lib.au_yolo_create(str(model).encode(),core_mask(core),int(query_perf),int(profiling))
        if not self.handle:_check(-1,self.lib)
        self.description=json.loads(self.lib.au_yolo_metadata(self.handle))
        self.description.update(backend='native-rknn-c-api',image_input='RGB uint8 NHWC; pass_through=0',input_zero_copy=False)
        self.shapes=[_shape_nchw(a) for a in self.description['outputs']]

    def infer(self, rgb):
        if not self.handle:
            raise RuntimeError('NativeYolo is closed')
        ptr=_rgb(rgb,640);outputs=[np.empty(shape,np.float32) for shape in self.shapes];native=(C.c_double*7)()
        pointers=(_PTR*len(outputs))(*(a.ctypes.data_as(_PTR) for a in outputs))
        start=time.perf_counter()
        _check(self.lib.au_yolo_infer(self.handle,ptr,rgb.nbytes,pointers,native),self.lib)
        result=_time_dict(native,start);result['profiling_enabled']=self.profiling
        return outputs,result

    def close(self):
        if self.handle:self.lib.au_yolo_close(self.handle);self.handle=None

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
