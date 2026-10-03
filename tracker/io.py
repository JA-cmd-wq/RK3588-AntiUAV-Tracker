"""MPP DMA-buffer video IO and RGA image transforms, without CPU pixel transforms.

Frame.rgb() is an RGB view into mapped DMA storage. The Frame must outlive its
view and all async users. Frame.close() releases the original decoded sample.
Only the small final NPU tensors are copied into NumPy arrays.
"""
from __future__ import annotations
import ctypes as C
from pathlib import Path
import numpy as np

_lib = C.CDLL(str(Path(__file__).with_name('libnative_io.so')))
_ptr, _int, _double = C.c_void_p, C.c_int, C.c_double
_lib.auv_last_error.restype = C.c_char_p
def _sig(name, restype, *args):
    fn = getattr(_lib, name); fn.restype = restype; fn.argtypes = list(args); return fn
_open = _sig('auv_decoder_open', _ptr, C.c_char_p)
_read = _sig('auv_decoder_read', _ptr, _ptr, C.POINTER(_int))
_dclose = _sig('auv_decoder_close', None, _ptr)
_fclose = _sig('auv_frame_close', None, _ptr)
_clone = _sig('auv_frame_clone', _ptr, _ptr, C.POINTER(_double))
_width = _sig('auv_frame_width', _int, _ptr)
_height = _sig('auv_frame_height', _int, _ptr)
_pts = _sig('auv_frame_pts', C.c_int64, _ptr)
_stride = _sig('auv_frame_stride', _int, _ptr)
_rgb = _sig('auv_frame_rgb', _ptr, _ptr)
_fmt = _sig('auv_frame_format_ms', _double, _ptr)
_crop = _sig('auv_frame_crop', _int, _ptr, _int, _int, _int, _int, _int, _int, _ptr, C.POINTER(_double))
_letter = _sig('auv_frame_letterbox', _int, _ptr, _int, _int, _ptr, C.POINTER(_double), C.POINTER(_int), C.POINTER(_int), C.POINTER(_double))
_eopen = _sig('auv_encoder_open', _ptr, C.c_char_p, _int, _int, _double, _int)
_ewrite = _sig('auv_encoder_write', _int, _ptr, _ptr, C.POINTER(_double), C.POINTER(_double))
_eclose = _sig('auv_encoder_close', _int, _ptr, C.POINTER(_double))
def _error(): return RuntimeError(_lib.auv_last_error().decode())
def _packed(rgb):
    a = np.clip(np.rint(rgb), 0, 255).astype(np.uint32)
    return int(a[0]) | int(a[1]) << 8 | int(a[2]) << 16

class Frame:
    def __init__(self, handle):
        self._handle = handle
        self.width, self.height = _width(handle), _height(handle)
        self.pts_ns = _pts(handle)
        self._view = None
        self.copy_ms = 0.0
    @property
    def format_ms(self): return _fmt(self._handle)
    def rgb(self):
        # A crop/encode may have ended CPU access after an earlier view was
        # created. Re-enter the DMA CPU synchronization interval each time.
        pointer = _rgb(self._handle)
        if not pointer: raise _error()
        if self._view is None:
            stride = _stride(self._handle)
            backing = (C.c_uint8 * (stride * self.height)).from_address(pointer)
            self._view = np.ndarray((self.height, self.width, 3), np.uint8,
                                    buffer=backing, strides=(stride, 3, 1))
        return self._view
    def crop(self, context, side, pad_rgb=(114, 114, 114)):
        output = np.empty((side, side, 3), np.uint8); ms = _double()
        if _crop(self._handle, *(int(v) for v in context), side, _packed(pad_rgb),
                 output.ctypes.data, C.byref(ms)): raise _error()
        return output, ms.value
    def clone_for_draw(self):
        """RGA-copy RGB into independent DMA storage; original stays immutable."""
        ms = _double()
        handle = _clone(self._handle, C.byref(ms))
        if not handle: raise _error()
        result = Frame(handle); result.copy_ms = ms.value
        return result
    def letterbox(self, side=640, pad=114):
        output=np.empty((side,side,3),np.uint8)
        scale,ms=_double(),_double();x,y=_int(),_int()
        if _letter(self._handle,side,_packed((pad,pad,pad)),output.ctypes.data,
                   C.byref(scale),C.byref(x),C.byref(y),C.byref(ms)): raise _error()
        return output, scale.value, x.value, y.value, ms.value
    def close(self):
        if self._handle:
            _fclose(self._handle);self._handle=None;self._view=None
    def __del__(self): self.close()

class Decoder:
    def __init__(self,path):
        self._handle=_open(str(path).encode())
        if not self._handle: raise _error()
    def read(self):
        eos=_int();handle=_read(self._handle,C.byref(eos))
        if eos.value<0: raise _error()
        return Frame(handle) if handle else None
    def close(self):
        if self._handle: _dclose(self._handle);self._handle=None
    def __del__(self): self.close()

class Encoder:
    def __init__(self,path,width,height,fps,bps=6000000):
        self._handle=_eopen(str(path).encode(),width,height,fps,bps)
        if not self._handle: raise _error()
    def write(self,frame):
        convert,submit=_double(),_double()
        if _ewrite(self._handle,frame._handle,C.byref(convert),C.byref(submit)): raise _error()
        return convert.value,submit.value
    def close(self):
        if not self._handle: return 0.0
        ms=_double();handle=self._handle;self._handle=None
        if _eclose(handle,C.byref(ms)): raise _error()
        return ms.value
