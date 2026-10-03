"""One or two independently owned YOLO contexts sharing a bounded job queue.

Only detector calls run in parallel. FEAR remains owned by the pipeline's one
tracking thread. Context destruction belongs to the caller after close joins
every worker. Use separate NativeYolo contexts on core 0 and core 2.
"""
from __future__ import annotations
import collections
import os
import threading
import time
from yolo_post import postprocess as default_postprocess


class AsyncDetector:
    def __init__(self,runtime,args,errors,*,postprocess=None):
        self.runtimes=list(runtime) if isinstance(runtime,(tuple,list)) else [runtime]
        if not 1<=len(self.runtimes)<=2:
            raise ValueError('Use one or two independent YOLO contexts')
        if len({id(item) for item in self.runtimes})!=len(self.runtimes):
            raise ValueError('Each YOLO worker requires a distinct runtime context')
        self.runtime=self.runtimes[0]  # single-worker compatibility
        self.args,self.errors=args,errors
        self.postprocess=postprocess or default_postprocess
        self.cv=threading.Condition()
        self.pending=collections.OrderedDict();self.pending_priority={}
        self.results={};self.inflight=set();self.stop=False
        self.calls=[];self.dropped=0;self.cpu_seconds=0.
        self.cpu_by_worker=[0. for _ in self.runtimes]
        self.calls_by_worker=[0 for _ in self.runtimes]
        self.threads=[threading.Thread(target=self.work,args=(index,),
                      name='YOLO' if index==0 else 'YOLO-2',daemon=True)
                      for index in range(len(self.runtimes))]
        self.thread=self.threads[0]
        for thread in self.threads:thread.start()

    def submit(self,frame,index,generation,priority=False):
        key=(generation,index)
        with self.cv:
            if self.stop:return
            if key in self.pending:
                if priority:
                    self.pending_priority[key]=True
                    self.pending.move_to_end(key,last=False)
                return
            if key in self.results or key in self.inflight:return
            self.pending[key]=frame;self.pending_priority[key]=priority
            if priority:self.pending.move_to_end(key,last=False)
            while len(self.pending)>self.args.yolo_queue:
                normal=next((key for key in self.pending if not self.pending_priority[key]),None)
                # Preserve an urgent SEARCH waiter against ordinary lookahead.
                # Multiple tracking urgents can coalesce to their newest two.
                if normal is None:normal=next(reversed(self.pending))
                del self.pending[normal];self.pending_priority.pop(normal)
                self.dropped+=1
            self.cv.notify_all()

    def work(self,worker):
        cpu_start=time.thread_time()
        try:
            cpu=getattr(self.args,'yolo_cpu',-1) if worker==0 else getattr(self.args,'yolo_second_cpu',-1)
            if cpu>=0:os.sched_setaffinity(0,{cpu})
            runtime=self.runtimes[worker]
            while True:
                with self.cv:
                    self.cv.wait_for(lambda:self.pending or self.stop)
                    if self.stop:return
                    key,frame=self.pending.popitem(last=False)
                    self.pending_priority.pop(key);self.inflight.add(key)
                begin=time.perf_counter()
                image,scale,left,top,pre_ms=frame.letterbox()
                outputs,timing=runtime.infer(image)
                start=time.perf_counter()
                candidates=self.postprocess(outputs,(scale,left,top,frame.width,frame.height),self.args.yolo_conf)
                post_ms=(time.perf_counter()-start)*1000
                result=dict(frame=frame,index=key[1],generation=key[0],worker=worker,
                            candidates=candidates,yolo_pre_ms=pre_ms,
                            yolo_api_ms=timing.get('total_ms',0),yolo_npu_ms=timing.get('npu_ms'),
                            yolo_post_ms=post_ms,yolo_ms=(time.perf_counter()-begin)*1000)
                with self.cv:
                    self.calls.append({key:value for key,value in result.items() if key!='frame'})
                    self.calls_by_worker[worker]+=1
                    self.inflight.discard(key)
                    if not self.stop:
                        self.results[key]=result
                        while len(self.results)>self.args.history_frames*3:
                            self.results.pop(next(iter(self.results)))
                    self.cv.notify_all()
        except BaseException as error:
            self.errors.put(error)
            with self.cv:self.stop=True;self.cv.notify_all()
        finally:
            with self.cv:
                self.cpu_by_worker[worker]=time.thread_time()-cpu_start
                self.cpu_seconds=sum(self.cpu_by_worker)
                self.cv.notify_all()

    def take(self,index,generation,wait=False):
        key=(generation,index)
        with self.cv:
            if wait:
                self.cv.wait_for(lambda:key in self.results or self.stop,timeout=30)
                if key not in self.results:raise RuntimeError('YOLO SEARCH job failed or timed out')
                return [self.results.pop(key)]
            keys=sorted(key for key in self.results if key[0]==generation and key[1]<=index)
            return [self.results.pop(key) for key in keys]

    def close(self):
        with self.cv:
            self.stop=True;self.pending.clear();self.pending_priority.clear();self.results.clear()
            self.cv.notify_all()
        for thread in self.threads:
            thread.join()
