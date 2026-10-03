#!/usr/bin/env python3
"""Ordered MPP/RGA/FEAR pipeline with an independent asynchronous YOLO worker."""
from __future__ import annotations
import argparse
import collections
import csv
import importlib.util
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
from tracker_core import FEARTracker, clamp_bbox
from run import BoxKalman, center, iou, sha256
from yolo_post import postprocess
from infer import NativeFear, NativeYolo, validate_cores
spec = importlib.util.spec_from_file_location('auv_io', HERE/'io.py')
hw = importlib.util.module_from_spec(spec); spec.loader.exec_module(hw)


def pin_cpu(cpu):
    if cpu>=0:os.sched_setaffinity(0,{cpu})


def forbid_cpu_pixels(*args, **kwargs):
    raise RuntimeError('CPU colorspace conversion/resize is prohibited in this pipeline')


class RGATracker:
    def __init__(self, runtime):
        self.runtime = runtime
        self.decode_core = FEARTracker(None, smooth=False)
        self.bbox = None
        self.mean_color = (114,114,114)

    def initialize(self, frame, box):
        t = time.perf_counter()
        self.bbox = clamp_bbox(box, (frame.height, frame.width))
        thumbnail, _ = frame.crop((0,0,frame.width,frame.height),256)
        self.mean_color = thumbnail.mean(axis=(0,1))
        x,y,w,h = self.bbox
        context = np.asarray((x-w*.2,y-h*.2,w*1.4,h*1.4), dtype=np.int32)
        rgb, rga_ms = frame.crop(context,128,self.mean_color)
        timings = self.runtime.template(rgb)
        return (time.perf_counter()-t)*1000, rga_ms, timings

    def update(self, frame, prediction):
        t = time.perf_counter()
        self.bbox = clamp_bbox(prediction,(frame.height,frame.width))
        x,y,w,h = self.bbox
        context = np.asarray((x-w*2,y-h*2,w*5,h*5),dtype=np.int32)
        # Extremely large false detections can extend beyond RGA's 8192 limit.
        # Keep the same center and use the actual capped mapping for decode.
        for origin,extent,position,size in ((0,2,x,w),(1,3,y,h)):
            if context[extent]>8192:
                context[extent]=8192;context[origin]=int(position+size/2-4096)
        format_before = frame.format_ms
        rgb,rga_ms = frame.crop(context,256,self.mean_color)
        rga_ms = max(0.,rga_ms-(frame.format_ms-format_before))
        before = time.perf_counter()
        bbox_map, logits, timings = self.runtime.search(rgb)
        infer_ms = (time.perf_counter()-before)*1000
        before = time.perf_counter()
        box, score = self.decode_core.decode(bbox_map,logits)
        sx,sy = context[2]/256, context[3]/256
        box = [round(float(box[0]*sx+context[0])), round(float(box[1]*sy+context[1])),
               max(3,round(float(box[2]*sx))), max(3,round(float(box[3]*sy)))]
        box = clamp_bbox(box,(frame.height,frame.width))
        return box,score,{'fear_pre_ms':rga_ms, 'fear_api_ms':infer_ms,
                         'fear_npu_ms':timings.get('npu_ms'),
                         'fear_post_ms':(time.perf_counter()-before)*1000,
                         'fear_ms':(time.perf_counter()-t)*1000,
                         'fear_input_ms':timings.get('inputs_ms',0),
                         'fear_output_ms':timings.get('outputs_ms',0),
                         'fear_exp_ms':timings.get('cpu_exp_ms',0)}


from detector_workers import AsyncDetector as DetectorWorkers


class AsyncDetector(DetectorWorkers):
    def __init__(self,runtime,args,errors):
        super().__init__(runtime,args,errors,postprocess=postprocess)


class Fusion:
    def __init__(self,detector,tracker,args):
        self.detector,self.tracker,self.args=detector,tracker,args
        self.state='SEARCH';self.generation=0;self.kalman=None
        self.misses=self.detector_misses=0
        self.last_detection_box=None;self.last_detection_frame=-100000
        self.ever_locked=False;self.losses=self.recoveries=self.coast_recoveries=0
        self.history=collections.OrderedDict();self.measurements=collections.OrderedDict();self.last_yolo=-100000;self.last_request=-100000

    def step(self,frame,index):
        a=self.args
        row=dict(frame=index,state_before=self.state,state=self.state,source='NONE',reason='',
                 bbox=None,prediction=None,fear_bbox=None,fear_score=None,score=0.,event='',
                 misses=0,detector_misses=0,fear_ms=0.,fear_pre_ms=0.,fear_api_ms=0.,
                 fear_npu_ms=None,fear_post_ms=0.,fear_init_ms=0.,fear_calls=0,
                 fear_input_ms=0.,fear_output_ms=0.,fear_exp_ms=0.,kalman_ms=0.,
                 yolo_ms=0.,yolo_pre_ms=0.,yolo_api_ms=0.,yolo_npu_ms=None,yolo_post_ms=0.,
                 yolo_called=0,yolo_source_frame=None,yolo_age=None,yolo_candidates=[],
                 yolo_conf=None,yolo_distance_px=None,yolo_size_rejected=0,stale_rejected=0)
        if self.state=='LOST':self.state='SEARCH'
        prediction=fear_box=None
        if self.state=='TRACK':
            t=time.perf_counter();prediction=self.kalman.predict()
            row['kalman_ms']+=(time.perf_counter()-t)*1000
            box,score,timings=self.tracker.update(frame,prediction)
            row.update(timings,fear_bbox=box.tolist(),fear_score=score,prediction=prediction.tolist(),fear_calls=1)
            if score>=a.fear_threshold:fear_box=box
            if fear_box is None or self.detector_misses>=getattr(a,'yolo_suspect_checks',2) or index-max(self.last_yolo,self.last_request)>=a.yolo_interval:
                self.detector.submit(frame,index,self.generation,priority=True)
                self.last_request=index
        self.history[index]=(frame,prediction.copy() if prediction is not None else None)
        while len(self.history)>a.history_frames:self.history.popitem(last=False)
        if self.state=='SEARCH':
            self.detector.submit(frame,index,self.generation,priority=True)
            detections=self.detector.take(index,self.generation,wait=True)
        else:
            detections=self.detector.take(index,self.generation)
        selected=None;selected_result=None;selected_source_box=None
        for result in detections:
            age=index-result['index']
            if age>a.max_yolo_age or result['index'] not in self.history or (self.state=='TRACK' and result['index']<=self.last_yolo):
                row['stale_rejected']+=1;continue
            for k in ('yolo_ms','yolo_pre_ms','yolo_api_ms','yolo_npu_ms','yolo_post_ms'):
                if result[k] is not None: row[k]=(row[k] or 0.)+result[k]
            row.update(yolo_called=row['yolo_called']+1,yolo_source_frame=result['index'],yolo_age=age,
                       yolo_candidates=result['candidates'])
            candidates=result['candidates'];eligible=candidates
            if self.last_detection_box is not None and index-self.last_detection_frame<=a.size_memory_frames:
                area=float(np.prod(self.last_detection_box[2:]))
                eligible=[c for c in candidates if 1/a.size_ratio_limit<=np.prod(c['bbox'][2:])/max(area,1.)<=a.size_ratio_limit]
                row['yolo_size_rejected']+=len(candidates)-len(eligible)
            candidate=None
            source_pred=self.history[result['index']][1]
            if eligible:
                if self.state=='SEARCH':candidate=max(eligible,key=lambda c:c['score'])
                elif source_pred is not None:
                    candidate=min(eligible,key=lambda c:np.linalg.norm(center(c['bbox'])-center(source_pred)))
                    distance=float(np.linalg.norm(center(candidate['bbox'])-center(source_pred)))
                    gate=max(a.association_gate_px,a.association_gate_scale*np.linalg.norm(source_pred[2:]))
                    gate*=1+min(self.misses,a.max_misses)/a.max_misses
                    row['yolo_distance_px']=distance
                    if distance>gate:candidate=None
            self.last_yolo=max(self.last_yolo,result['index'])
            # Only a confident detection elsewhere counts against FEAR. An empty or weak YOLO
            # result is no evidence: the detector may not see the target at this size or angle,
            # and low-score hits on a night sky are often stars.
            if candidate:self.detector_misses=0
            elif any(c['score']>=getattr(a,'disagree_conf',.5) for c in eligible):self.detector_misses+=1
            if candidate:
                selected_source_box=np.asarray(candidate['bbox'],float)
                translated=selected_source_box.copy()
                if age and prediction is not None and source_pred is not None:
                    source_measurement=self.measurements.get(result['index'])
                    if fear_box is not None and source_measurement is not None:
                        translated[:2]+=center(fear_box)-center(source_measurement)
                    else:translated[:2]+=center(prediction)-center(source_pred)
                selected={**candidate,'bbox':translated};selected_result=result
            else:selected=selected_result=selected_source_box=None
        if selected is not None and self.state=='TRACK' and (self.misses or fear_box is None):
            source=selected_result['index']
            ms,_,_=self.tracker.initialize(selected_result['frame'],selected_source_box)
            row['fear_init_ms']+=ms;row['fear_ms']+=ms
            replay_box=selected_source_box.copy();replay_score=1.
            for replay_index in range(source+1,index+1):
                replay_frame,replay_prediction=self.history[replay_index]
                replay_box,replay_score,rt=self.tracker.update(replay_frame,replay_prediction if replay_prediction is not None else replay_box)
                row['fear_calls']+=1
                for key in ('fear_ms','fear_pre_ms','fear_api_ms','fear_npu_ms','fear_post_ms','fear_input_ms','fear_output_ms','fear_exp_ms'):
                    if rt.get(key) is not None:row[key]=(row[key] or 0.)+rt[key]
            if source<index and replay_score<a.fear_threshold:
                selected=None;self.detector_misses+=1
                row['reason']='replay_low_confidence'
            else:
                selected['bbox']=replay_box
        if selected is not None:
            box=np.asarray(selected['bbox'])
            row.update(source='YOLO',bbox=box.tolist(),score=selected['score'],yolo_conf=selected['score'])
            if self.state=='SEARCH':
                ms,_,_=self.tracker.initialize(frame,box)
                row['fear_init_ms']+=ms;row['fear_ms']+=ms
                t=time.perf_counter();self.kalman=BoxKalman(box)
                row['kalman_ms']+=(time.perf_counter()-t)*1000
                row['event']='REACQUIRED' if self.ever_locked else 'INITIAL_LOCK'
                if self.ever_locked:self.recoveries+=1
                self.ever_locked=True;self.generation+=1
                self.history.clear();self.history[index]=(frame,np.asarray(box).copy())
                row['reason']='global_detection'
            else:
                if self.misses or fear_box is None:
                    self.coast_recoveries+=1;row['event']='COAST_RECOVERY'
                    # Reliable YOLO recovery resets old velocity rather than carrying
                    # velocity estimated during a lost/occluded interval into the new lock.
                    t=time.perf_counter();self.kalman=BoxKalman(box)
                else:
                    row['event']='YOLO_CORRECTION';t=time.perf_counter();self.kalman.update(box)
                row['kalman_ms']+=(time.perf_counter()-t)*1000
                if fear_box is not None and not self.misses:
                    # FEAR's box scale follows its template. When YOLO's box is a very different
                    # size, the target has grown or shrunk since the template was taken: refresh it.
                    ratio=float(np.prod(selected_source_box[2:]))/max(float(np.prod(fear_box[2:])),1.)
                    limit=getattr(a,'template_refresh_ratio',1.5)
                    if not 1/limit<=ratio<=limit:
                        ms,_,_=self.tracker.initialize(selected_result['frame'],selected_source_box)
                        row['fear_init_ms']+=ms;row['fear_ms']+=ms;row['event']='TEMPLATE_REFRESH'
                row['reason']='low_score_recovery' if fear_box is None else 'periodic_correction'
            self.tracker.bbox=clamp_bbox(box,(frame.height,frame.width))
            self.last_detection_box=box.copy();self.last_detection_frame=index
            self.misses=0;self.state='TRACK'
        elif self.state=='TRACK':
            if fear_box is not None and self.detector_misses<a.detector_miss_limit:
                t=time.perf_counter();self.kalman.update(fear_box)
                row['kalman_ms']+=(time.perf_counter()-t)*1000
                row.update(source='FEAR',bbox=fear_box.tolist(),score=row['fear_score'],reason='high_fear_score')
                self.misses=0
            else:
                self.misses+=1
                row['reason']='low_fear_score' if fear_box is None else 'detector_disagrees'
                if self.misses<=a.coast_frames:
                    row.update(source='KALMAN',bbox=clamp_bbox(prediction,(frame.height,frame.width)).tolist())
                if self.misses>=a.max_misses or (fear_box is not None and self.detector_misses>=a.detector_miss_limit):
                    self.state='LOST';self.losses+=1;self.kalman=None;self.generation+=1
                    row.update(source='NONE',bbox=None,event='LOST')
        else:row['reason']='no_detection'
        row.update(state=self.state,misses=self.misses,detector_misses=self.detector_misses)
        self.measurements[index]=None if row['bbox'] is None else np.asarray(row['bbox'],float)
        while len(self.measurements)>a.history_frames:self.measurements.popitem(last=False)
        return row


def draw(frame,row,fps,debug):
    rgb=frame.rgb()
    if row['bbox'] is not None:
        colors={'YOLO':(255,210,0),'FEAR':(60,240,70),'KALMAN':(30,120,255)}
        color=colors[row['source']] if debug else (40,240,80)
        x,y,w,h=map(round,row['bbox'])
        cv2.rectangle(rgb,(x,y),(x+w,y+h),color,2)
        cv2.putText(rgb,f"cls=0 score={row['score']:.2f}",(x,max(22,y-8)),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
    label=f"{row['state']} FPS {fps:.1f}"
    if debug:label+=f" / {row['source']} frame {row['frame']}"
    cv2.putText(rgb,label,(12,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)


def run(a):
    validate_cores(a.fear_core,a.yolo_core)
    if a.yolo_workers==2:validate_cores(a.fear_core,'2')
    if min(a.decode_queue,a.encode_queue,a.yolo_queue,a.history_frames)<1 or a.max_yolo_age>=a.history_frames:
        raise ValueError('Queues must be positive and history_frames greater than max_yolo_age')
    a.no_encode=a.no_encode or a.no_video
    cv2.setNumThreads(1)
    cv2.cvtColor=forbid_cpu_pixels;cv2.resize=forbid_cpu_pixels
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    gt=json.loads(Path(a.gt).read_text()) if a.gt else None
    runtime=NativeFear(a.template_model,a.search_model,core=a.fear_core,query_perf=True)
    yolos=[]
    try:
        yolos.append(NativeYolo(a.yolo_model,core=a.yolo_core,query_perf=True))
        if a.yolo_workers==2:yolos.append(NativeYolo(a.yolo_model,core='2',query_perf=True))
    except BaseException:
        for context in yolos:context.close()
        runtime.close();raise
    errors=queue.Queue();detector=AsyncDetector(yolos,a,errors)
    tracker=RGATracker(runtime);fusion=Fusion(detector,tracker,a)
    inputs=queue.Queue(a.decode_queue);outputs=queue.Queue(a.encode_queue)
    rows=[];decoder=hw.Decoder(a.video);encoder=None
    stop=threading.Event();flush=[0.];encoded=[0];stage_cpu={};start_unix=time.time();start=time.perf_counter()

    def put(q,item):
        while not stop.is_set():
            try:q.put(item,timeout=.2);return
            except queue.Full:pass
    def decode_worker():
        cpu_start=time.thread_time()
        try:
            pin_cpu(a.decode_cpu)
            index=0
            while not stop.is_set() and (a.max_frames==0 or index<a.max_frames):
                t=time.perf_counter();frame=decoder.read();ms=(time.perf_counter()-t)*1000
                if frame is None:break
                if index%a.yolo_interval==0 or fusion.state!='TRACK':
                    detector.submit(frame,index,fusion.generation)
                wait=time.perf_counter();put(inputs,(frame,index,ms));index+=1
            put(inputs,None)
        except BaseException as e:errors.put(e);stop.set()
        finally:stage_cpu['decode']=time.thread_time()-cpu_start
    def track_worker():
        cpu_start=time.thread_time()
        try:
            pin_cpu(a.track_cpu)
            while not stop.is_set():
                t=time.perf_counter()
                try:item=inputs.get(timeout=.2)
                except queue.Empty:continue
                wait_ms=(time.perf_counter()-t)*1000
                if item is None:break
                frame,index,decode_ms=item
                t=time.perf_counter();row=fusion.step(frame,index)
                row.update(track_ms=(time.perf_counter()-t)*1000,decode_ms=decode_ms,
                           format_ms=frame.format_ms,track_queue_wait_ms=wait_ms,
                           draw_ms=0.,encode_ms=0.,encode_convert_ms=0.,gt_exists=None,gt_valid=None,gt_iou=None,
                           gt_center_error_px=None,kalman_prediction_error_px=None)
                if gt is not None and index<len(gt['gt_rect']):
                    exists=bool(gt['exist'][index]);target=gt['gt_rect'][index]
                    valid=exists and len(target)==4 and target[2]>0 and target[3]>0
                    row.update(gt_exists=int(exists),gt_valid=int(valid))
                    if valid:
                        row['gt_iou']=iou(row['bbox'],target) if row['bbox'] is not None else 0.
                        if row['bbox'] is not None:row['gt_center_error_px']=float(np.linalg.norm(center(row['bbox'])-center(target)))
                        if row['prediction'] is not None:row['kalman_prediction_error_px']=float(np.linalg.norm(center(row['prediction'])-center(target)))
                rows.append(row)
                put(outputs,(frame,row))
            put(outputs,None)
        except BaseException as e:errors.put(e);stop.set()
        finally:stage_cpu['track']=time.thread_time()-cpu_start
    def encode_worker():
        nonlocal encoder
        cpu_start=time.thread_time()
        try:
            pin_cpu(a.encode_cpu)
            rolling=collections.deque(maxlen=30);last=time.perf_counter()
            while not stop.is_set():
                try:item=outputs.get(timeout=.2)
                except queue.Empty:continue
                if item is None:break
                frame,row=item
                if not a.no_encode:
                    if encoder is None:encoder=hw.Encoder(output,frame.width,frame.height,a.source_fps,a.bitrate)
                    t=time.perf_counter()
                    display_frame=frame.clone_for_draw()
                    draw(display_frame,row,len(rolling)/sum(rolling) if rolling else 0.,a.debug)
                    row['draw_ms']=(time.perf_counter()-t)*1000
                    row['format_ms']=frame.format_ms
                    convert,submit=encoder.write(display_frame)
                    row['encode_convert_ms']=convert;row['encode_ms']=submit
                encoded[0]+=1;now=time.perf_counter();rolling.append(now-last);last=now
                if encoded[0]%100==0:
                    print(json.dumps({'frames':encoded[0],'fps':encoded[0]/(now-start),'state':row['state']}),flush=True)
            if encoder is not None:flush[0]=encoder.close();encoder=None
        except BaseException as e:errors.put(e);stop.set()
        finally:stage_cpu['encode']=time.thread_time()-cpu_start
    threads=[threading.Thread(target=fn,name=name,daemon=True) for fn,name in
             ((decode_worker,'MPP decode'),(track_worker,'FEAR Kalman'),(encode_worker,'draw MPP encode'))]
    try:
        for thread in threads:thread.start()
        for thread in threads:
            while thread.is_alive():
                thread.join(.2)
                if not errors.empty():raise errors.get()
        detector.close()
        seconds=time.perf_counter()-start
        if not errors.empty():raise errors.get()
        if not rows:raise RuntimeError('No frames processed')
        fields=[k for k in rows[0] if k not in ('bbox','prediction','fear_bbox')]
        fields += [f'{p}_{axis}' for p in ('box','prediction','fear_box') for axis in ('x','y','w','h')]
        with output.with_suffix('.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
            for row in rows:
                serial={k:v for k,v in row.items() if k in fields}
                serial['yolo_candidates']=json.dumps(serial['yolo_candidates'],separators=(',',':'))
                for key,prefix in (('bbox','box'),('prediction','prediction'),('fear_bbox','fear_box')):
                    for j,axis in enumerate(('x','y','w','h')):serial[f'{prefix}_{axis}']=row[key][j] if row[key] is not None else None
                writer.writerow(serial)
        timing_keys=('decode_ms','format_ms','yolo_ms','yolo_pre_ms','yolo_api_ms','yolo_npu_ms','yolo_post_ms',
                     'fear_ms','fear_pre_ms','fear_api_ms','fear_npu_ms','fear_post_ms','fear_init_ms',
                     'kalman_ms','track_ms','draw_ms','encode_convert_ms','encode_ms','track_queue_wait_ms')
        means={k:sum(float(r.get(k) or 0.) for r in rows)/len(rows) for k in timing_keys}
        valid=[r for r in rows if r['gt_valid']]
        calls=detector.calls.copy()
        stage_cpu['yolo']=detector.cpu_seconds
        summary=dict(frames=len(rows),encoded_frames=encoded[0],end_to_end_fps=len(rows)/seconds,
                     yolo_workers=a.yolo_workers,yolo_calls_by_worker=detector.calls_by_worker,
                     yolo_cpu_seconds_by_worker=detector.cpu_by_worker,npu_core_masks={'fear':2,'yolo':[1,4] if a.yolo_workers==2 else [1]},
                     worker_cpu_seconds=stage_cpu,worker_cpu_utilization_percent={k:v/seconds*100 for k,v in stage_cpu.items()},
                     end_to_end_seconds=seconds,measurement_start_unix=start_unix,source_fps=a.source_fps,average_ms_per_frame=means,
                     encoder_flush_ms=flush[0],yolo_calls=len(calls),yolo_dropped_pending=detector.dropped,
                     yolo_actual_call_ms=float(np.mean([r['yolo_ms'] for r in calls])) if calls else None,
                     yolo_actual_pre_ms=float(np.mean([r['yolo_pre_ms'] for r in calls])) if calls else None,
                     yolo_actual_post_ms=float(np.mean([r['yolo_post_ms'] for r in calls])) if calls else None,
                     yolo_worker_ms_per_frame={k:sum(float(r.get(k) or 0.) for r in calls)/len(rows) for k in ('yolo_ms','yolo_pre_ms','yolo_api_ms','yolo_npu_ms','yolo_post_ms')},
                     fear_calls=sum(r['fear_calls'] for r in rows),
                     fear_actual_npu_ms=sum(float(r.get('fear_npu_ms') or 0.) for r in rows)/max(1,sum(r['fear_calls'] for r in rows)),
                     yolo_actual_api_ms=float(np.mean([r['yolo_api_ms'] for r in calls])) if calls else None,
                     yolo_actual_npu_ms=float(np.mean([r['yolo_npu_ms'] for r in calls if r['yolo_npu_ms'] is not None])) if any(r['yolo_npu_ms'] is not None for r in calls) else None,
                     lost_transitions=fusion.losses,search_reacquisitions=fusion.recoveries,coast_recoveries=fusion.coast_recoveries,
                     gt_valid_frames=len(valid),gt_mean_iou=float(np.mean([r['gt_iou'] for r in valid])) if valid else None,
                     gt_iou_ge_0_5=sum(r['gt_iou']>=.5 for r in valid)/len(valid) if valid else None,
                     gt_visible_missing_frames=sum(r['bbox'] is None for r in valid),
                     source_counts=dict(collections.Counter(r['source'] for r in rows)),config=vars(a),
                     cpu_colorspace_or_resize_calls=0,
                     model_sha256={p:sha256(p) for p in (a.yolo_model,a.template_model,a.search_model)},
                     timing_note='Thread busy wall times overlap: do not sum to derive FPS. Decode timer includes appsink wait; encode is appsrc submission/backpressure plus separate EOS flush. format_ms is RGA NV12→RGB; encode_convert_ms is RGA RGB→NV12. No CPU colorspace/resize; cv2 functions are disabled. MPP DMA→RGA frame path, but small RGB NPU tensors are copied into host arrays and CAPI inputs_set: partial zero-copy, not end-to-end zero-copy. FEAR sequential; only YOLO independent. Ground truth evaluation-only.')
        output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2))
        output.with_suffix('.yolo_calls.json').write_text(json.dumps(calls,indent=2))
        print(json.dumps(summary,indent=2),flush=True)
        return summary
    finally:
        stop.set();detector.close()
        # Native handles outlive every worker, including failure/cancellation paths.
        for thread in threads:
            if thread.ident is not None:thread.join()
        decoder.close()
        if encoder is not None:encoder.close()
        runtime.close()
        for context in yolos:context.close()


def parser():
    from run import parser as old_parser
    p=old_parser()
    p.description=__doc__
    p.set_defaults(yolo_core='0')
    p._option_string_actions['--yolo-core'].choices=('0',)
    p.add_argument('--source-fps',type=float,default=20.)
    p.add_argument('--decode-queue',type=int,default=8)
    p.add_argument('--encode-queue',type=int,default=4)
    p.add_argument('--yolo-queue',type=int,default=2)
    p.add_argument('--yolo-workers',type=int,choices=(1,2),default=2,help='Separate YOLO contexts on cores 0 and 2; FEAR core 1')
    p.add_argument('--yolo-second-cpu',type=int,default=5)
    p.add_argument('--max-yolo-age',type=int,default=4)
    p.add_argument('--yolo-suspect-checks',type=int,default=2,help='Run YOLO every frame after this many unmatched checks')
    p.add_argument('--disagree-conf',type=float,default=.5,help='YOLO score needed for a detection elsewhere to count against FEAR')
    p.add_argument('--template-refresh-ratio',type=float,default=1.5,help='Refresh the FEAR template when YOLO and FEAR box areas differ by more than this')
    p.add_argument('--history-frames',type=int,default=8)
    p.add_argument('--bitrate',type=int,default=6000000)
    for stage,cpu in (('decode',5),('track',6),('yolo',7),('encode',4)):
        p.add_argument('--'+stage+'-cpu',type=int,default=cpu,help='Linux CPU affinity; -1 disables')
    p.add_argument('--debug',action='store_true')
    p.add_argument('--no-encode',action='store_true')
    return p


if __name__=='__main__':run(parser().parse_args())
