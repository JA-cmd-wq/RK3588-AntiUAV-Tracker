#!/usr/bin/env python3
"""Board-only deterministic fake tests: no NPU, model inference, or video IO.

Run with the board's existing Python/NumPy/OpenCV environment. The native IO
module is imported but no device function is called. Blocking fake frames make
queue races reproducible, rather than relying on detector timing or hardware.
"""
from __future__ import annotations
import collections
import contextlib
import copy
import io
import json
import queue
import sys
import threading
import time
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import pipeline as p
from detector_workers import AsyncDetector as MultiDetector


def args(**changes):
    values=dict(yolo_queue=2,history_frames=8,yolo_conf=.25,fear_threshold=.7,
                yolo_interval=15,max_yolo_age=4,size_memory_frames=120,
                size_ratio_limit=3.,association_gate_px=120.,association_gate_scale=2.,
                max_misses=15,coast_frames=5,detector_miss_limit=3,yolo_suspect_checks=2,
                yolo_cpu=-1,yolo_second_cpu=-1,yolo_workers=1)
    values.update(changes)
    return SimpleNamespace(**values)


class FakeFrame:
    def __init__(self,index,width=320,height=180):
        self.index=index;self.width=width;self.height=height;self.format_ms=0.
        self.image=np.full((height,width,3),index%200,np.uint8)
    def rgb(self):return self.image
    def clone_for_draw(self):
        clone=FakeFrame(self.index,self.width,self.height);clone.image=self.image.copy()
        return clone
    def letterbox(self):return np.zeros((640,640,3),np.uint8),1.,0,0,0.


class BlockingFrame(FakeFrame):
    def __init__(self,index):
        super().__init__(index);self.entered=threading.Event();self.release=threading.Event()
    def letterbox(self):
        self.entered.set()
        if not self.release.wait(2):raise RuntimeError('Test failed to release fake detector')
        return super().letterbox()


class FakeRuntime:
    def infer(self,_):return [],{'total_ms':0.,'npu_ms':None}


class ScriptDetector:
    def __init__(self):self.results=[];self.submissions=[]
    def submit(self,frame,index,generation,priority=False):
        self.submissions.append((frame,index,generation,priority))
    def take(self,index,generation,wait=False):
        accepted=[r for r in self.results if r['generation']==generation and r['index']<=index]
        self.results=[r for r in self.results if r not in accepted]
        return accepted
    def result(self,frame,box=None,generation=1,score=.95):
        self.results.append(dict(frame=frame,index=frame.index,generation=generation,
                                 candidates=[] if box is None else [{'bbox':list(box),'score':score}],
                                 yolo_ms=0.,yolo_pre_ms=0.,yolo_api_ms=0.,yolo_npu_ms=None,yolo_post_ms=0.))


class ScriptTracker:
    def __init__(self,score=.99):
        self.score=score;self.replay_score=None;self.initialized=False
        self.bbox=None;self.calls=[];self.active=False
    def initialize(self,frame,box):
        if self.active:raise AssertionError('Parallel FEAR initialize')
        self.active=True
        self.calls.append(('init',frame.index,np.asarray(box,float).copy(),threading.get_ident()))
        self.bbox=np.asarray(box,float).copy();self.active=False;self.initialized=True
        return 0.,0.,{}
    def update(self,frame,prediction):
        if self.active:raise AssertionError('Parallel FEAR update')
        self.active=True
        self.calls.append(('update',frame.index,np.asarray(prediction,float).copy(),threading.get_ident()))
        box=np.asarray(prediction,float).copy();self.active=False
        score=self.replay_score if self.initialized and self.replay_score is not None else self.score
        return box,score,{'fear_ms':0.,'fear_pre_ms':0.,'fear_api_ms':0.,'fear_npu_ms':None,'fear_post_ms':0.}


def tracking_fusion(score=.99,**changes):
    detector=ScriptDetector();tracker=ScriptTracker(score)
    fusion=p.Fusion(detector,tracker,args(**changes));fusion.state='TRACK';fusion.generation=1
    fusion.ever_locked=True;fusion.kalman=p.BoxKalman([40,60,20,10])
    fusion.kalman.x[4:6]=[10.,0.]
    fusion.last_yolo=-1
    return fusion,detector,tracker


class AsyncQueueTests(unittest.TestCase):
    def make_worker(self):
        errors=queue.Queue();worker=p.AsyncDetector(FakeRuntime(),args(),errors)
        frame=BlockingFrame(0);worker.submit(frame,0,0)
        self.assertTrue(frame.entered.wait(1),'Fake YOLO worker did not start')
        return worker,frame,errors

    def test_decode_lookahead_cannot_evict_pending_search(self):
        with patch.object(p,'postprocess',return_value=[]):
            worker,blocking,errors=self.make_worker()
            try:
                worker.submit(FakeFrame(1),1,0,priority=True)
                worker.submit(FakeFrame(2),2,0)
                worker.submit(FakeFrame(3),3,0)
                with worker.cv:
                    self.assertIn((0,1),worker.pending,'Ordinary decode lookahead evicted SEARCH request; take(wait=True) will timeout')
                    self.assertLessEqual(len(worker.pending),2)
            finally:blocking.release.set();worker.close()
            self.assertTrue(errors.empty())

    def test_pending_search_is_protected_from_long_lookahead_flood(self):
        with patch.object(p,'postprocess',return_value=[]):
            worker,blocking,errors=self.make_worker()
            try:
                worker.submit(FakeFrame(1),1,0,priority=True)
                for index in range(2,30):worker.submit(FakeFrame(index),index,0)
                with worker.cv:self.assertIn((0,1),worker.pending)
            finally:blocking.release.set();worker.close()
            self.assertTrue(errors.empty())

    def test_existing_normal_request_can_upgrade_to_urgent(self):
        with patch.object(p,'postprocess',return_value=[]):
            worker,blocking,errors=self.make_worker()
            try:
                frame=FakeFrame(1);worker.submit(frame,1,0)
                worker.submit(frame,1,0,priority=True)
                for index in range(2,8):worker.submit(FakeFrame(index),index,0)
                with worker.cv:
                    self.assertIn((0,1),worker.pending)
                    self.assertTrue(worker.pending_priority[(0,1)])
                    self.assertEqual(len(worker.pending),2)
            finally:blocking.release.set();worker.close()
            self.assertTrue(errors.empty())

    def test_urgent_flood_coalesces_to_recent_requests(self):
        with patch.object(p,'postprocess',return_value=[]):
            worker,blocking,errors=self.make_worker()
            try:
                for index in range(1,10):worker.submit(FakeFrame(index),index,0,priority=True)
                with worker.cv:
                    self.assertEqual(set(worker.pending),{(0,8),(0,9)})
                    self.assertEqual(worker.dropped,7)
            finally:blocking.release.set();worker.close()
            self.assertTrue(errors.empty())

    def test_generation_filter_and_take_consume_result_once(self):
        with patch.object(p,'postprocess',return_value=[]):
            worker,blocking,errors=self.make_worker()
            try:
                with worker.cv:
                    worker.results[(0,4)]={'index':4,'generation':0}
                    worker.results[(1,5)]={'index':5,'generation':1}
                self.assertEqual(worker.take(5,1),[{'index':5,'generation':1}])
                self.assertEqual(worker.take(5,1),[])
                self.assertEqual(worker.take(3,0),[])
            finally:blocking.release.set();worker.close()
            self.assertTrue(errors.empty())

    def test_close_unblocks_search_wait(self):
        with patch.object(p,'postprocess',return_value=[]):
            worker,blocking,errors=self.make_worker();caught=[]
            waiter=threading.Thread(target=lambda:self._capture_search(worker,caught),daemon=True)
            waiter.start()
            blocking.release.set();worker.close();waiter.join(1)
            self.assertFalse(waiter.is_alive(),'SEARCH wait did not wake on detector shutdown')
            self.assertEqual(len(caught),1)
            self.assertIsInstance(caught[0],RuntimeError)

    def test_detector_error_wakes_waiter_and_is_reported(self):
        class BrokenRuntime:
            def infer(self,_):raise ValueError('Injected fake detector failure')
        errors=queue.Queue()
        with patch.object(p,'postprocess',return_value=[]):
            worker=p.AsyncDetector(BrokenRuntime(),args(),errors)
            try:
                worker.submit(FakeFrame(0),0,0)
                with self.assertRaises(RuntimeError):worker.take(0,0,wait=True)
                self.assertIsInstance(errors.get(timeout=1),ValueError)
            finally:worker.close()

    @staticmethod
    def _capture_search(worker,caught):
        try:worker.take(100,0,wait=True)
        except BaseException as e:caught.append(e)


class FusionChronologyTests(unittest.TestCase):
    def test_source_history_association_and_motion_transport(self):
        # Candidate is on the source prediction, while the current prediction
        # is beyond a deliberately small gate; association must use source time.
        fusion,detector,tracker=tracking_fusion(association_gate_px=5.,association_gate_scale=.01)
        f0=FakeFrame(0);source=np.array([20.,60.,20.,10.])
        fusion.history[0]=(f0,source.copy())
        detector.result(f0,source)
        row=fusion.step(FakeFrame(1),1)
        self.assertEqual(row['source'],'YOLO')
        self.assertEqual(row['yolo_age'],1)
        self.assertEqual(row['yolo_distance_px'],0.)
        np.testing.assert_allclose(row['bbox'],row['prediction'])
        self.assertFalse(any(c[0]=='init' for c in tracker.calls),'Periodic correction reinitialized template')

    def test_source_not_current_location_selects_nearest_candidate(self):
        fusion,detector,_=tracking_fusion(association_gate_px=200.)
        f0=FakeFrame(0);source=np.array([20.,60.,20.,10.]);fusion.history[0]=(f0,source.copy())
        detector.result(f0,source,score=.7)
        detector.results[0]['candidates'].append({'bbox':[50,60,20,10],'score':.99})
        row=fusion.step(FakeFrame(1),1)
        self.assertEqual(row['yolo_conf'],.7)

    def test_measured_motion_overrides_wrong_kalman_displacement(self):
        fusion,detector,tracker=tracking_fusion(association_gate_px=200.)
        source=FakeFrame(0);source_prediction=np.array([150.,60.,20.,10.])
        fusion.history[0]=(source,source_prediction)
        # Past final measurement was x=20; current FEAR is x=70. Kalman
        # prediction displacement is negative, representing a camera turn.
        fusion.measurements[0]=np.array([20.,60.,20.,10.])
        original_update=tracker.update
        def update(frame,prediction):
            box,score,timings=original_update(frame,prediction)
            return np.array([70.,60.,20.,10.]),score,timings
        tracker.update=update
        detector.result(source,[25,60,20,10])
        row=fusion.step(FakeFrame(1),1)
        self.assertEqual(row['source'],'YOLO')
        np.testing.assert_allclose(row['bbox'],[75,60,20,10])
        self.assertNotEqual(row['bbox'][0],-75.,'Old box transported using the wrong current/source prediction difference')

    def test_too_old_detection_cannot_change_state_or_template(self):
        fusion,detector,tracker=tracking_fusion(max_yolo_age=2)
        f0=FakeFrame(0);fusion.history[0]=(f0,np.array([40.,60.,20.,10.]))
        detector.result(f0,[40,60,20,10]);row=fusion.step(FakeFrame(3),3)
        self.assertEqual(row['stale_rejected'],1);self.assertEqual(row['source'],'FEAR')
        self.assertEqual(fusion.detector_misses,0);self.assertEqual(len(tracker.calls),1)

    def test_missing_source_history_rejects_result(self):
        fusion,detector,_=tracking_fusion()
        detector.result(FakeFrame(0),[40,60,20,10]);row=fusion.step(FakeFrame(1),1)
        self.assertEqual(row['stale_rejected'],1);self.assertEqual(row['source'],'FEAR')

    def test_negative_detection_counted_once_then_not_per_frame(self):
        fusion,detector,_=tracking_fusion()
        f0=FakeFrame(0);fusion.history[0]=(f0,np.array([40.,60.,20.,10.]))
        detector.result(f0,None)
        fusion.step(FakeFrame(1),1);self.assertEqual(fusion.detector_misses,1)
        fusion.step(FakeFrame(2),2);self.assertEqual(fusion.detector_misses,1)

    def test_previous_generation_negative_cannot_veto_current_track(self):
        fusion,detector,_=tracking_fusion()
        f0=FakeFrame(0);fusion.history[0]=(f0,np.array([40.,60.,20.,10.]))
        detector.result(f0,None,generation=0)
        row=fusion.step(FakeFrame(1),1)
        self.assertEqual(fusion.detector_misses,0);self.assertEqual(row['source'],'FEAR')

    def test_out_of_order_negative_older_than_applied_result_cannot_count_miss(self):
        fusion,detector,_=tracking_fusion()
        fusion.last_yolo=5
        f4=FakeFrame(4);fusion.history[4]=(f4,np.array([40.,60.,20.,10.]))
        detector.result(f4,None)
        row=fusion.step(FakeFrame(6),6)
        self.assertEqual(fusion.detector_misses,0,'Late negative replaced a newer accepted detector confirmation')
        self.assertEqual(row['source'],'FEAR');self.assertEqual(row['stale_rejected'],1)

    def test_out_of_order_positive_cannot_override_newer_applied_detection(self):
        fusion,detector,_=tracking_fusion()
        fusion.last_yolo=5
        f4=FakeFrame(4);fusion.history[4]=(f4,np.array([40.,60.,20.,10.]))
        detector.result(f4,[80,60,20,10])
        row=fusion.step(FakeFrame(6),6)
        self.assertEqual(row['source'],'FEAR');self.assertNotEqual(row['event'],'YOLO_CORRECTION')
        self.assertEqual(row['stale_rejected'],1)

    def test_newer_negative_in_same_batch_cancels_older_positive_correction(self):
        fusion,detector,_=tracking_fusion()
        first,second=FakeFrame(1),FakeFrame(2)
        fusion.history[1]=(first,np.array([40.,60.,20.,10.]))
        detector.result(first,[40,60,20,10]);detector.result(second,None)
        row=fusion.step(second,2)
        self.assertEqual(row['source'],'FEAR','Older positive remained selected after a newer unmatched detector result')
        self.assertNotEqual(row['event'],'YOLO_CORRECTION')
        self.assertEqual(fusion.last_yolo,2);self.assertEqual(fusion.detector_misses,1)

    def test_low_score_recovery_initializes_source_and_replays_in_order(self):
        fusion,detector,tracker=tracking_fusion(score=.2)
        tracker.replay_score=.9
        for index in range(3):fusion.history[index]=(FakeFrame(index),np.array([40.+index*10,60,20,10]))
        detector.result(fusion.history[0][0],[40,60,20,10])
        row=fusion.step(FakeFrame(3),3)
        calls=[(c[0],c[1]) for c in tracker.calls]
        self.assertEqual(calls,[('update',3),('init',0),('update',1),('update',2),('update',3)])
        self.assertEqual(len({c[3] for c in tracker.calls}),1,'Replay and normal tracking used different threads')
        self.assertEqual(row['event'],'COAST_RECOVERY')

    def test_low_confidence_replay_cannot_claim_recovery(self):
        fusion,detector,tracker=tracking_fusion(score=.2)
        for index in range(3):fusion.history[index]=(FakeFrame(index),np.array([40.+index*10,60,20,10]))
        detector.result(fusion.history[0][0],[40,60,20,10])
        row=fusion.step(FakeFrame(3),3)
        self.assertNotEqual(row['event'],'COAST_RECOVERY','Low-confidence replay claimed a successful current-frame recovery')
        self.assertEqual(fusion.coast_recoveries,0)
        self.assertEqual(row['source'],'KALMAN')

    def test_current_frame_detection_can_recover_without_replay(self):
        fusion,detector,tracker=tracking_fusion(score=.2)
        current=FakeFrame(1);detector.result(current,[50,60,20,10])
        row=fusion.step(current,1)
        self.assertEqual(row['source'],'YOLO');self.assertEqual(row['event'],'COAST_RECOVERY')
        self.assertEqual([(c[0],c[1]) for c in tracker.calls],[('update',1),('init',1)])

    def test_loss_changes_generation(self):
        fusion,detector,_=tracking_fusion(score=.2,max_misses=2,coast_frames=1)
        fusion.step(FakeFrame(1),1);row=fusion.step(FakeFrame(2),2)
        self.assertEqual(row['state'],'LOST');self.assertEqual(fusion.generation,2)

    def test_draw_on_clone_keeps_detection_source_pixels_immutable(self):
        source=FakeFrame(10);snapshot=source.image.copy();clone=source.clone_for_draw()
        p.draw(clone,dict(bbox=[40,40,30,20],source='FEAR',score=.9,state='TRACK',frame=10),99.,False)
        self.assertFalse(np.array_equal(clone.image,snapshot),'Fake draw did not change the clone')
        np.testing.assert_array_equal(source.image,snapshot)


class PipelineThreadTests(unittest.TestCase):
    def run_fake(self,directory,count=12,broken_encoder=False,slow_track=False):
        frames=[FakeFrame(i) for i in range(count)];encoded=[];closed=[]
        class Runtime(FakeRuntime):
            def __init__(self,*_,**kwargs):
                self.core=str(kwargs.get('core','0'));self.description={'core_mask':1<<int(self.core),'backend':'fake'}
            def close(self):closed.append('runtime')
        class Decoder:
            def __init__(self,*_):self.index=0
            def read(self):
                if self.index==len(frames):return None
                frame=frames[self.index];self.index+=1;return frame
            def close(self):closed.append('decoder')
        class Encoder:
            def __init__(self,*_):
                if broken_encoder:raise RuntimeError('Injected fake encoder failure')
            def write(self,frame):
                time.sleep(.002);encoded.append(frame.index);return 0.,0.
            def close(self):closed.append('encoder');return 0.
        class Tracker(ScriptTracker):
            def update(self,*values):
                if slow_track:time.sleep(.03)
                return super().update(*values)
        class Fusion(p.Fusion):
            def __init__(self,detector,tracker,a):
                super().__init__(detector,Tracker(),a)
                self.state='TRACK';self.generation=1;self.ever_locked=True
                self.kalman=p.BoxKalman([40,60,20,10]);self.last_yolo=0
        config=args(video='fake',output=str(Path(directory)/'result.mp4'),gt=None,
                    template_model='fake-template',search_model='fake-search',yolo_model='fake-yolo',
                    fear_core='1',yolo_core='0',decode_queue=2,encode_queue=1,
                    no_encode=False,no_video=False,source_fps=20.,bitrate=1000000,debug=False,max_frames=0,
                    decode_cpu=-1,track_cpu=-1,encode_cpu=-1,yolo_cpu=-1,yolo_second_cpu=-1,yolo_workers=2)
        with patch.object(p,'NativeFear',Runtime),patch.object(p,'NativeYolo',Runtime),\
             patch.object(p.hw,'Decoder',Decoder),patch.object(p.hw,'Encoder',Encoder),\
             patch.object(p,'Fusion',Fusion),patch.object(p,'sha256',return_value='fake'),\
             patch.object(p,'postprocess',return_value=[{'bbox':[40,60,20,10],'score':.95}]),\
             patch.object(p.cv2,'cvtColor'),patch.object(p.cv2,'resize'):
            with contextlib.redirect_stdout(io.StringIO()):
                summary=p.run(config)
        return summary,encoded,closed,frames

    def test_bounded_full_pipeline_preserves_every_frame_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            summary,encoded,closed,frames=self.run_fake(directory)
            self.assertEqual(summary['frames'],12);self.assertEqual(summary['encoded_frames'],12)
            self.assertEqual(encoded,list(range(12)))
            self.assertIn('encoder',closed);self.assertIn('decoder',closed)
            for frame in frames:self.assertTrue(np.all(frame.image==frame.index),'Draw thread mutated decoded source frame')
            with (Path(directory)/'result.csv').open() as stream:
                self.assertEqual(sum(1 for _ in stream),13)

    def test_encode_error_joins_tracking_before_releasing_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            try:
                with self.assertRaisesRegex(RuntimeError,'Injected fake encoder failure'):
                    self.run_fake(directory,count=50,broken_encoder=True,slow_track=True)
                alive=[thread for thread in threading.enumerate() if thread.name in ('MPP decode','FEAR Kalman','draw MPP encode') or thread.name.startswith('YOLO')]
                self.assertEqual(alive,[],'Pipeline returned/destroyed handles while a worker was still active')
            finally:
                # Cleanup a failing implementation's short-lived fake threads,
                # so this diagnostic does not interfere with another test.
                for thread in threading.enumerate():
                    if thread.name in ('MPP decode','FEAR Kalman','draw MPP encode') or thread.name.startswith('YOLO'):thread.join(1)


class TwoWorkerTests(unittest.TestCase):
    def worker(self):
        errors=queue.Queue();runtimes=[FakeRuntime(),FakeRuntime()]
        detector=MultiDetector(runtimes,args(),errors,postprocess=lambda *args:[])
        blocks=[BlockingFrame(0),BlockingFrame(1)]
        for frame in blocks:detector.submit(frame,frame.index,0)
        for frame in blocks:self.assertTrue(frame.entered.wait(1),'Independent detector context failed to start its job')
        return detector,blocks,errors

    def test_two_contexts_have_simultaneous_distinct_inflight_jobs(self):
        detector,blocks,errors=self.worker()
        try:
            with detector.cv:
                self.assertEqual(detector.inflight,{(0,0),(0,1)})
                self.assertEqual(len(detector.threads),2)
            detector.submit(blocks[0],0,0)
            with detector.cv:self.assertEqual(len(detector.pending),0,'Same job scheduled twice across workers')
        finally:
            for frame in blocks:frame.release.set()
            detector.close()
        self.assertTrue(errors.empty());self.assertEqual(sorted(row['index'] for row in detector.calls),[0,1])
        self.assertEqual(detector.calls_by_worker,[1,1])
        self.assertAlmostEqual(detector.cpu_seconds,sum(detector.cpu_by_worker),places=9)

    def test_close_joins_both_independent_workers(self):
        detector,blocks,errors=self.worker()
        closer=threading.Thread(target=detector.close)
        closer.start();time.sleep(.01)
        try:self.assertTrue(closer.is_alive(),'close returned while both fake inference jobs were still blocked')
        finally:
            for frame in blocks:frame.release.set()
            closer.join(1)
        self.assertFalse(closer.is_alive());self.assertFalse(any(thread.is_alive() for thread in detector.threads))
        self.assertTrue(errors.empty())

    def test_out_of_order_completed_batch_is_consumed_in_source_order(self):
        detector,blocks,errors=self.worker()
        try:
            with detector.cv:
                detector.results[(1,5)]={'index':5,'generation':1}
                detector.results[(1,3)]={'index':3,'generation':1}
            self.assertEqual([row['index'] for row in detector.take(6,1)],[3,5])
            self.assertEqual(detector.take(6,1),[])
        finally:
            for frame in blocks:frame.release.set()
            detector.close()
        self.assertTrue(errors.empty())

    def test_sharing_one_context_between_workers_is_rejected(self):
        runtime=FakeRuntime()
        with self.assertRaisesRegex(ValueError,'distinct runtime'):
            MultiDetector([runtime,runtime],args(),queue.Queue(),postprocess=lambda *args:[])


if __name__=='__main__':unittest.main(verbosity=2)
