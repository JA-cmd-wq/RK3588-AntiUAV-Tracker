#!/usr/bin/env python3
"""Read-only board NPU load sampler; run separately during pipeline benchmarks.

The debugfs load uses the driver's averaging window. It is utilization, distinct
from FPS and RKNN call latency. No inference is issued to inflate utilization.
"""
import argparse,csv,re,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--stop-file',required=True,type=Path);p.add_argument('--duration',type=float,default=120);a=p.parse_args()
a.output.parent.mkdir(parents=True,exist_ok=True)
start=time.monotonic()
with a.output.open('w') as f:
 w=csv.DictWriter(f,fieldnames=['unix_time','core0_percent','core1_percent','core2_percent','frequency_hz']);w.writeheader()
 while time.monotonic()-start<a.duration and not a.stop_file.exists():
  raw=Path('/sys/kernel/debug/rknpu/load').read_text();values=dict((int(k),int(v)) for k,v in re.findall(r'Core(\d+):\s*(\d+)%',raw))
  freq=Path('/sys/class/devfreq/fdab0000.npu/cur_freq').read_text().strip()
  w.writerow({'unix_time':time.time(),**{f'core{k}_percent':values.get(k) for k in range(3)},'frequency_hz':freq});f.flush();time.sleep(.1)
