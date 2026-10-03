"""Build reproducible real-video calibration; no random feature input."""
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from export_models import create_runtime

# extended_crop/raw_nchw live in the board runtime package (single copy in the repo).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tracker"))
from tracker_core import extended_crop, raw_nchw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--tracks", required=True, help="Demo metrics JSON, for selecting real object crops")
    p.add_argument("--weights", required=True)
    p.add_argument("--source-dir", default="upstream")
    p.add_argument("--output", default="calibration")
    p.add_argument("--samples", type=int, default=100)
    args = p.parse_args()
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=True)
    tracks = json.loads(Path(args.tracks).read_text())["frames"]
    selected = set(np.linspace(1, len(tracks)-1, args.samples, dtype=int).tolist())
    rt = create_runtime(args.weights, args.source_dir)
    cap = cv2.VideoCapture(args.video)
    templates, searches, manifest = [], [], []
    mean_color = None
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if mean_color is None:
            mean_color = np.mean(rgb, axis=(0,1))
        if index in selected:
            z = raw_nchw(extended_crop(rgb, tracks[index]["bbox"], 128, .2)[0])
            x = raw_nchw(extended_crop(rgb, tracks[index-1]["bbox"], 256, 2., mean_color)[0])
            f = rt.template(z)
            zp, xp, fp = [out / f"{index:06d}_{s}.npy" for s in ("template","search","feature")]
            for path, array in [(zp,z),(xp,x),(fp,f)]: np.save(path,array)
            templates.append(str(zp)); searches.append(str(xp)+" "+str(fp))
            manifest.append({"id":f"official_{index}","template":str(zp),"search":str(xp)})
        index += 1
    cap.release()
    (out / "template.txt").write_text("\n".join(templates)+"\n")
    (out / "search.txt").write_text("\n".join(searches)+"\n")
    (out / "manifest.json").write_text(json.dumps({"samples":manifest},indent=2))
    (out / "provenance.json").write_text(json.dumps({"video":args.video,"tracks":args.tracks,
        "samples":len(manifest),"indices":sorted(selected),"feature_backend":"original pretrained PyTorch",
        "note":"Object crops selected by FP16 demo trajectory; changing-view templates from each selected frame. Validation UAV clip must be held out."},indent=2))
    print(f"Saved {len(manifest)} calibration pairs",flush=True)


if __name__ == "__main__": main()
