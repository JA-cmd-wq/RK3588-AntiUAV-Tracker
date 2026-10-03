"""Replace Exp(x) = 1/sigmoid(-x)-1, with original FP32 ONNX comparison.

The identity is exact in real arithmetic, but finite precision still requires
RK3588 validation. The original model is never overwritten.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort
from export_models import metric


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="models/search.onnx")
    p.add_argument("--output",default="models/search_sigmoid.onnx")
    p.add_argument("--manifest",default="calibration/manifest.json")
    p.add_argument("--limit",type=int,default=20)
    a=p.parse_args()
    graph=onnx.load(a.input); nodes=[]; count=0
    for node in graph.graph.node:
        if node.op_type != "Exp":
            nodes.append(node); continue
        count+=1
        prefix=node.name+"_sigmoid_identity"
        one=prefix+"_one"
        graph.graph.initializer.append(numpy_helper.from_array(np.ones((1,),dtype=np.float32),one))
        for op,inp,out in [("Neg",[node.input[0]],prefix+"_neg"),
                           ("Sigmoid",[prefix+"_neg"],prefix+"_sigmoid"),
                           ("Reciprocal",[prefix+"_sigmoid"],prefix+"_reciprocal"),
                           ("Sub",[prefix+"_reciprocal",one],node.output[0])]:
            nodes.append(helper.make_node(op,inp,[out],name=prefix+"_"+op.lower()))
    if count!=1: raise ValueError(f"Expected one Exp, got {count}")
    del graph.graph.node[:]; graph.graph.node.extend(nodes)
    onnx.checker.check_model(graph); onnx.save(graph,a.output)
    opt=ort.SessionOptions();opt.intra_op_num_threads=4
    old=ort.InferenceSession(a.input,opt,providers=["CPUExecutionProvider"])
    new=ort.InferenceSession(a.output,opt,providers=["CPUExecutionProvider"])
    manifest=json.loads(Path(a.manifest).read_text())["samples"]
    reports=[]
    template=ort.InferenceSession(str(Path(a.input).parent/"template.onnx"),opt,providers=["CPUExecutionProvider"])
    for sample in manifest[:a.limit]:
        z=np.load(sample["template"]);x=np.load(sample["search"])
        f=template.run(None,{"template":z})[0]
        inputs={"search":x,"template_features":f}
        aa,bb=old.run(None,inputs),new.run(None,inputs)
        reports.append({"id":sample["id"],"outputs":[metric(v,w) for v,w in zip(aa,bb)]})
    report={"identity":"Exp(x) = Reciprocal(Sigmoid(Neg(x))) - 1", "exp_replaced":count,
            "weights_changed":False,"activation_weights_changed":False,"samples":reports}
    Path(a.output).with_suffix(".parity.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({"count":len(reports),"bbox_max_abs":max(r["outputs"][0]["max_abs"] for r in reports),
                     "bbox_min_cosine":min(r["outputs"][0]["cosine"] for r in reports)},indent=2),flush=True)


if __name__=="__main__": main()
