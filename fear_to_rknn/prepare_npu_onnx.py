"""Exact graph boundaries: input normalization in RKNN config, Exp in CPU decode."""
import argparse
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from export_models import metric

MEAN=np.array([.485,.456,.406],np.float32).reshape(1,3,1,1)
STD=np.array([.229,.224,.225],np.float32).reshape(1,3,1,1)


def rewrite(path,output,search=False):
    graph=onnx.load(path)
    normalize=[n for n in graph.graph.node if n.name.startswith("/normalize/")]
    if not normalize: raise ValueError("Missing known normalization subgraph")
    last=normalize[-1].output[0]
    # Only the ImageNet normalization output is replaced; image/feature weights stay identical.
    for node in graph.graph.node:
        for i,name in enumerate(node.input):
            if name==last: node.input[i]=graph.graph.input[0].name
    if search:
        exp=[n for n in graph.graph.node if n.op_type=="Exp"]
        if len(exp)!=1: raise ValueError("Expected one bbox Exp")
        graph.graph.output[0].name=exp[0].input[0]
    required={o.name for o in graph.graph.output};keep=[]
    for node in reversed(graph.graph.node):
        if required.intersection(node.output):
            keep.append(node);required.update(node.input)
    del graph.graph.node[:];graph.graph.node.extend(reversed(keep))
    used={name for n in graph.graph.node for name in n.input}
    initializers=[n for n in graph.graph.initializer if n.name in used]
    del graph.graph.initializer[:];graph.graph.initializer.extend(initializers)
    del graph.graph.value_info[:]
    onnx.checker.check_model(graph);onnx.save(graph,output)


def main():
    p=argparse.ArgumentParser();p.add_argument("--models",default="models")
    p.add_argument("--manifest",default="calibration/manifest.json");p.add_argument("--limit",type=int,default=20)
    a=p.parse_args();root=Path(a.models)
    for name in ("template","search"):rewrite(root/(name+".onnx"),root/(name+"_npu.onnx"),name=="search")
    opt=ort.SessionOptions();opt.intra_op_num_threads=4
    sessions={name:ort.InferenceSession(str(root/(name+".onnx")),opt,providers=["CPUExecutionProvider"])
              for name in ("template","search","template_npu","search_npu")}
    samples=json.loads(Path(a.manifest).read_text())["samples"][:a.limit];reports=[]
    for sample in samples:
        z=np.load(sample["template"]);x=np.load(sample["search"])
        f=sessions["template"].run(None,{"template":z})[0]
        fn=sessions["template_npu"].run(None,{"template":(z/255-MEAN)/STD})[0]
        yy=sessions["search"].run(None,{"search":x,"template_features":f})
        yn=sessions["search_npu"].run(None,{"search":(x/255-MEAN)/STD,"template_features":f})
        yn[0]=np.exp(yn[0])
        reports.append({"id":sample["id"],"template":metric(f,fn),"bbox":metric(yy[0],yn[0]),"cls":metric(yy[1],yn[1])})
    result={"transform":"ImageNet normalize in RKNN input config; bbox Exp in CPU postprocess",
            "weights_changed":False,"learned_activation_changed":False,"samples":reports}
    (root/"npu_boundary_parity.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:{"min_cosine":min(r[k]["cosine"] for r in reports),
                         "max_abs":max(r[k]["max_abs"] for r in reports)} for k in ("template","bbox","cls")},indent=2))


if __name__=="__main__":main()
