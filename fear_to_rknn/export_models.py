"""Fixed-shape FEAR-XS inference, preserving every pretrained operation/weight."""
import argparse
import collections
import collections.abc
import json
import pathlib
import sys
import numpy as np
import torch
from torch import nn


class FEARInference(nn.Module):
    def __init__(self, source_dir):
        super().__init__()
        sys.path.insert(0, str(pathlib.Path(source_dir).resolve()))
        # Upstream mobile-vision targets Python 3.7; this is an import compatibility
        # alias only and has no effect on model tensors or operations.
        for name in ("Mapping", "MutableMapping", "Sequence", "Iterable"):
            if not hasattr(collections, name):
                setattr(collections, name, getattr(collections.abc, name))
        from model_training.model.blocks import Encoder, AdjustLayer, BoxTower
        self.encoder = Encoder(pretrained=False)
        self.neck = AdjustLayer(112, 256)
        self.connect_model = BoxTower(inchannels=256, outchannels=256, towernum=2,
                                      conv_block="sep_conv", mobile=True)

    def get_features(self, x):
        for stage in self.encoder.stages[:4]:
            x = stage(x)
        return self.neck(x)


def load_model(weights_path, source_dir=None, variant="original"):
    if variant != "original":
        raise ValueError("No unvalidated approximate operator variant is provided")
    source_dir = source_dir or pathlib.Path(__file__).parent / "upstream"
    model = FEARInference(source_dir)
    checkpoint = torch.load(weights_path, map_location="cpu")
    state = {k[len("model."):]: v for k, v in checkpoint["state_dict"].items()
             if k.startswith("model.")}
    model.load_state_dict(state, strict=True)
    return model.eval()


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, x):
        return (x / 255.0 - self.mean) / self.std


class TemplateWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model, self.normalize = model, Normalize()

    def forward(self, template):
        return self.model.get_features(self.normalize(template))


class SearchWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model, self.normalize = model, Normalize()

    def forward(self, search, template_features):
        features = self.model.get_features(self.normalize(search))
        bbox, cls, _, _ = self.model.connect_model(features, template_features)
        return bbox, cls


class TorchRuntime:
    def __init__(self, model):
        self.template_model = TemplateWrapper(model).eval()
        self.search_model = SearchWrapper(model).eval()

    @torch.inference_mode()
    def template(self, raw):
        return self.template_model(torch.from_numpy(raw)).numpy()

    @torch.inference_mode()
    def search(self, raw, features):
        return [x.numpy() for x in self.search_model(torch.from_numpy(raw), torch.from_numpy(features))]


def create_runtime(weights_path, source_dir=None, variant="original"):
    torch.set_num_threads(4)
    return TorchRuntime(load_model(weights_path, source_dir, variant))


def metric(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return {"cosine": float(np.vdot(a.ravel(), b.ravel()) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30)),
            "max_abs": float(np.max(np.abs(a-b))), "mean_abs": float(np.mean(np.abs(a-b))),
            "relative_l2": float(np.linalg.norm(a-b) / max(np.linalg.norm(a), 1e-30))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source-dir", default="upstream")
    p.add_argument("--output", default="models")
    args = p.parse_args()
    torch.set_num_threads(4)
    out = pathlib.Path(args.output); out.mkdir(parents=True, exist_ok=True)
    model = load_model(args.weights, args.source_dir)
    tm, sm = TemplateWrapper(model).eval(), SearchWrapper(model).eval()
    rng = np.random.default_rng(20261002)
    z = torch.from_numpy(rng.integers(0, 256, (1,3,128,128)).astype("float32"))
    x = torch.from_numpy(rng.integers(0, 256, (1,3,256,256)).astype("float32"))
    with torch.inference_mode():
        f = tm(z); y = sm(x, f)
    f = f.clone()  # ONNX tracing uses autograd bookkeeping; leave inference-mode storage.
    torch.onnx.export(tm, (z,), out / "template.onnx", opset_version=12,
                      input_names=["template"], output_names=["template_features"], do_constant_folding=True)
    torch.onnx.export(sm, (x, f), out / "search.onnx", opset_version=12,
                      input_names=["search", "template_features"], output_names=["bbox", "cls"], do_constant_folding=True)
    import onnx, onnxruntime as ort
    report = {}
    for name, inputs, truth in [("template", {"template":z.numpy()}, [f.numpy()]),
                                ("search", {"search":x.numpy(), "template_features":f.numpy()}, [a.numpy() for a in y])]:
        graph = onnx.load(out / (name + ".onnx")); onnx.checker.check_model(graph)
        opt = ort.SessionOptions(); opt.intra_op_num_threads = 4
        sess = ort.InferenceSession(str(out / (name + ".onnx")), opt, providers=["CPUExecutionProvider"])
        pred = sess.run(None, inputs)
        report[name] = {"ops":dict(collections.Counter(n.op_type for n in graph.graph.node)),
                        "inputs": {a.name:[d.dim_value for d in a.type.tensor_type.shape.dim] for a in graph.graph.input},
                        "outputs": {a.name:[d.dim_value for d in a.type.tensor_type.shape.dim] for a in graph.graph.output},
                        "pytorch_onnx": [metric(a,b) for a,b in zip(truth,pred)]}
        for key,value in inputs.items():
            np.save(out / (name+"_"+key+".npy"),value)
            value.tofile(out / (name+"_"+key+".bin"))
        for i,a in enumerate(truth):
            np.save(out / (name+"_reference_"+str(i)+".npy"),a)
    (out / "export_report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
