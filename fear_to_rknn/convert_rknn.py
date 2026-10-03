"""Convert fixed FEAR-XS ONNX. Run only on RK3588/Linux with Toolkit2."""
import argparse
import pathlib
from rknn.api import RKNN


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="models")
    p.add_argument("--precision", choices=["fp16", "int8"], default="fp16")
    p.add_argument("--branch", choices=["template", "search", "both"], default="both")
    p.add_argument("--calibration", default="calibration")
    p.add_argument("--optimization-level", type=int, default=3)
    p.add_argument("--search-stem", default="search", help="ONNX/model basename for a validated search variant")
    p.add_argument("--template-stem", default="template")
    p.add_argument("--normalize-input", action="store_true", help="Use with *_npu.onnx: RGB raw input normalized by RKNN config")
    args = p.parse_args()
    root = pathlib.Path(args.models)
    for branch in (["template", "search"] if args.branch == "both" else [args.branch]):
        stem = args.search_stem if branch == "search" else args.template_stem
        dataset = str(pathlib.Path(args.calibration) / (branch+".txt"))
        if args.precision == "int8" and not pathlib.Path(dataset).is_file():
            raise FileNotFoundError(dataset)
        model = RKNN(verbose=True, verbose_file=str(root / (stem+"_"+args.precision+"_verbose.log")))
        try:
            # Images enter as RGB float32 0..255; normalization is in ONNX.
            # Feature input must stay unnormalized; do not add mean/std here.
            config = dict(target_platform="rk3588", optimization_level=args.optimization_level,
                          quantized_dtype="asymmetric_quantized-8", quantized_algorithm="normal")
            if args.normalize_input:
                config["mean_values"]=[[123.675,116.28,103.53]]
                config["std_values"]=[[58.395,57.12,57.375]]
                if branch == "search":
                    config["mean_values"].append([0]*256)
                    config["std_values"].append([1]*256)
            model.config(**config)
            assert model.load_onnx(model=str(root / (stem+".onnx"))) == 0
            assert model.build(do_quantization=args.precision == "int8",
                               dataset=dataset if args.precision == "int8" else None) == 0
            assert model.export_rknn(str(root / (stem+"_"+args.precision+".rknn"))) == 0
        finally:
            model.release()


if __name__ == "__main__":
    main()
