"""
Export the trained context encoder to ONNX for on-device (phone) benchmarking.

Produces ``encoder.onnx`` containing the context encoder (the part that runs at
inference time on the edge device). By default it exports a randomly-initialized
CNN encoder; pass a checkpoint to export trained weights.
"""

import argparse
import torch
from model import VisionJEPA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="cnn", choices=["cnn", "resnet18"])
    ap.add_argument("--ckpt", default=None, help="optional state_dict for the full VisionJEPA")
    ap.add_argument("--out", default="encoder.onnx")
    args = ap.parse_args()

    model = VisionJEPA(backbone=args.backbone)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    encoder = model.context_encoder.eval()

    dummy = torch.rand(1, 3, 64, 64)
    torch.onnx.export(
        encoder, dummy, args.out,
        input_names=["image"], output_names=["embedding"],
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    n = sum(p.numel() for p in encoder.parameters())
    print(f"exported {args.out}  |  backbone={args.backbone}  params={n:,}")


if __name__ == "__main__":
    main()
