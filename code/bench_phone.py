"""
On-device (phone) inference benchmark for the exported encoder.

Runs ``encoder.onnx`` with ONNX Runtime and reports single-image inference
latency, throughput, model size, and peak process memory. Designed to run on an
Android phone via Termux (real ARM-CPU edge numbers), and works on a laptop too.

--- HOW TO RUN ON AN ANDROID PHONE (Termux) ---
  1. Install the Termux app (F-Droid build recommended).
  2. In Termux:
        pkg update && pkg install python
        pip install onnxruntime numpy
  3. Copy encoder.onnx and this script into Termux storage
     (e.g. `termux-setup-storage` then place them under ~/storage/shared, or scp).
  4. Run:
        python bench_phone.py --onnx encoder.onnx --runs 200
  5. Record the printed latency / memory. For a rough energy figure, note the
     battery percentage before and after a long run (e.g. --runs 20000).
"""

import argparse
import os
import time

import numpy as np


def peak_rss_mb():
    try:
        import resource
        # ru_maxrss is KB on Linux/Android, bytes on macOS
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / 1024.0 if kb > 1e6 else kb / 1024.0
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="encoder.onnx")
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--threads", type=int, default=0, help="0 = ORT default")
    args = ap.parse_args()

    import onnxruntime as ort
    so = ort.SessionOptions()
    if args.threads > 0:
        so.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(args.onnx, sess_options=so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    x = np.random.rand(1, 3, 64, 64).astype(np.float32)
    for _ in range(args.warmup):
        sess.run(None, {iname: x})

    lat = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        sess.run(None, {iname: x})
        lat.append((time.perf_counter() - t0) * 1000.0)
    lat = np.array(lat)

    size_mb = os.path.getsize(args.onnx) / (1024 ** 2)
    print(f"model            : {args.onnx} ({size_mb:.2f} MB on disk)")
    print(f"runs             : {args.runs} (warmup {args.warmup})")
    print(f"latency (ms)     : mean {lat.mean():.2f}  std {lat.std():.2f}  "
          f"p50 {np.percentile(lat,50):.2f}  p90 {np.percentile(lat,90):.2f}")
    print(f"throughput (img/s): {1000.0/lat.mean():.1f}")
    print(f"peak process RSS : {peak_rss_mb():.1f} MB")


if __name__ == "__main__":
    main()
