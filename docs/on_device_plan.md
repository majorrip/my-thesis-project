# On-Device / Low-Power Study — Plan and Pre-Registration

*Recorded before running, so the evaluation criteria are fixed in advance.*

## Goal
Deliver the "low-power on-device" contribution the thesis motivates, using real
hardware (an Android phone + a laptop) rather than a datacenter GPU. Two parts:

## Part A — Memory-bounded training ("firebreak")
**Claim under test:** layer-wise local training (with `detach()` between encoder
blocks) keeps peak *training* memory approximately constant as encoder depth
grows, whereas end-to-end backprop's peak memory grows with depth — enabling
training of deeper encoders within a fixed on-device memory budget.

- **Script:** `code/evaluate_memory.py` (run on Colab GPU for real
  `torch.cuda` peak memory; laptop/CPU reports an analytical estimate).
- **Metric:** peak training memory vs. depth for `e2e` and `local`; plus the
  effective rank of the learned embedding (to confirm local training still
  yields structured, non-collapsed features).
- **Success criterion (pre-registered):** at the largest depth that `e2e` can
  train, `local` uses substantially less peak memory (target: a clear,
  monotonic gap that widens with depth), while its embedding effective rank
  stays well above 1 (not collapsed).

## Part B — Real on-phone inference
**Claim under test:** the trained encoder runs on a real Android phone at
practical latency and memory.

- **Scripts:** `code/export_onnx.py` (produces `encoder.onnx`) and
  `code/bench_phone.py` (run on the phone via Termux with ONNX Runtime).
- **Metrics:** single-image latency (mean/std/p90), throughput, model size on
  disk, peak process memory; optional rough energy via battery-drain over a
  long run.
- **Success criterion (pre-registered):** the encoder loads and runs on the
  phone; we report the measured latency/memory as-is (this part is a
  measurement, not a pass/fail hypothesis).

## Integrity rules
- Decide criteria before running (done, above). Report all configs and seeds.
- Any hyperparameters are tuned on a validation split, never the test set.
- Negative or null results are reported plainly.
